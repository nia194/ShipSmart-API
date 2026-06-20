"""
Workflow engine (UC3, D-engine) — hand-rolled, deterministic, no framework.

The orchestrator depends on the ``WorkflowEngine`` **Protocol**, never on a
concrete engine. ``StateMachineEngine`` is the hand-rolled implementation: it
runs nodes (plain ``async`` functions) sequentially, and forks independent nodes
in parallel via ``asyncio.gather`` with a **deterministic merge** so output is
reproducible regardless of completion order. Control flow is always the code's —
never a model's.

A ``LangGraphEngine`` (or Temporal, etc.) could be added later as an alternative
adapter behind the same Protocol with zero change to agents, nodes, or routes —
this is the "engine fungibility" seam (and why ``langgraph`` is not a dependency).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from app.workflow.state import WorkflowState, _now

# A node is a pure-ish step: state in, new state out (side effects at the edges).
NodeFn = Callable[[WorkflowState], Awaitable[WorkflowState]]


@runtime_checkable
class WorkflowEngine(Protocol):
    """Port for executing workflow steps."""

    async def run_step(self, state: WorkflowState, node: NodeFn) -> WorkflowState: ...

    async def run_parallel(
        self, state: WorkflowState, nodes: list[NodeFn],
    ) -> WorkflowState: ...


class StateMachineEngine:
    """Deterministic, hand-rolled execution of workflow nodes."""

    async def run_step(self, state: WorkflowState, node: NodeFn) -> WorkflowState:
        return await node(state)

    async def run_parallel(
        self, state: WorkflowState, nodes: list[NodeFn],
    ) -> WorkflowState:
        """Run ``nodes`` concurrently on independent copies, then merge in order.

        Each node sees the same base state (a deep copy) so they cannot interfere.
        Results are merged in the FIXED order of ``nodes`` — changed non-decision
        fields are copied over, and each node's newly appended decisions are
        concatenated in that order — so the merged state is deterministic no
        matter which task finishes first. Nodes are expected to write disjoint
        fields (e.g. landed-cost vs. routing).
        """
        if not nodes:
            return state

        base = state
        results = await asyncio.gather(
            *(node(base.model_copy(deep=True)) for node in nodes)
        )

        merged = base.model_copy(deep=True)
        base_decisions = len(base.decisions)
        for result in results:  # deterministic: the order of `nodes`
            for field in WorkflowState.model_fields:
                if field == "decisions":
                    continue
                new_value = getattr(result, field)
                if new_value != getattr(base, field):
                    setattr(merged, field, new_value)
            merged.decisions.extend(result.decisions[base_decisions:])

        merged.updated_at = _now()
        return merged
