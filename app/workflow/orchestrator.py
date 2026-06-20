"""
Workflow orchestrator (UC3) — the multi-agent stage graph.

``DurableWorkflow`` wires the specialist agents into the fixed stage sequence and
runs them through the injected ``WorkflowEngine``:

    classify → (landed-cost ‖ routing) → compliance (+UC2 critic) → documentation

Landed-cost and routing are independent, so they run in parallel; everything else
is sequential. Control flow lives here in plain, auditable Python — never in a
model. The branch/interrupt for unverified-high-risk shipments and the durable
checkpointer are Phase 3 (UC4); this phase runs straight through to ``completed``
(the seam is marked below).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.audit import AuditSink
from app.domain.adapters import DomainProviders
from app.llm.router import LLMRouter
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import VectorStore
from app.workflow.engine import WorkflowEngine
from app.workflow.nodes import (
    classification_node,
    compliance_node,
    documentation_node,
    landed_cost_node,
    routing_node,
)
from app.workflow.state import WorkflowState, _now


@dataclass(frozen=True)
class WorkflowDeps:
    """Everything the workflow needs, injected (ports + LLM/RAG + audit)."""

    providers: DomainProviders
    llm_router: LLMRouter
    embedding_provider: EmbeddingProvider
    vector_store: VectorStore
    audit_sink: AuditSink | None = None
    compliance_critique_max_rounds: int | None = None


class DurableWorkflow:
    """Sequences the specialist agents into the workflow graph via the engine."""

    def __init__(self, *, engine: WorkflowEngine, deps: WorkflowDeps) -> None:
        self._engine = engine
        self._classify = classification_node(deps.providers.classification)
        self._landed_cost = landed_cost_node(deps.providers.duty)
        self._routing = routing_node(deps.providers.carrier)
        self._compliance = compliance_node(
            llm_router=deps.llm_router,
            embedding_provider=deps.embedding_provider,
            vector_store=deps.vector_store,
            audit_sink=deps.audit_sink,
            critique_max_rounds=deps.compliance_critique_max_rounds,
        )
        self._documentation = documentation_node(deps.providers.doc_renderer)

    async def process(self, state: WorkflowState) -> WorkflowState:
        """Run the full graph and return the finished state."""
        state.status = "running"
        state.decisions.append("workflow:start")

        state = await self._engine.run_step(state, self._classify)
        state = await self._engine.run_parallel(state, [self._landed_cost, self._routing])
        state = await self._engine.run_step(state, self._compliance)

        # ── Phase 3 (UC4) seam ────────────────────────────────────────────────
        # The unverified-high-risk interrupt (suspend → human review → resume) is
        # inserted here, before documentation. Phase 2 runs straight through.

        state = await self._engine.run_step(state, self._documentation)

        state.status = "completed"
        state.decisions.append("workflow:complete")
        state.updated_at = _now()
        return state
