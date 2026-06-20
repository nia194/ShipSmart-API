"""
Workflow orchestration + durability (UC3/UC4).

A deterministic, hand-rolled state machine (a ``WorkflowEngine`` Protocol +
``StateMachineEngine``) sequences the specialist agents; durability and the
human-in-the-loop interrupt/resume (checkpointer + review-queue ports) wrap the
graph. The engine is swappable — a LangGraph adapter could be added later behind
the same Protocol — but control flow stays deterministic code, never a model.

Phase 2 (UC3) populates the engine, nodes, orchestrator, and state; Phase 3 (UC4)
adds the checkpointer + review-queue ports and the interrupt/resume lifecycle.
"""

from __future__ import annotations

from app.workflow.checkpointer import (
    InMemoryCheckpointer,
    SqliteCheckpointer,
    WorkflowCheckpointer,
    create_checkpointer,
)
from app.workflow.engine import StateMachineEngine, WorkflowEngine
from app.workflow.orchestrator import DurableWorkflow, WorkflowDeps
from app.workflow.review_queue import (
    InMemoryReviewQueue,
    ReviewItem,
    ReviewQueue,
)
from app.workflow.state import ComplianceSummary, WorkflowState, WorkflowStatus

__all__ = [
    "ComplianceSummary",
    "DurableWorkflow",
    "InMemoryCheckpointer",
    "InMemoryReviewQueue",
    "ReviewItem",
    "ReviewQueue",
    "SqliteCheckpointer",
    "StateMachineEngine",
    "WorkflowCheckpointer",
    "WorkflowDeps",
    "WorkflowEngine",
    "WorkflowState",
    "WorkflowStatus",
    "create_checkpointer",
]
