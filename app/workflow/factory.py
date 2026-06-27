"""Assemble a :class:`DurableWorkflow` from its dependencies.

Shared by the workflow route (``app.api.routes.workflow``) and the concierge
bridge (``app.agents.concierge.service``) so both wire the multi-agent graph and
its compliance/HITL gating identically — one definition of "how the workflow is
built," fed from ``app.state`` in either entry point.
"""

from __future__ import annotations

from app.core.audit import AuditSink
from app.core.config import settings
from app.domain.adapters import DomainProviders, default_providers
from app.llm.router import LLMRouter
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import VectorStore
from app.workflow.checkpointer import WorkflowCheckpointer
from app.workflow.engine import StateMachineEngine
from app.workflow.orchestrator import DurableWorkflow, WorkflowDeps
from app.workflow.review_queue import ReviewQueue


def build_workflow(
    *,
    llm_router: LLMRouter,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    audit_sink: AuditSink | None = None,
    providers: DomainProviders | None = None,
    checkpointer: WorkflowCheckpointer | None = None,
    review_queue: ReviewQueue | None = None,
) -> DurableWorkflow:
    """Build the durable multi-agent workflow with compliance + HITL gating from config."""
    return DurableWorkflow(
        engine=StateMachineEngine(),
        deps=WorkflowDeps(
            providers=providers or default_providers(),
            llm_router=llm_router,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            audit_sink=audit_sink,
            compliance_critique_max_rounds=settings.compliance_critique_max_rounds,
            compliance_explicit_enabled=settings.compliance_explicit_enabled,
            checkpointer=checkpointer,
            review_queue=review_queue,
            high_risk_areas=settings.workflow_high_risk_areas_set,
        ),
    )
