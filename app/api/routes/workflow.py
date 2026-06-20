"""
Workflow route (UC3) — the multi-agent processing endpoint.

``POST /api/v1/workflow/process`` runs a shipment through the full stage graph
(classify → landed-cost ‖ routing → compliance(+UC2) → documentation) and returns
the finished state with its decision trail.

Like the compliance route it needs only ``llm_router`` + ``rag`` on ``app.state``
(503 if missing) — the compliance stage uses them; the other stages use the
deterministic mock domain adapters (taken from ``app.state.domain`` when wired,
else built on demand). 404 when the workflow feature is disabled.

The durable lifecycle (``GET /workflow/{id}``, ``POST /workflow/{id}/review``) is
Phase 3 (UC4); this phase exposes ``process`` only, which runs to completion.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request

from app.core.config import settings
from app.core.errors import AppError
from app.core.rate_limit import limiter
from app.domain.adapters import DomainProviders, default_providers
from app.llm.router import LLMRouter
from app.schemas.workflow import WorkflowProcessRequest, WorkflowResponse
from app.workflow.engine import StateMachineEngine
from app.workflow.orchestrator import DurableWorkflow, WorkflowDeps
from app.workflow.state import WorkflowState

router = APIRouter(prefix="/workflow", tags=["workflow"])


@router.post("/process", response_model=WorkflowResponse)
@limiter.limit(settings.rate_limit_workflow)
async def process_workflow(
    body: WorkflowProcessRequest, request: Request,
) -> WorkflowResponse:
    """Run a shipment through the multi-agent workflow and return the result."""
    if not settings.workflow_enabled:
        raise AppError(status_code=404, message="Workflow endpoint is disabled")

    llm_router: LLMRouter | None = getattr(request.app.state, "llm_router", None)
    if llm_router is None:
        raise AppError(status_code=503, message="LLM router is not initialized")

    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        raise AppError(status_code=503, message="RAG pipeline is not initialized")

    providers: DomainProviders = (
        getattr(request.app.state, "domain", None) or default_providers()
    )

    state = WorkflowState(
        workflow_id=uuid.uuid4().hex,
        request_id=getattr(request.state, "request_id", ""),
        origin_country=body.origin_country,
        destination_country=body.destination_country,
        declared_value_usd=body.declared_value_usd,
        weight_lbs=body.weight_lbs,
        description=body.description,
        category=body.category,
    )

    workflow = DurableWorkflow(
        engine=StateMachineEngine(),
        deps=WorkflowDeps(
            providers=providers,
            llm_router=llm_router,
            embedding_provider=rag["embedding_provider"],
            vector_store=rag["vector_store"],
            audit_sink=getattr(request.app.state, "audit_sink", None),
            compliance_critique_max_rounds=settings.compliance_critique_max_rounds,
        ),
    )

    result = await workflow.process(state)
    return WorkflowResponse.from_state(result)
