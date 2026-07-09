"""
Workflow route (UC3 + UC4) — process, inspect, and review.

- ``POST /api/v1/workflow/process`` runs a shipment through the stage graph
  (classify → landed-cost ‖ routing → compliance(+UC2) → documentation). If a
  high-risk area can't be verified it suspends (``awaiting_review``).
- ``GET  /api/v1/workflow/{id}`` returns the current persisted state.
- ``POST /api/v1/workflow/{id}/review`` submits an officer determination
  (``cleared`` → continue to documentation; ``blocked`` → terminate).

Needs ``llm_router`` + ``rag`` on ``app.state`` (the compliance stage uses them);
the other stages use the deterministic mock domain adapters. The checkpointer +
review queue are process-lifetime singletons on ``app.state`` (wired in
``bootstrap``; lazily created here if absent) so a workflow can be resumed across
requests. 404 when the feature is disabled.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request

from app.core.config import settings
from app.core.errors import AppError
from app.core.kill_switch import require_feature
from app.core.rate_limit import limiter
from app.core.scope import enforce_scope
from app.domain.adapters import DomainProviders, default_providers
from app.llm.router import LLMRouter
from app.schemas.workflow import (
    WorkflowProcessRequest,
    WorkflowResponse,
    WorkflowReviewRequest,
)
from app.workflow.checkpointer import WorkflowCheckpointer, create_checkpointer
from app.workflow.factory import build_workflow
from app.workflow.orchestrator import DurableWorkflow
from app.workflow.review_queue import InMemoryReviewQueue, ReviewQueue
from app.workflow.state import WorkflowState

router = APIRouter(
    prefix="/workflow",
    tags=["workflow"],
    dependencies=[Depends(require_feature("workflow"))],
)


def _require_enabled() -> None:
    if not settings.workflow_enabled:
        raise AppError(status_code=404, message="Workflow endpoint is disabled")


def _checkpointer(request: Request) -> WorkflowCheckpointer:
    """Shared checkpointer singleton (lazily created + cached on app.state)."""
    cp = getattr(request.app.state, "workflow_checkpointer", None)
    if cp is None:
        cp = create_checkpointer(settings.workflow_durable, settings.workflow_checkpoint_path)
        request.app.state.workflow_checkpointer = cp
    return cp


def _review_queue(request: Request) -> ReviewQueue:
    rq = getattr(request.app.state, "review_queue", None)
    if rq is None:
        rq = InMemoryReviewQueue()
        request.app.state.review_queue = rq
    return rq


def _build_workflow(request: Request) -> DurableWorkflow:
    """Assemble a DurableWorkflow from app.state (503 if LLM/RAG not wired)."""
    llm_router: LLMRouter | None = getattr(request.app.state, "llm_router", None)
    if llm_router is None:
        raise AppError(status_code=503, message="LLM router is not initialized")
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        raise AppError(status_code=503, message="RAG pipeline is not initialized")

    providers: DomainProviders = (
        getattr(request.app.state, "domain", None) or default_providers()
    )
    return build_workflow(
        llm_router=llm_router,
        embedding_provider=rag["embedding_provider"],
        vector_store=rag["vector_store"],
        audit_sink=getattr(request.app.state, "audit_sink", None),
        providers=providers,
        checkpointer=_checkpointer(request),
        review_queue=_review_queue(request),
    )


@router.post("/process", response_model=WorkflowResponse)
@limiter.limit(settings.rate_limit_workflow)
async def process_workflow(
    body: WorkflowProcessRequest, request: Request,
) -> WorkflowResponse:
    """Run a shipment through the workflow; may suspend for human review."""
    _require_enabled()
    # Domestic-only deployments reject cross-border shipments (no-op when worldwide).
    enforce_scope(body.origin_country, body.destination_country)
    workflow = _build_workflow(request)
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
    return WorkflowResponse.from_state(await workflow.process(state))


@router.get("/{workflow_id}", response_model=WorkflowResponse)
@limiter.limit(settings.rate_limit_workflow)
async def get_workflow(workflow_id: str, request: Request) -> WorkflowResponse:
    """Return the current persisted state for a workflow (404 if unknown)."""
    _require_enabled()
    state = _checkpointer(request).load(workflow_id)
    if state is None:
        raise AppError(status_code=404, message=f"Workflow not found: {workflow_id}")
    return WorkflowResponse.from_state(state)


@router.post("/{workflow_id}/review", response_model=WorkflowResponse)
@limiter.limit(settings.rate_limit_workflow)
async def review_workflow(
    workflow_id: str, body: WorkflowReviewRequest, request: Request,
) -> WorkflowResponse:
    """Submit a human determination for a workflow awaiting review."""
    _require_enabled()
    workflow = _build_workflow(request)
    state = workflow.load(workflow_id)
    if state is None:
        raise AppError(status_code=404, message=f"Workflow not found: {workflow_id}")
    if state.status != "awaiting_review":
        raise AppError(
            status_code=409,
            message=f"Workflow {workflow_id} is not awaiting review (status={state.status})",
        )
    resumed = await workflow.resume(state, determination=body.determination, note=body.note)
    return WorkflowResponse.from_state(resumed)
