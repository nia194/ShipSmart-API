"""
AI Advisor Routes.
Endpoints for shipping advice, tracking guidance, and recommendations.

These endpoints combine RAG context, tool execution, and LLM reasoning
to provide structured, debuggable AI-assisted features.
"""

from __future__ import annotations

from fastapi import APIRouter, Header, Request

from app.core.config import settings
from app.core.errors import AppError
from app.core.rate_limit import limiter
from app.llm.router import TASK_REASONING, TASK_SYNTHESIS, LLMRouter
from app.schemas.advisor import (
    RecommendationRequest,
    RecommendationResponse,
    ServiceOption,
    ShippingAdvisorRequest,
    ShippingAdvisorResponse,
    TrackingAdvisorRequest,
    TrackingAdvisorResponse,
)
from app.services.java_client import JavaApiClient
from app.services.recommendation_service import generate_recommendations
from app.services.shipping_advisor_service import get_shipping_advice
from app.services.tracking_advisor_service import get_tracking_guidance

router = APIRouter(prefix="/advisor", tags=["advisor"])


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    if authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip() or None
    return authorization.strip() or None


# ── Shipping Advisor ─────────────────────────────────────────────────────────

@router.post("/shipping", response_model=ShippingAdvisorResponse)
@limiter.limit(settings.rate_limit_advisor)
async def shipping_advisor(
    body: ShippingAdvisorRequest, request: Request,
) -> ShippingAdvisorResponse:
    """Get AI-powered shipping advice.

    Combines RAG context, address/quote tools, and LLM reasoning.
    """
    rag = getattr(request.app.state, "rag", None)
    tool_registry = getattr(request.app.state, "tool_registry", None)
    llm_router: LLMRouter | None = getattr(request.app.state, "llm_router", None)

    if rag is None or tool_registry is None:
        raise AppError(
            status_code=503,
            message="RAG pipeline or tool registry not initialized",
        )

    # Shipping advisor reasons over context + tool results → reasoning task
    reasoning_client = (
        llm_router.for_task(TASK_REASONING) if llm_router else rag["llm_client"]
    )

    advice = await get_shipping_advice(
        query=body.query,
        context=body.context,
        embedding_provider=rag["embedding_provider"],
        vector_store=rag["vector_store"],
        llm_client=reasoning_client,
        tool_registry=tool_registry,
    )

    return ShippingAdvisorResponse(
        answer=advice.answer,
        reasoning_summary=advice.reasoning_summary,
        tools_used=advice.tools_used,
        sources=advice.sources,
        context_used=advice.context_used,
    )


# ── Tracking Advisor ────────────────────────────────────────────────────────

@router.post("/tracking", response_model=TrackingAdvisorResponse)
@limiter.limit(settings.rate_limit_advisor)
async def tracking_advisor(
    body: TrackingAdvisorRequest, request: Request,
) -> TrackingAdvisorResponse:
    """Get guidance on delivery/tracking issues.

    Combines RAG context, optional address validation, and LLM reasoning.
    """
    rag = getattr(request.app.state, "rag", None)
    tool_registry = getattr(request.app.state, "tool_registry", None)
    llm_router: LLMRouter | None = getattr(request.app.state, "llm_router", None)

    if rag is None or tool_registry is None:
        raise AppError(
            status_code=503,
            message="RAG pipeline or tool registry not initialized",
        )

    reasoning_client = (
        llm_router.for_task(TASK_REASONING) if llm_router else rag["llm_client"]
    )

    guidance = await get_tracking_guidance(
        issue=body.issue,
        context=body.context,
        embedding_provider=rag["embedding_provider"],
        vector_store=rag["vector_store"],
        llm_client=reasoning_client,
        tool_registry=tool_registry,
    )

    return TrackingAdvisorResponse(
        guidance=guidance.guidance,
        issue_summary=guidance.issue_summary,
        tools_used=guidance.tools_used,
        sources=guidance.sources,
        next_steps=guidance.next_steps,
    )


# ── Recommendations ──────────────────────────────────────────────────────────

@router.post("/recommendation", response_model=RecommendationResponse)
@limiter.limit(settings.rate_limit_advisor)
async def get_recommendation(
    body: RecommendationRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> RecommendationResponse:
    """Get ranked service recommendations from quote preview results.

    Takes a list of services and returns scored recommendations with explanations.
    If `services` is empty but `context.shipment_request_id` is provided, the
    list is hydrated from the Java API (Authorization header is forwarded).
    """
    # Recommendation summary is light synthesis over scored options.
    llm_router: LLMRouter | None = getattr(request.app.state, "llm_router", None)
    rag = getattr(request.app.state, "rag", None)
    llm_client = (
        llm_router.for_task(TASK_SYNTHESIS)
        if llm_router
        else (rag["llm_client"] if rag else None)
    )

    http_client = getattr(request.app.state, "http_client", None)
    java_client = JavaApiClient(http_client) if http_client is not None else None

    recommendations = await generate_recommendations(
        services=body.services,
        context=body.context,
        llm_client=llm_client,
        java_client=java_client,
        auth_token=_bearer_token(authorization),
    )

    return RecommendationResponse(
        primary_recommendation=ServiceOption(
            service_name=recommendations.primary_recommendation.service_name,
            price_usd=recommendations.primary_recommendation.price_usd,
            estimated_days=recommendations.primary_recommendation.estimated_days,
            recommendation_type=recommendations.primary_recommendation.recommendation_type.value,
            explanation=recommendations.primary_recommendation.explanation,
            score=recommendations.primary_recommendation.score,
        ),
        alternatives=[
            ServiceOption(
                service_name=alt.service_name,
                price_usd=alt.price_usd,
                estimated_days=alt.estimated_days,
                recommendation_type=alt.recommendation_type.value,
                explanation=alt.explanation,
                score=alt.score,
            )
            for alt in recommendations.alternatives
        ],
        summary=recommendations.summary,
        metadata=recommendations.metadata,
    )
