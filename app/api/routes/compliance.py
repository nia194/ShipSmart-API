"""
Compliance route (UC2).

A new, additive ``POST /api/v1/compliance/check`` that runs the deterministic
compliance flow (structural rules + grounded fixed-area investigation) with an
optional model-in-the-loop critic, and returns an ADVISORY verdict plus the full
reasoning trace.

Dependency policy (keyless-friendly): this flow reasons via ``retrieve_area`` +
structural checks + a grounded summary — it calls no MCP tools — so it requires
only ``llm_router`` + ``rag`` on ``app.state`` (503 if missing) and does NOT
require ``tool_registry``. That keeps the endpoint working out-of-the-box in local
/ keyless dev (no ``SHIPSMART_MCP_URL`` needed).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.agents.compliance import Shipment, check_compliance
from app.core.config import settings
from app.core.errors import AppError
from app.core.kill_switch import require_feature
from app.core.rate_limit import limiter
from app.core.scope import enforce_scope
from app.llm.router import LLMRouter
from app.schemas.compliance import (
    ComplianceFinding,
    ComplianceRequest,
    ComplianceResponse,
)

router = APIRouter(
    prefix="/compliance",
    tags=["compliance"],
    dependencies=[Depends(require_feature("compliance"))],
)


@router.post("/check", response_model=ComplianceResponse)
@limiter.limit(settings.rate_limit_compliance)
async def check_compliance_endpoint(
    body: ComplianceRequest, request: Request,
) -> ComplianceResponse:
    """Review a shipment for compliance concerns (advisory).

    Returns the advisory verdict, findings (flags / grounded info / honest
    unverified gaps), a grounded summary, cited sources, and the decision trail.
    503 when the LLM router or RAG pipeline is not initialized; 404 when the
    compliance feature is disabled. Does not require the MCP tool registry.
    """
    if not settings.compliance_enabled:
        raise AppError(status_code=404, message="Compliance endpoint is disabled")

    llm_router: LLMRouter | None = getattr(request.app.state, "llm_router", None)
    if llm_router is None:
        raise AppError(status_code=503, message="LLM router is not initialized")

    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        raise AppError(status_code=503, message="RAG pipeline is not initialized")

    # Domestic-only deployments reject cross-border shipments (no-op when worldwide).
    enforce_scope(body.origin_country, body.destination_country)

    shipment = Shipment(
        origin_country=body.origin_country,
        destination_country=body.destination_country,
        declared_value_usd=body.declared_value_usd,
        weight_lbs=body.weight_lbs,
        description=body.description,
        category=body.category,
    )

    result = await check_compliance(
        shipment,
        llm_router=llm_router,
        embedding_provider=rag["embedding_provider"],
        vector_store=rag["vector_store"],
        audit_sink=getattr(request.app.state, "audit_sink", None),
        request_id=getattr(request.state, "request_id", ""),
    )

    return ComplianceResponse(
        verdict=result.verdict,
        summary=result.summary,
        findings=[
            ComplianceFinding(
                area=f.area, status=f.status, kind=f.kind,
                detail=f.detail, sources=f.sources,
            )
            for f in result.findings
        ],
        sources=result.sources,
        decisions=result.decisions,
        critique_rounds=result.critique_rounds,
        provider=result.provider,
    )
