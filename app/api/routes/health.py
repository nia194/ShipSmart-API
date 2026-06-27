"""Health and readiness check routes."""

from datetime import UTC, datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.core.config import settings

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: str


class ReadyResponse(BaseModel):
    status: str
    rag_mode: str = "normal"
    rag_hybrid: bool = False
    guardrails_enabled: bool = True
    agent_enabled: bool = True
    compliance_enabled: bool = True
    concierge_enabled: bool = False
    # Effective concierge recall backend: "memory" | "postgres" | "disabled"
    # (disabled = the configured store failed to wire, so recall is off).
    conversation_store: str = "memory"
    workflow_enabled: bool = False
    workflow_durable: bool = False
    # Resolved LLM failover chain per task (provider names), e.g.
    # {"reasoning": ["openai", "gemini", "echo"], "synthesis": ["openai", "echo"]}.
    llm_chains: dict[str, list[str]] = Field(default_factory=dict)


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Liveness check. Used by Render health checks."""
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        timestamp=datetime.now(tz=UTC).isoformat(),
    )


@router.get("/ready", response_model=ReadyResponse, tags=["health"])
async def ready(request: Request) -> ReadyResponse:
    """Readiness check. Returns 200 when the service can accept traffic, and
    reports the resolved LLM failover chain (A) + active retrieval/guardrail
    flags so operators can confirm what's wired without reading logs."""
    llm_router = getattr(request.app.state, "llm_router", None)
    conversation_store = (
        getattr(settings, "conversation_store", "memory")
        if getattr(request.app.state, "conversation_store", None) is not None
        else "disabled"
    )
    return ReadyResponse(
        status="ready",
        rag_mode=getattr(settings, "rag_mode", "normal"),
        rag_hybrid=getattr(settings, "rag_hybrid", False),
        guardrails_enabled=getattr(settings, "guardrails_enabled", True),
        agent_enabled=getattr(settings, "agent_enabled", True),
        compliance_enabled=getattr(settings, "compliance_enabled", True),
        concierge_enabled=getattr(settings, "concierge_enabled", False),
        conversation_store=conversation_store,
        workflow_enabled=getattr(settings, "workflow_enabled", False),
        workflow_durable=getattr(settings, "workflow_durable", False),
        llm_chains=llm_router.describe_chains() if llm_router else {},
    )
