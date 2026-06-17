"""
Agent (Concierge) route.

A new, additive `POST /api/v1/agent/run` that runs the model-driven, read-only
tool-calling loop (`app.services.agent_service.run_agent`) over the MCP tools +
`retrieve_rag`. Existing `/orchestration/run` and `/advisor/*` are unchanged.

Day-1 is read-only end to end: the agent plans, retrieves, and calls read-only
tools — it never persists.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.core.config import settings
from app.core.errors import AppError
from app.core.rate_limit import limiter
from app.llm.router import LLMRouter
from app.schemas.agent import AgentRequest, AgentResponse, AgentStep
from app.services.agent_service import run_agent

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/run", response_model=AgentResponse)
@limiter.limit(settings.rate_limit_agent)
async def run_agent_endpoint(body: AgentRequest, request: Request) -> AgentResponse:
    """Run the concierge agent for a free-text request.

    Returns the grounded answer plus the reasoning trace (steps, tools_used,
    sources, decisions, provider). Returns 503 when the MCP tool registry is not
    initialized (same as orchestration); 404 when the agent feature is disabled.
    """
    if not settings.agent_enabled:
        raise AppError(status_code=404, message="Agent endpoint is disabled")

    registry = getattr(request.app.state, "tool_registry", None)
    if registry is None:
        raise AppError(status_code=503, message="Tool registry is not initialized")

    llm_router: LLMRouter | None = getattr(request.app.state, "llm_router", None)
    if llm_router is None:
        raise AppError(status_code=503, message="LLM router is not initialized")

    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        raise AppError(status_code=503, message="RAG pipeline is not initialized")

    result = await run_agent(
        body.query,
        body.context,
        registry=registry,
        llm_router=llm_router,
        embedding_provider=rag["embedding_provider"],
        vector_store=rag["vector_store"],
        request_id=getattr(request.state, "request_id", ""),
    )

    return AgentResponse(
        answer=result.answer,
        steps=[AgentStep(**s) for s in result.steps],
        tools_used=result.tools_used,
        sources=result.sources,
        decisions=result.decisions,
        provider=result.provider,
    )
