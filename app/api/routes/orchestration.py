"""
Orchestration routes.
Handles tool-based orchestration: discover tools, execute tools, run queries.

This service is NOT the system-of-record for shipments or quotes.
Spring Boot owns transactional logic. Python provides AI-assist and tooling.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.errors import AppError
from app.core.rate_limit import limiter
from app.llm.router import TASK_REASONING, LLMRouter
from app.services.orchestration_service import (
    execute_tool,
    run_orchestration,
)

router = APIRouter(prefix="/orchestration", tags=["orchestration"])


# ── Schemas ──────────────────────────────────────────────────────────────────

class OrchestrationRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    tool: str | None = Field(
        None,
        description="Explicit tool name. If omitted, tool is auto-selected from query.",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the tool.",
    )


class OrchestrationResponse(BaseModel):
    type: str
    tool_used: str | None = None
    answer: str
    data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: list[dict[str, Any]]


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/run", response_model=OrchestrationResponse)
@limiter.limit(settings.rate_limit_orchestration)
async def run_workflow(
    body: OrchestrationRequest, request: Request,
) -> OrchestrationResponse:
    """Run an orchestration workflow.

    If `tool` is specified, executes that tool directly.
    Otherwise, auto-selects a tool based on the query text. Selection uses
    deterministic regex rules first; if those miss and a reasoning LLM is
    configured, it falls back to LLM-assisted selection.
    If no tool matches at all, returns a direct_answer signal.
    """
    registry = getattr(request.app.state, "tool_registry", None)
    if registry is None:
        raise AppError(status_code=503, message="Tool registry is not initialized")

    llm_router: LLMRouter | None = getattr(request.app.state, "llm_router", None)
    reasoning_client = llm_router.for_task(TASK_REASONING) if llm_router else None

    if body.tool:
        # Explicit tool execution
        result = await execute_tool(body.tool, body.params, registry)
    else:
        # Auto-select tool from query (regex → LLM fallback)
        result = await run_orchestration(
            body.query, body.params, registry, llm_client=reasoning_client,
        )

    return OrchestrationResponse(
        type=result.type,
        tool_used=result.tool_used,
        answer=result.answer,
        data=result.data,
        metadata=result.metadata,
    )


@router.get("/tools", response_model=list[ToolSchema])
async def list_tools(request: Request) -> list[ToolSchema]:
    """List all registered tools and their schemas."""
    registry = getattr(request.app.state, "tool_registry", None)
    if registry is None:
        raise AppError(status_code=503, message="Tool registry is not initialized")

    return [
        ToolSchema(**schema)
        for schema in registry.list_schemas()
    ]
