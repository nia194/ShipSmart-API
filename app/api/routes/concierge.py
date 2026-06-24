"""Conversational Concierge route — POST /api/v1/concierge/chat.

A stateful, multi-turn slot-filling chat (distinct from the one-shot
``/agent/run``). Gathers the shipment slots, never re-asks for ones already
present, then dispatches to an existing deterministic worker and echoes the full
merged state so a client can patch its form. 404 when disabled; 503 when the LLM
router / RAG pipeline is not initialized.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.agents.concierge.models import ConversationState
from app.agents.concierge.service import run_concierge
from app.core.config import settings
from app.core.errors import AppError
from app.core.rate_limit import limiter
from app.llm.router import LLMRouter
from app.schemas.concierge import ConciergeRequest, ConciergeResponse, ConciergeState

router = APIRouter(prefix="/concierge", tags=["concierge"])


@router.post("/chat", response_model=ConciergeResponse)
@limiter.limit(settings.rate_limit_concierge)
async def concierge_chat(body: ConciergeRequest, request: Request) -> ConciergeResponse:
    """Run one concierge turn over the client-sent conversation state."""
    if not settings.concierge_enabled:
        raise AppError(status_code=404, message="Concierge endpoint is disabled")

    llm_router: LLMRouter | None = getattr(request.app.state, "llm_router", None)
    rag = getattr(request.app.state, "rag", None)
    if llm_router is None or rag is None:
        raise AppError(status_code=503, message="LLM router / RAG pipeline is not initialized")

    state = ConversationState.from_wire(body.state.model_dump() if body.state else None)
    result = await run_concierge(
        body.message, state,
        llm_router=llm_router,
        embedding_provider=rag["embedding_provider"],
        vector_store=rag["vector_store"],
        audit_sink=getattr(request.app.state, "audit_sink", None),
        tool_registry=getattr(request.app.state, "tool_registry", None),
        request_id=getattr(request.state, "request_id", ""),
    )

    return ConciergeResponse(
        reply=result.reply,
        state=ConciergeState(**result.state.to_wire()),
        clarification=result.clarification,
        dispatched_to=result.dispatched_to,
        sources=result.sources,
        decisions=result.decisions,
        provider=result.provider,
    )
