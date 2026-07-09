"""Conversational Concierge route — POST /api/v1/concierge/chat.

A stateful, multi-turn slot-filling chat (distinct from the one-shot
``/agent/run``). Gathers the shipment slots, never re-asks for ones already
present, then dispatches to an existing deterministic worker and echoes the full
merged state so a client can patch its form.

Conversation memory is server-side and **best-effort**: each turn is persisted by
an anonymous ``session_id`` (minted here if absent) so a chat can be RECALLED via
``GET /concierge/{session_id}`` after a page reload. Persistence never blocks a
turn — a store error is logged and the reply is returned anyway. 404 when disabled;
503 when the LLM router / RAG pipeline is not initialized.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, Request

from app.agents.concierge.models import ConversationState
from app.agents.concierge.service import reconcile_recall, run_concierge
from app.conversations.store import ConversationMessage, ConversationStore
from app.core.config import settings
from app.core.errors import AppError
from app.core.kill_switch import require_feature
from app.core.rate_limit import limiter
from app.llm.router import LLMRouter
from app.schemas.concierge import (
    ConciergeHistoryResponse,
    ConciergeMessage,
    ConciergeRequest,
    ConciergeResponse,
    ConciergeState,
)
from app.workflow.factory import build_workflow
from app.workflow.orchestrator import DurableWorkflow

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/concierge",
    tags=["concierge"],
    dependencies=[Depends(require_feature("concierge"))],
)


def _store(request: Request) -> ConversationStore | None:
    return getattr(request.app.state, "conversation_store", None)


def _workflow(request: Request, rag: dict, llm_router: LLMRouter) -> DurableWorkflow | None:
    """Assemble the multi-agent workflow ONLY when international + compliance + workflow
    are all on — so the concierge bridge can drive it. ``None`` everywhere else keeps
    today's domestic / flags-off behavior intact."""
    if settings.is_domestic_scope:
        return None
    if not (settings.workflow_enabled and settings.compliance_explicit_enabled):
        return None
    return build_workflow(
        llm_router=llm_router,
        embedding_provider=rag["embedding_provider"],
        vector_store=rag["vector_store"],
        audit_sink=getattr(request.app.state, "audit_sink", None),
        providers=getattr(request.app.state, "domain", None),
        checkpointer=getattr(request.app.state, "workflow_checkpointer", None),
        review_queue=getattr(request.app.state, "review_queue", None),
    )


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

    session_id = body.session_id or uuid.uuid4().hex
    store = _store(request)

    client_state = ConversationState.from_wire(body.state.model_dump() if body.state else None)
    state = client_state
    if store is not None:
        stored = await _safe_load(store, session_id)
        state = reconcile_recall(client_state, stored)

    result = await run_concierge(
        body.message, state,
        llm_router=llm_router,
        embedding_provider=rag["embedding_provider"],
        vector_store=rag["vector_store"],
        audit_sink=getattr(request.app.state, "audit_sink", None),
        tool_registry=getattr(request.app.state, "tool_registry", None),
        workflow=_workflow(request, rag, llm_router),
        reply_to=body.reply_to,
        recent_history=body.recent_history,
        request_id=getattr(request.state, "request_id", ""),
    )

    if store is not None:
        await _persist_turn(store, session_id, state, body.message, result)

    return ConciergeResponse(
        reply=result.reply,
        state=ConciergeState(**result.state.to_wire()),
        session_id=session_id,
        clarification=result.clarification,
        dispatched_to=result.dispatched_to,
        sources=result.sources,
        decisions=result.decisions,
        provider=result.provider,
    )


@router.get("/{session_id}", response_model=ConciergeHistoryResponse)
@limiter.limit(settings.rate_limit_concierge)
async def concierge_history(session_id: str, request: Request) -> ConciergeHistoryResponse:
    """Return the persisted transcript + merged state so a client can rehydrate."""
    if not settings.concierge_enabled:
        raise AppError(status_code=404, message="Concierge endpoint is disabled")
    store = _store(request)
    if store is None:
        raise AppError(status_code=404, message="Conversation memory is disabled")
    record = await _safe_load(store, session_id)
    if record is None:
        raise AppError(status_code=404, message=f"Conversation not found: {session_id}")
    return ConciergeHistoryResponse(
        session_id=record.session_id,
        state=ConciergeState(
            slots=record.slots,
            intent=record.intent,
            status=record.status,
            turns=record.turns,
        ),
        messages=[
            ConciergeMessage(role=m.role, content=m.content, created_at=m.created_at)
            for m in record.messages
        ],
    )


async def _safe_load(store: ConversationStore, session_id: str):
    """Load a conversation; never let a store error break the request."""
    try:
        return await store.load(session_id, limit=settings.conversation_max_messages)
    except Exception as exc:  # noqa: BLE001 - recall is best-effort
        logger.warning("Conversation load failed for %s: %s", session_id, exc)
        return None


async def _persist_turn(
    store: ConversationStore,
    session_id: str,
    pre_state: ConversationState,
    message: str,
    result,
) -> None:
    """Upsert the recall snapshot + append the user/assistant turns (best-effort)."""
    post = result.state
    delta = {
        k: v for k, v in post.slots.items()
        if pre_state.slots.get(k) != v
    }
    try:
        await store.upsert_state(
            session_id,
            status=post.status,
            intent=post.intent,
            slots=dict(post.slots),
            turns=post.turns,
            last_dispatched_to=result.dispatched_to,
        )
        await store.append_messages(
            session_id,
            [
                ConversationMessage(role="user", content=message, slots_delta=delta),
                ConversationMessage(
                    role="assistant", content=result.reply, decisions=list(result.decisions),
                ),
            ],
        )
    except Exception as exc:  # noqa: BLE001 - persistence must never break a turn
        logger.warning("Conversation persist failed for %s: %s", session_id, exc)
