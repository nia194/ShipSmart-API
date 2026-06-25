"""``run_concierge`` — the deterministic concierge pipeline.

guardrails-at-the-LLM-edge → extract → ``fold_turn`` → required-slot check
(clarify only what's missing, never re-ask for present slots) → deterministic
dispatch to an existing worker → reply + full-state echo. The model never
books/quotes/verdicts; assembling a worker's typed input is pure code.
"""

from __future__ import annotations

import logging

from app.agents.compliance.models import Shipment
from app.agents.compliance.service import check_compliance
from app.agents.concierge.extract import extract
from app.agents.concierge.models import ConciergeResult, ConversationState, Slots
from app.agents.concierge.state import clarification_for, fold_turn, missing_required
from app.core.audit import AuditSink
from app.core.config import settings
from app.core.scope import violates_domestic_scope
from app.llm.router import LLMRouter
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import VectorStore
from app.services.agent_service import run_agent

logger = logging.getLogger(__name__)

# slot → advisor-context key. origin/destination are free-text location labels
# mapped verbatim into the legacy *_zip context keys (no geocoding).
_ADVISOR_CONTEXT_KEYS = {
    "origin": "origin_zip",
    "destination": "destination_zip",
    "weight_lbs": "weight_lbs",
    "length_in": "length_in",
    "width_in": "width_in",
    "height_in": "height_in",
    "drop_off_date": "drop_off_date",
    "expected_delivery_date": "expected_delivery_date",
}


def _advisor_context(slots: Slots) -> dict:
    return {
        key: slots[slot]
        for slot, key in _ADVISOR_CONTEXT_KEYS.items()
        if slots.get(slot) not in (None, "")
    }


def _shipment_from_slots(slots: Slots) -> Shipment:
    # Domestic-only deployments pin both ends to the home country; worldwide keeps
    # the legacy default of US when a country slot is unfilled.
    if settings.is_domestic_scope:
        origin = destination = settings.home_country
    else:
        origin = slots.get("origin_country") or "US"
        destination = slots.get("destination_country") or "US"
    return Shipment(
        origin_country=origin,
        destination_country=destination,
        declared_value_usd=float(slots.get("declared_value_usd") or 0.0),
        weight_lbs=float(slots.get("weight_lbs") or 0.0),
        description=(slots.get("description") or ""),
        category=slots.get("category"),
    )


async def run_concierge(
    message: str,
    state: ConversationState | None = None,
    *,
    llm_router: LLMRouter,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    audit_sink: AuditSink | None = None,
    tool_registry=None,
    request_id: str = "",
) -> ConciergeResult:
    """Run one concierge turn and return the reply + the full merged state."""
    state = state or ConversationState()
    decisions: list[str] = ["concierge:plan"]

    new_intent, entities = await extract(message, llm_router, request_id=request_id)
    state = fold_turn(state, entities)
    intent = new_intent or state.intent or "advice"
    state = state.with_(intent=intent, turns=state.turns + 1)
    decisions.append(f"concierge:intent:{intent}")

    # Domestic-only: a user who explicitly named an international origin/destination
    # gets an honest "we ship domestically" reply rather than a silent rewrite.
    offending = violates_domestic_scope(
        state.slots.get("origin_country", ""), state.slots.get("destination_country", ""),
    )
    if offending is not None:
        decisions.append("concierge:scope:domestic_only")
        reply = (
            f"I can only help with shipments within {settings.home_country} on this "
            f"deployment — {offending} isn't supported."
        )
        return ConciergeResult(
            reply=reply, state=state.with_(status="answered"),
            dispatched_to="scope_blocked", decisions=decisions,
        )

    missing = missing_required(state.slots, intent)
    # Domestic-only: destination country defaults to home, so never ask for it.
    if settings.is_domestic_scope:
        missing = [slot for slot in missing if slot != "destination_country"]
    if missing:
        slot = missing[0]
        clarification = clarification_for(slot)
        decisions.append(f"concierge:clarify:{slot}")
        state = state.with_(status="gathering", pending_clarification=slot)
        return ConciergeResult(
            reply=clarification, state=state,
            clarification=clarification, decisions=decisions,
        )

    decisions.append("concierge:ready")
    state = state.with_(status="answered", pending_clarification=None)
    return await _dispatch(
        intent, message, state, decisions,
        llm_router=llm_router, embedding_provider=embedding_provider,
        vector_store=vector_store, audit_sink=audit_sink,
        tool_registry=tool_registry, request_id=request_id,
    )


async def _dispatch(
    intent: str,
    message: str,
    state: ConversationState,
    decisions: list[str],
    *,
    llm_router: LLMRouter,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    audit_sink: AuditSink | None,
    tool_registry,
    request_id: str,
) -> ConciergeResult:
    decisions.append(f"concierge:dispatch:{intent}")

    # The explicit compliance pass is an additive feature. When switched off, a
    # compliance intent falls through to the normal agent/RAG path below — still
    # grounded by the compliance corpus + guardrails (the lightweight default
    # checks), just without the hard UC2 verdict.
    if intent == "compliance" and not settings.compliance_explicit_enabled:
        decisions.append("concierge:compliance:explicit_skipped")

    if intent == "compliance" and settings.compliance_explicit_enabled:
        result = await check_compliance(
            _shipment_from_slots(state.slots),
            llm_router=llm_router, embedding_provider=embedding_provider,
            vector_store=vector_store, audit_sink=audit_sink, request_id=request_id,
        )
        decisions.extend(result.decisions)
        return ConciergeResult(
            reply=result.summary, state=state, dispatched_to="compliance",
            sources=result.sources, decisions=decisions, provider=result.provider,
        )

    # quote / tracking / advice → the existing read-only agent (tools + RAG).
    if tool_registry is not None:
        agent = await run_agent(
            message, _advisor_context(state.slots),
            registry=tool_registry, llm_router=llm_router,
            embedding_provider=embedding_provider, vector_store=vector_store,
            request_id=request_id,
        )
        decisions.extend(agent.decisions)
        return ConciergeResult(
            reply=agent.answer, state=state, dispatched_to="agent",
            sources=agent.sources, decisions=decisions, provider=agent.provider,
        )

    decisions.append("concierge:dispatch:summary_fallback")
    return ConciergeResult(
        reply=_summary_reply(state.slots), state=state,
        dispatched_to="summary", decisions=decisions,
    )


def _summary_reply(slots: Slots) -> str:
    have = [f"{k.replace('_', ' ')}: {v}" for k, v in slots.items() if v not in (None, "")]
    body = "; ".join(have) if have else "your shipment details"
    return (
        f"Got it — I have {body}. The live shipping tools aren't connected right now, "
        "so I can't pull live options, but everything you told me is captured."
    )
