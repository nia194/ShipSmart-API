"""``run_concierge`` — the deterministic concierge pipeline.

guardrails-at-the-LLM-edge → extract → ``fold_turn`` → required-slot check
(clarify only what's missing, never re-ask for present slots) → deterministic
dispatch to an existing worker → reply + full-state echo. The model never
books/quotes/verdicts; assembling a worker's typed input is pure code.
"""

from __future__ import annotations

import logging
import re
import uuid

from app.agents.compliance.models import Shipment
from app.agents.compliance.service import check_compliance
from app.agents.concierge.extract import extract_nlu, is_greeting
from app.agents.concierge.models import ConciergeResult, ConversationState, Slots
from app.agents.concierge.reply import (
    compose_gathering_reply,
    compose_ready_summary,
    correction_note,
)
from app.agents.concierge.state import (
    REQUIRED_SLOTS,
    apply_corrections,
    choose_intent,
    clarification_for,
    fold_turn,
    is_empty,
    missing_required,
)
from app.conversations.store import ConversationRecord
from app.core.audit import AuditSink
from app.core.config import settings
from app.core.scope import violates_domestic_scope
from app.llm.errors import LLMError
from app.llm.reply_context import render_reference_block
from app.llm.router import TASK_REASONING, LLMRouter
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import VectorStore
from app.services.agent_service import run_agent
from app.workflow.orchestrator import DurableWorkflow
from app.workflow.state import WorkflowState

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


def _is_keyless(llm_router: LLMRouter) -> bool:
    """True when no real LLM is wired (echo/scripted/unset).

    Keyless, the read-only agent can only return generic boilerplate, so the
    deterministic ready-summary is strictly more useful for the user.
    """
    try:
        client = llm_router.for_task(TASK_REASONING)
    except Exception:
        return True
    return getattr(client, "provider_name", "") in ("", "echo", "scripted")


# get_quote_preview requires US ZIPs + package dimensions, but the concierge gathers
# free-text places and (per its required slots) no dimensions. Resolve common cities to
# a representative ZIP and assume a standard box when none is given, so an origin/
# destination/weight conversation can actually produce a quote (the README's "add
# dimensions for a sharper rate"). Unknown cities pass through unchanged.
_CITY_ZIP: dict[str, str] = {
    "new york": "10001", "nyc": "10001", "los angeles": "90001", "la": "90001",
    "chicago": "60601", "houston": "77001", "phoenix": "85001", "philadelphia": "19101",
    "san antonio": "78201", "san diego": "92101", "dallas": "75201", "austin": "73301",
    "san francisco": "94102", "sf": "94102", "seattle": "98101", "denver": "80202",
    "boston": "02108", "miami": "33101", "atlanta": "30301", "reno": "89501",
    "portland": "97201", "las vegas": "89101", "detroit": "48201", "minneapolis": "55401",
    "nashville": "37201", "washington": "20001", "dc": "20001",
}
_DEFAULT_DIMS_IN = {"length_in": 12.0, "width_in": 9.0, "height_in": 6.0}


def _to_zip(value: str) -> str:
    """Best-effort place→ZIP: a ZIP-shaped value passes through; a known city maps to
    a representative ZIP; anything else is returned unchanged."""
    v = value.strip()
    m = re.match(r"^(\d{5})", v)
    if m:
        return m.group(1)
    key = re.sub(r",\s*[a-z]{2}$", "", v.lower())
    return _CITY_ZIP.get(key, value)


def _agent_query_for(intent: str, message: str, slots: Slots) -> str:
    """Shape an explicit query for the read-only agent from the conversation intent.

    The raw last turn is often terse ("about 5 lbs") and a poor agent prompt; an
    intent-shaped question makes the agent act on the gathered slots.
    """
    if intent == "quote":
        return "What are my best shipping options and the price for this shipment?"
    if intent == "tracking":
        ref = slots.get("tracking_reference")
        if ref:
            return f"What is the latest status of tracking number {ref}?"
    return message


def _advisor_context(slots: Slots) -> dict:
    ctx = {
        key: slots[slot]
        for slot, key in _ADVISOR_CONTEXT_KEYS.items()
        if slots.get(slot) not in (None, "")
    }
    for zip_key in ("origin_zip", "destination_zip"):
        if zip_key in ctx:
            ctx[zip_key] = _to_zip(str(ctx[zip_key]))
    # A quote needs dimensions; assume a standard box when the user gave none.
    if "weight_lbs" in ctx:
        for dim_key, default in _DEFAULT_DIMS_IN.items():
            ctx.setdefault(dim_key, default)
    return ctx


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


def reconcile_recall(
    client_state: ConversationState, stored: ConversationRecord | None,
) -> ConversationState:
    """Rehydrate from the server snapshot ONLY when the client has nothing.

    Active sessions keep the client-owned draft as the source of truth (today's
    behavior): the client echoes its merged form+chat state each turn. A FRESH
    client (no slots, turn 0) that supplies a known ``session_id`` is a page-reload
    recall — restore the persisted snapshot so the conversation continues instead
    of starting over.
    """
    if stored is None:
        return client_state
    if client_state.slots or client_state.turns:
        return client_state
    return ConversationState(
        slots=dict(stored.slots),
        intent=stored.intent,
        status=stored.status,
        turns=stored.turns,
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
    workflow: DurableWorkflow | None = None,
    reply_to: object | None = None,
    recent_history: list | None = None,
    request_id: str = "",
) -> ConciergeResult:
    """Run one concierge turn and return the reply + the full merged state.

    ``workflow`` is the assembled multi-agent workflow, passed in by the route ONLY
    when international + compliance + workflow are all enabled. When ``None`` the
    bridge can't fire — preserving today's domestic / flags-off behavior exactly.
    """
    state = state or ConversationState()
    decisions: list[str] = ["concierge:plan"]

    # Optional reply-to / recent-turns reference (bounded), used only to resolve references
    # in the message — the merged slots + worker results stay authoritative.
    reference_block = render_reference_block(reply_to, recent_history)
    if reference_block:
        decisions.append("concierge:reply_to")

    nlu = await extract_nlu(
        message, state.slots, llm_router,
        reference_block=reference_block, request_id=request_id,
    )

    # Pure greeting / smalltalk with nothing extracted and no prior context: welcome
    # and orient the user, instead of dispatching an empty query to the agent (which
    # would return generic RAG boilerplate).
    if is_greeting(message) and not nlu.slots and not nlu.intents and not state.slots:
        decisions.append("concierge:greeting")
        reply = (
            "Hi! I'm your shipping assistant. Tell me what you'd like to ship and where "
            '— e.g. "from Atlanta to Seattle, 12 lb" — and I\'ll line up options, check '
            "customs/compliance, or track a package."
        )
        return ConciergeResult(
            reply=reply, state=state.with_(status="gathering", turns=state.turns + 1),
            decisions=decisions,
        )

    state = fold_turn(state, nlu.slots)                 # new mentions: gap-fill / newest-wins
    state = apply_corrections(state, nlu.corrections)   # explicit overrides win outright
    intent = choose_intent(nlu.intents, state.intent)
    state = state.with_(intent=intent, turns=state.turns + 1)
    decisions.append(f"concierge:intent:{intent}")
    secondary = [i for i in nlu.intents if i != intent]
    if nlu.corrections:
        decisions.append("concierge:correction")

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
    # A required slot the user gave but too vaguely (LLM-flagged) needs confirming.
    # Keyless → no ambiguities, so this is a no-op on the deterministic path.
    required = REQUIRED_SLOTS.get(intent, ())
    ambiguous = [
        a for a in nlu.ambiguities
        if a in required and a not in missing and not is_empty(state.slots.get(a))
    ]
    if missing or ambiguous:
        slot = missing[0] if missing else ambiguous[0]
        tag = "clarify" if missing else "disambiguate"
        clarification = clarification_for(slot)
        if not missing:  # disambiguation: re-confirm a vague-but-present value
            clarification = f"Just to confirm — {clarification[0].lower()}{clarification[1:]}"
        decisions.append(f"concierge:{tag}:{slot}")
        state = state.with_(status="gathering", pending_clarification=slot)
        reply = await compose_gathering_reply(
            clarification, state.slots,
            corrections=nlu.corrections, secondary_intents=secondary,
            llm_router=llm_router, request_id=request_id,
        )
        return ConciergeResult(
            reply=reply, state=state,
            clarification=clarification, decisions=decisions,
        )

    decisions.append("concierge:ready")
    state = state.with_(status="answered", pending_clarification=None)
    try:
        return await _dispatch(
            intent, message, state, decisions,
            corrections=nlu.corrections,
            llm_router=llm_router, embedding_provider=embedding_provider,
            vector_store=vector_store, audit_sink=audit_sink,
            tool_registry=tool_registry, workflow=workflow, request_id=request_id,
        )
    except LLMError as exc:
        # A provider failure mid-dispatch (bad/expired key, outage, rate limit) must
        # not 502 a user-facing chat. Degrade to the deterministic ready-summary —
        # the same graceful fallback the keyless path and reply-polish already use.
        logger.warning("concierge: dispatch degraded to summary (LLM error: %s)", exc)
        decisions.append("concierge:dispatch:llm_degraded")
        reply = await compose_ready_summary(
            state.slots, intent, corrections=nlu.corrections,
            llm_router=None, request_id=request_id,
        )
        return ConciergeResult(
            reply=reply, state=state, dispatched_to="summary", decisions=decisions,
        )


async def _dispatch(
    intent: str,
    message: str,
    state: ConversationState,
    decisions: list[str],
    *,
    corrections: Slots,
    llm_router: LLMRouter,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    audit_sink: AuditSink | None,
    tool_registry,
    workflow: DurableWorkflow | None,
    request_id: str,
) -> ConciergeResult:
    decisions.append(f"concierge:dispatch:{intent}")
    ack = correction_note(corrections)  # "Updated weight to 15 lb. " — prefixes worker output

    # Drive the full multi-agent workflow for an international shipment when the
    # route supplied one (⇒ international + compliance + workflow all enabled).
    # This is the "compliance-workflows ON" path; it never fires domestically or
    # with the flags off (workflow is then None).
    if workflow is not None and _should_run_workflow(intent, state.slots):
        return await _dispatch_workflow(workflow, state, decisions, ack, request_id)

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
            reply=ack + result.summary, state=state, dispatched_to="compliance",
            sources=result.sources, decisions=decisions, provider=result.provider,
        )

    # quote / tracking / advice → the existing read-only agent (tools + RAG) when a
    # real LLM is wired. Keyless, the agent only returns generic boilerplate, so fall
    # through to the deterministic ready-summary below (grounded + genuinely useful).
    if tool_registry is not None and not _is_keyless(llm_router):
        # A terse last turn ("about 5 lbs") is a poor agent query — give the agent an
        # intent-shaped question so it acts on the gathered slots instead of refusing.
        agent_query = _agent_query_for(intent, message, state.slots)
        agent = await run_agent(
            agent_query, _advisor_context(state.slots),
            registry=tool_registry, llm_router=llm_router,
            embedding_provider=embedding_provider, vector_store=vector_store,
            request_id=request_id,
        )
        decisions.extend(agent.decisions)
        return ConciergeResult(
            reply=ack + agent.answer, state=state, dispatched_to="agent",
            sources=agent.sources, decisions=decisions, provider=agent.provider,
        )

    decisions.append("concierge:dispatch:summary_fallback")
    reply = await compose_ready_summary(
        state.slots, intent, corrections=corrections,
        llm_router=llm_router, request_id=request_id,
    )
    return ConciergeResult(
        reply=reply, state=state, dispatched_to="summary", decisions=decisions,
    )


def _should_run_workflow(intent: str, slots: Slots) -> bool:
    """True for an international shipment-processing intent with a known destination.

    Domestic deployments and non-shipment intents (tracking/advice) never trigger
    the full workflow. ``workflow is not None`` (checked by the caller) already
    guarantees compliance + workflow are enabled.
    """
    if settings.is_domestic_scope or intent not in ("quote", "compliance"):
        return False
    dest = (slots.get("destination_country") or "").strip().upper()
    if not dest:
        return False
    origin = (slots.get("origin_country") or "US").strip().upper()
    return dest != origin


async def _dispatch_workflow(
    workflow: DurableWorkflow,
    state: ConversationState,
    decisions: list[str],
    ack: str,
    request_id: str,
) -> ConciergeResult:
    """Run the multi-agent workflow over the gathered slots and narrate the result."""
    slots = state.slots
    ws = WorkflowState(
        workflow_id=uuid.uuid4().hex,
        request_id=request_id,
        origin_country=(slots.get("origin_country") or "US"),
        destination_country=slots["destination_country"],
        declared_value_usd=float(slots.get("declared_value_usd") or 0.0),
        weight_lbs=float(slots.get("weight_lbs") or 0.0),
        description=slots.get("description") or "",
        category=slots.get("category"),
    )
    result_ws = await workflow.process(ws)
    decisions.extend(result_ws.decisions)
    return ConciergeResult(
        reply=ack + _workflow_summary(result_ws),
        state=state,
        dispatched_to="workflow",
        decisions=decisions,
        provider=result_ws.compliance.provider if result_ws.compliance else "",
    )


def _workflow_summary(ws: WorkflowState) -> str:
    """A conversational summary of a (possibly suspended) workflow run."""
    if ws.status == "awaiting_review":
        areas = ", ".join(ws.pending_review_areas) or "a compliance item"
        return (
            f"This international shipment to {ws.destination_country} needs a quick human "
            f"compliance review ({areas}) before it can proceed — I've queued it for an "
            f"officer. Reference: {ws.workflow_id}."
        )
    parts: list[str] = []
    if ws.recommended_carrier is not None:
        c = ws.recommended_carrier
        parts.append(
            f"best option {c.carrier} {c.service} ~${c.price_usd:.0f} in {c.estimated_days}d"
        )
    if ws.landed_cost is not None:
        parts.append(f"landed cost ~${ws.landed_cost.total_landed_usd:.0f}")
    if ws.compliance is not None:
        parts.append(f"compliance {ws.compliance.verdict.replace('_', ' ')}")
    if ws.hs_code:
        parts.append(f"HS {ws.hs_code}")
    if ws.documents:
        parts.append(f"{len(ws.documents)} document(s) drafted")
    body = "; ".join(parts) if parts else "processed end to end"
    return (
        f"Done — I ran the full process for your international shipment to "
        f"{ws.destination_country}: {body}. (workflow {ws.workflow_id})"
    )
