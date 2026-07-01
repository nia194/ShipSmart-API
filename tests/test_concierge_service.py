"""run_concierge pipeline tests — keyless (EchoClient + in-memory RAG)."""

from __future__ import annotations

from app.agents.concierge.models import ConversationState
from app.agents.concierge.service import run_concierge
from app.llm.client import EchoClient
from app.llm.router import TASK_FALLBACK, TASK_REASONING, TASK_SYNTHESIS, LLMRouter
from app.rag.embeddings import LocalHashEmbedding
from app.rag.vector_store import InMemoryVectorStore


def _deps():
    echo = EchoClient()
    return {
        "llm_router": LLMRouter(
            clients={TASK_REASONING: echo, TASK_SYNTHESIS: echo, TASK_FALLBACK: echo},
            fallback=echo,
        ),
        "embedding_provider": LocalHashEmbedding(dims=16),
        "vector_store": InMemoryVectorStore(),
    }


async def test_clarifies_for_missing_slot():
    res = await run_concierge("I want to ship something", None, **_deps())
    assert res.clarification
    assert res.dispatched_to is None
    assert any(d.startswith("concierge:clarify:") for d in res.decisions)


async def test_does_not_reask_when_slots_present():
    # form-provided slots (the hybrid-sync case): compliance has what it needs.
    state = ConversationState(
        slots={"destination_country": "BR", "description": "camera drone"},
        intent="compliance",
    )
    res = await run_concierge("is this compliant?", state, **_deps())
    assert not any(d.startswith("concierge:clarify:") for d in res.decisions)
    assert res.dispatched_to == "compliance"
    assert res.state.status == "answered"


async def test_compliance_intent_falls_through_when_explicit_off(monkeypatch):
    # Switch off ⇒ the explicit compliance pass is skipped; the compliance intent
    # falls through to the normal flow (no tool_registry here ⇒ summary fallback).
    import app.agents.concierge.service as svc

    monkeypatch.setattr(svc.settings, "compliance_explicit_enabled", False)
    state = ConversationState(
        slots={"destination_country": "BR", "description": "camera drone"},
        intent="compliance",
    )
    res = await run_concierge("is this compliant?", state, **_deps())
    assert res.dispatched_to != "compliance"
    assert "concierge:compliance:explicit_skipped" in res.decisions


async def test_domestic_scope_blocks_international_destination(monkeypatch):
    # Domestic deployment + an explicit international destination ⇒ graceful refusal.
    import app.agents.concierge.service as svc

    monkeypatch.setattr(svc.settings, "shipping_scope", "domestic", raising=False)
    monkeypatch.setattr(svc.settings, "domestic_country", "US", raising=False)
    state = ConversationState(
        slots={"destination_country": "DE", "description": "camera drone"},
        intent="compliance",
    )
    res = await run_concierge("is this compliant?", state, **_deps())
    assert res.dispatched_to == "scope_blocked"
    assert "concierge:scope:domestic_only" in res.decisions


async def test_domestic_scope_skips_destination_clarification(monkeypatch):
    # destination_country is normally required for compliance; domestic mode defaults
    # it to home, so the concierge must not stop to ask for it.
    import app.agents.concierge.service as svc

    monkeypatch.setattr(svc.settings, "shipping_scope", "domestic", raising=False)
    monkeypatch.setattr(svc.settings, "domestic_country", "US", raising=False)
    state = ConversationState(slots={"description": "phone charger"}, intent="compliance")
    res = await run_concierge("is this compliant?", state, **_deps())
    assert not any(d.startswith("concierge:clarify:") for d in res.decisions)
    assert res.dispatched_to == "compliance"


async def test_merges_and_echoes_full_state():
    state = ConversationState(slots={"origin": "Atlanta, GA"})
    res = await run_concierge("ship from Atlanta to Seattle weighing 10 lb", state, **_deps())
    assert res.state.slots["origin"] == "Atlanta, GA"   # equivalent restatement kept
    assert res.state.slots["destination"]
    assert res.state.slots["weight_lbs"] == 10.0


async def test_reply_to_tags_decision_path():
    # A reply with a referenced message tags the trail; bounded reference is built even
    # keyless (the tag is set from the reference, independent of the LLM NLU path).
    state = ConversationState(
        slots={"origin": "Atlanta", "destination": "Seattle", "weight_lbs": 10.0},
        intent="quote",
    )
    res = await run_concierge(
        "why not the cheaper one?", state,
        reply_to={"role": "assistant", "text": "FedEx fastest, LuggageToShip cheapest"},
        recent_history=[{"role": "user", "text": "show options"}],
        **_deps(),
    )
    assert "concierge:reply_to" in res.decisions


async def test_no_reply_to_has_no_tag():
    res = await run_concierge("ship from Atlanta to Seattle weighing 10 lb", None, **_deps())
    assert "concierge:reply_to" not in res.decisions


# ── greeting / smalltalk + keyless dispatch (regression) ──────────────────────
async def test_greeting_returns_friendly_intro():
    # A pure greeting must be welcomed + oriented, not dispatched to the agent (which
    # keyless returns generic RAG boilerplate).
    res = await run_concierge("hi", None, **_deps())
    assert "concierge:greeting" in res.decisions
    assert res.dispatched_to is None
    assert res.reply.startswith("Hi!")
    assert res.state.status == "gathering"


async def test_greeting_with_content_is_not_smalltalk():
    # A greeting carrying shipping content must flow normally, not be swallowed.
    res = await run_concierge(
        "hi, ship from Atlanta to Seattle weighing 10 lb", None, **_deps(),
    )
    assert "concierge:greeting" not in res.decisions
    assert res.state.slots.get("destination") == "Seattle"


async def test_keyless_dispatch_uses_summary_not_agent_boilerplate():
    # Even with a tool registry present, keyless (EchoClient) must skip the agent —
    # whose keyless answer is generic boilerplate — and return a deterministic summary.
    state = ConversationState(
        slots={"origin": "Boston", "destination": "Miami", "weight_lbs": 10.0},
        intent="quote",
    )
    res = await run_concierge("get me a quote", state, tool_registry=object(), **_deps())
    assert res.dispatched_to == "summary"
    assert "Based on available shipping information" not in res.reply
    assert "I have" in res.reply


def test_advisor_context_resolves_city_zip_and_default_dims():
    # A quote needs US ZIPs + dimensions; the concierge gathers cities + no dims.
    from app.agents.concierge.service import _advisor_context

    ctx = _advisor_context({"origin": "Chicago", "destination": "Denver", "weight_lbs": 5.0})
    assert ctx["origin_zip"] == "60601"
    assert ctx["destination_zip"] == "80202"
    assert ctx["length_in"] == 12.0 and ctx["width_in"] == 9.0 and ctx["height_in"] == 6.0


def test_advisor_context_keeps_real_zip_and_given_dims():
    from app.agents.concierge.service import _advisor_context

    ctx = _advisor_context(
        {"origin": "94102", "destination": "10001", "weight_lbs": 3.0, "length_in": 20.0}
    )
    assert ctx["origin_zip"] == "94102" and ctx["destination_zip"] == "10001"
    assert ctx["length_in"] == 20.0  # user-given dim preserved
    assert ctx["width_in"] == 9.0    # missing dims defaulted


def test_agent_query_for_shapes_an_explicit_query():
    from app.agents.concierge.service import _agent_query_for

    q = _agent_query_for("quote", "about 5 lbs", {})
    assert "option" in q.lower() and "5 lbs" not in q
    assert "ABC123" in _agent_query_for("tracking", "where is it", {"tracking_reference": "ABC123"})
    assert _agent_query_for("advice", "cheapest carrier?", {}) == "cheapest carrier?"


async def test_dispatch_degrades_to_summary_on_llm_provider_error(monkeypatch):
    # A live provider that errors mid-dispatch (bad/expired key, outage) must NOT 502
    # a user-facing chat — it degrades to the deterministic summary.
    import app.agents.concierge.service as svc
    from app.llm.errors import AuthError

    async def boom_agent(*args, **kwargs):
        raise AuthError(provider="openai")

    monkeypatch.setattr(svc, "run_agent", boom_agent)

    class _Provider:  # non-keyless ⇒ dispatch actually attempts the agent
        provider_name = "openai"

        async def complete(self, messages, **kw):
            return ""

    router = LLMRouter(
        clients={TASK_REASONING: _Provider(), TASK_SYNTHESIS: _Provider(),
                 TASK_FALLBACK: EchoClient()},
        fallback=EchoClient(),
    )
    state = ConversationState(
        slots={"origin": "Boston", "destination": "Miami", "weight_lbs": 10.0},
        intent="quote",
    )
    res = await run_concierge(
        "get me a quote", state, llm_router=router,
        embedding_provider=LocalHashEmbedding(dims=16),
        vector_store=InMemoryVectorStore(), tool_registry=object(),
    )
    assert res.dispatched_to == "summary"
    assert "concierge:dispatch:llm_degraded" in res.decisions
    assert "I have" in res.reply
