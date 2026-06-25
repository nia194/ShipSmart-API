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
