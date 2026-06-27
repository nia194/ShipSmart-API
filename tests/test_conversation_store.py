"""Conversation store + concierge recall tests — keyless (in-memory store)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.core.config as config_mod
from app.agents.concierge.models import ConversationState
from app.agents.concierge.service import reconcile_recall
from app.conversations.store import (
    ConversationMessage,
    ConversationRecord,
    InMemoryConversationStore,
    PostgresConversationStore,
    create_conversation_store,
)
from app.llm.client import EchoClient
from app.llm.router import TASK_FALLBACK, TASK_REASONING, TASK_SYNTHESIS, LLMRouter
from app.main import app
from app.rag.embeddings import LocalHashEmbedding
from app.rag.vector_store import InMemoryVectorStore


# ── store unit tests ─────────────────────────────────────────────────────────
async def test_inmemory_round_trips_state_and_messages():
    store = InMemoryConversationStore()
    await store.upsert_state(
        "s1", status="answered", intent="quote",
        slots={"origin": "Atlanta", "weight_lbs": 12.0}, turns=2,
        last_dispatched_to="agent",
    )
    await store.append_messages("s1", [
        ConversationMessage(role="user", content="ship 12 lb"),
        ConversationMessage(role="assistant", content="where to?"),
    ])
    rec = await store.load("s1")
    assert rec is not None
    assert rec.intent == "quote"
    assert rec.slots["weight_lbs"] == 12.0
    assert [m.role for m in rec.messages] == ["user", "assistant"]


async def test_inmemory_upsert_preserves_created_at_and_caps_recall():
    store = InMemoryConversationStore()
    await store.upsert_state("s2", status="gathering", intent=None, slots={}, turns=1,
                             last_dispatched_to=None)
    first = await store.load("s2")
    await store.upsert_state("s2", status="answered", intent="quote", slots={"origin": "X"},
                             turns=2, last_dispatched_to="agent")
    second = await store.load("s2")
    assert first.created_at == second.created_at  # created_at is sticky across upserts
    for i in range(5):
        await store.append_messages("s2", [ConversationMessage(role="user", content=f"m{i}")])
    capped = await store.load("s2", limit=3)
    assert len(capped.messages) == 3
    assert capped.messages[-1].content == "m4"  # newest kept, oldest-first order


async def test_load_unknown_session_is_none():
    assert await InMemoryConversationStore().load("nope") is None


def test_factory_defaults_to_memory_and_postgres_by_name():
    assert isinstance(create_conversation_store(), InMemoryConversationStore)
    assert isinstance(create_conversation_store("weird"), InMemoryConversationStore)
    assert isinstance(
        create_conversation_store("postgres", "postgresql://x"), PostgresConversationStore,
    )


def test_postgres_requires_dsn():
    with pytest.raises(ValueError):
        create_conversation_store("postgres", "")


# ── recall reconcile (pure) ──────────────────────────────────────────────────
def test_reconcile_returns_client_when_no_store_record():
    cs = ConversationState(slots={"origin": "Atlanta"})
    assert reconcile_recall(cs, None) is cs


def test_reconcile_prefers_active_client_state():
    cs = ConversationState(slots={"origin": "Atlanta"}, turns=1)
    stored = ConversationRecord(session_id="s", slots={"origin": "Boston"}, turns=3)
    assert reconcile_recall(cs, stored).slots["origin"] == "Atlanta"


def test_reconcile_rehydrates_fresh_client_from_store():
    fresh = ConversationState()  # page reload: empty slots, turn 0
    stored = ConversationRecord(
        session_id="s", slots={"origin": "Boston"}, intent="quote",
        status="answered", turns=3,
    )
    out = reconcile_recall(fresh, stored)
    assert out.slots["origin"] == "Boston"
    assert out.intent == "quote"
    assert out.turns == 3


# ── route: session id + recall ───────────────────────────────────────────────
@pytest.fixture
def _wired(monkeypatch):
    monkeypatch.setattr(config_mod.settings, "concierge_enabled", True, raising=False)
    echo = EchoClient()
    app.state.llm_router = LLMRouter(
        clients={TASK_REASONING: echo, TASK_SYNTHESIS: echo, TASK_FALLBACK: echo},
        fallback=echo,
    )
    app.state.rag = {
        "embedding_provider": LocalHashEmbedding(dims=16),
        "vector_store": InMemoryVectorStore(),
        "llm_client": echo,
    }
    app.state.audit_sink = None
    app.state.tool_registry = None
    app.state.conversation_store = InMemoryConversationStore()
    yield
    app.state.conversation_store = None


def test_chat_mints_and_echoes_session_id(_wired):
    client = TestClient(app)
    r = client.post(
        "/api/v1/concierge/chat",
        json={"message": "ship from Atlanta to Seattle, 10 lb"},
    )
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert sid
    # history is queryable and replays the turn
    h = client.get(f"/api/v1/concierge/{sid}")
    assert h.status_code == 200
    body = h.json()
    assert body["session_id"] == sid
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
    assert body["state"]["slots"]["destination"]


def test_recall_continues_after_reload(_wired):
    client = TestClient(app)
    sid = client.post(
        "/api/v1/concierge/chat",
        json={"message": "ship from Atlanta to Seattle weighing 10 lb"},
    ).json()["session_id"]
    # "reload": a fresh client (no state) re-sends the known session_id
    r = client.post(
        "/api/v1/concierge/chat",
        json={"message": "what's the cheapest option?", "session_id": sid},
    )
    assert r.status_code == 200
    slots = r.json()["state"]["slots"]
    assert slots.get("destination")          # recalled from the server snapshot
    assert slots.get("weight_lbs") == 10.0


def test_history_404_for_unknown_session(_wired):
    client = TestClient(app)
    assert client.get("/api/v1/concierge/deadbeef").status_code == 404
