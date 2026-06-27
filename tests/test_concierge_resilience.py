"""Resilience: a conversation-store failure must never break a chat turn.

Persistence is best-effort (the route wraps load + persist in try/except); a turn
should still return its reply + merged state when the store is down.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.core.config as config_mod
from app.llm.client import EchoClient
from app.llm.router import TASK_FALLBACK, TASK_REASONING, TASK_SYNTHESIS, LLMRouter
from app.main import app
from app.rag.embeddings import LocalHashEmbedding
from app.rag.vector_store import InMemoryVectorStore


class _BoomStore:
    """A ConversationStore whose every operation fails (simulates the DB being down)."""

    async def load(self, *args, **kwargs):
        raise RuntimeError("conversation store down")

    async def upsert_state(self, *args, **kwargs):
        raise RuntimeError("conversation store down")

    async def append_messages(self, *args, **kwargs):
        raise RuntimeError("conversation store down")


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
    app.state.conversation_store = _BoomStore()
    yield
    app.state.conversation_store = None


def test_concierge_turn_survives_conversation_store_failure(_wired):
    client = TestClient(app)
    r = client.post(
        "/api/v1/concierge/chat",
        json={
            "message": "ship from Atlanta to Seattle, 10 lb",
            "session_id": "11111111111111111111111111111111",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reply"]  # the turn succeeded despite recall/persist failing
    assert body["state"]["slots"].get("destination")
