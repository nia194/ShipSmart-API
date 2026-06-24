"""Route tests for POST /api/v1/concierge/chat (keyless app.state wiring)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.core.config as config_mod
from app.llm.client import EchoClient
from app.llm.router import TASK_FALLBACK, TASK_REASONING, TASK_SYNTHESIS, LLMRouter
from app.main import app
from app.rag.embeddings import LocalHashEmbedding
from app.rag.vector_store import InMemoryVectorStore


@pytest.fixture(autouse=True)
def _state(monkeypatch):
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
    yield


client = TestClient(app)


def test_chat_clarifies_when_thin():
    r = client.post("/api/v1/concierge/chat", json={"message": "I want to ship something"})
    assert r.status_code == 200
    data = r.json()
    assert data["clarification"]
    assert data["state"]["status"] == "gathering"
    assert any(d.startswith("concierge:clarify:") for d in data["decisions"])


def test_chat_merges_and_echoes_state():
    r = client.post("/api/v1/concierge/chat", json={
        "message": "ship from Atlanta to Seattle weighing 10 lb",
        "state": {"slots": {"priority": "speed"}},
    })
    assert r.status_code == 200
    slots = r.json()["state"]["slots"]
    assert slots["destination"]
    assert slots["weight_lbs"] == 10.0
    assert slots["priority"] == "speed"  # prior slot preserved through the turn


def test_chat_404_when_disabled(monkeypatch):
    monkeypatch.setattr(config_mod.settings, "concierge_enabled", False, raising=False)
    r = client.post("/api/v1/concierge/chat", json={"message": "hello"})
    assert r.status_code == 404


def test_chat_empty_message_422():
    r = client.post("/api/v1/concierge/chat", json={"message": ""})
    assert r.status_code == 422


def test_ready_reports_concierge_enabled():
    r = client.get("/ready")
    assert r.status_code == 200
    assert "concierge_enabled" in r.json()
