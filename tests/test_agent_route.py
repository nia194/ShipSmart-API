"""Tests for the agent (Concierge) route — POST /api/v1/agent/run.

Wires `app.state.tool_registry` to a MockTransport-backed RemoteToolRegistry and
`app.state.rag` / `app.state.llm_router` to keyless components (EchoClient), so
the route runs end to end without a live MCP or API keys — mirroring the
orchestration route tests.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

import app.core.config as config_mod
from app.llm.client import EchoClient
from app.llm.router import TASK_FALLBACK, TASK_REASONING, TASK_SYNTHESIS, LLMRouter
from app.main import app
from app.rag.embeddings import LocalHashEmbedding
from app.rag.vector_store import InMemoryVectorStore
from app.services.mcp_client import create_remote_registry
from tests.conftest import build_mcp_mock_transport


@pytest.fixture(autouse=True)
def _setup_app_state():
    """Hydrate tool registry + RAG pipeline + LLM router on app.state per test."""
    transport = build_mcp_mock_transport()
    registry = asyncio.run(
        create_remote_registry(base_url="http://mcp.test", api_key="", transport=transport)
    )
    echo = EchoClient()
    app.state.tool_registry = registry
    app.state.llm_router = LLMRouter(
        clients={TASK_REASONING: echo, TASK_SYNTHESIS: echo, TASK_FALLBACK: echo},
        fallback=echo,
    )
    app.state.rag = {
        "embedding_provider": LocalHashEmbedding(dims=16),
        "vector_store": InMemoryVectorStore(),
        "llm_client": echo,
    }
    yield
    asyncio.run(registry.aclose())
    app.state.tool_registry = None


client = TestClient(app)


# ── Happy path ───────────────────────────────────────────────────────────────


def test_agent_run_returns_answer_and_trace():
    response = client.post("/api/v1/agent/run", json={
        "query": "How do I ship a power bank?",
    })
    assert response.status_code == 200
    data = response.json()
    # Echo reasoning client has no native tool calling → text-fallback path.
    assert data["answer"]
    assert "agent:fallback:text" in data["decisions"]
    assert data["provider"] == "echo"
    assert isinstance(data["steps"], list)
    assert isinstance(data["tools_used"], list)
    assert isinstance(data["sources"], list)


def test_agent_run_accepts_context():
    response = client.post("/api/v1/agent/run", json={
        "query": "Is my address deliverable?",
        "context": {"street": "123 Main St", "city": "NYC", "state": "NY", "zip_code": "10001"},
    })
    assert response.status_code == 200
    assert response.json()["answer"]


# ── Validation ───────────────────────────────────────────────────────────────


def test_agent_run_empty_query_422():
    response = client.post("/api/v1/agent/run", json={"query": ""})
    assert response.status_code == 422


def test_agent_run_missing_body_422():
    response = client.post("/api/v1/agent/run")
    assert response.status_code == 422


# ── 503 when registry missing ────────────────────────────────────────────────


def test_agent_run_503_without_registry():
    app.state.tool_registry = None
    response = client.post("/api/v1/agent/run", json={"query": "hello"})
    assert response.status_code == 503


# ── Feature flag ─────────────────────────────────────────────────────────────


def test_agent_run_404_when_disabled(monkeypatch):
    monkeypatch.setattr(config_mod.settings, "agent_enabled", False, raising=False)
    response = client.post("/api/v1/agent/run", json={"query": "hello"})
    assert response.status_code == 404


# ── /ready surfaces agent_enabled ────────────────────────────────────────────


def test_ready_reports_agent_enabled():
    response = client.get("/ready")
    assert response.status_code == 200
    assert "agent_enabled" in response.json()
