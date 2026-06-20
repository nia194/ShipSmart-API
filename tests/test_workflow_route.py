"""Tests for the workflow route — POST /api/v1/workflow/process (UC3).

Keyless end-to-end: app.state wired to EchoClient + LocalHashEmbedding +
InMemoryVectorStore + the default mock domain providers. The feature is OFF by
default, so the fixture enables it; one test asserts the disabled 404.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.core.config as config_mod
from app.core.audit import InMemoryAuditSink
from app.domain.adapters import default_providers
from app.llm.client import EchoClient
from app.llm.router import TASK_FALLBACK, TASK_REASONING, TASK_SYNTHESIS, LLMRouter
from app.main import app
from app.rag.embeddings import LocalHashEmbedding
from app.rag.vector_store import InMemoryVectorStore

_VALID = {
    "origin_country": "US",
    "destination_country": "DE",
    "declared_value_usd": 1000,
    "weight_lbs": 5,
    "description": "a 20000mAh power bank",
}


@pytest.fixture(autouse=True)
def _setup_app_state(monkeypatch):
    """Enable the workflow flag and wire keyless deps onto app.state."""
    monkeypatch.setattr(config_mod.settings, "workflow_enabled", True, raising=False)
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
    app.state.domain = default_providers()
    app.state.audit_sink = InMemoryAuditSink()
    yield


client = TestClient(app)


# ── Happy path ────────────────────────────────────────────────────────────────


def test_workflow_process_returns_completed_result():
    response = client.post("/api/v1/workflow/process", json=_VALID)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["workflow_id"]
    assert data["hs_code"] == "8507.60"
    assert data["recommended_carrier"] is not None
    assert data["compliance"]["verdict"]
    assert "workflow:start" in data["decisions"]
    assert "workflow:complete" in data["decisions"]


# ── Validation ────────────────────────────────────────────────────────────────


def test_workflow_process_invalid_country_422():
    bad = {**_VALID, "destination_country": "DEU"}
    assert client.post("/api/v1/workflow/process", json=bad).status_code == 422


# ── 503 when dependencies missing ─────────────────────────────────────────────


def test_workflow_process_503_without_llm_router():
    app.state.llm_router = None
    assert client.post("/api/v1/workflow/process", json=_VALID).status_code == 503


def test_workflow_process_503_without_rag():
    app.state.rag = None
    assert client.post("/api/v1/workflow/process", json=_VALID).status_code == 503


# ── Feature flag ──────────────────────────────────────────────────────────────


def test_workflow_process_404_when_disabled(monkeypatch):
    monkeypatch.setattr(config_mod.settings, "workflow_enabled", False, raising=False)
    assert client.post("/api/v1/workflow/process", json=_VALID).status_code == 404


# ── /ready surfaces workflow_enabled ──────────────────────────────────────────


def test_ready_reports_workflow_enabled():
    assert "workflow_enabled" in client.get("/ready").json()
