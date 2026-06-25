"""Tests for the compliance route — POST /api/v1/compliance/check.

Wires app.state.llm_router + app.state.rag to keyless components (EchoClient,
LocalHashEmbedding, InMemoryVectorStore) so the route runs end to end without a
live MCP or API keys. Notably, the compliance route does NOT require
app.state.tool_registry (the keyless-friendly dependency policy), which these
tests assert by leaving it unset.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.core.config as config_mod
from app.core.audit import InMemoryAuditSink
from app.llm.client import EchoClient
from app.llm.router import TASK_FALLBACK, TASK_REASONING, TASK_SYNTHESIS, LLMRouter
from app.main import app
from app.rag.embeddings import LocalHashEmbedding
from app.rag.vector_store import InMemoryVectorStore

_VALID = {
    "origin_country": "US",
    "destination_country": "DE",
    "declared_value_usd": 0,
    "description": "20000mAh power bank",
}


@pytest.fixture(autouse=True)
def _setup_app_state():
    """Hydrate llm_router + rag + audit sink on app.state; leave tool_registry unset."""
    echo = EchoClient()
    app.state.tool_registry = None  # compliance must work WITHOUT the MCP registry
    app.state.llm_router = LLMRouter(
        clients={TASK_REASONING: echo, TASK_SYNTHESIS: echo, TASK_FALLBACK: echo},
        fallback=echo,
    )
    app.state.rag = {
        "embedding_provider": LocalHashEmbedding(dims=16),
        "vector_store": InMemoryVectorStore(),
        "llm_client": echo,
    }
    app.state.audit_sink = InMemoryAuditSink()
    yield


client = TestClient(app)


# ── Happy path (works without tool_registry) ──────────────────────────────────


def test_compliance_check_returns_verdict_and_trace():
    response = client.post("/api/v1/compliance/check", json=_VALID)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] in {"action_required", "review_recommended", "advisory"}
    assert "compliance:plan" in data["decisions"]
    assert isinstance(data["findings"], list) and data["findings"]
    assert data["provider"] == "echo"


# ── Validation ────────────────────────────────────────────────────────────────


def test_compliance_check_invalid_country_code_422():
    bad = {**_VALID, "origin_country": "USA"}  # must be ISO alpha-2
    assert client.post("/api/v1/compliance/check", json=bad).status_code == 422


def test_compliance_check_missing_body_422():
    assert client.post("/api/v1/compliance/check").status_code == 422


# ── 503 when dependencies missing ─────────────────────────────────────────────


def test_compliance_check_503_without_llm_router():
    app.state.llm_router = None
    assert client.post("/api/v1/compliance/check", json=_VALID).status_code == 503


def test_compliance_check_503_without_rag():
    app.state.rag = None
    assert client.post("/api/v1/compliance/check", json=_VALID).status_code == 503


# ── Feature flag ──────────────────────────────────────────────────────────────


def test_compliance_check_404_when_disabled(monkeypatch):
    monkeypatch.setattr(config_mod.settings, "compliance_enabled", False, raising=False)
    assert client.post("/api/v1/compliance/check", json=_VALID).status_code == 404


# ── Shipping scope (domestic-only deployment) ─────────────────────────────────


def test_compliance_check_422_cross_border_when_domestic(monkeypatch):
    monkeypatch.setattr(config_mod.settings, "shipping_scope", "domestic", raising=False)
    monkeypatch.setattr(config_mod.settings, "domestic_country", "US", raising=False)
    # _VALID is US -> DE → cross-border → rejected in a domestic-only deployment.
    assert client.post("/api/v1/compliance/check", json=_VALID).status_code == 422


def test_compliance_check_allows_domestic_shipment_when_domestic(monkeypatch):
    monkeypatch.setattr(config_mod.settings, "shipping_scope", "domestic", raising=False)
    monkeypatch.setattr(config_mod.settings, "domestic_country", "US", raising=False)
    domestic = {**_VALID, "destination_country": "US"}
    assert client.post("/api/v1/compliance/check", json=domestic).status_code == 200


# ── /ready surfaces compliance_enabled ────────────────────────────────────────


def test_ready_reports_compliance_enabled():
    response = client.get("/ready")
    assert response.status_code == 200
    assert "compliance_enabled" in response.json()
