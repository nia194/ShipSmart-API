"""Tests for the workflow routes (UC3 + UC4).

  POST /api/v1/workflow/process        — run; may suspend for human review
  GET  /api/v1/workflow/{id}           — current persisted state
  POST /api/v1/workflow/{id}/review    — officer determination

Keyless end-to-end: app.state wired to EchoClient + LocalHashEmbedding + an empty
InMemoryVectorStore + the default mock domain providers, plus fresh checkpointer /
review-queue singletons per test. With the empty store the default high-risk areas
are unverified, so a plain process suspends (awaiting_review); tests that want a
straight-through completion clear WORKFLOW_HIGH_RISK_AREAS. The slowapi limiter is
disabled here so the lifecycle tests don't trip the per-minute cap.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.core.config as config_mod
from app.core.audit import InMemoryAuditSink
from app.core.rate_limit import limiter
from app.domain.adapters import default_providers
from app.llm.client import EchoClient
from app.llm.router import TASK_FALLBACK, TASK_REASONING, TASK_SYNTHESIS, LLMRouter
from app.main import app
from app.rag.embeddings import LocalHashEmbedding
from app.rag.vector_store import InMemoryVectorStore
from app.workflow.checkpointer import InMemoryCheckpointer
from app.workflow.review_queue import InMemoryReviewQueue

_VALID = {
    "origin_country": "US",
    "destination_country": "DE",
    "declared_value_usd": 1000,
    "weight_lbs": 5,
    "description": "a 20000mAh power bank",
}


@pytest.fixture(autouse=True)
def _setup_app_state(monkeypatch):
    """Enable workflow, wire keyless deps + fresh durability singletons, no limiter."""
    monkeypatch.setattr(config_mod.settings, "workflow_enabled", True, raising=False)
    monkeypatch.setattr(limiter, "enabled", False, raising=False)
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
    app.state.workflow_checkpointer = InMemoryCheckpointer()
    app.state.review_queue = InMemoryReviewQueue()
    yield


client = TestClient(app)


# ── process: interrupt (default high-risk + empty KB) ─────────────────────────


def test_process_suspends_for_review_on_high_risk():
    data = client.post("/api/v1/workflow/process", json=_VALID).json()
    assert data["status"] == "awaiting_review"
    assert data["pending_review_areas"]
    assert "workflow:interrupt:human_review" in data["decisions"]


def test_process_completes_when_no_high_risk_configured(monkeypatch):
    monkeypatch.setattr(config_mod.settings, "workflow_high_risk_areas", "", raising=False)
    data = client.post("/api/v1/workflow/process", json=_VALID).json()
    assert data["status"] == "completed"
    assert data["hs_code"] == "8507.60"
    assert "workflow:complete" in data["decisions"]


# ── Shipping scope (domestic-only deployment) ─────────────────────────────────


def test_process_422_cross_border_when_domestic(monkeypatch):
    monkeypatch.setattr(config_mod.settings, "shipping_scope", "domestic", raising=False)
    monkeypatch.setattr(config_mod.settings, "domestic_country", "US", raising=False)
    # _VALID is US -> DE → rejected before the workflow runs.
    assert client.post("/api/v1/workflow/process", json=_VALID).status_code == 422


# ── GET /{id} ─────────────────────────────────────────────────────────────────


def test_get_returns_persisted_state():
    wid = client.post("/api/v1/workflow/process", json=_VALID).json()["workflow_id"]
    got = client.get(f"/api/v1/workflow/{wid}")
    assert got.status_code == 200
    assert got.json()["workflow_id"] == wid
    assert got.json()["status"] == "awaiting_review"


def test_get_unknown_id_404():
    assert client.get("/api/v1/workflow/does-not-exist").status_code == 404


# ── review → resume ───────────────────────────────────────────────────────────


def test_review_cleared_completes():
    wid = client.post("/api/v1/workflow/process", json=_VALID).json()["workflow_id"]
    resp = client.post(f"/api/v1/workflow/{wid}/review",
                       json={"determination": "cleared", "note": "ok"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["officer_determination"] == "cleared"
    assert data["documents"]


def test_review_blocked_terminates():
    wid = client.post("/api/v1/workflow/process", json=_VALID).json()["workflow_id"]
    data = client.post(f"/api/v1/workflow/{wid}/review",
                       json={"determination": "blocked"}).json()
    assert data["status"] == "blocked"
    assert data["documents"] == []


def test_review_unknown_id_404():
    resp = client.post("/api/v1/workflow/nope/review", json={"determination": "cleared"})
    assert resp.status_code == 404


def test_review_conflict_when_not_awaiting(monkeypatch):
    monkeypatch.setattr(config_mod.settings, "workflow_high_risk_areas", "", raising=False)
    wid = client.post("/api/v1/workflow/process", json=_VALID).json()["workflow_id"]  # completes
    resp = client.post(f"/api/v1/workflow/{wid}/review", json={"determination": "cleared"})
    assert resp.status_code == 409


def test_review_invalid_determination_422():
    wid = client.post("/api/v1/workflow/process", json=_VALID).json()["workflow_id"]
    resp = client.post(f"/api/v1/workflow/{wid}/review", json={"determination": "maybe"})
    assert resp.status_code == 422


# ── validation / dependencies / flag ──────────────────────────────────────────


def test_process_invalid_country_422():
    assert client.post(
        "/api/v1/workflow/process", json={**_VALID, "destination_country": "DEU"}
    ).status_code == 422


def test_process_503_without_llm_router():
    app.state.llm_router = None
    assert client.post("/api/v1/workflow/process", json=_VALID).status_code == 503


def test_process_503_without_rag():
    app.state.rag = None
    assert client.post("/api/v1/workflow/process", json=_VALID).status_code == 503


def test_process_404_when_disabled(monkeypatch):
    monkeypatch.setattr(config_mod.settings, "workflow_enabled", False, raising=False)
    assert client.post("/api/v1/workflow/process", json=_VALID).status_code == 404


def test_ready_reports_workflow_flags():
    body = client.get("/ready").json()
    assert "workflow_enabled" in body and "workflow_durable" in body
