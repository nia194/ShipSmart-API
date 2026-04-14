"""
Failure scenario tests for pre-deployment validation.
Tests graceful degradation when components are missing or fail.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


# ── Missing RAG State ──────────────────────────────────────────────────────


def test_rag_query_without_state():
    """RAG endpoint returns 503 when state is not initialized."""
    # Remove RAG state if present
    if hasattr(app.state, "rag"):
        saved = app.state.rag
        delattr(app.state, "rag")
    else:
        saved = None

    try:
        response = client.post("/api/v1/rag/query", json={
            "query": "test",
            "top_k": 3,
        })
        assert response.status_code == 503
    finally:
        if saved is not None:
            app.state.rag = saved


# ── Empty Services for Recommendation ─────────────────────────────────────


def test_recommendation_empty_services():
    """Recommendation endpoint handles empty services gracefully."""
    response = client.post("/api/v1/advisor/recommendation", json={
        "services": [],
    })
    assert response.status_code == 200
    data = response.json()
    assert data["primary_recommendation"]["service_name"] == "N/A"
    assert "No shipping services available" in data["summary"]


# ── Single Service Recommendation ─────────────────────────────────────────


def test_recommendation_single_service():
    """Recommendation with one service returns it as primary, no alternatives."""
    response = client.post("/api/v1/advisor/recommendation", json={
        "services": [
            {"service": "Ground", "price_usd": 9.99, "estimated_days": 5},
        ],
    })
    assert response.status_code == 200
    data = response.json()
    assert data["primary_recommendation"]["service_name"] == "Ground"
    assert len(data["alternatives"]) == 0


# ── Invalid Request Bodies ────────────────────────────────────────────────


def test_shipping_advisor_missing_query():
    """Shipping advisor rejects missing query field."""
    response = client.post("/api/v1/advisor/shipping", json={})
    assert response.status_code == 422


def test_tracking_advisor_missing_issue():
    """Tracking advisor rejects missing issue field."""
    response = client.post("/api/v1/advisor/tracking", json={})
    assert response.status_code == 422


def test_recommendation_missing_services():
    """Recommendation rejects missing services field."""
    response = client.post("/api/v1/advisor/recommendation", json={
        "context": {"fragile": True},
    })
    assert response.status_code == 422


def test_recommendation_partial_service_still_works():
    """Recommendation handles services with missing optional fields gracefully."""
    response = client.post("/api/v1/advisor/recommendation", json={
        "services": [{"service": "Ground"}],
    })
    # Defaults to price_usd=0.0, estimated_days=0 — still processes
    assert response.status_code == 200
    data = response.json()
    assert data["primary_recommendation"]["service_name"] == "Ground"


# ── Health & Readiness Under Load ─────────────────────────────────────────


def test_health_always_responds():
    """Health endpoint never fails."""
    for _ in range(10):
        response = client.get("/health")
        assert response.status_code == 200


def test_ready_always_responds():
    """Ready endpoint never fails."""
    for _ in range(10):
        response = client.get("/ready")
        assert response.status_code == 200
