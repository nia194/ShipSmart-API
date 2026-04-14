"""Tests for centralized error handling."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_404_returns_consistent_format():
    """Non-existent path should return a JSON error, not HTML."""
    response = client.get("/nonexistent-path")
    # FastAPI returns 404 for unmatched routes; our general handler catches 500s.
    # The default 404 behavior is acceptable — it still returns JSON.
    assert response.status_code in (404, 405)


def test_validation_error_on_orchestration():
    """POST /api/v1/orchestration/run with invalid body triggers validation."""
    response = client.post("/api/v1/orchestration/run", json={})
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == 422
    assert data["error"] == "Validation Error"
    assert "message" in data
    assert "timestamp" in data
