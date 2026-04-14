"""Tests for the /api/v1/info endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_info_returns_metadata():
    response = client.get("/api/v1/info")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "shipsmart-api-python"
    assert data["version"] == "0.1.0"
    assert "env" in data
    assert "llm_provider" in data
    assert "rag_provider" in data
    assert "embedding_provider" in data


def test_info_does_not_leak_secrets():
    response = client.get("/api/v1/info")
    data = response.json()
    text = str(data)
    assert "api_key" not in text.lower()
    assert "secret" not in text.lower()
