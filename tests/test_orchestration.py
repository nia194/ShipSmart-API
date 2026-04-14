"""Tests for orchestration endpoint and service."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.providers.mock_provider import MockShippingProvider
from app.tools.address_tools import ValidateAddressTool
from app.tools.quote_tools import GetQuotePreviewTool
from app.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def _setup_tool_registry():
    """Initialize tool registry for tests (lifespan doesn't run with TestClient)."""
    provider = MockShippingProvider()
    registry = ToolRegistry()
    registry.register(ValidateAddressTool(provider))
    registry.register(GetQuotePreviewTool(provider))
    app.state.tool_registry = registry
    yield


client = TestClient(app)


# ── GET /api/v1/orchestration/tools ─────────────────────────────────────────

def test_list_tools():
    response = client.get("/api/v1/orchestration/tools")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    names = [t["name"] for t in data]
    assert "validate_address" in names
    assert "get_quote_preview" in names


# ── POST /api/v1/orchestration/run — explicit tool ──────────────────────────

def test_run_validate_address_explicit():
    response = client.post("/api/v1/orchestration/run", json={
        "query": "Check this address",
        "tool": "validate_address",
        "params": {
            "street": "123 Main St",
            "city": "New York",
            "state": "NY",
            "zip_code": "10001",
        },
    })
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "tool_result"
    assert data["tool_used"] == "validate_address"
    assert data["data"]["is_valid"] is True
    assert "answer" in data


def test_run_quote_preview_explicit():
    response = client.post("/api/v1/orchestration/run", json={
        "query": "Get a quote",
        "tool": "get_quote_preview",
        "params": {
            "origin_zip": "90210",
            "destination_zip": "10001",
            "weight_lbs": 5.0,
            "length_in": 12.0,
            "width_in": 8.0,
            "height_in": 6.0,
        },
    })
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "tool_result"
    assert data["tool_used"] == "get_quote_preview"
    assert len(data["data"]["services"]) == 3


# ── POST /api/v1/orchestration/run — auto-select ────────────────────────────

def test_run_auto_select_address():
    response = client.post("/api/v1/orchestration/run", json={
        "query": "Validate this address for me",
        "params": {
            "street": "456 Oak Ave",
            "city": "San Francisco",
            "state": "CA",
            "zip_code": "94102",
        },
    })
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "tool_result"
    assert data["tool_used"] == "validate_address"


def test_run_auto_select_quote():
    response = client.post("/api/v1/orchestration/run", json={
        "query": "What's the shipping cost for this package?",
        "params": {
            "origin_zip": "90210",
            "destination_zip": "10001",
            "weight_lbs": 5.0,
            "length_in": 12.0,
            "width_in": 8.0,
            "height_in": 6.0,
        },
    })
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "tool_result"
    assert data["tool_used"] == "get_quote_preview"


def test_run_no_tool_match():
    response = client.post("/api/v1/orchestration/run", json={
        "query": "What is the meaning of life?",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "direct_answer"
    assert data["tool_used"] is None


# ── Error cases ──────────────────────────────────────────────────────────────

def test_run_unknown_tool():
    response = client.post("/api/v1/orchestration/run", json={
        "query": "test",
        "tool": "nonexistent_tool",
    })
    assert response.status_code == 404


def test_run_missing_params():
    response = client.post("/api/v1/orchestration/run", json={
        "query": "test",
        "tool": "validate_address",
        "params": {},
    })
    assert response.status_code == 422


def test_run_empty_query():
    response = client.post("/api/v1/orchestration/run", json={
        "query": "",
    })
    assert response.status_code == 422


def test_run_missing_body():
    response = client.post("/api/v1/orchestration/run")
    assert response.status_code == 422
