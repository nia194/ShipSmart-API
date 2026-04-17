"""Tests for AI advisor endpoints.

Tools are served by a MockTransport-backed `RemoteToolRegistry` (see
`tests/conftest.py`) so the suite runs without a live ShipSmart-MCP service.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.llm.client import create_llm_client
from app.main import app
from app.rag.embeddings import create_embedding_provider
from app.rag.ingestion import ingest_documents, load_documents
from app.rag.vector_store import create_vector_store
from app.services.mcp_client import create_remote_registry
from tests.conftest import build_mcp_mock_transport


@pytest.fixture(autouse=True)
def _setup_app_state():
    """Initialize app state for advisor tests."""
    # RAG
    embedding_provider = create_embedding_provider()
    vector_store = create_vector_store()
    llm_client = create_llm_client()

    app.state.rag = {
        "embedding_provider": embedding_provider,
        "vector_store": vector_store,
        "llm_client": llm_client,
    }

    # Tools — hydrate RemoteToolRegistry from a MockTransport
    transport = build_mcp_mock_transport()
    registry = asyncio.run(
        create_remote_registry(
            base_url="http://mcp.test",
            api_key="",
            transport=transport,
        )
    )
    app.state.tool_registry = registry

    # Ingest documents for RAG
    async def ingest():
        docs = load_documents("data/documents")
        if docs:
            await ingest_documents(
                docs,
                embedding_provider,
                vector_store,
                chunk_size=500,
                chunk_overlap=50,
            )

    asyncio.run(ingest())
    yield
    # Cleanup
    asyncio.run(registry.aclose())
    asyncio.run(vector_store.clear())
    app.state.tool_registry = None


client = TestClient(app)


# ── Shipping Advisor ────────────────────────────────────────────────────────

def test_shipping_advisor_general_question():
    response = client.post("/api/v1/advisor/shipping", json={
        "query": "What carriers are available for shipping?",
    })
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "reasoning_summary" in data
    assert isinstance(data["tools_used"], list)
    assert isinstance(data["sources"], list)


def test_shipping_advisor_with_address():
    response = client.post("/api/v1/advisor/shipping", json={
        "query": "Is this address valid?",
        "context": {
            "street": "123 Main St",
            "city": "New York",
            "state": "NY",
            "zip_code": "10001",
        },
    })
    assert response.status_code == 200
    data = response.json()
    assert "validate_address" in data["tools_used"]


def test_shipping_advisor_with_quote_context():
    response = client.post("/api/v1/advisor/shipping", json={
        "query": "What shipping options are available?",
        "context": {
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
    assert "get_quote_preview" in data["tools_used"]
    assert data["answer"]


def test_shipping_advisor_empty_query():
    response = client.post("/api/v1/advisor/shipping", json={
        "query": "",
    })
    assert response.status_code == 422


# ── Tracking Advisor ───────────────────────────────────────────────────────

def test_tracking_advisor_general_issue():
    response = client.post("/api/v1/advisor/tracking", json={
        "issue": "What should I do if my package is delayed?",
    })
    assert response.status_code == 200
    data = response.json()
    assert "guidance" in data
    assert "issue_summary" in data
    assert isinstance(data["tools_used"], list)
    assert isinstance(data["next_steps"], list)


def test_tracking_advisor_with_address():
    response = client.post("/api/v1/advisor/tracking", json={
        "issue": "Package not being delivered to my apartment",
        "context": {
            "street": "456 Oak Ave Apt 5B",
            "city": "San Francisco",
            "state": "CA",
            "zip_code": "94102",
        },
    })
    assert response.status_code == 200
    data = response.json()
    # Address validation tool may or may not be called depending on LLM
    assert "guidance" in data


def test_tracking_advisor_empty_issue():
    response = client.post("/api/v1/advisor/tracking", json={
        "issue": "",
    })
    assert response.status_code == 422


# ── Recommendations ───────────────────────────────────────────────────────

def test_recommendation_basic():
    services = [
        {"service": "Ground", "price_usd": 9.99, "estimated_days": 5},
        {"service": "Express", "price_usd": 19.99, "estimated_days": 2},
        {"service": "Overnight", "price_usd": 49.99, "estimated_days": 1},
    ]
    response = client.post("/api/v1/advisor/recommendation", json={
        "services": services,
    })
    assert response.status_code == 200
    data = response.json()
    assert "primary_recommendation" in data
    assert "alternatives" in data
    assert "summary" in data
    # Primary should have cheapest or best value
    assert data["primary_recommendation"]["service_name"]
    assert data["primary_recommendation"]["price_usd"] > 0
    assert data["primary_recommendation"]["estimated_days"] > 0


def test_recommendation_with_context():
    services = [
        {"service": "Ground", "price_usd": 9.99, "estimated_days": 5},
        {"service": "Overnight", "price_usd": 49.99, "estimated_days": 1},
    ]
    response = client.post("/api/v1/advisor/recommendation", json={
        "services": services,
        "context": {
            "fragile": True,
        },
    })
    assert response.status_code == 200
    data = response.json()
    assert "primary_recommendation" in data
    # When fragile, explanation should mention it
    assert data["primary_recommendation"]["explanation"]


def test_recommendation_empty_services():
    response = client.post("/api/v1/advisor/recommendation", json={
        "services": [],
    })
    assert response.status_code == 200
    data = response.json()
    assert "primary_recommendation" in data
    assert data["primary_recommendation"]["service_name"] == "N/A"


def test_recommendation_single_service():
    services = [
        {"service": "Standard", "price_usd": 15.0, "estimated_days": 3},
    ]
    response = client.post("/api/v1/advisor/recommendation", json={
        "services": services,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["primary_recommendation"]["service_name"] == "Standard"
    assert len(data["alternatives"]) == 0


def test_recommendation_missing_body():
    response = client.post("/api/v1/advisor/recommendation")
    assert response.status_code == 422
