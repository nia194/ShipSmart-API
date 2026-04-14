"""
Integration tests for full system flows.
Tests the combination of RAG + tools + LLM + recommendations.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.llm.client import create_llm_client
from app.main import app
from app.providers.mock_provider import MockShippingProvider
from app.rag.embeddings import create_embedding_provider
from app.rag.ingestion import ingest_documents, load_documents
from app.rag.vector_store import create_vector_store
from app.tools.address_tools import ValidateAddressTool
from app.tools.quote_tools import GetQuotePreviewTool
from app.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def _setup_app_state():
    """Initialize app state for integration tests."""
    # RAG
    embedding_provider = create_embedding_provider()
    vector_store = create_vector_store()
    llm_client = create_llm_client()

    app.state.rag = {
        "embedding_provider": embedding_provider,
        "vector_store": vector_store,
        "llm_client": llm_client,
    }

    # Tools
    provider = MockShippingProvider()
    registry = ToolRegistry()
    registry.register(ValidateAddressTool(provider))
    registry.register(GetQuotePreviewTool(provider))
    app.state.tool_registry = registry

    # Ingest documents
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
    asyncio.run(vector_store.clear())


client = TestClient(app)


# ── Full Flow Tests ──────────────────────────────────────────────────────

def test_shipping_advisor_full_flow_with_quote():
    """Full flow: user asks about shipping → advisor uses quote tool."""
    response = client.post("/api/v1/advisor/shipping", json={
        "query": "What's the best option for a 5 lb package?",
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

    # Verify response structure
    assert "answer" in data
    assert "tools_used" in data
    assert "sources" in data
    assert "context_used" in data

    # Verify tools were used
    assert "get_quote_preview" in data["tools_used"]

    # Verify RAG context was used
    assert data["context_used"] is True
    assert len(data["sources"]) > 0


def test_tracking_advisor_full_flow_with_address():
    """Full flow: user reports issue → advisor validates address."""
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

    # Verify response structure
    assert "guidance" in data
    assert "issue_summary" in data
    assert "next_steps" in data
    assert isinstance(data["next_steps"], list)

    # Verify RAG context was used
    assert len(data["sources"]) > 0


def test_recommendation_flow():
    """Full flow: quote options → scoring → recommendations."""
    response = client.post("/api/v1/advisor/recommendation", json={
        "services": [
            {"service": "Ground", "price_usd": 9.99, "estimated_days": 5},
            {"service": "Express", "price_usd": 19.99, "estimated_days": 2},
            {"service": "Overnight", "price_usd": 49.99, "estimated_days": 1},
        ],
        "context": {"fragile": True},
    })

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "primary_recommendation" in data
    assert "alternatives" in data
    assert "summary" in data

    # Verify recommendation is scored
    primary = data["primary_recommendation"]
    assert primary["service_name"]
    assert primary["price_usd"] > 0
    assert primary["estimated_days"] > 0
    assert primary["recommendation_type"]
    assert primary["explanation"]
    assert primary["score"] >= 0

    # Verify alternatives exist
    assert len(data["alternatives"]) > 0

    # Verify metadata
    assert data["metadata"]["num_options"] == 3


# ── Endpoint Availability Tests ──────────────────────────────────────────

def test_advisor_endpoints_available():
    """Verify all advisor endpoints are registered."""
    # These should not 404
    assert client.options("/api/v1/advisor/shipping").status_code in (200, 405)
    assert client.options("/api/v1/advisor/tracking").status_code in (200, 405)
    assert client.options("/api/v1/advisor/recommendation").status_code in (200, 405)


def test_advisor_endpoints_require_post():
    """Advisor endpoints should only accept POST."""
    # GET requests should fail (not Found or Method Not Allowed)
    assert client.get("/api/v1/advisor/shipping").status_code in (404, 405)
    assert client.get("/api/v1/advisor/tracking").status_code in (404, 405)
    assert client.get("/api/v1/advisor/recommendation").status_code in (404, 405)


# ── Error Handling Tests ─────────────────────────────────────────────────

def test_advisor_invalid_request_validation():
    """Advisor endpoints validate request shape."""
    # Missing required query
    response = client.post("/api/v1/advisor/shipping", json={})
    assert response.status_code == 422


def test_recommendation_missing_services():
    """Recommendation endpoint requires services array."""
    response = client.post("/api/v1/advisor/recommendation", json={
        "context": {"fragile": True},
    })
    assert response.status_code == 422


# ── RAG Integration Tests ────────────────────────────────────────────────

def test_advisor_uses_rag_context():
    """Verify advisor retrieves RAG context."""
    response = client.post("/api/v1/advisor/shipping", json={
        "query": "What carriers are available?",
    })

    assert response.status_code == 200
    data = response.json()

    # Should have retrieved context
    assert data["context_used"] is True
    assert len(data["sources"]) > 0

    # Sources should have chunk info
    for source in data["sources"]:
        assert "source" in source
        assert "chunk_index" in source
        assert "score" in source
        # Score should be numeric (cosine similarity may vary slightly outside [0,1])
        assert isinstance(source["score"], (int, float))


# ── Tool Integration Tests ───────────────────────────────────────────────

def test_advisor_executes_tools():
    """Verify tools are executed when context provided."""
    response = client.post("/api/v1/advisor/shipping", json={
        "query": "Validate this address",
        "context": {
            "street": "123 Main St",
            "city": "New York",
            "state": "NY",
            "zip_code": "10001",
        },
    })

    assert response.status_code == 200
    data = response.json()

    # Should have used validate_address tool
    assert "validate_address" in data["tools_used"]


def test_advisor_skips_tools_when_not_needed():
    """Verify tools are not executed if context not provided."""
    response = client.post("/api/v1/advisor/shipping", json={
        "query": "What carriers are available?",
    })

    assert response.status_code == 200
    data = response.json()

    # Should not have used tools (no context)
    assert len(data["tools_used"]) == 0


# ── Response Shape Tests ─────────────────────────────────────────────────

def test_shipping_advisor_response_shape():
    """Verify shipping advisor response has correct shape."""
    response = client.post("/api/v1/advisor/shipping", json={
        "query": "Help me ship a package",
    })

    assert response.status_code == 200
    data = response.json()

    # Required fields
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0
    assert isinstance(data["reasoning_summary"], str)
    assert isinstance(data["tools_used"], list)
    assert isinstance(data["sources"], list)
    assert isinstance(data["context_used"], bool)


def test_tracking_advisor_response_shape():
    """Verify tracking advisor response has correct shape."""
    response = client.post("/api/v1/advisor/tracking", json={
        "issue": "My package is lost",
    })

    assert response.status_code == 200
    data = response.json()

    # Required fields
    assert isinstance(data["guidance"], str)
    assert len(data["guidance"]) > 0
    assert isinstance(data["issue_summary"], str)
    assert isinstance(data["tools_used"], list)
    assert isinstance(data["sources"], list)
    assert isinstance(data["next_steps"], list)


def test_recommendation_response_shape():
    """Verify recommendation response has correct shape."""
    response = client.post("/api/v1/advisor/recommendation", json={
        "services": [
            {"service": "Ground", "price_usd": 9.99, "estimated_days": 5},
        ],
    })

    assert response.status_code == 200
    data = response.json()

    # Required fields
    assert "primary_recommendation" in data
    assert "alternatives" in data
    assert "summary" in data
    assert "metadata" in data

    # Check primary recommendation structure
    primary = data["primary_recommendation"]
    assert "service_name" in primary
    assert "price_usd" in primary
    assert "estimated_days" in primary
    assert "recommendation_type" in primary
    assert "explanation" in primary
    assert "score" in primary


# ── Database/State Tests ────────────────────────────────────────────────

def test_rag_state_initialized():
    """Verify RAG state is available to advisors."""
    response = client.post("/api/v1/advisor/shipping", json={
        "query": "Help me",
    })

    # Should not 503 Service Unavailable
    assert response.status_code == 200


def test_tool_registry_initialized():
    """Verify tool registry is available to advisors."""
    response = client.post("/api/v1/advisor/shipping", json={
        "query": "Validate address",
        "context": {
            "street": "123 Main",
            "city": "LA",
            "state": "CA",
            "zip_code": "90001",
        },
    })

    # Should not 503 Service Unavailable
    assert response.status_code == 200
    assert "validate_address" in response.json()["tools_used"]
