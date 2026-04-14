"""Tests for RAG endpoints and integration."""

import pytest
from fastapi.testclient import TestClient

from app.llm.client import create_llm_client
from app.main import app
from app.rag.embeddings import create_embedding_provider
from app.rag.vector_store import create_vector_store


@pytest.fixture(autouse=True)
def _setup_rag_state():
    """Initialize RAG state on app for tests (lifespan doesn't run with TestClient)."""
    app.state.rag = {
        "embedding_provider": create_embedding_provider(),
        "vector_store": create_vector_store(),
        "llm_client": create_llm_client(),
    }
    yield
    app.state.rag["vector_store"]._chunks.clear()


client = TestClient(app)


def test_rag_query_endpoint():
    """Test POST /api/v1/rag/query returns expected shape."""
    response = client.post("/api/v1/rag/query", json={"query": "What carriers are supported?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "metadata" in data
    assert isinstance(data["sources"], list)


def test_rag_query_empty_query():
    """Validation rejects empty query."""
    response = client.post("/api/v1/rag/query", json={"query": ""})
    assert response.status_code == 422


def test_rag_ingest_endpoint():
    """Test POST /api/v1/rag/ingest loads seed documents."""
    response = client.post("/api/v1/rag/ingest")
    assert response.status_code == 200
    data = response.json()
    assert "chunks_ingested" in data
    assert "total_chunks" in data
    assert data["chunks_ingested"] > 0


def test_rag_query_after_ingest():
    """After ingestion, queries return sources."""
    # Ingest first
    client.post("/api/v1/rag/ingest")
    # Query
    response = client.post("/api/v1/rag/query", json={"query": "What is dimensional weight?"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["sources"]) > 0
    assert data["metadata"]["chunks_retrieved"] > 0


def test_rag_query_missing_body():
    """Missing body returns 422."""
    response = client.post("/api/v1/rag/query")
    assert response.status_code == 422
