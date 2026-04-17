"""Tests for advisor services (RAG + tool + LLM combination).

Tool execution now runs against a `RemoteToolRegistry` backed by
`httpx.MockTransport` — see `tests/conftest.py::mcp_tool_registry`.
"""

import asyncio

import pytest

from app.llm.client import create_llm_client
from app.rag.embeddings import create_embedding_provider
from app.rag.ingestion import ingest_documents, load_documents
from app.rag.vector_store import create_vector_store
from app.services.recommendation_service import generate_recommendations
from app.services.shipping_advisor_service import get_shipping_advice
from app.services.tracking_advisor_service import get_tracking_guidance


@pytest.fixture
def rag_state():
    """Set up RAG pipeline for tests."""

    async def _setup():
        embedding_provider = create_embedding_provider()
        vector_store = create_vector_store()
        llm_client = create_llm_client()

        # Ingest documents
        docs = load_documents("data/documents")
        if docs:
            await ingest_documents(
                docs,
                embedding_provider,
                vector_store,
                chunk_size=500,
                chunk_overlap=50,
            )

        return {
            "embedding_provider": embedding_provider,
            "vector_store": vector_store,
            "llm_client": llm_client,
        }, vector_store

    state, vector_store = asyncio.run(_setup())
    yield state
    asyncio.run(vector_store.clear())


# ── Shipping Advisor ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_shipping_advice_with_tools(rag_state, mcp_tool_registry):
    """Test shipping advice with tool execution."""
    advice = await get_shipping_advice(
        query="What are the shipping options for this package?",
        context={
            "origin_zip": "90210",
            "destination_zip": "10001",
            "weight_lbs": 5.0,
            "length_in": 12.0,
            "width_in": 8.0,
            "height_in": 6.0,
        },
        embedding_provider=rag_state["embedding_provider"],
        vector_store=rag_state["vector_store"],
        llm_client=rag_state["llm_client"],
        tool_registry=mcp_tool_registry,
    )
    assert "get_quote_preview" in advice.tools_used
    assert advice.answer
    assert len(advice.sources) > 0
    assert advice.context_used


@pytest.mark.asyncio
async def test_shipping_advice_address_validation(rag_state, mcp_tool_registry):
    """Test shipping advice with address validation."""
    advice = await get_shipping_advice(
        query="Validate this address",
        context={
            "street": "123 Main St",
            "city": "New York",
            "state": "NY",
            "zip_code": "10001",
        },
        embedding_provider=rag_state["embedding_provider"],
        vector_store=rag_state["vector_store"],
        llm_client=rag_state["llm_client"],
        tool_registry=mcp_tool_registry,
    )
    assert "validate_address" in advice.tools_used
    assert advice.answer


@pytest.mark.asyncio
async def test_shipping_advice_rag_only(rag_state):
    """Test shipping advice with RAG but no tools."""
    advice = await get_shipping_advice(
        query="What shipping carriers are available?",
        embedding_provider=rag_state["embedding_provider"],
        vector_store=rag_state["vector_store"],
        llm_client=rag_state["llm_client"],
    )
    assert len(advice.tools_used) == 0
    assert advice.answer
    assert advice.context_used


# ── Tracking Advisor ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tracking_guidance_rag_focused(rag_state, mcp_tool_registry):
    """Test tracking guidance (mostly RAG-driven)."""
    guidance = await get_tracking_guidance(
        issue="What should I do if my package is delayed?",
        embedding_provider=rag_state["embedding_provider"],
        vector_store=rag_state["vector_store"],
        llm_client=rag_state["llm_client"],
        tool_registry=mcp_tool_registry,
    )
    assert guidance.guidance
    assert guidance.issue_summary
    assert isinstance(guidance.next_steps, list)
    assert len(guidance.sources) > 0


@pytest.mark.asyncio
async def test_tracking_guidance_with_address(rag_state, mcp_tool_registry):
    """Test tracking guidance with address validation."""
    guidance = await get_tracking_guidance(
        issue="Package not being delivered to my apartment",
        context={
            "street": "456 Oak Ave",
            "city": "San Francisco",
            "state": "CA",
            "zip_code": "94102",
        },
        embedding_provider=rag_state["embedding_provider"],
        vector_store=rag_state["vector_store"],
        llm_client=rag_state["llm_client"],
        tool_registry=mcp_tool_registry,
    )
    assert guidance.guidance
    # May or may not use validate_address depending on LLM
    assert isinstance(guidance.tools_used, list)


# ── Recommendations ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_recommendations_cheapest(rag_state):
    """Test that cheapest service is recommended."""
    services = [
        {"service": "Ground", "price_usd": 5.0, "estimated_days": 5},
        {"service": "Express", "price_usd": 15.0, "estimated_days": 2},
        {"service": "Overnight", "price_usd": 50.0, "estimated_days": 1},
    ]
    recommendations = await generate_recommendations(
        services,
        llm_client=rag_state["llm_client"],
    )
    assert recommendations.primary_recommendation.service_name == "Ground"
    assert recommendations.primary_recommendation.recommendation_type.value == "cheapest"


@pytest.mark.asyncio
async def test_recommendations_fastest(rag_state):
    """Test that fastest service is recommended when preferred."""
    services = [
        {"service": "Ground", "price_usd": 9.99, "estimated_days": 5},
        {"service": "Express", "price_usd": 19.99, "estimated_days": 2},
        {"service": "Overnight", "price_usd": 49.99, "estimated_days": 1},
    ]
    recommendations = await generate_recommendations(
        services,
        context={"urgent": True},
        llm_client=rag_state["llm_client"],
    )
    # Overnight is fastest, should be in alternatives or primary
    services_recommended = [recommendations.primary_recommendation.service_name]
    services_recommended += [a.service_name for a in recommendations.alternatives]
    assert "Overnight" in services_recommended


@pytest.mark.asyncio
async def test_recommendations_with_fragile_context(rag_state):
    """Test recommendations with fragile context."""
    services = [
        {"service": "Ground", "price_usd": 9.99, "estimated_days": 5},
        {"service": "Express", "price_usd": 19.99, "estimated_days": 2},
    ]
    recommendations = await generate_recommendations(
        services,
        context={"fragile": True},
        llm_client=rag_state["llm_client"],
    )
    assert recommendations.primary_recommendation.explanation
    assert "fragile" in recommendations.primary_recommendation.explanation.lower()


@pytest.mark.asyncio
async def test_recommendations_empty_services(rag_state):
    """Test recommendations with no services."""
    recommendations = await generate_recommendations(
        [],
        llm_client=rag_state["llm_client"],
    )
    assert recommendations.primary_recommendation.service_name == "N/A"
    assert len(recommendations.alternatives) == 0
