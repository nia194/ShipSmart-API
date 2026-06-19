"""Tests for iterative RAG (G) with fake retrievers/clients (no network/DB)."""

from __future__ import annotations

import pytest

import app.core.config as config_mod
from app.llm.client import EchoClient
from app.rag.embeddings import LocalHashEmbedding
from app.rag.iterative import _UNCOVERED_REFUSAL, iterative_rag
from app.rag.vector_store import InMemoryVectorStore, SearchResult
from app.services.rag_service import rag_query


def _chunk(text="UPS Ground info", source="ups", score=0.9) -> SearchResult:
    return SearchResult(text=text, source=source, chunk_index=0, score=score)


def _retriever(sequence: list[list[SearchResult]]):
    """Async retriever returning sequence[i] on call i, then [] forever."""
    state = {"i": 0}

    async def _r(query: str, k: int) -> list[SearchResult]:
        i = state["i"]
        state["i"] += 1
        return sequence[i] if i < len(sequence) else []

    return _r


# ── loop bounds / coverage ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stops_at_max_steps_and_refuses_when_uncovered():
    res = await iterative_rag("q", retriever=_retriever([]), llm_client=EchoClient(), max_steps=3)
    assert res.steps == 3                       # bounded
    assert res.answer == _UNCOVERED_REFUSAL     # refuse, don't guess (D)
    assert res.answer_source == "rule"
    assert "iterative:uncovered_refusal" in res.decisions


@pytest.mark.asyncio
async def test_covered_on_first_step_stops_early():
    res = await iterative_rag(
        "q", retriever=_retriever([[_chunk()]]), llm_client=EchoClient(), max_steps=3,
    )
    assert res.steps == 1
    assert res.answer                            # Echo answered
    assert res.sources                           # grounded sources returned


@pytest.mark.asyncio
async def test_reformulates_then_covers():
    res = await iterative_rag(
        "q", retriever=_retriever([[], [_chunk()]]), llm_client=EchoClient(), max_steps=3,
    )
    assert res.steps == 2
    assert "iterative:reformulate" in res.decisions


# ── tool escalation ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_escalates_to_tools_when_query_needs_ground_truth(mcp_tool_registry):
    res = await iterative_rag(
        "what will this shipment cost",
        retriever=_retriever([]),          # no KB coverage
        llm_client=EchoClient(),
        tool_registry=mcp_tool_registry,
        context={"origin_zip": "90210", "destination_zip": "10001", "weight_lbs": 5.0},
        max_steps=2,
    )
    assert "get_quote_preview" in res.tools_used
    assert any(d.startswith("iterative:tools:") for d in res.decisions)
    # tool results cover the question → not the uncovered refusal
    assert res.answer != _UNCOVERED_REFUSAL


# ── normal vs iterative via the service ───────────────────────────────────────


@pytest.mark.asyncio
async def test_rag_service_iterative_mode_tags_mode(monkeypatch):
    monkeypatch.setattr(config_mod.settings, "rag_mode", "iterative", raising=False)
    res = await rag_query(
        "anything", LocalHashEmbedding(dims=16), InMemoryVectorStore(), EchoClient(),
    )
    assert res.metadata["decision_path"]["mode"] == "iterative"
    assert "steps" in res.metadata


@pytest.mark.asyncio
async def test_rag_service_normal_mode_is_single_shot(monkeypatch):
    monkeypatch.setattr(config_mod.settings, "rag_mode", "normal", raising=False)
    res = await rag_query(
        "anything", LocalHashEmbedding(dims=16), InMemoryVectorStore(), EchoClient(),
    )
    assert res.metadata["decision_path"]["mode"] == "normal"
    assert "steps" not in res.metadata
