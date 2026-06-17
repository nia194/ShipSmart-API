"""Tests for the agent's retrieval coverage signal (Phase 1).

The ``retrieve_rag`` observation fed back to the model leads with a coverage
signal (top_score / covered / chunk_count) so the agent can reason over result
quality. ``covered`` reuses the deterministic RAG layer's grounding threshold
(``app.rag.iterative._covered``) — read, not re-invented.
"""

from __future__ import annotations

import pytest

from app.llm.client import ToolCall
from app.rag.embeddings import LocalHashEmbedding
from app.rag.vector_store import InMemoryVectorStore, SearchResult, StoredChunk
from app.services.agent_service import (
    CoverageSignal,
    _dispatch,
    _format_rag_observation,
    coverage_of,
)

_KB_TEXT = "Power banks contain lithium batteries; ship ground, declare watt-hours."


# ── coverage_of: reads scores + grounding threshold ─────────────────────────


def test_coverage_of_strong():
    cov = coverage_of([SearchResult(text="t", source="s", chunk_index=0, score=0.82)])
    assert cov.covered is True
    assert cov.top_score == pytest.approx(0.82)
    assert cov.chunk_count == 1


def test_coverage_of_weak_when_no_chunks():
    cov = coverage_of([])
    assert cov.covered is False
    assert cov.top_score == 0.0
    assert cov.chunk_count == 0


def test_coverage_of_weak_when_not_grounded():
    # A returned chunk that does not clear the grounding threshold (score <= 0).
    cov = coverage_of([SearchResult(text="t", source="s", chunk_index=0, score=0.0)])
    assert cov.covered is False
    assert cov.top_score == 0.0
    assert cov.chunk_count == 1


# ── observation rendering leads with the coverage line ──────────────────────


def test_observation_includes_coverage_fields():
    cov = CoverageSignal(top_score=0.5, covered=True, chunk_count=2)
    text = _format_rag_observation(
        [SearchResult(text="body", source="kb", chunk_index=0, score=0.5)], cov,
    )
    assert "coverage:" in text
    assert "top_score=0.500" in text
    assert "covered=true" in text
    assert "chunks=2" in text


def test_observation_empty_results_still_reports_coverage():
    cov = coverage_of([])
    text = _format_rag_observation([], cov)
    assert "covered=false" in text
    assert "chunks=0" in text


# ── retrieve_rag dispatch surfaces coverage for both query qualities ─────────


async def _seeded_store(embed: LocalHashEmbedding) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    vec = (await embed.embed([_KB_TEXT]))[0]
    await store.add([StoredChunk(text=_KB_TEXT, source="hazmat", chunk_index=0, embedding=vec)])
    return store


@pytest.mark.asyncio
async def test_retrieve_rag_observation_well_covered():
    embed = LocalHashEmbedding(dims=16)
    store = await _seeded_store(embed)
    # Exact-match query → identical embedding → cosine 1.0 → covered.
    call = ToolCall(id="c1", name="retrieve_rag", arguments={"query": _KB_TEXT})
    observation, chunks, coverage = await _dispatch(
        call, registry=None, embedding_provider=embed, vector_store=store,
    )
    assert "covered=true" in observation
    assert coverage is not None and coverage.covered is True
    assert coverage.chunk_count == len(chunks) == 1


@pytest.mark.asyncio
async def test_retrieve_rag_observation_poorly_covered():
    embed = LocalHashEmbedding(dims=16)
    store = InMemoryVectorStore()  # empty → nothing retrieved
    call = ToolCall(id="c1", name="retrieve_rag", arguments={"query": "drone germany"})
    observation, chunks, coverage = await _dispatch(
        call, registry=None, embedding_provider=embed, vector_store=store,
    )
    assert "covered=false" in observation
    assert "chunks=0" in observation
    assert coverage is not None and coverage.covered is False
    assert chunks == []
