"""Tests for the shared grounding primitive (app/rag/grounding.py).

``coverage_of`` / ``CoverageSignal`` were relocated here from the agent service
(re-exported there for back-compat); ``retrieve_area`` is the bounded,
deterministic single-area retrieval the agent loop and compliance flow share.
There is no LLM in its control flow. Hermetic + keyless.
"""

from __future__ import annotations

import pytest

from app.rag.embeddings import EmbeddingProvider
from app.rag.grounding import AreaRetrieval, CoverageSignal, coverage_of, retrieve_area
from app.rag.vector_store import InMemoryVectorStore, SearchResult, StoredChunk

# ── coverage_of: one definition of "covered" (reuses _covered: any score > 0) ──


def test_coverage_of_strong():
    cov = coverage_of([SearchResult(text="t", source="s", chunk_index=0, score=0.82)])
    assert cov.covered is True
    assert cov.top_score == pytest.approx(0.82)
    assert cov.chunk_count == 1


def test_coverage_of_weak_when_empty():
    cov = coverage_of([])
    assert cov.covered is False
    assert cov.top_score == 0.0
    assert cov.chunk_count == 0


def test_coverage_signal_reexported_from_agent_service():
    # The agent service's public surface is unchanged after the relocation.
    from app.services import agent_service

    assert agent_service.CoverageSignal is CoverageSignal
    assert agent_service.coverage_of is coverage_of


# ── retrieve_area: bounded, conditional, non-degenerate (deterministic) ────────


class _FixedEmbedding(EmbeddingProvider):
    """Deterministic embedding: every text -> the same unit vector (cosine 1.0)."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    @property
    def dimensions(self) -> int:
        return 3


async def test_retrieve_area_single_pass_covered():
    store = InMemoryVectorStore()
    await store.add([
        StoredChunk(text="Power banks ship ground.", source="kb", chunk_index=0,
                    embedding=[1.0, 0.0, 0.0]),
    ])
    out = await retrieve_area(
        "lithium_battery", "power bank",
        embedding_provider=_FixedEmbedding(), vector_store=store,
    )
    assert isinstance(out, AreaRetrieval)
    assert out.covered is True
    assert out.attempts == 1
    assert out.decisions == ["lithium_battery:retrieve:1", "lithium_battery:covered"]


async def test_retrieve_area_single_pass_uncovered_when_empty():
    out = await retrieve_area(
        "customs_docs", "cn23 form",
        embedding_provider=_FixedEmbedding(), vector_store=InMemoryVectorStore(),
    )
    assert out.covered is False
    assert out.attempts == 1
    assert out.decisions == ["customs_docs:retrieve:1", "customs_docs:uncovered"]


async def test_retrieve_area_reformulates_then_bounds():
    seen: list[str] = []

    def reformulate(area: str, query: str, coverage: CoverageSignal) -> str:
        seen.append(query)
        return f"{query} more specific {len(seen)}"

    out = await retrieve_area(
        "import_restriction", "brazil drone",
        embedding_provider=_FixedEmbedding(), vector_store=InMemoryVectorStore(),
        max_retrievals=2, reformulate=reformulate,
    )
    assert out.covered is False
    assert out.attempts == 2  # bounded by max_retrievals
    assert out.decisions == [
        "import_restriction:retrieve:1",
        "import_restriction:reformulate",
        "import_restriction:retrieve:2",
        "import_restriction:uncovered",
    ]


async def test_retrieve_area_rejects_identical_reformulation():
    out = await retrieve_area(
        "value_threshold", "de minimis",
        embedding_provider=_FixedEmbedding(), vector_store=InMemoryVectorStore(),
        max_retrievals=3, reformulate=lambda a, q, c: "de minimis",  # identical → rejected
    )
    assert out.covered is False
    assert out.decisions == [
        "value_threshold:retrieve:1",
        "value_threshold:reformulate",
        "value_threshold:retrieve:rejected",
        "value_threshold:uncovered",
    ]
