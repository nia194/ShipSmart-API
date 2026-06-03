"""Tests for hybrid retrieval (F): fusion, BM25, dispatch, dense-only default."""

from __future__ import annotations

import pytest

import app.core.config as config_mod
from app.rag.embeddings import LocalHashEmbedding
from app.rag.hybrid import _bm25_search, fuse, hybrid_retrieve
from app.rag.retrieval import retrieve_auto
from app.rag.vector_store import InMemoryVectorStore, SearchResult, StoredChunk


def _sr(source: str, score: float, idx: int = 0) -> SearchResult:
    return SearchResult(text=f"text-{source}", source=source, chunk_index=idx, score=score)


# ── fusion math ──────────────────────────────────────────────────────────────


def test_fuse_alpha_weights_dense_vs_sparse():
    dense = [_sr("d", 0.9)]
    sparse = [_sr("s", 5.0)]
    # alpha=1.0 → dense only
    top_dense = fuse(dense, sparse, alpha=1.0, top_k=2)
    assert top_dense[0].source == "d"
    assert dict((r.source, round(r.score, 3)) for r in top_dense) == {"d": 1.0, "s": 0.0}
    # alpha=0.0 → sparse only
    top_sparse = fuse(dense, sparse, alpha=0.0, top_k=2)
    assert top_sparse[0].source == "s"


def test_fuse_combines_shared_chunk():
    # same chunk in both lists → gets both contributions
    dense = [_sr("a", 1.0), _sr("b", 0.5)]
    sparse = [_sr("a", 9.0), _sr("c", 1.0)]
    fused = {r.source: r.score for r in fuse(dense, sparse, alpha=0.5, top_k=5)}
    # "a" is top in both → normalized 1.0 each side → 0.5*1 + 0.5*1 = 1.0
    assert fused["a"] == pytest.approx(1.0)
    assert fused["a"] >= fused["b"]
    assert fused["a"] >= fused["c"]


def test_fuse_clamps_alpha():
    out = fuse([_sr("d", 1.0)], [_sr("s", 1.0)], alpha=5.0, top_k=2)  # clamped to 1.0
    assert {r.source for r in out if r.score > 0} == {"d"}


# ── BM25 sparse ──────────────────────────────────────────────────────────────


def test_bm25_ranks_lexical_match_first():
    chunks = [
        StoredChunk(text="UPS Ground is the cheapest economy service",
                    source="ups", chunk_index=0, embedding=[]),
        StoredChunk(text="FedEx Overnight is the fastest premium option",
                    source="fedex", chunk_index=0, embedding=[]),
        StoredChunk(text="Pack fragile items with bubble wrap",
                    source="packing", chunk_index=0, embedding=[]),
    ]
    results = _bm25_search("cheapest economy", chunks, top_k=2)
    assert results
    assert results[0].source == "ups"


# ── dispatcher ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retrieve_auto_dense_by_default(monkeypatch):
    monkeypatch.setattr(config_mod.settings, "rag_hybrid", False, raising=False)
    _, mode = await retrieve_auto("q", LocalHashEmbedding(dims=16), InMemoryVectorStore(), top_k=3)
    assert mode == "dense"


@pytest.mark.asyncio
async def test_retrieve_auto_hybrid_when_enabled(monkeypatch):
    monkeypatch.setattr(config_mod.settings, "rag_hybrid", True, raising=False)
    monkeypatch.setattr(config_mod.settings, "rag_hybrid_alpha", 0.5, raising=False)
    _, mode = await retrieve_auto("q", LocalHashEmbedding(dims=16), InMemoryVectorStore(), top_k=3)
    assert mode == "hybrid"


# ── integration: hybrid over the in-memory store ─────────────────────────────


@pytest.mark.asyncio
async def test_hybrid_retrieve_memory_store_uses_bm25():
    ep = LocalHashEmbedding(dims=16)
    vs = InMemoryVectorStore()
    texts = [
        "UPS Ground is the cheapest economy service",
        "FedEx Overnight is the fastest premium option",
        "Pack fragile items with bubble wrap",
    ]
    sources = ["ups", "fedex", "packing"]
    embs = await ep.embed(texts)
    await vs.add([
        StoredChunk(text=t, source=s, chunk_index=0, embedding=e)
        for t, s, e in zip(texts, sources, embs, strict=True)
    ])
    # alpha=0 → sparse (BM25) dominates → the lexical match wins
    results = await hybrid_retrieve("cheapest economy", ep, vs, top_k=2, alpha=0.0)
    assert results[0].source == "ups"
    # SearchResult shape preserved
    assert hasattr(results[0], "text") and hasattr(results[0], "chunk_index")
