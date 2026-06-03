"""
Hybrid retrieval — dense + sparse (F).

Dense (pgvector cosine) is strong on paraphrase; sparse lexical (BM25 / Postgres
full-text) catches exact tokens — carrier names, service codes, tracking numbers.
With RAG_HYBRID=true we run both and fuse them by RAG_HYBRID_ALPHA (the dense
weight). RAG_HYBRID=false keeps dense-only behavior unchanged.

Sparse backend selection (no new config):
  * VECTOR_STORE_TYPE=pgvector → the Infra SQL function ``match_rag_chunks_lexical``
    (tsvector + GIN, ts_rank_cd) via ``PGVectorStore.search_lexical``.
  * in-memory store → in-process BM25 (rank_bm25) over the loaded chunks.

Retrieval-quality tradeoffs (defaults kept: chunk_size=500, top_k=3, overlap=50):
  * chunk too big → noisy/diluted matches; too small → a single rule is split
    across chunks and missed.
  * top_k too low → the relevant chunk isn't retrieved; too high → noise + cost.
  * overlap protects rules that straddle a chunk boundary.
"""

from __future__ import annotations

import logging
import re

from app.rag.embeddings import EmbeddingProvider
from app.rag.retrieval import retrieve
from app.rag.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def _key(r: SearchResult) -> tuple[str, int]:
    return (r.source, r.chunk_index)


def _bm25_search(query: str, chunks: list, top_k: int) -> list[SearchResult]:
    """In-process BM25 over loaded chunks (used by the in-memory store)."""
    if not chunks:
        return []
    from rank_bm25 import BM25Okapi

    corpus = [_tokenize(getattr(c, "text", "")) for c in chunks]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(scores, chunks, strict=True), key=lambda p: p[0], reverse=True)
    out: list[SearchResult] = []
    for score, c in ranked[:top_k]:
        if score <= 0:
            continue
        out.append(SearchResult(
            text=c.text, source=c.source, chunk_index=c.chunk_index, score=float(score),
        ))
    return out


async def _sparse_retrieve(query: str, vector_store: VectorStore, top_k: int) -> list[SearchResult]:
    """Dispatch to the available sparse backend (pgvector lexical or BM25)."""
    if hasattr(vector_store, "search_lexical"):
        try:
            return await vector_store.search_lexical(query, top_k)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - network/DB dependent
            logger.warning("Lexical search failed, sparse side empty: %s", exc)
            return []
    if hasattr(vector_store, "all_chunks"):
        return _bm25_search(query, vector_store.all_chunks(), top_k)  # type: ignore[attr-defined]
    return []


def _minmax(scores: dict[tuple[str, int], float]) -> dict[tuple[str, int], float]:
    """Min-max normalize to [0,1]; a single/degenerate set maps to 1.0."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi <= lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def fuse(
    dense: list[SearchResult],
    sparse: list[SearchResult],
    alpha: float,
    top_k: int,
) -> list[SearchResult]:
    """Fuse dense + sparse by normalized weighted score.

    ``score = alpha * dense_norm + (1 - alpha) * sparse_norm`` per chunk
    (keyed by source + chunk_index). A chunk present in only one list scores 0
    on the missing side. alpha is clamped to [0,1].
    """
    alpha = max(0.0, min(1.0, alpha))
    d_norm = _minmax({_key(r): r.score for r in dense})
    s_norm = _minmax({_key(r): r.score for r in sparse})

    reps: dict[tuple[str, int], SearchResult] = {}
    for r in dense:
        reps.setdefault(_key(r), r)
    for r in sparse:
        reps.setdefault(_key(r), r)

    fused = [
        SearchResult(
            text=r.text, source=r.source, chunk_index=r.chunk_index,
            score=alpha * d_norm.get(k, 0.0) + (1.0 - alpha) * s_norm.get(k, 0.0),
        )
        for k, r in reps.items()
    ]
    fused.sort(key=lambda x: x.score, reverse=True)
    return fused[:top_k]


async def hybrid_retrieve(
    query: str,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    *,
    top_k: int = 3,
    alpha: float = 0.5,
    candidate_k: int | None = None,
) -> list[SearchResult]:
    """Dense + sparse retrieval fused to ``top_k`` results.

    Pulls a wider candidate set from each side, then fuses. If no sparse backend
    is available the result is exactly dense-only (graceful degradation).
    """
    cand = candidate_k or max(top_k * 3, top_k)
    dense = await retrieve(query, embedding_provider, vector_store, top_k=cand)
    sparse = await _sparse_retrieve(query, vector_store, cand)
    if not sparse:
        return dense[:top_k]
    fused = fuse(dense, sparse, alpha, top_k)
    logger.info(
        "Hybrid retrieve: dense=%d sparse=%d fused=%d (alpha=%.2f)",
        len(dense), len(sparse), len(fused), alpha,
    )
    return fused
