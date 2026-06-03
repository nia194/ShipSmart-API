"""
Retrieval pipeline.
Embeds a query, searches the vector store, and returns relevant chunks.
"""

from __future__ import annotations

import logging

from app.core.cache import rag_cache
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


async def retrieve(
    query: str,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    top_k: int = 3,
) -> list[SearchResult]:
    """Embed a query and retrieve the most relevant chunks.

    Args:
        query: The user's question or search text.
        embedding_provider: Provider for generating the query embedding.
        vector_store: Store to search.
        top_k: Number of results to return.

    Returns:
        List of SearchResult objects, sorted by relevance.
    """
    if vector_store.count() == 0:
        logger.warning("Vector store is empty — no documents ingested")
        return []

    # Check cache
    cache_key = rag_cache.make_key(query, top_k, vector_store.count())
    cached = rag_cache.get(cache_key)
    if cached is not None:
        logger.debug("RAG cache hit for query (top_k=%d)", top_k)
        return cached

    embeddings = await embedding_provider.embed([query])
    query_embedding = embeddings[0]

    results = await vector_store.search(query_embedding, top_k=top_k)
    logger.info("Retrieved %d chunks for query (top_k=%d)", len(results), top_k)

    rag_cache.set(cache_key, results)
    return results


async def retrieve_auto(
    query: str,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    top_k: int = 3,
) -> tuple[list[SearchResult], str]:
    """Dispatch dense vs hybrid retrieval based on config (F).

    Returns ``(results, mode)`` where mode is ``"dense"`` (today) or ``"hybrid"``.
    RAG_HYBRID defaults false, so this is dense-only and identical to ``retrieve``
    unless hybrid is explicitly enabled. Hybrid is lazy-imported so rank_bm25 is
    only loaded when actually used.
    """
    from app.core.config import settings

    if getattr(settings, "rag_hybrid", False):
        from app.rag.hybrid import hybrid_retrieve

        results = await hybrid_retrieve(
            query, embedding_provider, vector_store,
            top_k=top_k, alpha=getattr(settings, "rag_hybrid_alpha", 0.5),
        )
        return results, "hybrid"

    return await retrieve(query, embedding_provider, vector_store, top_k=top_k), "dense"
