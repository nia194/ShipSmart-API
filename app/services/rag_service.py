"""
RAG service — orchestrates retrieval and LLM completion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.llm.client import LLMClient
from app.llm.prompts import build_rag_prompt
from app.rag.embeddings import EmbeddingProvider
from app.rag.retrieval import retrieve
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    answer: str
    sources: list[dict]
    metadata: dict


async def rag_query(
    query: str,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    llm_client: LLMClient,
    top_k: int = 3,
) -> RAGResult:
    """Execute a full RAG query: retrieve context, build prompt, call LLM.

    Args:
        query: The user's question.
        embedding_provider: For embedding the query.
        vector_store: For retrieving relevant chunks.
        llm_client: For generating the answer.
        top_k: Number of chunks to retrieve.

    Returns:
        RAGResult with answer, sources, and metadata.
    """
    # Retrieve
    results = await retrieve(query, embedding_provider, vector_store, top_k=top_k)

    context_chunks = [r.text for r in results]
    sources = [
        {"source": r.source, "chunk_index": r.chunk_index, "score": round(r.score, 4)}
        for r in results
    ]

    # Build prompt and call LLM
    messages = build_rag_prompt(query, context_chunks)
    answer = await llm_client.complete(messages)

    logger.info(
        "RAG query completed: %d sources, answer length=%d",
        len(sources), len(answer),
    )

    return RAGResult(
        answer=answer,
        sources=sources,
        metadata={
            "chunks_retrieved": len(results),
            "store_size": vector_store.count(),
        },
    )
