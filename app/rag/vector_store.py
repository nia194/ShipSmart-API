"""
Vector store abstraction.
In-memory store for development; designed for easy swap to a real vector DB later.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StoredChunk:
    """A chunk stored in the vector store with its embedding."""
    text: str
    source: str
    chunk_index: int
    embedding: list[float]


@dataclass
class SearchResult:
    """A single search result with similarity score."""
    text: str
    source: str
    chunk_index: int
    score: float


class VectorStore(ABC):
    """Abstract interface for vector storage and retrieval."""

    @abstractmethod
    async def add(self, chunks: list[StoredChunk]) -> int:
        """Add chunks with embeddings. Returns number added."""

    @abstractmethod
    async def search(self, query_embedding: list[float], top_k: int = 3) -> list[SearchResult]:
        """Search for the most similar chunks."""

    @abstractmethod
    async def clear(self) -> None:
        """Remove all stored chunks."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored chunks."""


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store using cosine similarity.

    Suitable for development and small document sets.
    NOT suitable for production with large corpora.
    """

    def __init__(self) -> None:
        self._chunks: list[StoredChunk] = []

    async def add(self, chunks: list[StoredChunk]) -> int:
        self._chunks.extend(chunks)
        logger.info(
            "Added %d chunks to in-memory store (total: %d)", len(chunks), len(self._chunks),
        )
        return len(chunks)

    async def search(self, query_embedding: list[float], top_k: int = 3) -> list[SearchResult]:
        if not self._chunks:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        scores: list[tuple[float, StoredChunk]] = []
        for chunk in self._chunks:
            chunk_vec = np.array(chunk.embedding, dtype=np.float32)
            chunk_norm = np.linalg.norm(chunk_vec)
            if chunk_norm == 0:
                continue
            similarity = float(np.dot(query_vec, chunk_vec) / (query_norm * chunk_norm))
            scores.append((similarity, chunk))

        scores.sort(key=lambda x: x[0], reverse=True)

        return [
            SearchResult(
                text=chunk.text,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                score=score,
            )
            for score, chunk in scores[:top_k]
        ]

    async def clear(self) -> None:
        count = len(self._chunks)
        self._chunks.clear()
        logger.info("Cleared %d chunks from in-memory store", count)

    def count(self) -> int:
        return len(self._chunks)


def create_vector_store() -> VectorStore:
    """Factory: create the configured vector store.

    Supported backends:
      - "memory" (default): InMemoryVectorStore — fast, non-persistent.
      - "pgvector": PGVectorStore — Direct asyncpg to Postgres + pgvector extension.
        Requires DATABASE_URL to be set. Caller must `await store.connect()`
        during startup.
      - "mcp": MCPVectorStore — Postgres + pgvector via Supabase MCP HTTP server.
        Requires MCP_SERVER_URL to be set. Caller must `await store.connect()`
        during startup.
    """
    from app.core.config import settings  # local import to avoid cycles

    backend = (settings.vector_store_type or "memory").lower().strip()
    if backend == "pgvector":
        from app.rag.pgvector_store import PGVectorStore
        return PGVectorStore(dsn=settings.database_url, table=settings.pgvector_table)
    elif backend == "mcp":
        from app.rag.mcp_vector_store import MCPVectorStore
        return MCPVectorStore(
            mcp_server_url=settings.mcp_server_url,
            table=settings.pgvector_table,
            mcp_api_key=settings.mcp_api_key,
        )
    return InMemoryVectorStore()
