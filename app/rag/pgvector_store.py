"""
PostgreSQL + pgvector implementation of VectorStore.

Requires the `vector` extension in the target Postgres database and a table
matching the schema in supabase/migrations/<ts>_create_rag_chunks.sql.
The embedding column dimension must match the active embedding provider's
`dimensions` property.
"""

from __future__ import annotations

import json
import logging

import asyncpg

from app.rag.vector_store import SearchResult, StoredChunk, VectorStore

logger = logging.getLogger(__name__)


def _to_pgvector_literal(embedding: list[float]) -> str:
    """Format a Python list as a pgvector literal string.

    pgvector accepts vectors as text in the form '[1.0,2.0,3.0]'. Using a
    text literal avoids needing to register a custom asyncpg codec.
    """
    return "[" + ",".join(f"{float(x):.8f}" for x in embedding) + "]"


class PGVectorStore(VectorStore):
    """Persistent vector store backed by Postgres + pgvector."""

    def __init__(self, dsn: str, table: str = "rag_chunks") -> None:
        if not dsn:
            raise ValueError(
                "PGVectorStore requires a non-empty DSN. "
                "Set DATABASE_URL when VECTOR_STORE_TYPE=pgvector."
            )
        self._dsn = dsn
        self._table = table
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Open the connection pool. Call once during startup."""
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            dsn=self._dsn, min_size=1, max_size=5, command_timeout=30,
        )
        logger.info("PGVectorStore connected (table=%s)", self._table)

    async def disconnect(self) -> None:
        """Close the connection pool. Call once during shutdown."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("PGVectorStore disconnected")

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError(
                "PGVectorStore is not connected. Call connect() during startup."
            )
        return self._pool

    async def add(self, chunks: list[StoredChunk]) -> int:
        if not chunks:
            return 0
        pool = self._require_pool()
        sql = (
            f"INSERT INTO {self._table} "
            "(source, chunk_index, text, embedding, metadata) "
            "VALUES ($1, $2, $3, $4::vector, $5::jsonb) "
            "ON CONFLICT (source, chunk_index) DO UPDATE SET "
            "text = EXCLUDED.text, "
            "embedding = EXCLUDED.embedding, "
            "metadata = EXCLUDED.metadata"
        )
        async with pool.acquire() as conn:
            async with conn.transaction():
                for chunk in chunks:
                    await conn.execute(
                        sql,
                        chunk.source,
                        chunk.chunk_index,
                        chunk.text,
                        _to_pgvector_literal(chunk.embedding),
                        json.dumps({}),
                    )
        logger.info("PGVectorStore: upserted %d chunks", len(chunks))
        return len(chunks)

    async def search(self, query_embedding: list[float], top_k: int = 3) -> list[SearchResult]:
        pool = self._require_pool()
        sql = (
            f"SELECT source, chunk_index, text, "
            f"1 - (embedding <=> $1::vector) AS score "
            f"FROM {self._table} "
            f"ORDER BY embedding <=> $1::vector ASC "
            f"LIMIT $2"
        )
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                sql, _to_pgvector_literal(query_embedding), top_k,
            )
        return [
            SearchResult(
                text=row["text"],
                source=row["source"],
                chunk_index=row["chunk_index"],
                score=float(row["score"]),
            )
            for row in rows
        ]

    async def clear(self) -> None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {self._table}")
        logger.info("PGVectorStore: cleared table %s", self._table)

    def count(self) -> int:
        # Sync method per ABC; we expose an async helper for accurate counts.
        # Returning -1 here signals "unknown without async query".
        return -1

    async def count_async(self) -> int:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(f"SELECT COUNT(*) AS n FROM {self._table}")
        return int(row["n"]) if row else 0
