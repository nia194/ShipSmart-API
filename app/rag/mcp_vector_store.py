"""
MCP-based pgvector store via Supabase MCP server.

Connects to Supabase pgvector through the MCP HTTP endpoint instead of direct asyncpg.
This allows FastAPI to leverage the Supabase MCP server configured in .mcp.json.

Usage:
    Set in .env:
        VECTOR_STORE_TYPE=mcp
        MCP_SERVER_URL=https://mcp.supabase.com/mcp?project_ref=...
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.rag.vector_store import SearchResult, StoredChunk, VectorStore

logger = logging.getLogger(__name__)


def _to_pgvector_literal(embedding: list[float]) -> str:
    """Format a Python list as a pgvector literal string.

    pgvector accepts vectors as text in the form '[1.0,2.0,3.0]'.
    """
    return "[" + ",".join(f"{float(x):.8f}" for x in embedding) + "]"


class MCPVectorStore(VectorStore):
    """Vector store backed by Supabase pgvector via MCP HTTP server.

    Connects through the Supabase MCP endpoint instead of direct asyncpg.
    Requires:
        - MCP_SERVER_URL: HTTP endpoint of Supabase MCP server
        - MCP_API_KEY: Optional API key for MCP server (if required)
    """

    def __init__(
        self,
        mcp_server_url: str,
        table: str = "rag_chunks",
        mcp_api_key: str | None = None,
    ) -> None:
        if not mcp_server_url:
            raise ValueError(
                "MCPVectorStore requires MCP_SERVER_URL. "
                "Set it in .env or via environment variable."
            )
        self._mcp_server_url = mcp_server_url
        self._table = table
        self._mcp_api_key = mcp_api_key
        self._client = httpx.AsyncClient(timeout=30.0)
        self._chunk_count = 0

    async def _execute_mcp_query(self, sql: str, params: list[Any] | None = None) -> Any:
        """Execute a SQL query via MCP server."""
        payload = {
            "action": "execute_sql",
            "query": sql,
            "params": params or [],
        }

        headers = {"Content-Type": "application/json"}
        if self._mcp_api_key:
            headers["Authorization"] = f"Bearer {self._mcp_api_key}"

        try:
            response = await self._client.post(
                self._mcp_server_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("MCP query failed: %s", e)
            raise RuntimeError(f"MCP vector store error: {e}") from e

    async def connect(self) -> None:
        """Verify MCP connection is available."""
        try:
            # Test connection with a simple query
            await self._execute_mcp_query("SELECT 1")
            logger.info("MCPVectorStore connected (table=%s)", self._table)

            # Try to get existing chunk count
            try:
                result = await self._execute_mcp_query(f"SELECT COUNT(*) as count FROM {self._table}")
                if result and "rows" in result and len(result["rows"]) > 0:
                    self._chunk_count = result["rows"][0].get("count", 0)
                    logger.info("Vector store has %d existing chunks", self._chunk_count)
            except Exception as e:
                logger.warning("Could not count existing chunks: %s", e)
        except Exception as e:
            logger.error("MCPVectorStore connection failed: %s", e)
            raise

    async def disconnect(self) -> None:
        """Close MCP client."""
        await self._client.aclose()
        logger.info("MCPVectorStore disconnected")

    async def add(self, chunks: list[StoredChunk]) -> int:
        """Add chunks to pgvector via MCP."""
        if not chunks:
            return 0

        count = 0
        for chunk in chunks:
            sql = (
                f"INSERT INTO {self._table} "
                "(source, chunk_index, text, embedding, metadata) "
                "VALUES ($1, $2, $3, $4::vector, $5::jsonb) "
                "ON CONFLICT (source, chunk_index) DO UPDATE SET "
                "text = EXCLUDED.text, "
                "embedding = EXCLUDED.embedding, "
                "metadata = EXCLUDED.metadata"
            )

            try:
                await self._execute_mcp_query(
                    sql,
                    [
                        chunk.source,
                        chunk.chunk_index,
                        chunk.text,
                        _to_pgvector_literal(chunk.embedding),
                        json.dumps({}),
                    ],
                )
                count += 1
            except Exception as e:
                logger.error("Failed to add chunk %s:%d: %s", chunk.source, chunk.chunk_index, e)

        self._chunk_count += count
        logger.info("MCPVectorStore: upserted %d chunks (total: %d)", count, self._chunk_count)
        return count

    async def search(self, query_embedding: list[float], top_k: int = 3) -> list[SearchResult]:
        """Search for similar chunks via MCP."""
        sql = (
            f"SELECT source, chunk_index, text, "
            f"1 - (embedding <=> $1::vector) AS score "
            f"FROM {self._table} "
            f"ORDER BY embedding <=> $1::vector ASC "
            f"LIMIT $2"
        )

        try:
            result = await self._execute_mcp_query(
                sql,
                [_to_pgvector_literal(query_embedding), top_k],
            )

            if not result or "rows" not in result:
                return []

            return [
                SearchResult(
                    text=row.get("text", ""),
                    source=row.get("source", ""),
                    chunk_index=row.get("chunk_index", 0),
                    score=float(row.get("score", 0.0)),
                )
                for row in result["rows"]
            ]
        except Exception as e:
            logger.error("Search query failed: %s", e)
            return []

    async def clear(self) -> None:
        """Clear all chunks via MCP."""
        sql = f"DELETE FROM {self._table}"

        try:
            await self._execute_mcp_query(sql)
            self._chunk_count = 0
            logger.info("Cleared all chunks from vector store")
        except Exception as e:
            logger.error("Clear operation failed: %s", e)

    def count(self) -> int:
        """Return cached chunk count."""
        return self._chunk_count

    async def count_async(self) -> int:
        """Return current chunk count from database."""
        sql = f"SELECT COUNT(*) as count FROM {self._table}"

        try:
            result = await self._execute_mcp_query(sql)
            if result and "rows" in result and len(result["rows"]) > 0:
                return result["rows"][0].get("count", 0)
        except Exception as e:
            logger.error("Count query failed: %s", e)

        return self._chunk_count
