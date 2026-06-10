"""SQL-contract tests for PGVectorStore — no live Postgres required.

A fake asyncpg pool/connection records the SQL each method issues and returns
canned rows. This locks the wire contract the system depends on without a DB:
 * dense search uses the pgvector cosine operator (`embedding <=> $1::vector`),
 * lexical search calls the Infra function `match_rag_chunks_lexical($1, $2)`
   selecting exactly `source, chunk_index, text, score` (the shape ShipSmart-Test's
   cross-repo contract test also guards),
 * add() upserts on the (source, chunk_index) natural key.
"""

from __future__ import annotations

import pytest

pytest.importorskip("asyncpg")  # module imports asyncpg at top

from app.rag.pgvector_store import PGVectorStore, _to_pgvector_literal  # noqa: E402
from app.rag.vector_store import StoredChunk  # noqa: E402

# ── Fake asyncpg pool / connection ───────────────────────────────────────────


class _Acquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _Tx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, rows):
        self.rows = rows
        self.fetched: list[tuple] = []
        self.executed: list[tuple] = []

    async def fetch(self, sql, *args):
        self.fetched.append((sql, args))
        return self.rows

    async def execute(self, sql, *args):
        self.executed.append((sql, args))
        return "OK"

    async def fetchrow(self, sql, *args):
        self.fetched.append((sql, args))
        return {"n": len(self.rows)}

    def transaction(self):
        return _Tx()


class _FakePool:
    def __init__(self, rows):
        self.conn = _FakeConn(rows)

    def acquire(self):
        return _Acquire(self.conn)


def _store(rows=None) -> PGVectorStore:
    store = PGVectorStore(dsn="postgresql://test/db", table="rag_chunks")
    store._pool = _FakePool(rows or [])
    return store


# ── Construction / guards ────────────────────────────────────────────────────


def test_empty_dsn_rejected():
    with pytest.raises(ValueError, match="non-empty DSN"):
        PGVectorStore(dsn="")


async def test_methods_require_connection():
    store = PGVectorStore(dsn="postgresql://test/db")  # never connected → no pool
    with pytest.raises(RuntimeError, match="not connected"):
        await store.search([0.1, 0.2], top_k=3)


def test_pgvector_literal_formats_list():
    assert _to_pgvector_literal([1, 2.5]) == "[1.00000000,2.50000000]"


# ── SQL contract ─────────────────────────────────────────────────────────────


async def test_dense_search_uses_cosine_operator_and_maps_rows():
    rows = [{"source": "guides/x.md", "chunk_index": 2, "text": "hi", "score": 0.9}]
    store = _store(rows)
    results = await store.search([0.1, 0.2, 0.3], top_k=5)

    sql, args = store._pool.conn.fetched[-1]
    assert "embedding <=> $1::vector" in sql      # pgvector cosine distance
    assert "LIMIT $2" in sql
    assert args[1] == 5
    assert results[0].source == "guides/x.md"
    assert results[0].chunk_index == 2
    assert results[0].score == 0.9


async def test_lexical_search_calls_infra_function_with_canonical_columns():
    rows = [{"source": "faq.md", "chunk_index": 0, "text": "dim weight", "score": 1.2}]
    store = _store(rows)
    results = await store.search_lexical("dimensional weight", top_k=3)

    sql, args = store._pool.conn.fetched[-1]
    assert "match_rag_chunks_lexical($1, $2)" in sql            # the Infra function
    assert "source, chunk_index, text, score" in sql            # exact column contract
    assert args == ("dimensional weight", 3)
    assert results[0].score == 1.2


async def test_add_upserts_on_source_chunk_index():
    store = _store()
    n = await store.add([
        StoredChunk(text="hello", source="a.md", chunk_index=0, embedding=[0.1, 0.2]),
    ])
    assert n == 1
    sql, _args = store._pool.conn.executed[-1]
    assert "INSERT INTO rag_chunks" in sql
    assert "ON CONFLICT (source, chunk_index)" in sql


async def test_count_async_reads_count_query():
    store = _store([{"x": 1}, {"x": 2}])
    assert await store.count_async() == 2
    sql, _ = store._pool.conn.fetched[-1]
    assert "COUNT(*)" in sql
