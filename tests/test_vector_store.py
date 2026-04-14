"""Tests for in-memory vector store."""

import pytest

from app.rag.vector_store import InMemoryVectorStore, StoredChunk


@pytest.fixture
def store():
    return InMemoryVectorStore()


@pytest.mark.asyncio
async def test_add_and_count(store):
    assert store.count() == 0
    chunks = [
        StoredChunk(text="hello", source="a.txt", chunk_index=0, embedding=[1.0, 0.0, 0.0]),
        StoredChunk(text="world", source="a.txt", chunk_index=1, embedding=[0.0, 1.0, 0.0]),
    ]
    added = await store.add(chunks)
    assert added == 2
    assert store.count() == 2


@pytest.mark.asyncio
async def test_search_returns_most_similar(store):
    chunks = [
        StoredChunk(text="match", source="a.txt", chunk_index=0, embedding=[1.0, 0.0, 0.0]),
        StoredChunk(text="different", source="a.txt", chunk_index=1, embedding=[0.0, 1.0, 0.0]),
    ]
    await store.add(chunks)
    results = await store.search([1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].text == "match"
    assert results[0].score > 0.99


@pytest.mark.asyncio
async def test_search_empty_store(store):
    results = await store.search([1.0, 0.0], top_k=3)
    assert results == []


@pytest.mark.asyncio
async def test_clear(store):
    chunks = [StoredChunk(text="x", source="a.txt", chunk_index=0, embedding=[1.0])]
    await store.add(chunks)
    assert store.count() == 1
    await store.clear()
    assert store.count() == 0


@pytest.mark.asyncio
async def test_search_respects_top_k(store):
    chunks = [
        StoredChunk(text=f"chunk-{i}", source="a.txt", chunk_index=i, embedding=[float(i), 0.0])
        for i in range(5)
    ]
    await store.add(chunks)
    results = await store.search([4.0, 0.0], top_k=2)
    assert len(results) == 2
