"""Tests for document chunking."""

from app.rag.chunking import Chunk, chunk_text


def test_chunk_text_basic():
    text = "A" * 100
    chunks = chunk_text(text, source="test.txt", chunk_size=40, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source == "test.txt" for c in chunks)
    assert chunks[0].index == 0
    assert chunks[1].index == 1


def test_chunk_text_overlap():
    text = "abcdefghij" * 10  # 100 chars
    chunks = chunk_text(text, chunk_size=40, chunk_overlap=10)
    # With overlap, second chunk starts at position 30 (40 - 10)
    assert len(chunks) >= 3


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_small():
    text = "short text"
    chunks = chunk_text(text, chunk_size=500)
    assert len(chunks) == 1
    assert chunks[0].text == "short text"


def test_chunk_text_source_metadata():
    chunks = chunk_text("hello world", source="doc.md", chunk_size=500)
    assert chunks[0].source == "doc.md"
    assert chunks[0].index == 0
