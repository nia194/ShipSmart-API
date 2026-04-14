"""Tests for expanded RAG knowledge base — ingestion, retrieval, and source metadata."""

from pathlib import Path

import pytest

from app.rag.chunking import chunk_text
from app.rag.embeddings import create_embedding_provider
from app.rag.ingestion import ingest_documents, load_documents
from app.rag.retrieval import retrieve
from app.rag.vector_store import create_vector_store

DOCS_PATH = Path(__file__).resolve().parent.parent / "data" / "documents"


# ── Document loading ─────────────────────────────────────────────────────────


def test_load_documents_finds_all_files():
    """load_documents should find .md and .txt files in all subdirectories."""
    docs = load_documents(DOCS_PATH)
    assert len(docs) >= 12, f"Expected 12+ documents, found {len(docs)}"


def test_load_documents_uses_relative_paths():
    """Source names should include subdirectory paths, not just filenames."""
    docs = load_documents(DOCS_PATH)
    sources = [name for name, _ in docs]
    # Should contain subdirectory-prefixed paths
    has_subdir = any("/" in s for s in sources)
    assert has_subdir, f"Expected subdirectory paths in sources, got: {sources[:5]}"


def test_load_documents_includes_carriers():
    """Should load carrier-specific documents from carriers/ subdirectory."""
    docs = load_documents(DOCS_PATH)
    sources = [name for name, _ in docs]
    carrier_docs = [s for s in sources if s.startswith("carriers/")]
    assert len(carrier_docs) >= 4, f"Expected 4+ carrier docs, found {len(carrier_docs)}"


def test_load_documents_includes_guides():
    """Should load guide documents from guides/ subdirectory."""
    docs = load_documents(DOCS_PATH)
    sources = [name for name, _ in docs]
    guide_docs = [s for s in sources if s.startswith("guides/")]
    assert len(guide_docs) >= 3, f"Expected 3+ guide docs, found {len(guide_docs)}"


def test_load_documents_includes_scenarios():
    """Should load scenario documents from scenarios/ subdirectory."""
    docs = load_documents(DOCS_PATH)
    sources = [name for name, _ in docs]
    scenario_docs = [s for s in sources if s.startswith("scenarios/")]
    assert len(scenario_docs) >= 2, f"Expected 2+ scenario docs, found {len(scenario_docs)}"


def test_load_documents_includes_policies():
    """Should load policy documents from policies/ subdirectory."""
    docs = load_documents(DOCS_PATH)
    sources = [name for name, _ in docs]
    policy_docs = [s for s in sources if s.startswith("policies/")]
    assert len(policy_docs) >= 2, f"Expected 2+ policy docs, found {len(policy_docs)}"


# ── Chunking ─────────────────────────────────────────────────────────────────


def test_expanded_corpus_chunk_count():
    """Expanded corpus should produce significantly more chunks than 2-doc baseline."""
    docs = load_documents(DOCS_PATH)
    all_chunks = []
    for source, content in docs:
        chunks = chunk_text(content, source=source, chunk_size=500, chunk_overlap=50)
        all_chunks.extend(chunks)
    # 2-doc baseline was ~8-12 chunks; expanded should be 100+
    assert len(all_chunks) >= 50, f"Expected 50+ chunks, got {len(all_chunks)}"


def test_chunk_sources_retain_subdirectory_path():
    """Chunk source metadata should include the subdirectory path."""
    docs = load_documents(DOCS_PATH)
    for source, content in docs:
        chunks = chunk_text(content, source=source, chunk_size=500, chunk_overlap=50)
        if chunks:
            assert chunks[0].source == source
            break


# ── Ingestion ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_expanded_corpus():
    """Full ingestion pipeline should process all documents without error."""
    docs = load_documents(DOCS_PATH)
    embedding_provider = create_embedding_provider()
    vector_store = create_vector_store()

    count = await ingest_documents(
        documents=docs,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        chunk_size=500,
        chunk_overlap=50,
    )
    assert count >= 50, f"Expected 50+ chunks ingested, got {count}"
    assert vector_store.count() == count


# ── Retrieval ────────────────────────────────────────────────────────────────


@pytest.fixture
async def populated_store():
    """Ingest expanded corpus and return (embedding_provider, vector_store)."""
    docs = load_documents(DOCS_PATH)
    ep = create_embedding_provider()
    vs = create_vector_store()
    await ingest_documents(docs, ep, vs, chunk_size=500, chunk_overlap=50)
    return ep, vs


@pytest.mark.asyncio
async def test_retrieval_returns_results(populated_store):
    """Query should return results from the expanded corpus."""
    ep, vs = populated_store
    results = await retrieve("What is dimensional weight?", ep, vs, top_k=3)
    assert len(results) > 0


@pytest.mark.asyncio
async def test_retrieval_sources_have_subdirectory_paths(populated_store):
    """Retrieved results should have source names with subdirectory paths."""
    ep, vs = populated_store
    results = await retrieve("UPS shipping services", ep, vs, top_k=5)
    assert len(results) > 0
    sources = [r.source for r in results]
    has_subdir = any("/" in s for s in sources)
    assert has_subdir, f"Expected subdirectory paths in sources, got: {sources}"


@pytest.mark.asyncio
async def test_retrieval_multiple_sources(populated_store):
    """Broad query should retrieve from multiple source documents."""
    ep, vs = populated_store
    results = await retrieve("compare shipping carriers", ep, vs, top_k=5)
    unique_sources = {r.source for r in results}
    assert len(unique_sources) >= 2, f"Expected 2+ unique sources, got: {unique_sources}"
