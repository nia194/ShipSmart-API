"""
Document ingestion pipeline.
Loads documents, chunks them, generates embeddings, and stores in the vector store.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.rag.chunking import Chunk, chunk_text
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import StoredChunk, VectorStore

logger = logging.getLogger(__name__)


def load_documents(directory: str | Path) -> list[tuple[str, str]]:
    """Load .txt and .md files from a directory, recursing into subdirectories.

    Source names use relative paths from the documents root so that
    subdirectory structure is visible in retrieval results
    (e.g. "carriers/ups-overview.md" instead of just "ups-overview.md").

    Returns:
        List of (relative_path, content) tuples sorted by path.
    """
    path = Path(directory)
    if not path.is_dir():
        logger.warning("Documents directory not found: %s", path)
        return []

    docs: list[tuple[str, str]] = []
    for ext in ("**/*.txt", "**/*.md"):
        for file in sorted(path.glob(ext)):
            content = file.read_text(encoding="utf-8").strip()
            if content:
                # Use relative path from documents root as source name
                relative = file.relative_to(path).as_posix()
                docs.append((relative, content))
                logger.debug("Loaded document: %s (%d chars)", relative, len(content))

    logger.info("Loaded %d documents from %s", len(docs), path)
    return docs


async def ingest_documents(
    documents: list[tuple[str, str]],
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> int:
    """Chunk documents, embed them, and store in the vector store.

    Args:
        documents: List of (source_name, text_content) tuples.
        embedding_provider: Provider for generating embeddings.
        vector_store: Store for persisting chunks + embeddings.
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        Total number of chunks ingested.
    """
    all_chunks: list[Chunk] = []
    for source, content in documents:
        chunks = chunk_text(
            content, source=source, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.info("No chunks to ingest")
        return 0

    logger.info("Embedding %d chunks...", len(all_chunks))
    texts = [c.text for c in all_chunks]
    embeddings = await embedding_provider.embed(texts)

    stored = [
        StoredChunk(
            text=chunk.text,
            source=chunk.source,
            chunk_index=chunk.index,
            embedding=emb,
        )
        for chunk, emb in zip(all_chunks, embeddings, strict=True)
    ]

    count = await vector_store.add(stored)
    logger.info("Ingested %d chunks into vector store", count)
    return count
