"""
Document chunking.
Splits text into overlapping chunks for embedding and retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    """A chunk of text with metadata about its origin."""
    text: str
    source: str
    index: int


def chunk_text(
    text: str,
    source: str = "",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Split text into overlapping chunks by character count.

    Args:
        text: The full document text.
        source: Identifier for the source document (e.g., filename).
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of Chunk objects.
    """
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text_slice = text[start:end].strip()
        if chunk_text_slice:
            chunks.append(Chunk(text=chunk_text_slice, source=source, index=idx))
            idx += 1
        start += chunk_size - chunk_overlap

    return chunks
