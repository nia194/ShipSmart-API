"""
Embedding provider abstraction.
Supports OpenAI embeddings and a local hash-based placeholder for development.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract interface for generating text embeddings."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of the embeddings."""


class LocalHashEmbedding(EmbeddingProvider):
    """Deterministic hash-based embedding for local development.

    WARNING: This is NOT a real embedding model. It produces fixed-dimension
    vectors from text hashes. Useful only for testing the pipeline architecture.
    Semantic similarity will NOT work meaningfully.
    """

    def __init__(self, dims: int = 256):
        self._dims = dims
        logger.warning(
            "Using LocalHashEmbedding — not suitable for production. "
            "Set EMBEDDING_PROVIDER=openai for real embeddings."
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            # Expand hash bytes to fill the required dimensions
            rng = np.random.default_rng(seed=int.from_bytes(digest[:8]))
            vec = rng.standard_normal(self._dims).tolist()
            # Normalize to unit length
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            result.append(vec)
        return result

    @property
    def dimensions(self) -> int:
        return self._dims


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI text-embedding API provider."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", dims: int = 256):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dims = dims

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            input=texts,
            model=self._model,
            dimensions=self._dims,
        )
        return [item.embedding for item in response.data]

    @property
    def dimensions(self) -> int:
        return self._dims


def create_embedding_provider() -> EmbeddingProvider:
    """Factory: create the configured embedding provider."""
    provider = settings.embedding_provider.lower()

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set"
            )
        return OpenAIEmbedding(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dims=settings.embedding_dimensions,
        )

    # Default: local placeholder
    return LocalHashEmbedding(dims=settings.embedding_dimensions)
