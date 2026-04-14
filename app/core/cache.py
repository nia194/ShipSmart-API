"""
Simple in-memory TTL cache.
No external dependencies — just a dict with expiration timestamps.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class TTLCache:
    """In-memory cache with per-entry TTL (Time to Live) expiration."""

    def __init__(self, default_ttl: int = 120, max_size: int = 256) -> None:
        self._store: dict[str, tuple[float, Any]] = {}
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Get a cached value. Returns None if missing or expired."""
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None

        expires_at, value = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            self._misses += 1
            return None

        self._hits += 1
        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value with optional custom TTL (seconds)."""
        if len(self._store) >= self._max_size:
            self._evict_expired()
            if len(self._store) >= self._max_size:
                # Remove oldest entry
                oldest_key = next(iter(self._store))
                del self._store[oldest_key]

        expires_at = time.monotonic() + (ttl if ttl is not None else self._default_ttl)
        self._store[key] = (expires_at, value)

    def _evict_expired(self) -> None:
        """Remove all expired entries."""
        now = time.monotonic()
        expired = [k for k, (exp, _) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()

    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        return {
            "size": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "max_size": self._max_size,
        }

    @staticmethod
    def make_key(*parts: Any) -> str:
        """Create a deterministic cache key from arbitrary parameters/arguments."""
        raw = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# Shared cache instances
recommendation_cache = TTLCache(default_ttl=300, max_size=128)
rag_cache = TTLCache(default_ttl=120, max_size=64)
