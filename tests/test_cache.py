"""
Tests for TTL cache and caching integration in services.
"""

import time

import pytest

from app.core.cache import TTLCache, rag_cache, recommendation_cache


# ── TTLCache Unit Tests ────────────────────────────────────────────────────


def test_cache_set_and_get():
    cache = TTLCache(default_ttl=10)
    cache.set("k1", "v1")
    assert cache.get("k1") == "v1"


def test_cache_miss_returns_none():
    cache = TTLCache()
    assert cache.get("nonexistent") is None


def test_cache_expired_entry():
    cache = TTLCache(default_ttl=1)
    cache.set("k", "v", ttl=0)
    # TTL=0 means expires at monotonic() + 0 → already expired
    time.sleep(0.01)
    assert cache.get("k") is None


def test_cache_eviction_on_max_size():
    cache = TTLCache(default_ttl=60, max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)  # should evict "a"
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_cache_clear():
    cache = TTLCache()
    cache.set("x", 1)
    cache.set("y", 2)
    cache.clear()
    assert cache.get("x") is None
    assert cache.get("y") is None


def test_cache_stats():
    cache = TTLCache()
    cache.set("a", 1)
    cache.get("a")   # hit
    cache.get("b")   # miss
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size"] == 1


def test_cache_make_key_deterministic():
    k1 = TTLCache.make_key("query", 3, 10)
    k2 = TTLCache.make_key("query", 3, 10)
    assert k1 == k2


def test_cache_make_key_different_inputs():
    k1 = TTLCache.make_key("query1", 3)
    k2 = TTLCache.make_key("query2", 3)
    assert k1 != k2


# ── Shared Instance Tests ─────────────────────────────────────────────────


def test_recommendation_cache_exists():
    assert recommendation_cache._default_ttl == 300
    assert recommendation_cache._max_size == 128


def test_rag_cache_exists():
    assert rag_cache._default_ttl == 120
    assert rag_cache._max_size == 64


# ── Recommendation Caching Integration ─────────────────────────────────────


@pytest.mark.asyncio
async def test_recommendation_caching():
    """Verify generate_recommendations uses cache on repeat calls."""
    from app.core.cache import recommendation_cache
    from app.services.recommendation_service import generate_recommendations

    recommendation_cache.clear()

    services = [
        {"service": "Ground", "price_usd": 9.99, "estimated_days": 5},
        {"service": "Express", "price_usd": 19.99, "estimated_days": 2},
    ]

    result1 = await generate_recommendations(services)
    stats1 = recommendation_cache.stats()

    result2 = await generate_recommendations(services)
    stats2 = recommendation_cache.stats()

    # Second call should be a cache hit
    assert stats2["hits"] == stats1["hits"] + 1
    assert result1.primary_recommendation.service_name == result2.primary_recommendation.service_name
