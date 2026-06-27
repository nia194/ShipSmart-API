"""Proves the slowapi rate limiter actually rejects bursts.

The suite disables the limiter globally (tests/conftest.py::_disable_rate_limiter) so
the wider suite isn't order-sensitive; this test re-enables it to verify the limiter
itself works — closing the gap that disabling it would otherwise leave untested.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.rate_limit import limiter
from app.main import app

client = TestClient(app)


def test_advisor_rate_limit_returns_429_after_burst(monkeypatch):
    monkeypatch.setattr(limiter, "enabled", True)  # override the suite-wide disable

    statuses = [
        client.post("/api/v1/advisor/shipping", json={"query": "hi there"}).status_code
        for _ in range(15)
    ]

    assert statuses[0] != 429, "the first request should be allowed"
    assert 429 in statuses, f"expected a 429 within a 15-request burst, got {statuses}"
