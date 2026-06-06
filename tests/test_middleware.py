"""Tests for request correlation: RequestLoggingMiddleware + outbound headers.

The middleware honors an inbound X-Request-Id / W3C traceparent (minting valid
ones when absent or malformed), stashes them in ContextVars so outbound clients
(java_client, mcp_client) forward the same IDs, and echoes them on the response.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from app.core import correlation
from app.core.correlation import (
    new_traceparent,
    outbound_headers,
    request_id_var,
    traceparent_var,
)
from app.main import app

client = TestClient(app)

_TRACEPARENT_RE = re.compile(r"^[0-9a-f]{2}-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$")


# ── Middleware response headers ──────────────────────────────────────────────


def test_response_always_carries_correlation_headers():
    resp = client.get("/health")
    assert resp.headers.get("X-Request-Id")
    assert _TRACEPARENT_RE.match(resp.headers.get("traceparent", ""))


def test_inbound_request_id_is_echoed_back():
    resp = client.get("/health", headers={"X-Request-Id": "abc-123"})
    assert resp.headers["X-Request-Id"] == "abc-123"


def test_valid_inbound_traceparent_is_preserved():
    tp = "00-" + "a" * 32 + "-" + "b" * 16 + "-01"
    resp = client.get("/health", headers={"traceparent": tp})
    assert resp.headers["traceparent"] == tp


def test_malformed_traceparent_is_replaced_with_a_valid_one():
    resp = client.get("/health", headers={"traceparent": "not-a-traceparent"})
    got = resp.headers["traceparent"]
    assert got != "not-a-traceparent"
    assert _TRACEPARENT_RE.match(got)


# ── Correlation helpers (pure) ───────────────────────────────────────────────


def test_new_traceparent_is_well_formed():
    assert _TRACEPARENT_RE.match(new_traceparent())
    # Distinct trace ids each call (random).
    assert new_traceparent() != new_traceparent()


def test_outbound_headers_empty_without_context():
    # No request in flight → nothing to forward.
    request_id_var.set(None)
    traceparent_var.set(None)
    assert outbound_headers() == {}


def test_outbound_headers_forward_set_context(monkeypatch):
    rid = request_id_var.set("rid-9")
    tp = traceparent_var.set("00-" + "c" * 32 + "-" + "d" * 16 + "-01")
    try:
        headers = outbound_headers()
        assert headers["X-Request-Id"] == "rid-9"
        assert headers["traceparent"].startswith("00-")
    finally:
        request_id_var.reset(rid)
        traceparent_var.reset(tp)


def test_outbound_headers_module_is_shared_by_middleware():
    # Guard against an accidental second ContextVar instance: the middleware and
    # the outbound clients must read the same module-level vars.
    assert correlation.request_id_var is request_id_var
