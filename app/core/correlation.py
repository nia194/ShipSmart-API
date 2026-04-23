"""Correlation-ID context for request-scoped propagation.

Read by middleware, read by outbound clients (java_client, MCP client) so
the same X-Request-Id / traceparent flow across every service hop.
"""
from __future__ import annotations

import contextvars
import secrets

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None,
)
traceparent_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "traceparent", default=None,
)


def new_traceparent() -> str:
    trace_id = secrets.token_hex(16)
    span_id = secrets.token_hex(8)
    return f"00-{trace_id}-{span_id}-01"


def outbound_headers() -> dict[str, str]:
    """Return the X-Request-Id / traceparent headers to forward downstream."""
    headers: dict[str, str] = {}
    rid = request_id_var.get()
    if rid:
        headers["X-Request-Id"] = rid
    tp = traceparent_var.get()
    if tp:
        headers["traceparent"] = tp
    return headers
