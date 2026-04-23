"""Request logging middleware.

Honors inbound X-Request-Id and W3C traceparent (mints them if absent),
stores them in ContextVars so outbound clients can forward the same IDs,
and echoes them back as response headers so the browser can surface them.
"""
from __future__ import annotations

import logging
import re
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.correlation import (
    new_traceparent,
    request_id_var,
    traceparent_var,
)

logger = logging.getLogger("shipsmart.requests")

_TRACEPARENT_RE = re.compile(r"^[0-9a-f]{2}-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:  # noqa: ANN001
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        traceparent = request.headers.get("traceparent")
        if not traceparent or not _TRACEPARENT_RE.match(traceparent):
            traceparent = new_traceparent()

        rid_token = request_id_var.set(request_id)
        tp_token = traceparent_var.set(traceparent)
        request.state.request_id = request_id
        request.state.traceparent = traceparent

        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(rid_token)
            traceparent_var.reset(tp_token)
        duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "%s %s → %d (%.1fms) [%s]",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )
        response.headers["X-Request-Id"] = request_id
        response.headers["traceparent"] = traceparent
        return response
