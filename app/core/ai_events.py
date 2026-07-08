"""AI-event sink — the append-only, PII-safe observability plane (guardrails §5.8/§7.5).

Mirrors the ``app.core.audit`` sink port. Every event is built through
:func:`build_ai_event`, which pseudonymizes identity and redacts free text at
write time, so **raw PII never reaches a sink**. Sinks are append-only and
best-effort (emitting must never break a request). A durable ``PostgresAIEventSink``
(writing the ``ai_audit_log`` table from ShipSmart-Infra) is the documented future
adapter — a swap, not an architecture change.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from app.schemas.ai_event import AIEvent
from app.security.pii import pseudonymize, redact, redact_value

logger = logging.getLogger("shipsmart.ai_events")


@runtime_checkable
class AIEventSink(Protocol):
    """Port for emitting AI events. Implementations must never raise."""

    def emit(self, event: AIEvent) -> None: ...


class LoggingAIEventSink:
    """Default sink — one structured log line per event; best-effort."""

    def emit(self, event: AIEvent) -> None:
        try:
            logger.info("ai_event %s", event.model_dump())
        except Exception:  # noqa: BLE001 - observability must never break a request
            pass


class InMemoryAIEventSink:
    """Append-only in-memory capture — for tests and small in-process trails."""

    def __init__(self) -> None:
        self._events: list[AIEvent] = []

    def emit(self, event: AIEvent) -> None:
        self._events.append(event)

    @property
    def events(self) -> list[AIEvent]:
        return list(self._events)


def create_ai_event_sink(kind: str = "logging") -> AIEventSink:
    """Factory: ``logging`` (default) or ``memory``. Unknown -> logging + warn."""
    normalized = (kind or "logging").strip().lower()
    if normalized == "memory":
        return InMemoryAIEventSink()
    if normalized != "logging":
        logger.warning("Unknown AI_EVENT_SINK %r — defaulting to logging", kind)
    return LoggingAIEventSink()


def build_ai_event(
    *,
    request_id: str = "",
    session_id: str | None = None,
    route: str = "",
    intent: str = "",
    provider: str = "",
    model: str = "",
    embedding_version: str = "",
    decisions: list[str] | None = None,
    tool_calls: list[str] | None = None,
    source_ids: list[str] | None = None,
    guardrail_events: list[str] | None = None,
    latency_ms: float = 0.0,
    token_count: int = 0,
    cost_estimate_usd: float = 0.0,
    secret: str,
) -> AIEvent:
    """Build a PII-safe AIEvent: pseudonymize identity, redact free text."""
    return AIEvent(
        request_id=request_id,
        session_id_hash=pseudonymize(session_id, secret=secret, kind="session"),
        route=route,
        intent=redact(intent),
        provider=provider,
        model=model,
        embedding_version=embedding_version,
        decisions=list(decisions or []),
        tool_calls=redact_value(list(tool_calls or [])),
        source_ids=list(source_ids or []),
        guardrail_events=list(guardrail_events or []),
        latency_ms=latency_ms,
        token_count=token_count,
        cost_estimate_usd=cost_estimate_usd,
    )


def record_ai_event(sink: AIEventSink, *, secret: str, **fields) -> AIEvent:
    """Build + emit a PII-safe AIEvent through ``sink`` (best-effort)."""
    event = build_ai_event(secret=secret, **fields)
    sink.emit(event)
    return event
