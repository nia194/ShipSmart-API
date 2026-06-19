"""
Audit + tracing foundation — emergent auditability, made first-class.

Auditability in ShipSmart is *emergent*: every meaningful branch appends a
namespaced decision tag (``agent:*``, ``compliance:*``, ``critique:*``,
``workflow:*``), and every request carries correlation IDs (``X-Request-Id`` +
W3C ``traceparent``, see ``app.core.correlation``). This module elevates that into
a thin, first-class, swappable layer so the system's actions — and, for the
human-in-the-loop workflow, *every person's* determinations — compose into one
replayable trail.

Design:
  * :class:`AuditEvent` — a typed, frozen, append-only record (who/what/when +
    correlation keys + optional payload). One decision tag = one event.
  * :class:`AuditSink` — a Protocol (port). :class:`LoggingAuditSink` (default)
    emits a structured log line; :class:`InMemoryAuditSink` captures events for
    tests and small in-process trails. A persistent ``PostgresAuditSink`` is a
    future adapter — a swap, not an architecture change.

Best-practice properties: append-only, structured, **non-blocking / best-effort**
(emitting an event must never break a request), and PII/secret-safe (callers pass
only already-safe fields, mirroring the no-secrets `/info` contract).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, runtime_checkable

logger = logging.getLogger("shipsmart.audit")

# Who took the action. ``actor_name`` carries the specific identity
# (e.g. "compliance", "officer") so the trail reads as "everything the system
# AND every person did".
Actor = Literal["system", "agent", "human"]


@dataclass(frozen=True)
class AuditEvent:
    """One append-only audit record. ``ts`` defaults to now (UTC, ISO-8601)."""

    event: str  # the decision tag, e.g. "workflow:interrupt:human_review"
    actor: Actor = "system"
    actor_name: str = ""
    workflow_id: str = ""
    request_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    ts: str = ""

    def __post_init__(self) -> None:
        if not self.ts:
            object.__setattr__(self, "ts", datetime.now(tz=UTC).isoformat())

    def as_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "actor": self.actor,
            "actor_name": self.actor_name,
            "event": self.event,
            "workflow_id": self.workflow_id,
            "request_id": self.request_id,
            "payload": self.payload,
        }


@runtime_checkable
class AuditSink(Protocol):
    """Port for emitting audit events. Implementations must never raise."""

    def emit(self, event: AuditEvent) -> None: ...


class LoggingAuditSink:
    """Default sink — one structured log line per event; best-effort (never raises)."""

    def emit(self, event: AuditEvent) -> None:
        try:
            logger.info("audit %s", event.as_dict())
        except Exception:  # noqa: BLE001 - auditing must never break a request
            pass


class InMemoryAuditSink:
    """Captures events in memory — for tests and small in-process trails."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def emit(self, event: AuditEvent) -> None:
        self._events.append(event)

    @property
    def events(self) -> list[AuditEvent]:
        return list(self._events)


def create_audit_sink(kind: str = "logging") -> AuditSink:
    """Factory: build the configured audit sink.

    ``logging`` (default) → :class:`LoggingAuditSink`; ``memory`` →
    :class:`InMemoryAuditSink`. Unknown values fall back to logging with a warning
    (config never crashes the app). A persistent backend is a future adapter.
    """
    normalized = (kind or "logging").strip().lower()
    if normalized == "memory":
        return InMemoryAuditSink()
    if normalized != "logging":
        logger.warning("Unknown AUDIT_SINK %r — defaulting to logging", kind)
    return LoggingAuditSink()
