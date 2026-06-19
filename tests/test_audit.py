"""Tests for the audit + tracing foundation (app/core/audit.py).

Hermetic + keyless. Covers the typed event, the sink port + adapters, and the
config-driven factory. Auditing must be append-only and never raise.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from app.core.audit import (
    AuditEvent,
    InMemoryAuditSink,
    LoggingAuditSink,
    create_audit_sink,
)


def test_audit_event_defaults_timestamp_and_serializes():
    ev = AuditEvent(
        event="compliance:flag:lithium_battery", actor="agent", actor_name="compliance",
        workflow_id="wf-1", request_id="rid-1",
    )
    assert ev.ts  # auto-filled ISO-8601 UTC
    payload = ev.as_dict()
    assert payload["event"] == "compliance:flag:lithium_battery"
    assert payload["actor"] == "agent"
    assert payload["actor_name"] == "compliance"
    assert payload["workflow_id"] == "wf-1"
    assert payload["request_id"] == "rid-1"
    assert payload["ts"] == ev.ts


def test_audit_event_is_append_only_frozen():
    ev = AuditEvent(event="workflow:resume")
    with pytest.raises(FrozenInstanceError):
        ev.event = "tampered"  # type: ignore[misc]


def test_in_memory_sink_captures_in_order():
    sink = InMemoryAuditSink()
    sink.emit(AuditEvent(event="workflow:classify:done"))
    sink.emit(AuditEvent(event="workflow:interrupt:human_review", actor="system"))
    sink.emit(
        AuditEvent(event="workflow:review:determination", actor="human", actor_name="officer")
    )
    assert [e.event for e in sink.events] == [
        "workflow:classify:done",
        "workflow:interrupt:human_review",
        "workflow:review:determination",
    ]
    # `.events` returns a copy — callers cannot mutate the trail in place.
    sink.events.clear()
    assert len(sink.events) == 3


def test_logging_sink_never_raises():
    LoggingAuditSink().emit(AuditEvent(event="workflow:resume"))  # must not raise


def test_factory_selects_sink_and_defaults_gracefully():
    assert isinstance(create_audit_sink("memory"), InMemoryAuditSink)
    assert isinstance(create_audit_sink("logging"), LoggingAuditSink)
    assert isinstance(create_audit_sink("unknown"), LoggingAuditSink)  # graceful default
    assert isinstance(create_audit_sink(""), LoggingAuditSink)
