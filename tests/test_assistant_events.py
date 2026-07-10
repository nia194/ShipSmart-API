"""Assistant/product audit-event tests (Product Roadmap §13)."""

from __future__ import annotations

import pytest

from app.core import assistant_events as ev
from app.core.ai_events import InMemoryAIEventSink

SECRET = "test-secret"


def test_records_a_pii_safe_event_on_the_intent_field():
    sink = InMemoryAIEventSink()
    ev.record_assistant_event(
        sink,
        event_type=ev.FORM_PATCH_APPLIED,
        secret=SECRET,
        session_id="sess-9",
        detail="origin.city",
        intent_label="auto",
    )
    e = sink.events[0]
    assert e.route == "assistant"
    assert e.intent == "form_patch_applied:origin.city:auto"
    # identity pseudonymized, never raw
    assert e.session_id_hash and "sess-9" not in e.session_id_hash


def test_tool_call_event_carries_tool_names_hashed_safe():
    sink = InMemoryAIEventSink()
    ev.record_assistant_event(
        sink,
        event_type=ev.TOOL_CALL_COMPLETED,
        secret=SECRET,
        tool_calls=["get_quote_preview"],
        latency_ms=42.0,
    )
    e = sink.events[0]
    assert e.intent == "tool_call_completed"
    assert e.tool_calls == ["get_quote_preview"] and e.latency_ms == 42.0


def test_unknown_event_type_is_rejected():
    with pytest.raises(ValueError):
        ev.record_assistant_event(
            InMemoryAIEventSink(), event_type="not_an_event", secret=SECRET
        )


def test_all_roadmap_events_are_recordable():
    sink = InMemoryAIEventSink()
    for event_type in ev.ASSISTANT_EVENT_TYPES:
        ev.record_assistant_event(sink, event_type=event_type, secret=SECRET)
    assert len(sink.events) == len(ev.ASSISTANT_EVENT_TYPES)
    assert {e.intent for e in sink.events} == ev.ASSISTANT_EVENT_TYPES


def test_redacts_pii_in_the_detail():
    sink = InMemoryAIEventSink()
    ev.record_assistant_event(
        sink,
        event_type=ev.ASSISTANT_MESSAGE_RECEIVED,
        secret=SECRET,
        detail="contact me at jane.doe@example.com",
    )
    assert "jane.doe@example.com" not in sink.events[0].intent
