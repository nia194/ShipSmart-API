"""The AIEvent schema + the append-only, PII-safe AI-event sink (guardrails §5.8/§7.5)."""

from __future__ import annotations

from app.core.ai_events import (
    InMemoryAIEventSink,
    LoggingAIEventSink,
    build_ai_event,
    create_ai_event_sink,
    record_ai_event,
)
from app.schemas.ai_event import PROMPT_VERSION, SCHEMA_VERSION, AIEvent

SECRET = "test-secret"


def test_ai_event_defaults_carry_versions():
    e = AIEvent(request_id="r1", route="/agent/run", provider="openai", model="gpt-4o")
    assert e.prompt_version == PROMPT_VERSION and e.schema_version == SCHEMA_VERSION
    assert e.decisions == [] and e.session_id_hash is None


def test_build_pseudonymizes_identity_and_redacts_free_text():
    e = build_ai_event(
        request_id="r1",
        session_id="anon-session-42",
        route="/api/v1/agent/run",
        intent="ship to 123 Main Street for jane@example.com",
        provider="openai",
        model="gpt-4o",
        decisions=["agent:plan"],
        secret=SECRET,
    )
    # identity pseudonymized (never the raw session id)
    assert e.session_id_hash and e.session_id_hash.startswith("sess_")
    assert "anon-session-42" not in (e.session_id_hash or "")
    # free text carries no raw PII
    assert "123 Main Street" not in e.intent and "jane@example.com" not in e.intent
    assert "[REDACTED_ADDRESS]" in e.intent


def test_inmemory_sink_is_append_only_capture():
    sink = InMemoryAIEventSink()
    record_ai_event(sink, secret=SECRET, request_id="r1", route="/x")
    record_ai_event(sink, secret=SECRET, request_id="r2", route="/y")
    assert [ev.request_id for ev in sink.events] == ["r1", "r2"]


def test_logging_sink_never_raises_and_factory():
    assert isinstance(create_ai_event_sink("memory"), InMemoryAIEventSink)
    assert isinstance(create_ai_event_sink("nonsense"), LoggingAIEventSink)
    # emitting must never raise even on odd input
    create_ai_event_sink("logging").emit(AIEvent(request_id="r"))
