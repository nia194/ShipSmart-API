"""Explicit-feedback endpoint tests (§6.6 / Layer-6 online loop — F10)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.routes import feedback as feedback_route
from app.core.ai_events import InMemoryAIEventSink
from app.core.config import settings
from app.main import app

client = TestClient(app)


@pytest.fixture()
def sink(monkeypatch):
    mem = InMemoryAIEventSink()
    monkeypatch.setattr(feedback_route, "_sink", mem)
    return mem


def test_feedback_is_recorded_as_a_pii_safe_ai_event(sink):
    r = client.post(
        "/api/v1/feedback",
        json={
            "rating": "down",
            "session_id": "sess-123",
            "message_id": "msg-9",
            "category": "wrong_answer",
            "comment": "It quoted the wrong price. Reach me at jane.doe@example.com.",
        },
    )
    assert r.status_code == 202 and r.json() == {"status": "recorded"}

    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.route == "/api/v1/feedback"
    assert event.intent == "feedback:down:wrong_answer"
    assert event.request_id == "msg-9"
    # identity pseudonymized, never raw
    assert event.session_id_hash and "sess-123" not in event.session_id_hash
    # the comment is stored REDACTED — the raw email must never reach a sink
    assert "jane.doe@example.com" not in event.feedback_comment
    assert "wrong price" in event.feedback_comment


def test_feedback_up_without_optionals(sink):
    r = client.post("/api/v1/feedback", json={"rating": "up"})
    assert r.status_code == 202
    assert sink.events[0].intent == "feedback:up"
    assert sink.events[0].feedback_comment == ""


def test_feedback_validates_rating_and_caps_comment(sink):
    assert client.post("/api/v1/feedback", json={"rating": "meh"}).status_code == 422
    too_long = {"rating": "up", "comment": "x" * 2001}
    assert client.post("/api/v1/feedback", json=too_long).status_code == 422
    assert sink.events == []


def test_feedback_404s_when_disabled(sink, monkeypatch):
    monkeypatch.setattr(settings, "feedback_enabled", False)
    assert client.post("/api/v1/feedback", json={"rating": "up"}).status_code == 404
    assert sink.events == []
