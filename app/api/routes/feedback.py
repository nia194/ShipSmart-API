"""Explicit user feedback (Governance & Guardrails §6.6 — the Layer-6 online loop).

The intake side of the online eval loop: a thumbs-up/down (plus an optional
category and free-text comment) on an AI reply. The record enters the same
append-only, PII-safe AIEvent stream as every other AI event — identity is
pseudonymized and the comment is redacted at build time, so raw PII never
reaches a sink. ShipSmart-Test's promotion pipeline samples these events into
the review queue; a reviewed case lands in a dataset as
``provenance: online_promoted``.

Deliberately fire-and-forget for the caller (202): feedback must never block or
break the product surface that asked for it.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.ai_events import AIEventSink, create_ai_event_sink, record_ai_event
from app.core.config import settings
from app.core.errors import AppError

router = APIRouter(prefix="/feedback", tags=["feedback"])

FEEDBACK_INTENT_PREFIX = "feedback"

_sink: AIEventSink | None = None


def _get_sink() -> AIEventSink:
    global _sink
    if _sink is None:
        _sink = create_ai_event_sink(settings.ai_event_sink)
    return _sink


class FeedbackRequest(BaseModel):
    rating: Literal["up", "down"]
    session_id: str | None = None
    message_id: str = Field(default="", max_length=100)
    category: str = Field(default="", max_length=50)     # e.g. "wrong_answer", "great_answer"
    comment: str = Field(default="", max_length=2000)


class FeedbackResponse(BaseModel):
    status: str = "recorded"


@router.post("", response_model=FeedbackResponse, status_code=202)
async def submit_feedback(body: FeedbackRequest) -> FeedbackResponse:
    """Record one explicit feedback signal as a PII-safe, append-only AIEvent."""
    if not settings.feedback_enabled:
        raise AppError(status_code=404, message="Feedback endpoint is disabled")

    intent = f"{FEEDBACK_INTENT_PREFIX}:{body.rating}"
    if body.category:
        intent += f":{body.category}"

    record_ai_event(
        _get_sink(),
        secret=settings.pseudonym_secret,
        request_id=body.message_id,
        session_id=body.session_id,
        route="/api/v1/feedback",
        intent=intent,
        feedback_comment=body.comment,
    )
    return FeedbackResponse()
