"""Request/response schemas for the Conversational Concierge endpoint.

The ``state`` is client-owned and echoed verbatim each turn (no server store).
Its ``slots`` are the shared shipment-context superset (form + chat).
"""

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.chat import ReplyMessage


class ConciergeState(BaseModel):
    """The conversation state echoed back to (and re-sent by) the client."""

    slots: dict[str, Any] = Field(default_factory=dict)
    intent: str | None = None
    status: str = "gathering"
    pending_clarification: str | None = None
    turns: int = 0


class ConciergeRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    state: ConciergeState | None = None
    # Optional anonymous session id for server-side recall. Absent on the first
    # turn → the server mints one and echoes it back; the client persists it
    # (e.g. localStorage) and re-sends it so the chat survives a page reload.
    session_id: str | None = None
    # WhatsApp-style "reply to a message": the replied-to message + a few recent turns,
    # used only to resolve references in `message`. Bounded server-side; never authoritative
    # over the live shipment slots / worker results.
    reply_to: ReplyMessage | None = None
    recent_history: list[ReplyMessage] | None = None


class ConciergeResponse(BaseModel):
    reply: str
    state: ConciergeState
    session_id: str | None = None
    clarification: str | None = None
    dispatched_to: str | None = None
    sources: list[dict] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    provider: str = ""


class ConciergeMessage(BaseModel):
    """One persisted transcript turn, replayed on recall."""

    role: str
    content: str
    created_at: str = ""


class ConciergeHistoryResponse(BaseModel):
    """Persisted conversation, returned by GET /concierge/{session_id} for recall."""

    session_id: str
    state: ConciergeState
    messages: list[ConciergeMessage] = Field(default_factory=list)
