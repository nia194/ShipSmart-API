"""Assistant + product audit events (Product Roadmap §13 — extends the AIEvent sink).

The roadmap names a fixed set of product-level audit events (assistant message,
form-patch proposed/applied/undone, tool call requested/completed, quote search,
recommendation rendered, user action clicked). Rather than a parallel logger,
these are thin typed builders over the existing PII-safe append-only AIEvent sink
(``app.core.ai_events``): identity is pseudonymized and free text redacted at
write time. The event name rides the ``intent`` field, so no new decision-tag
namespace is introduced (the §4.2 decision-tag contract stays unchanged).

The patch-undo rate this makes queryable is both a §14 product metric and a §5.6
safety signal (a high undo rate means the extraction is unsafe).
"""

from __future__ import annotations

from app.core.ai_events import AIEventSink, record_ai_event
from app.schemas.ai_event import AIEvent

# The fixed §13 event vocabulary (event name -> ridden on AIEvent.intent).
ASSISTANT_MESSAGE_RECEIVED = "assistant_message_received"
FORM_PATCH_PROPOSED = "form_patch_proposed"
FORM_PATCH_APPLIED = "form_patch_applied"
FORM_PATCH_UNDONE = "form_patch_undone"
TOOL_CALL_REQUESTED = "tool_call_requested"
TOOL_CALL_COMPLETED = "tool_call_completed"
QUOTE_SEARCH_REQUESTED = "quote_search_requested"
QUOTE_SEARCH_COMPLETED = "quote_search_completed"
RECOMMENDATION_RENDERED = "recommendation_rendered"
USER_ACTION_CLICKED = "user_action_clicked"

ASSISTANT_EVENT_TYPES = frozenset(
    {
        ASSISTANT_MESSAGE_RECEIVED,
        FORM_PATCH_PROPOSED,
        FORM_PATCH_APPLIED,
        FORM_PATCH_UNDONE,
        TOOL_CALL_REQUESTED,
        TOOL_CALL_COMPLETED,
        QUOTE_SEARCH_REQUESTED,
        QUOTE_SEARCH_COMPLETED,
        RECOMMENDATION_RENDERED,
        USER_ACTION_CLICKED,
    }
)

ASSISTANT_ROUTE = "assistant"


def record_assistant_event(
    sink: AIEventSink,
    *,
    event_type: str,
    secret: str,
    session_id: str | None = None,
    request_id: str = "",
    detail: str = "",
    tool_calls: list[str] | None = None,
    intent_label: str = "",
    latency_ms: float = 0.0,
) -> AIEvent:
    """Record one §13 product audit event through the PII-safe AIEvent sink.

    ``event_type`` must be a known event; ``detail`` (e.g. an apply-policy or a
    field name) is redacted with the rest of the intent text. Best-effort — the
    sink itself never breaks the request that emitted the event.
    """
    if event_type not in ASSISTANT_EVENT_TYPES:
        raise ValueError(f"unknown assistant event {event_type!r} (not in ASSISTANT_EVENT_TYPES)")
    intent = event_type
    if detail:
        intent = f"{event_type}:{detail}"
    if intent_label:
        intent = f"{intent}:{intent_label}"
    return record_ai_event(
        sink,
        secret=secret,
        route=ASSISTANT_ROUTE,
        intent=intent,
        session_id=session_id,
        request_id=request_id,
        tool_calls=tool_calls or [],
        latency_ms=latency_ms,
    )
