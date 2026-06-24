"""Domain types for the Conversational Concierge.

``ConversationState`` is the wire-simple, client-owned state echoed back each
turn. Its ``slots`` are the shipment-context SUPERSET both the conventional form
and the chat populate — so the Hybrid Form ⇄ Chat Sync client can map them 1:1
onto its ``ShipmentDraft``. The server does NOT track per-field provenance; the
client owns that and maps to/from its ``Tracked<>`` wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

Slots = dict[str, Any]

# The shipment-context superset. Flat + typed on the wire.
SLOT_KEYS: tuple[str, ...] = (
    "origin", "destination",
    "origin_country", "destination_country",
    "drop_off_date", "expected_delivery_date",
    "weight_lbs", "length_in", "width_in", "height_in",
    "category", "description", "declared_value_usd",
    "priority", "tracking_reference",
)


@dataclass(frozen=True)
class ConversationState:
    """The conversation's typed, client-owned state (echoed every turn)."""

    slots: Slots = field(default_factory=dict)
    intent: str | None = None
    status: str = "gathering"          # "gathering" | "answered"
    pending_clarification: str | None = None
    turns: int = 0

    @staticmethod
    def from_wire(data: dict | None) -> ConversationState:
        """Build state from the client payload, keeping only known slot keys."""
        data = data or {}
        slots = {k: v for k, v in (data.get("slots") or {}).items() if k in SLOT_KEYS}
        return ConversationState(
            slots=slots,
            intent=data.get("intent"),
            status=data.get("status") or "gathering",
            pending_clarification=data.get("pending_clarification"),
            turns=int(data.get("turns") or 0),
        )

    def to_wire(self) -> dict:
        return {
            "slots": dict(self.slots),
            "intent": self.intent,
            "status": self.status,
            "pending_clarification": self.pending_clarification,
            "turns": self.turns,
        }

    def with_(self, **changes: Any) -> ConversationState:
        return replace(self, **changes)


@dataclass(frozen=True)
class ConciergeResult:
    """The outcome of one concierge turn."""

    reply: str
    state: ConversationState
    clarification: str | None = None
    dispatched_to: str | None = None
    sources: list[dict] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    provider: str = ""
