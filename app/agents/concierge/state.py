"""The pure reducer + slot policy for the Conversational Concierge.

Everything here is deterministic and side-effect-free — the load-bearing
invariant: the conversation is an interface, the merge/clarify logic is code.
"""

from __future__ import annotations

from typing import Any

from app.agents.concierge.models import ConversationState, Slots

# Required slots per intent (drives clarification + the "don't re-ask" check).
# origin_country defaults to "US" at compliance dispatch, so it is NOT required.
REQUIRED_SLOTS: dict[str, tuple[str, ...]] = {
    "quote": ("origin", "destination", "weight_lbs"),
    "compliance": ("destination_country", "description"),
    "tracking": ("tracking_reference",),
    "advice": (),
}

CLARIFICATIONS: dict[str, str] = {
    "origin": "Where are you shipping from?",
    "destination": "Where's it going?",
    "weight_lbs": "About how much does it weigh (in lbs)?",
    "destination_country": "Which country is it going to?",
    "description": "What are you shipping?",
    "tracking_reference": "What's your tracking number?",
}


def is_empty(value: Any) -> bool:
    """Treat None / "" / blank strings / empty collections as empty."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return False


def normalize(value: Any) -> Any:
    """Case/space-insensitive normalization for conflict comparison."""
    if isinstance(value, str):
        return " ".join(value.strip().lower().split())
    return value


def equivalent(a: Any, b: Any) -> bool:
    """Equal after normalization, treating "Atlanta" == "Atlanta, GA"."""
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return True
    if isinstance(na, str) and isinstance(nb, str):
        ha, hb = na.split(",")[0].strip(), nb.split(",")[0].strip()
        if ha and ha == hb:
            return True
    return False


def fold_turn(state: ConversationState, extracted: Slots) -> ConversationState:
    """Pure reducer: merge newly-extracted entities into the slots.

    Empty never overwrites non-empty; newest non-empty wins; an extracted value
    equivalent (after normalization) to the existing one is a no-op.
    """
    slots = dict(state.slots)
    for key, value in (extracted or {}).items():
        if is_empty(value):
            continue
        existing = slots.get(key)
        if existing is not None and equivalent(existing, value):
            continue
        slots[key] = value
    return state.with_(slots=slots)


def missing_required(slots: Slots, intent: str | None) -> list[str]:
    """Required slots for the intent that are still empty (preserves order)."""
    required = REQUIRED_SLOTS.get(intent or "advice", ())
    return [k for k in required if is_empty(slots.get(k))]


def clarification_for(slot: str) -> str:
    return CLARIFICATIONS.get(slot, f"Could you tell me the {slot.replace('_', ' ')}?")
