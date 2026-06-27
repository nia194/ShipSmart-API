"""Bounded conversation-reference context for the chat "reply-to-a-message" feature.

Turns a replied-to message + a few recent turns into ONE fenced, size-bounded
``<conversation_reference>`` block for inclusion in a grounded prompt. This is a
LOWER-priority reference than the live shipment/quote/tool context — it exists only to
help the model resolve what the current question refers to (e.g. "the cheaper one"); the
grounding rules in ``app.llm.guardrails`` keep the current data authoritative. Reused by
the shipping advisor and the concierge.

Bounding is enforced HERE (turn cap + per-text truncation) so no client can bloat the
prompt, and the untrusted message text is stripped of any fence/role tokens — including
this module's own tags — so it cannot break out of its block.
"""

from __future__ import annotations

import re
from typing import Any

# Bounds (kept conservative to avoid prompt bloat; easily promoted to settings later).
REPLY_MAX_CHARS = 800
HISTORY_MAX_TURNS = 6
HISTORY_TURN_MAX_CHARS = 400

_ROLES = ("user", "assistant")
# Strip the existing prompt fences AND this module's reference tags from untrusted text.
_FENCE = re.compile(
    r"</?\s*(user_input|retrieved_chunk|tool_results|system|assistant|"
    r"conversation_reference|reply_to|recent_turns)\b[^>]*>",
    re.I,
)


def _role(role: Any) -> str:
    r = str(role or "").strip().lower()
    return r if r in _ROLES else "user"


def _clean(text: Any, limit: int) -> str:
    s = "" if text is None else str(text)
    return _FENCE.sub("[tag]", s).strip()[:limit]


def _pair(msg: Any, limit: int) -> tuple[str, str] | None:
    """Normalize a {role, text} dict OR a pydantic-like (.role/.text) into (role, text)."""
    if msg is None:
        return None
    if isinstance(msg, dict):
        role, text = msg.get("role"), msg.get("text")
    else:
        role, text = getattr(msg, "role", None), getattr(msg, "text", None)
    cleaned = _clean(text, limit)
    if not cleaned:
        return None
    return _role(role), cleaned


def bound_history(history: list | None) -> list[tuple[str, str]]:
    """Last ``HISTORY_MAX_TURNS`` turns, each truncated; empties dropped. Order preserved."""
    pairs: list[tuple[str, str]] = []
    for msg in (history or [])[-HISTORY_MAX_TURNS:]:
        pair = _pair(msg, HISTORY_TURN_MAX_CHARS)
        if pair:
            pairs.append(pair)
    return pairs


def render_reference_block(reply_to: Any = None, history: list | None = None) -> str:
    """Build the fenced ``<conversation_reference>`` block, or ``""`` when there's nothing."""
    parts: list[str] = []
    rt = _pair(reply_to, REPLY_MAX_CHARS)
    if rt:
        parts.append(f'<reply_to role="{rt[0]}">\n{rt[1]}\n</reply_to>')
    pairs = bound_history(history)
    if pairs:
        lines = "\n".join(f"[{role}] {text}" for role, text in pairs)
        parts.append(f"<recent_turns>\n{lines}\n</recent_turns>")
    if not parts:
        return ""
    return "<conversation_reference>\n" + "\n".join(parts) + "\n</conversation_reference>"
