"""Proactive reply composition (NLG) for the Conversational Concierge.

The concierge should *guide*, not just answer: confirm what it captured, ask only
the next missing detail, acknowledge corrections, and — once it has enough —
summarize progress and suggest the next best action. These are deterministic,
grounded templates (keyless-safe + testable). An OPTIONAL model polish rephrases
the template more naturally when a real provider is wired; it may only REPHRASE,
never invent facts (it is grounded by the same slot summary), and any failure
falls back to the template. This is where the assistant stops feeling like a
generic chatbot and starts feeling like a shipment setup assistant.
"""

from __future__ import annotations

from app.agents.concierge.models import Slots
from app.llm.guardrails import assemble
from app.llm.router import TASK_SYNTHESIS, LLMRouter

# Human labels + display order for a captured-fields summary.
_SUMMARY_ORDER: tuple[tuple[str, str], ...] = (
    ("origin", "from {}"),
    ("destination", "to {}"),
    ("weight_lbs", "{} lb"),
    ("length_in", None),  # dims handled together below
    ("declared_value_usd", "${}"),
    ("drop_off_date", "drop-off {}"),
    ("expected_delivery_date", "by {}"),
    ("priority", "{} priority"),
    ("description", "{}"),
)

_FRIENDLY: dict[str, str] = {
    "origin": "origin",
    "destination": "destination",
    "weight_lbs": "weight",
    "destination_country": "destination country",
    "origin_country": "origin country",
    "description": "what you're shipping",
    "tracking_reference": "tracking number",
    "drop_off_date": "drop-off date",
    "expected_delivery_date": "delivery date",
    "declared_value_usd": "declared value",
}

# What a completed intent unlocks → the next-best-action suggestion.
_NEXT_BEST: dict[str, str] = {
    "quote": "I can pull live shipping options now — or add package dimensions for a sharper rate.",
    "compliance": "I can run a compliance check on this shipment now.",
    "tracking": "I can look up the latest status for that tracking number.",
    "advice": "Ask me anything about this shipment and I'll help.",
}


def summarize_slots(slots: Slots) -> str:
    """A compact, human one-liner of the known shipment, e.g. 'from Atlanta to Seattle, 12 lb'."""
    parts: list[str] = []
    for key, tmpl in _SUMMARY_ORDER:
        if key == "length_in":
            if all(slots.get(d) not in (None, "") for d in ("length_in", "width_in", "height_in")):
                parts.append(
                    f"{slots['length_in']}×{slots['width_in']}×{slots['height_in']} in"
                )
            continue
        value = slots.get(key)
        if value in (None, "") or tmpl is None:
            continue
        parts.append(tmpl.format(value))
    return ", ".join(parts)


def correction_note(corrections: Slots) -> str:
    """A short 'Updated X to Y.' acknowledgement, or '' when nothing was corrected."""
    if not corrections:
        return ""
    bits = [f"{_FRIENDLY.get(k, k.replace('_', ' '))} to {v}" for k, v in corrections.items()]
    return "Updated " + ", ".join(bits) + ". "


def _secondary_suffix(secondary_intents: list[str]) -> str:
    extra = [i for i in secondary_intents if i in ("compliance", "quote", "tracking")]
    if "compliance" in extra:
        return " (I can also check customs/compliance for this when you're ready.)"
    return ""


async def compose_gathering_reply(
    next_clarification: str,
    slots: Slots,
    *,
    corrections: Slots | None = None,
    secondary_intents: list[str] | None = None,
    llm_router: LLMRouter | None = None,
    request_id: str = "",
) -> str:
    """Confirm what's captured + ask the next missing detail (optionally polished)."""
    summary = summarize_slots(slots)
    prefix = correction_note(corrections or {})
    lead = f"{prefix}Got it — {summary}. " if summary else prefix
    template = f"{lead}{next_clarification}{_secondary_suffix(secondary_intents or [])}"
    return await _maybe_polish(template, summary, llm_router, request_id)


async def compose_ready_summary(
    slots: Slots,
    intent: str,
    *,
    corrections: Slots | None = None,
    llm_router: LLMRouter | None = None,
    request_id: str = "",
) -> str:
    """Everything required is present but no live worker is wired — summarize + suggest."""
    summary = summarize_slots(slots) or "your shipment details"
    prefix = correction_note(corrections or {})
    suggestion = _NEXT_BEST.get(intent, _NEXT_BEST["advice"])
    template = f"{prefix}I have {summary}. {suggestion}"
    return await _maybe_polish(template, summary, llm_router, request_id)


async def _maybe_polish(
    template: str, grounding: str, llm_router: LLMRouter | None, request_id: str,
) -> str:
    """Rephrase the template naturally via the synthesis model — facts-locked.

    Keyless/echo or any error → the deterministic template stands.
    """
    if llm_router is None:
        return template
    try:
        client = llm_router.for_task(TASK_SYNTHESIS)
    except Exception:
        return template
    if getattr(client, "provider_name", "") in ("", "echo", "scripted"):
        return template
    system = (
        "Rephrase the assistant's draft reply to a shipping customer so it sounds warm, "
        "concise, and proactive. Do NOT add, remove, or change any facts, numbers, or the "
        "question being asked. Keep it to 1-2 sentences. Return only the rephrased reply.\n"
        f"Known shipment facts (do not contradict): {grounding or 'none yet'}"
    )
    assembled = assemble(
        system_prompt=system, user_text=template, contexts=[], request_id=request_id,
    )
    if assembled.blocked:
        return template
    try:
        out = (await client.complete(assembled.messages) or "").strip()
    except Exception:
        return template
    if not out:
        return template
    # Guard against a degenerate rephrase: a clarifying question must survive the
    # polish. If the template asks something ("…?") but the rephrase dropped the
    # question, keep the deterministic template rather than emit an apology.
    if "?" in template and "?" not in out:
        return template
    return out
