"""Deterministic apply-policy for assistant form patches (Product Roadmap §6).

The apply_policy gate is decided by CODE from rule-derived extraction confidence
and field risk — never by LLM self-report (models are systematically miscalibrated
about their own extractions, and a policy gate fed by uncalibrated confidence is a
policy gate that does not exist).

  auto    — high-confidence, non-risky facts: flash the field + offer Undo.
  confirm — a risky field (money/legal/safety weight) or medium confidence.
  none    — advisory intents or low confidence: never mutate the form from prose.
"""

from __future__ import annotations

from app.schemas.typed_outputs import ApplyPolicy

# Fields whose value carries money/legal/safety weight — always confirmed, never auto.
RISKY_FIELDS = frozenset(
    {
        "declared_value",
        "declared_value_usd",
        "customs_category",
        "hs_code",
        "hazmat",
        "contents",
        "insurance",
    }
)

# Intents that read/advise and must never trigger a form mutation.
ADVISORY_INTENTS = frozenset(
    {
        "recommendation",
        "policy_question",
        "compare_options",
        "tracking_question",
        "general_question",
    }
)

AUTO_CONFIDENCE = 0.85
CONFIRM_CONFIDENCE = 0.50


def is_risky_field(field_path: str) -> bool:
    """True if a patch to ``field_path`` needs explicit confirmation."""
    low = field_path.lower()
    leaf = low.rsplit(".", 1)[-1]
    return leaf in RISKY_FIELDS or any(risky in low for risky in RISKY_FIELDS)


def decide_apply_policy(
    *, intent: str | None, field_paths: list[str], confidence: float
) -> ApplyPolicy:
    """Rule-derived apply policy for a proposed form patch (never LLM self-reported)."""
    # Advisory intents (and empty patches) never mutate the form.
    if intent in ADVISORY_INTENTS or not field_paths:
        return "none"
    # Any risky field in the patch forces confirmation regardless of confidence.
    if any(is_risky_field(f) for f in field_paths):
        return "confirm"
    if confidence >= AUTO_CONFIDENCE:
        return "auto"
    if confidence >= CONFIRM_CONFIDENCE:
        return "confirm"
    return "none"
