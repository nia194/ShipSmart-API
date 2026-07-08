"""Misuse / evasion gate (Governance & Guardrails §7.1 — the dual-use policy).

A well-grounded shipping advisor without an evasion policy is a smuggling
consultant with citations. This gate refuses to help *evade* shipping rules
(undeclared/under-declared/mislabeled shipments, customs circumvention) even when
the topic is in-scope shipping, and redirects to the compliant path. It never
refuses a legitimate "how do I declare X correctly?" question.

Deterministic + keyless; input is normalized first (§6.3) so obfuscated variants
are caught. Emits ``guardrail:misuse_refused``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.security.normalization import normalize

MISUSE_TAG = "guardrail:misuse_refused"

# Evasion INTENT patterns — these express avoiding a rule, not following it.
_EVASION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("undeclared", re.compile(
        r"\b(without|avoid(?:ing)?|skip(?:ping)?|not?)\s+declar", re.I)),
    ("underdeclare", re.compile(
        r"\bunder[-\s]?declar|\bdeclare\s+(?:a\s+)?lower|"
        r"\blower\s+the\s+(?:declared\s+)?value", re.I)),
    ("mislabel", re.compile(r"\bmis[-\s]?(?:label|declar|classif)", re.I)),
    ("past_customs", re.compile(
        r"\b(?:get|sneak|slip|smuggle)\b[^.?!]{0,40}\b(?:past|through|by)\s+customs", re.I)),
    ("evade_customs", re.compile(
        r"\b(?:evade|circumvent|bypass|dodge|get around)\b[^.?!]{0,30}"
        r"\b(?:customs|declaration|screening|duty|duties|tax)", re.I)),
    ("hide", re.compile(
        r"\bhide\b[^.?!]{0,30}\b(?:from customs|the contents|what.?s inside)", re.I)),
    ("split_to_avoid", re.compile(
        r"\bsplit\b[^.?!]{0,40}\b(?:to (?:avoid|dodge|stay under)|threshold|de.?minimis)", re.I)),
    ("smuggle", re.compile(r"\bsmuggl", re.I)),
    ("falsify", re.compile(
        r"\bfalsif|\bfake\s+(?:the\s+)?(?:invoice|declaration|manifest|value)", re.I)),
]

_REFUSAL = (
    "I can't help with avoiding or circumventing customs declaration or shipping "
    "rules. If it helps, I can explain how to ship the item **correctly** — the "
    "declaration and documentation it needs, and what declaring costs."
)


@dataclass
class MisuseVerdict:
    is_misuse: bool
    matched: list[str] = field(default_factory=list)
    refusal: str = ""
    tags: list[str] = field(default_factory=list)


def check_misuse(text: str) -> MisuseVerdict:
    """Return a refusal verdict if ``text`` asks to evade a shipping/customs rule."""
    norm = normalize(text or "")
    matched = [name for name, pat in _EVASION_PATTERNS if pat.search(norm)]
    if not matched:
        return MisuseVerdict(is_misuse=False)
    return MisuseVerdict(
        is_misuse=True, matched=matched, refusal=_REFUSAL, tags=[MISUSE_TAG]
    )
