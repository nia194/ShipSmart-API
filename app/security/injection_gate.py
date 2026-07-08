"""Prompt-injection gate with severity (Governance & Guardrails §5.2).

Keeps detection but separates the OUTCOME into severity tiers so the request
boundary can react proportionally: ``block`` (instruction override / prompt
exfiltration), ``neutralize`` (data-fence breakout), ``warn`` (soft manipulation),
``allow``. Input is normalized first (§6.3). Emits ``guardrail:injection``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.security.normalization import normalize

INJECTION_TAG = "guardrail:injection"

_TIERS: list[tuple[str, list[re.Pattern[str]]]] = [
    ("block", [
        re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions", re.I),
        re.compile(r"\bdisregard\s+(?:the\s+)?(?:system|previous|above)\b", re.I),
        re.compile(r"\byou are now\b|\bact as\b[^.?!]{0,30}(?:DAN|jailbreak|unrestricted)", re.I),
        re.compile(r"\b(?:reveal|print|show|repeat|leak)\b[^.?!]{0,30}"
                   r"(?:system prompt|your instructions|the prompt|your rules)", re.I),
        re.compile(r"\boverride\b[^.?!]{0,20}(?:guardrails|safety|rules|instructions)", re.I),
    ]),
    ("neutralize", [
        re.compile(r"```[^`]*(?:system|assistant)\s*:", re.I),
        re.compile(r"\bend of (?:document|context|data)\b", re.I),
        re.compile(r"\bthe (?:above|following) is (?:a )?(?:new )?(?:instruction|system)", re.I),
    ]),
    ("warn", [
        re.compile(r"\bhypothetically\b|\bfor (?:a )?(?:test|research|experiment)\b", re.I),
        re.compile(r"\bpretend\b|\brole.?play\b", re.I),
    ]),
]


@dataclass
class InjectionVerdict:
    severity: str  # block | neutralize | warn | allow
    matched: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @property
    def should_block(self) -> bool:
        return self.severity == "block"


def classify_injection(text: str) -> InjectionVerdict:
    """Classify the injection severity of ``text`` (most severe tier wins)."""
    norm = normalize(text or "")
    for tier, patterns in _TIERS:
        if any(p.search(norm) for p in patterns):
            return InjectionVerdict(severity=tier, matched=[tier], tags=[INJECTION_TAG])
    return InjectionVerdict(severity="allow")
