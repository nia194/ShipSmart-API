"""Ingestion-time scan + quarantine (Governance & Guardrails §5.7).

Prompt-injection defenses fence retrieved chunks at *query* time, but a poisoned
document should never enter the store in the first place. This scans a document
for injection text at ingestion and quarantines suspects (they are not embedded).
Reuses the same detector as the query-time guardrails. Emits
``guardrail:quarantined_chunk``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.llm.guardrails import detect_injection

QUARANTINE_TAG = "guardrail:quarantined_chunk"


@dataclass
class IngestionVerdict:
    safe: bool
    reasons: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


def scan_document(text: str) -> IngestionVerdict:
    """Return unsafe (quarantine) if ``text`` contains prompt-injection content."""
    hits = detect_injection(text or "")
    if hits:
        return IngestionVerdict(safe=False, reasons=hits, tags=[QUARANTINE_TAG])
    return IngestionVerdict(safe=True)


def partition(docs: dict[str, str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Split {source: text} into (safe_to_ingest, quarantined{source: reasons})."""
    safe: dict[str, str] = {}
    quarantined: dict[str, list[str]] = {}
    for source, text in docs.items():
        verdict = scan_document(text)
        if verdict.safe:
            safe[source] = text
        else:
            quarantined[source] = verdict.reasons
    return safe, quarantined
