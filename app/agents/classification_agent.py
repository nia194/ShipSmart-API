"""
Classification agent (UC3) — pick an HS code for the goods.

Deterministic: asks the injected ``ClassificationProvider`` for ranked candidates
and selects the top one. No LLM (per the keyless-determinism invariant —
classification is part of the deterministic core). The workflow node wraps this
and emits the ``workflow:classify:*`` decision tags.
"""

from __future__ import annotations

from app.domain.models import HsCandidate
from app.domain.ports import ClassificationProvider


def classify(
    description: str, *, provider: ClassificationProvider,
) -> tuple[list[HsCandidate], HsCandidate]:
    """Return ``(candidates, chosen)`` — chosen is the highest-confidence candidate.

    The provider returns candidates ranked by confidence (with an explicit
    low-confidence fallback when nothing matches), so the choice is deterministic.
    """
    candidates = provider.candidates(description)
    chosen = candidates[0]
    return candidates, chosen
