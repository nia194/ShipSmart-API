"""
Mock classification adapter (UC3) — deterministic, keyless.

Substring-matches a goods description against the ``HS_TABLE`` keyword rows and
returns the matches ranked by confidence. When nothing matches it returns the
explicit low-confidence fallback (``9999.99``) — an honest "needs manual review"
rather than a fabricated code. Implements ``ClassificationProvider``.
"""

from __future__ import annotations

from app.domain.data.hs_codes import HS_FALLBACK, HS_TABLE
from app.domain.models import HsCandidate


class MockClassificationAdapter:
    """Keyword-driven HS classification over the mock table."""

    def candidates(self, description: str) -> list[HsCandidate]:
        low = (description or "").lower()
        matches = [
            HsCandidate(hs_code=hs, title=title, confidence=conf)
            for keywords, hs, title, conf in HS_TABLE
            if any(kw in low for kw in keywords)
        ]
        if not matches:
            code, title, conf = HS_FALLBACK
            return [HsCandidate(hs_code=code, title=title, confidence=conf)]
        return sorted(matches, key=lambda c: c.confidence, reverse=True)
