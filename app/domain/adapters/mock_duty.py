"""
Mock duty-rate adapter (UC3) — deterministic, keyless.

Computes a landed-cost estimate from the mock duty/tax table: duty by HS chapter
× destination, the destination's import tax (VAT/GST), and a USMCA trade
preference (0% duty between members). Implements ``DutyRateProvider``. All
arithmetic is pure and rounded for stable, reproducible output.
"""

from __future__ import annotations

from app.domain.data.duty_rates import (
    DEFAULT_DUTY,
    DUTY_BY_CHAPTER,
    IMPORT_TAX,
    USMCA,
)
from app.domain.models import DutyQuote

_DEFAULT_TAX: tuple[str, float] = ("Import tax", 0.10)


class MockDutyRateAdapter:
    """Landed-cost estimator over the mock duty/tax table."""

    def rate(
        self, hs_code: str, origin: str, destination: str, value_usd: float,
    ) -> DutyQuote:
        dest = (destination or "").strip().upper()
        org = (origin or "").strip().upper()
        chapter = (hs_code or "").strip()[:2]

        duty_pct = DUTY_BY_CHAPTER.get(chapter, DEFAULT_DUTY).get(
            dest, DEFAULT_DUTY.get(dest, 0.05)
        )
        trade_note = ""
        if duty_pct > 0 and org in USMCA and dest in USMCA and org != dest:
            duty_pct = 0.0
            trade_note = "USMCA-qualifying origin → 0% duty"

        tax_label, tax_pct = IMPORT_TAX.get(dest, _DEFAULT_TAX)

        value = max(0.0, float(value_usd))
        duty_usd = round(value * duty_pct, 2)
        # Import tax is assessed on the duty-inclusive value (common practice).
        tax_usd = round((value + duty_usd) * tax_pct, 2)
        total = round(value + duty_usd + tax_usd, 2)

        return DutyQuote(
            hs_code=hs_code, destination=dest, value_usd=round(value, 2),
            duty_pct=duty_pct, duty_usd=duty_usd,
            tax_label=tax_label, tax_pct=tax_pct, tax_usd=tax_usd,
            total_landed_usd=total, trade_note=trade_note,
        )
