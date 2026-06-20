"""
Landed-cost agent (UC3) — estimate duty + import tax for the shipment.

Deterministic pass-through to the injected ``DutyRateProvider`` for the chosen HS
code and lane. No LLM. The workflow node wraps this and emits the
``workflow:landed_cost:*`` decision tags.
"""

from __future__ import annotations

from app.domain.models import DutyQuote
from app.domain.ports import DutyRateProvider


def estimate(
    hs_code: str, origin: str, destination: str, value_usd: float,
    *, provider: DutyRateProvider,
) -> DutyQuote:
    """Return the landed-cost quote (duty + import tax + total) for the lane."""
    return provider.rate(hs_code, origin, destination, value_usd)
