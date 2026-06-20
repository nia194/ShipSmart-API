"""
Mock carrier adapter (UC3) — deterministic, keyless.

Computes carrier service options from the mock rate table (base + per-pound, with
an international surcharge + ETA). Implements ``CarrierProvider``. A real backend
would call the live carrier APIs via ShipSmart-MCP's ``get_quote_preview`` — that
seam is the ``MCPCarrierAdapter`` stub below (ship the mock; wire it later).
"""

from __future__ import annotations

from app.domain.data.carrier_rates import CARRIER_TABLE, INTERNATIONAL_SURCHARGE
from app.domain.models import CarrierQuote


class MockCarrierAdapter:
    """Carrier quoting over the mock rate table."""

    def quotes(self, origin: str, destination: str, weight_lbs: float) -> list[CarrierQuote]:
        international = (origin or "").strip().upper() != (destination or "").strip().upper()
        weight = max(0.0, float(weight_lbs))
        out: list[CarrierQuote] = []
        for carrier, service, base, per_lb, days_dom, days_intl in CARRIER_TABLE:
            price = base + per_lb * weight
            if international:
                price *= INTERNATIONAL_SURCHARGE
            out.append(
                CarrierQuote(
                    carrier=carrier, service=service,
                    price_usd=round(price, 2),
                    estimated_days=days_intl if international else days_dom,
                )
            )
        return out


class MCPCarrierAdapter:
    """Documented seam: a real carrier adapter backed by ShipSmart-MCP.

    Not wired in this phase — calling the live MCP ``get_quote_preview`` belongs
    here so the workflow can swap mock → real with no change to the routing agent
    or the orchestrator. Raises until implemented so it can never silently no-op.
    """

    def quotes(self, origin: str, destination: str, weight_lbs: float) -> list[CarrierQuote]:
        raise NotImplementedError(
            "MCPCarrierAdapter is a future seam — use MockCarrierAdapter for now"
        )
