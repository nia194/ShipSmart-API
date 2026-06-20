"""
Mock carrier-rate table (UC3) — realistic, attributed, easy to find/replace.

Drives ``MockCarrierAdapter``. Each carrier service has a base charge, a per-pound
rate, and domestic/international transit estimates. International shipments apply a
surcharge multiplier and the international ETA. Mirrors the MCP ``get_quote_preview``
shape (carrier / service / price_usd / estimated_days). NOT binding quotes.
"""

from __future__ import annotations

# (carrier, service, base_usd, per_lb_usd, days_domestic, days_international)
CARRIER_TABLE: list[tuple[str, str, float, float, int, int]] = [
    ("GlobalPost", "Economy", 6.50, 0.55, 6, 12),
    ("SwiftEx", "Ground", 8.99, 0.65, 4, 9),
    ("SwiftEx", "Express", 18.00, 1.20, 2, 4),
    ("AeroFreight", "Priority", 27.50, 1.80, 1, 3),
]

# Surcharge applied to the price for international lanes (customs handling, fuel).
INTERNATIONAL_SURCHARGE: float = 1.4
