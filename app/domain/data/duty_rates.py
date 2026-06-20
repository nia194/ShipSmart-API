"""
Mock duty + import-tax table (UC3) — realistic, attributed, easy to find/replace.

Drives ``MockDutyRateAdapter``: customs duty by HS chapter × destination, the
destination's import tax (VAT/GST), and a simple trade-agreement preference
(USMCA). Representative real-world values for a credible demo — NOT authoritative.
"""

from __future__ import annotations

# Supported demo destinations (ISO-3166 alpha-2).
DESTINATIONS: tuple[str, ...] = ("US", "DE", "GB", "CA", "AU", "JP")

# Destination import tax: (label, rate). US has no federal VAT (sales tax varies).
IMPORT_TAX: dict[str, tuple[str, float]] = {
    "US": ("Sales tax (varies by state)", 0.00),
    "DE": ("VAT", 0.19),
    "GB": ("VAT", 0.20),
    "CA": ("GST", 0.05),
    "AU": ("GST", 0.10),
    "JP": ("Consumption tax", 0.10),
}

# Customs duty % by HS chapter (first two digits of the HS code) × destination.
# Electronics (84/85) ~0% under the WTO Information Technology Agreement; apparel
# (61/62) and footwear (64) carry real duty; wine (22) is destination-specific.
DUTY_BY_CHAPTER: dict[str, dict[str, float]] = {
    "84": {"US": 0.00, "DE": 0.00, "GB": 0.00, "CA": 0.00, "AU": 0.00, "JP": 0.00},
    "85": {"US": 0.00, "DE": 0.00, "GB": 0.00, "CA": 0.00, "AU": 0.05, "JP": 0.00},
    "88": {"US": 0.00, "DE": 0.00, "GB": 0.00, "CA": 0.00, "AU": 0.00, "JP": 0.00},
    "61": {"US": 0.16, "DE": 0.12, "GB": 0.12, "CA": 0.18, "AU": 0.05, "JP": 0.09},
    "62": {"US": 0.16, "DE": 0.12, "GB": 0.12, "CA": 0.18, "AU": 0.05, "JP": 0.09},
    "64": {"US": 0.20, "DE": 0.17, "GB": 0.16, "CA": 0.18, "AU": 0.05, "JP": 0.30},
    "42": {"US": 0.08, "DE": 0.03, "GB": 0.03, "CA": 0.08, "AU": 0.05, "JP": 0.10},
    "22": {"US": 0.06, "DE": 0.14, "GB": 0.14, "CA": 0.05, "AU": 0.05, "JP": 0.15},
    "33": {"US": 0.00, "DE": 0.00, "GB": 0.00, "CA": 0.065, "AU": 0.05, "JP": 0.00},
    "90": {"US": 0.025, "DE": 0.00, "GB": 0.00, "CA": 0.00, "AU": 0.05, "JP": 0.00},
    "91": {"US": 0.05, "DE": 0.045, "GB": 0.045, "CA": 0.05, "AU": 0.05, "JP": 0.00},
    "95": {"US": 0.00, "DE": 0.047, "GB": 0.047, "CA": 0.08, "AU": 0.05, "JP": 0.035},
    "49": {"US": 0.00, "DE": 0.00, "GB": 0.00, "CA": 0.00, "AU": 0.00, "JP": 0.00},
    "09": {"US": 0.00, "DE": 0.075, "GB": 0.00, "CA": 0.00, "AU": 0.05, "JP": 0.20},
    "21": {"US": 0.064, "DE": 0.09, "GB": 0.06, "CA": 0.11, "AU": 0.05, "JP": 0.10},
}

# Duty applied when the HS chapter isn't in the table (a sensible non-zero default).
DEFAULT_DUTY: dict[str, float] = {
    "US": 0.035, "DE": 0.04, "GB": 0.04, "CA": 0.05, "AU": 0.05, "JP": 0.045,
}

# Trade-agreement preference: members shipping to one another get 0% duty.
USMCA: frozenset[str] = frozenset({"US", "CA", "MX"})
