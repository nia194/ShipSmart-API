"""
Mock shipping provider for development and testing.
Returns deterministic, realistic-looking results without hitting external APIs.
"""

from __future__ import annotations

import logging
import re

from app.providers.base import ProviderResult
from app.providers.shipping_provider import (
    AddressInput,
    QuotePreviewInput,
    ShippingProvider,
)

logger = logging.getLogger(__name__)

# Simple US zip code pattern for mock validation
_ZIP_PATTERN = re.compile(r"^\d{5}(-\d{4})?$")


class MockShippingProvider(ShippingProvider):
    """Deterministic mock provider for development.

    WARNING: This provider returns fake data. It does not contact any
    real carrier API. Use for pipeline testing only.
    """

    def __init__(self) -> None:
        logger.warning(
            "Using MockShippingProvider — no real carrier integration. "
            "Set SHIPPING_PROVIDER to a real provider for production."
        )

    @property
    def name(self) -> str:
        return "mock"

    async def health_check(self) -> bool:
        return True

    async def validate_address(self, address: AddressInput) -> ProviderResult:
        issues: list[str] = []

        if not address.street.strip():
            issues.append("Street is required")
        if not address.city.strip():
            issues.append("City is required")
        if not address.state.strip():
            issues.append("State is required")
        if not _ZIP_PATTERN.match(address.zip_code):
            issues.append(f"Invalid zip code format: {address.zip_code}")

        if issues:
            return ProviderResult(
                success=False,
                data={"is_valid": False, "issues": issues},
                provider=self.name,
                error="; ".join(issues),
            )

        # Mock normalization: title-case street/city, upper state
        normalized = {
            "street": address.street.strip().title(),
            "city": address.city.strip().title(),
            "state": address.state.strip().upper()[:2],
            "zip_code": address.zip_code.strip(),
            "country": address.country.upper(),
        }

        return ProviderResult(
            success=True,
            data={
                "is_valid": True,
                "normalized_address": normalized,
                "deliverable": True,
                "address_type": "residential",
            },
            provider=self.name,
        )

    async def get_quote_preview(
        self, shipment: QuotePreviewInput,
    ) -> ProviderResult:
        # Mock rate calculation based on weight and dimensions
        dim_weight = (
            shipment.length_in * shipment.width_in * shipment.height_in
        ) / 139  # standard DIM factor
        billable_weight = max(shipment.weight_lbs, dim_weight)

        base_rate = 5.99
        weight_rate = billable_weight * 0.45
        ground_price = round(base_rate + weight_rate, 2)

        services = [
            {
                "service": "Ground",
                "carrier": "MockCarrier",
                "price_usd": ground_price,
                "estimated_days": 5,
            },
            {
                "service": "Express",
                "carrier": "MockCarrier",
                "price_usd": round(ground_price * 1.8, 2),
                "estimated_days": 2,
            },
            {
                "service": "Overnight",
                "carrier": "MockCarrier",
                "price_usd": round(ground_price * 3.2, 2),
                "estimated_days": 1,
            },
        ]

        return ProviderResult(
            success=True,
            data={
                "billable_weight_lbs": round(billable_weight, 2),
                "dim_weight_lbs": round(dim_weight, 2),
                "actual_weight_lbs": shipment.weight_lbs,
                "services": services,
                "disclaimer": "Preview only — not a binding quote. "
                "Final rates from Spring Boot /api/v1/quotes.",
            },
            provider=self.name,
        )
