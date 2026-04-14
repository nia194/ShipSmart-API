"""
Shipping provider abstraction.
Defines the interface for carrier-facing operations: address validation,
quote previews, dropoff locations, etc.

Concrete implementations (mock, UPS, FedEx, etc.) implement this interface.
Tools call providers through this abstraction — never directly.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

from app.providers.base import Provider, ProviderResult


@dataclass
class AddressInput:
    """Standardized address for validation."""

    street: str
    city: str
    state: str
    zip_code: str
    country: str = "US"


@dataclass
class QuotePreviewInput:
    """Input for a shipping quote preview."""

    origin_zip: str
    destination_zip: str
    weight_lbs: float
    length_in: float
    width_in: float
    height_in: float


class ShippingProvider(Provider):
    """Abstract interface for shipping/carrier providers."""

    @abstractmethod
    async def validate_address(self, address: AddressInput) -> ProviderResult:
        """Validate and normalize a shipping address."""

    @abstractmethod
    async def get_quote_preview(
        self, shipment: QuotePreviewInput,
    ) -> ProviderResult:
        """Return a non-binding quote preview for a shipment."""
