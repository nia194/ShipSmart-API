"""
DHL Express shipping provider.
Integrates with DHL Express MyDHL API for rating.

API docs:
  - Rates: https://developer.dhl.com/api-reference/dhl-express-mydhl-api#get-rates
  - Auth: Basic auth (DHL_API_KEY as username, DHL_API_SECRET as password)

Requires:
  DHL_API_KEY, DHL_API_SECRET, DHL_ACCOUNT_NUMBER

Limitations:
  - DHL Express API does not offer a standalone address validation endpoint.
    validate_address() returns a basic format check only (not a real DHL call).
  - DHL Express is primarily international. Domestic US rates may not be available.
"""

from __future__ import annotations

import logging
import re
from datetime import date

import httpx

from app.core.config import settings
from app.providers.base import ProviderResult
from app.providers.shipping_provider import (
    AddressInput,
    QuotePreviewInput,
    ShippingProvider,
)

logger = logging.getLogger(__name__)

_ZIP_PATTERN = re.compile(r"^\d{5}(-\d{4})?$")

# DHL product codes for common services
_DHL_PRODUCTS: dict[str, tuple[str, int]] = {
    "P": ("DHL Express Worldwide", 4),
    "D": ("DHL Express Worldwide (Doc)", 4),
    "K": ("DHL Express 9:00", 2),
    "E": ("DHL Express 12:00", 2),
    "N": ("DHL Domestic Express", 2),
    "T": ("DHL Express 10:30", 2),
    "U": ("DHL Express Worldwide (EU)", 4),
    "Y": ("DHL Express 12:00 (Doc)", 2),
}


class DHLProvider(ShippingProvider):
    """DHL Express MyDHL API provider.

    Uses Basic auth (API key + secret).
    Calls the Rate endpoint for quote previews.
    Address validation is local-only (DHL has no public AV API).
    """

    def __init__(self) -> None:
        self._base_url = settings.dhl_base_url.rstrip("/")
        self._api_key = settings.dhl_api_key
        self._api_secret = settings.dhl_api_secret
        self._account_number = settings.dhl_account_number
        logger.info("DHLProvider initialized (base_url=%s)", self._base_url)

    @property
    def name(self) -> str:
        return "dhl"

    async def health_check(self) -> bool:
        """Check if DHL API is reachable with a lightweight rates request."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self._base_url}/mydhlapi/rates",
                    params={
                        "accountNumber": self._account_number or "000000000",
                        "originCountryCode": "US",
                        "originPostalCode": "10001",
                        "destinationCountryCode": "GB",
                        "destinationPostalCode": "SW1A1AA",
                        "weight": "1",
                        "length": "10",
                        "width": "10",
                        "height": "10",
                        "plannedShippingDate": date.today().isoformat(),
                        "isCustomsDeclarable": "false",
                        "unitOfMeasurement": "imperial",
                    },
                    auth=(self._api_key, self._api_secret),
                )
            return resp.status_code in (200, 400)  # 400 = reachable but bad params
        except Exception:
            return False

    async def validate_address(self, address: AddressInput) -> ProviderResult:
        """Local-only address validation.

        DHL Express does not provide a public address validation API.
        This performs basic format checks only. This is documented as a
        limitation — not a real DHL API call.
        """
        issues: list[str] = []

        if not address.street.strip():
            issues.append("Street is required")
        if not address.city.strip():
            issues.append("City is required")
        if not address.state.strip():
            issues.append("State is required")
        if address.country == "US" and not _ZIP_PATTERN.match(address.zip_code):
            issues.append(f"Invalid US zip code format: {address.zip_code}")
        elif not address.zip_code.strip():
            issues.append("Postal code is required")

        if issues:
            return ProviderResult(
                success=False,
                data={"is_valid": False, "issues": issues},
                provider=self.name,
                error="; ".join(issues),
            )

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
                "address_type": "unknown",
                "_note": "Local format check only — DHL does not offer address validation API",
            },
            provider=self.name,
        )

    async def get_quote_preview(self, shipment: QuotePreviewInput) -> ProviderResult:
        """Get rate estimates via DHL Express MyDHL API — Rates endpoint.

        Endpoint: GET /mydhlapi/rates
        Uses query parameters for a lightweight rate request.
        """
        try:
            dim_weight = (
                shipment.length_in * shipment.width_in * shipment.height_in
            ) / 139
            billable_weight = max(shipment.weight_lbs, dim_weight)

            params = {
                "accountNumber": self._account_number,
                "originCountryCode": "US",
                "originPostalCode": shipment.origin_zip,
                "destinationCountryCode": "US",
                "destinationPostalCode": shipment.destination_zip,
                "weight": str(round(shipment.weight_lbs, 1)),
                "length": str(round(shipment.length_in)),
                "width": str(round(shipment.width_in)),
                "height": str(round(shipment.height_in)),
                "plannedShippingDate": date.today().isoformat(),
                "isCustomsDeclarable": "false",
                "unitOfMeasurement": "imperial",
            }

            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(
                    f"{self._base_url}/mydhlapi/rates",
                    params=params,
                    auth=(self._api_key, self._api_secret),
                )

            if resp.status_code != 200:
                error_body = resp.text[:300]
                return ProviderResult(
                    success=False,
                    data={"services": []},
                    provider=self.name,
                    error=f"DHL Rates API error: HTTP {resp.status_code} — {error_body}",
                )

            body = resp.json()
            products = body.get("products", [])

            services = []
            for product in products:
                product_code = product.get("productCode", "")
                product_name_raw = product.get("productName", "")
                service_name, est_days = _DHL_PRODUCTS.get(
                    product_code, (product_name_raw or f"DHL {product_code}", 5)
                )

                # Extract total price
                total_price = 0.0
                for price_entry in product.get("totalPrice", []):
                    if price_entry.get("currencyType") == "PULCL":
                        total_price = float(price_entry.get("price", 0))
                        break
                if total_price == 0.0:
                    price_list = product.get("totalPrice", [])
                    if price_list:
                        total_price = float(price_list[0].get("price", 0))

                # Extract delivery estimate
                delivery = product.get("deliveryCapabilities", {})
                est_transit = delivery.get("totalTransitDays")
                if est_transit:
                    est_days = int(est_transit)

                services.append({
                    "service": service_name,
                    "carrier": "DHL",
                    "price_usd": round(total_price, 2),
                    "estimated_days": est_days,
                    "service_code": product_code,
                })

            services.sort(key=lambda s: s["price_usd"])

            return ProviderResult(
                success=True,
                data={
                    "billable_weight_lbs": round(billable_weight, 2),
                    "dim_weight_lbs": round(dim_weight, 2),
                    "actual_weight_lbs": shipment.weight_lbs,
                    "services": services,
                    "disclaimer": "DHL Express rate estimates — not a binding quote. "
                    "DHL Express is primarily international; domestic US "
                    "rates may be limited. Final rates from Spring Boot /api/v1/quotes.",
                },
                provider=self.name,
            )

        except httpx.HTTPError as exc:
            logger.error("DHL rating network error: %s", exc)
            return ProviderResult(
                success=False,
                data={"services": []},
                provider=self.name,
                error=f"Network error: {exc}",
            )
