"""
USPS shipping provider.
Integrates with the NEW USPS API platform (developer.usps.com).

API docs:
  - Addresses: https://developer.usps.com/apis/addresses
  - Domestic Prices: https://developer.usps.com/apis/domestic-prices
  - Auth: OAuth2 client_credentials

IMPORTANT: This uses the NEW USPS APIs, NOT the deprecated Web Tools.

Requires:
  USPS_CLIENT_ID, USPS_CLIENT_SECRET
"""

from __future__ import annotations

import logging
import time

import httpx

from app.core.config import settings
from app.providers.base import ProviderResult
from app.providers.shipping_provider import (
    AddressInput,
    QuotePreviewInput,
    ShippingProvider,
)

logger = logging.getLogger(__name__)

# USPS mail class → human-readable name and typical transit days
_USPS_SERVICES: dict[str, tuple[str, int]] = {
    "PRIORITY_MAIL_EXPRESS": ("USPS Priority Mail Express", 2),
    "PRIORITY_MAIL": ("USPS Priority Mail", 3),
    "USPS_GROUND_ADVANTAGE": ("USPS Ground Advantage", 5),
    "FIRST_CLASS_MAIL": ("USPS First-Class Mail", 3),
    "PARCEL_SELECT": ("USPS Parcel Select", 7),
    "LIBRARY_MAIL": ("USPS Library Mail", 9),
    "MEDIA_MAIL": ("USPS Media Mail", 8),
    "USPS_RETAIL_GROUND": ("USPS Retail Ground", 7),
}


class USPSProvider(ShippingProvider):
    """USPS new API platform provider.

    Uses OAuth2 client_credentials flow for authentication.
    Calls the Addresses API for validation and the Domestic Prices API
    for rate estimates.
    """

    def __init__(self) -> None:
        self._base_url = settings.usps_base_url.rstrip("/")
        self._client_id = settings.usps_client_id
        self._client_secret = settings.usps_client_secret
        self._access_token: str = ""
        self._token_expires_at: float = 0.0
        logger.info("USPSProvider initialized (base_url=%s)", self._base_url)

    @property
    def name(self) -> str:
        return "usps"

    async def health_check(self) -> bool:
        try:
            await self._ensure_token()
            return bool(self._access_token)
        except Exception:
            return False

    async def validate_address(self, address: AddressInput) -> ProviderResult:
        """Validate address via USPS Addresses API v3.

        Endpoint: GET /addresses/v3/address
        """
        try:
            await self._ensure_token()

            params = {
                "streetAddress": address.street,
                "city": address.city,
                "state": address.state,
                "ZIPCode": address.zip_code,
            }

            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{self._base_url}/addresses/v3/address",
                    params=params,
                    headers=self._auth_headers(),
                )

            if resp.status_code != 200:
                return ProviderResult(
                    success=False,
                    data={"is_valid": False, "issues": [f"USPS API returned {resp.status_code}"]},
                    provider=self.name,
                    error=f"USPS Addresses API error: HTTP {resp.status_code}",
                )

            body = resp.json()
            addr = body.get("address", {})

            # USPS returns the standardized address if valid
            if addr.get("streetAddress"):
                normalized = {
                    "street": addr.get("streetAddress", address.street),
                    "city": addr.get("city", address.city),
                    "state": addr.get("state", address.state),
                    "zip_code": addr.get("ZIPCode", address.zip_code),
                    "country": "US",
                }

                # Check for additional info
                delivery_point = addr.get("deliveryPoint", "")
                addr_type = "residential"  # USPS primarily delivers residential
                if body.get("business"):
                    addr_type = "commercial"

                zip_plus4 = addr.get("ZIPPlus4", "")
                if zip_plus4:
                    normalized["zip_code"] = f"{normalized['zip_code']}-{zip_plus4}"

                return ProviderResult(
                    success=True,
                    data={
                        "is_valid": True,
                        "normalized_address": normalized,
                        "deliverable": True,
                        "address_type": addr_type,
                        "delivery_point": delivery_point,
                    },
                    provider=self.name,
                )

            # Address not found or invalid
            corrections = body.get("addressCorrections", [])
            issues = [c.get("description", "Unknown issue") for c in corrections]
            if not issues:
                issues = ["Address could not be validated by USPS"]

            return ProviderResult(
                success=False,
                data={"is_valid": False, "issues": issues},
                provider=self.name,
                error="; ".join(issues),
            )

        except httpx.HTTPError as exc:
            logger.error("USPS address validation network error: %s", exc)
            return ProviderResult(
                success=False,
                data={"is_valid": False, "issues": ["USPS API unreachable"]},
                provider=self.name,
                error=f"Network error: {exc}",
            )

    async def get_quote_preview(self, shipment: QuotePreviewInput) -> ProviderResult:
        """Get domestic rate estimates via USPS Domestic Prices API v3.

        Endpoint: GET /prices/v3/base-rates/search
        """
        try:
            await self._ensure_token()

            dim_weight = (
                shipment.length_in * shipment.width_in * shipment.height_in
            ) / 166  # USPS uses 166 DIM factor
            billable_weight = max(shipment.weight_lbs, dim_weight)

            params = {
                "originZIPCode": shipment.origin_zip,
                "destinationZIPCode": shipment.destination_zip,
                "weight": str(round(shipment.weight_lbs, 2)),
                "length": str(round(shipment.length_in, 1)),
                "width": str(round(shipment.width_in, 1)),
                "height": str(round(shipment.height_in, 1)),
                "mailClass": "ALL",
                "processingCategory": "MACHINABLE",
                "rateIndicator": "DR",  # Dimensional Rectangular
                "destinationEntryFacilityType": "NONE",
                "priceType": "RETAIL",
            }

            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(
                    f"{self._base_url}/prices/v3/base-rates/search",
                    params=params,
                    headers=self._auth_headers(),
                )

            if resp.status_code != 200:
                error_body = resp.text[:300]
                return ProviderResult(
                    success=False,
                    data={"services": []},
                    provider=self.name,
                    error=f"USPS Prices API error: HTTP {resp.status_code} — {error_body}",
                )

            body = resp.json()
            rates = body.get("rates", [])

            services = []
            seen_classes: set[str] = set()
            for rate in rates:
                mail_class = rate.get("mailClass", "")
                # Deduplicate by mail class (take first/cheapest variant)
                if mail_class in seen_classes:
                    continue
                seen_classes.add(mail_class)

                service_name, est_days = _USPS_SERVICES.get(
                    mail_class, (f"USPS {mail_class}", 5)
                )
                price = float(rate.get("price", 0))

                # Use commitment if available
                commitment = rate.get("commitment", {})
                sched_days = commitment.get("scheduledDeliveryDate")
                if not sched_days:
                    name_val = commitment.get("name", "")
                    if name_val and name_val.replace(" ", "").replace("Day", "").isdigit():
                        est_days = int(name_val.replace(" ", "").replace("Day", "")[0])

                services.append({
                    "service": service_name,
                    "carrier": "USPS",
                    "price_usd": round(price, 2),
                    "estimated_days": est_days,
                    "service_code": mail_class,
                })

            services.sort(key=lambda s: s["price_usd"])

            return ProviderResult(
                success=True,
                data={
                    "billable_weight_lbs": round(billable_weight, 2),
                    "dim_weight_lbs": round(dim_weight, 2),
                    "actual_weight_lbs": shipment.weight_lbs,
                    "services": services,
                    "disclaimer": "USPS rate estimates — not a binding quote. "
                    "USPS max weight is 70 lbs. "
                    "Final rates from Spring Boot /api/v1/quotes.",
                },
                provider=self.name,
            )

        except httpx.HTTPError as exc:
            logger.error("USPS pricing network error: %s", exc)
            return ProviderResult(
                success=False,
                data={"services": []},
                provider=self.name,
                error=f"Network error: {exc}",
            )

    # ── OAuth2 token management ─────────────────────────────────────────────

    async def _ensure_token(self) -> None:
        """Acquire or refresh the OAuth2 access token if expired."""
        if self._access_token and time.time() < self._token_expires_at - 60:
            return

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self._base_url}/oauth2/v3/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        if resp.status_code != 200:
            self._access_token = ""
            raise RuntimeError(f"USPS OAuth2 token request failed: HTTP {resp.status_code}")

        token_data = resp.json()
        self._access_token = token_data.get("access_token", "")
        expires_in = int(token_data.get("expires_in", 3600))
        self._token_expires_at = time.time() + expires_in
        logger.info("USPS OAuth2 token acquired (expires_in=%ds)", expires_in)

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }
