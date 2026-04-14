"""
FedEx shipping provider.
Integrates with FedEx Developer API for address validation and rating.

API docs:
  - Rate: https://developer.fedex.com/api/en-us/catalog/rate/docs.html
  - Address Validation: https://developer.fedex.com/api/en-us/catalog/address-validation/docs.html
  - Auth: OAuth2 client_credentials (client_id + client_secret)

Requires:
  FEDEX_CLIENT_ID, FEDEX_CLIENT_SECRET, FEDEX_ACCOUNT_NUMBER
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

# FedEx service type → human-readable name and typical transit days
_FEDEX_SERVICES: dict[str, tuple[str, int]] = {
    "FEDEX_GROUND": ("FedEx Ground", 5),
    "GROUND_HOME_DELIVERY": ("FedEx Home Delivery", 5),
    "FEDEX_EXPRESS_SAVER": ("FedEx Express Saver", 3),
    "FEDEX_2_DAY": ("FedEx 2Day", 2),
    "FEDEX_2_DAY_AM": ("FedEx 2Day A.M.", 2),
    "STANDARD_OVERNIGHT": ("FedEx Standard Overnight", 1),
    "PRIORITY_OVERNIGHT": ("FedEx Priority Overnight", 1),
    "FIRST_OVERNIGHT": ("FedEx First Overnight", 1),
}


class FedExProvider(ShippingProvider):
    """FedEx Developer API provider.

    Uses OAuth2 client_credentials flow for authentication.
    Calls the Rate API v1 for quote previews and Address Validation API v1
    for address normalization.
    """

    def __init__(self) -> None:
        self._base_url = settings.fedex_base_url.rstrip("/")
        self._client_id = settings.fedex_client_id
        self._client_secret = settings.fedex_client_secret
        self._account_number = settings.fedex_account_number
        self._access_token: str = ""
        self._token_expires_at: float = 0.0
        logger.info("FedExProvider initialized (base_url=%s)", self._base_url)

    @property
    def name(self) -> str:
        return "fedex"

    async def health_check(self) -> bool:
        try:
            await self._ensure_token()
            return bool(self._access_token)
        except Exception:
            return False

    async def validate_address(self, address: AddressInput) -> ProviderResult:
        """Validate address via FedEx Address Validation API v1.

        Endpoint: POST /address/v1/addresses/resolve
        """
        try:
            await self._ensure_token()

            payload = {
                "addressesToValidate": [
                    {
                        "address": {
                            "streetLines": [address.street],
                            "city": address.city,
                            "stateOrProvinceCode": address.state,
                            "postalCode": address.zip_code,
                            "countryCode": address.country,
                        },
                    },
                ],
            }

            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{self._base_url}/address/v1/addresses/resolve",
                    json=payload,
                    headers=self._auth_headers(),
                )

            if resp.status_code != 200:
                return ProviderResult(
                    success=False,
                    data={"is_valid": False, "issues": [f"FedEx API returned {resp.status_code}"]},
                    provider=self.name,
                    error=f"FedEx Address Validation API error: HTTP {resp.status_code}",
                )

            body = resp.json()
            results = body.get("output", {}).get("resolvedAddresses", [])

            if not results:
                return ProviderResult(
                    success=False,
                    data={"is_valid": False, "issues": ["No resolved address returned"]},
                    provider=self.name,
                    error="FedEx returned no address candidates",
                )

            resolved = results[0]
            classification = resolved.get("classification", "UNKNOWN")
            is_valid = classification in ("RESOLVED", "MIXED")

            if is_valid:
                street_lines = resolved.get("streetLinesToken", [address.street])
                normalized = {
                    "street": " ".join(street_lines) if isinstance(street_lines, list)
                    else street_lines,
                    "city": resolved.get("city", address.city),
                    "state": resolved.get("stateOrProvinceCode", address.state),
                    "zip_code": resolved.get("postalCode", address.zip_code),
                    "country": resolved.get("countryCode", address.country),
                }
                addr_attrs = resolved.get("attributes", {})
                is_residential = addr_attrs.get("Residential") == "true"
                addr_type = "residential" if is_residential else "commercial"

                return ProviderResult(
                    success=True,
                    data={
                        "is_valid": True,
                        "normalized_address": normalized,
                        "deliverable": True,
                        "address_type": addr_type,
                    },
                    provider=self.name,
                )

            return ProviderResult(
                success=False,
                data={"is_valid": False, "issues": [f"Classification: {classification}"]},
                provider=self.name,
                error=f"Address not resolved (classification={classification})",
            )

        except httpx.HTTPError as exc:
            logger.error("FedEx address validation network error: %s", exc)
            return ProviderResult(
                success=False,
                data={"is_valid": False, "issues": ["FedEx API unreachable"]},
                provider=self.name,
                error=f"Network error: {exc}",
            )

    async def get_quote_preview(self, shipment: QuotePreviewInput) -> ProviderResult:
        """Get rate estimates via FedEx Rate API v1.

        Endpoint: POST /rate/v1/rates/quotes
        """
        try:
            await self._ensure_token()

            dim_weight = (
                shipment.length_in * shipment.width_in * shipment.height_in
            ) / 139
            billable_weight = max(shipment.weight_lbs, dim_weight)

            payload = {
                "accountNumber": {"value": self._account_number},
                "requestedShipment": {
                    "shipper": {
                        "address": {
                            "postalCode": shipment.origin_zip,
                            "countryCode": "US",
                        },
                    },
                    "recipient": {
                        "address": {
                            "postalCode": shipment.destination_zip,
                            "countryCode": "US",
                        },
                    },
                    "pickupType": "DROPOFF_AT_FEDEX_LOCATION",
                    "rateRequestType": ["LIST"],
                    "requestedPackageLineItems": [
                        {
                            "weight": {
                                "units": "LB",
                                "value": round(shipment.weight_lbs, 1),
                            },
                            "dimensions": {
                                "length": round(shipment.length_in),
                                "width": round(shipment.width_in),
                                "height": round(shipment.height_in),
                                "units": "IN",
                            },
                        },
                    ],
                },
            }

            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    f"{self._base_url}/rate/v1/rates/quotes",
                    json=payload,
                    headers=self._auth_headers(),
                )

            if resp.status_code != 200:
                return ProviderResult(
                    success=False,
                    data={"services": []},
                    provider=self.name,
                    error=f"FedEx Rate API error: HTTP {resp.status_code}",
                )

            body = resp.json()
            rate_details = body.get("output", {}).get("rateReplyDetails", [])

            services = []
            for rd in rate_details:
                service_type = rd.get("serviceType", "")
                service_name, est_days = _FEDEX_SERVICES.get(
                    service_type, (f"FedEx {service_type}", 5)
                )

                # Extract price from rated shipment details
                shipment_details = rd.get("ratedShipmentDetails", [{}])
                if shipment_details:
                    total_charges = shipment_details[0].get("totalNetCharge", 0)
                    price = float(total_charges) if total_charges else 0
                else:
                    price = 0

                # Use commit transit days if available
                commit = rd.get("commit", {})
                transit_days = commit.get("transitDays", {})
                if isinstance(transit_days, dict):
                    days_str = transit_days.get("description", "")
                    if days_str.isdigit():
                        est_days = int(days_str)

                services.append({
                    "service": service_name,
                    "carrier": "FedEx",
                    "price_usd": round(price, 2),
                    "estimated_days": est_days,
                    "service_code": service_type,
                })

            services.sort(key=lambda s: s["price_usd"])

            return ProviderResult(
                success=True,
                data={
                    "billable_weight_lbs": round(billable_weight, 2),
                    "dim_weight_lbs": round(dim_weight, 2),
                    "actual_weight_lbs": shipment.weight_lbs,
                    "services": services,
                    "disclaimer": "FedEx rate estimates — not a binding quote. "
                    "Final rates from Spring Boot /api/v1/quotes.",
                },
                provider=self.name,
            )

        except httpx.HTTPError as exc:
            logger.error("FedEx rating network error: %s", exc)
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
                f"{self._base_url}/oauth/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        if resp.status_code != 200:
            self._access_token = ""
            raise RuntimeError(f"FedEx OAuth2 token request failed: HTTP {resp.status_code}")

        token_data = resp.json()
        self._access_token = token_data.get("access_token", "")
        expires_in = int(token_data.get("expires_in", 3600))
        self._token_expires_at = time.time() + expires_in
        logger.info("FedEx OAuth2 token acquired (expires_in=%ds)", expires_in)

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
            "X-locale": "en_US",
        }
