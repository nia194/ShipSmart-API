"""
UPS shipping provider.
Integrates with UPS Developer API for address validation and rating.

API docs:
  - Rating: https://developer.ups.com/api/reference/rating
  - Address Validation: https://developer.ups.com/api/reference/addressvalidation
  - Auth: https://developer.ups.com/get-started (OAuth2 client_credentials)

Requires:
  UPS_CLIENT_ID, UPS_CLIENT_SECRET, UPS_ACCOUNT_NUMBER
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

# UPS service code → human-readable name and typical transit days
_UPS_SERVICES: dict[str, tuple[str, int]] = {
    "03": ("UPS Ground", 5),
    "02": ("UPS 2nd Day Air", 2),
    "01": ("UPS Next Day Air", 1),
    "12": ("UPS 3 Day Select", 3),
    "13": ("UPS Next Day Air Saver", 1),
    "14": ("UPS Next Day Air Early", 1),
    "59": ("UPS 2nd Day Air A.M.", 2),
}


class UPSProvider(ShippingProvider):
    """UPS Developer API provider.

    Uses OAuth2 client_credentials flow for authentication.
    Calls the Rating API for quote previews and the Address Validation API
    for address normalization.
    """

    def __init__(self) -> None:
        self._base_url = settings.ups_base_url.rstrip("/")
        self._client_id = settings.ups_client_id
        self._client_secret = settings.ups_client_secret
        self._account_number = settings.ups_account_number
        self._access_token: str = ""
        self._token_expires_at: float = 0.0
        logger.info("UPSProvider initialized (base_url=%s)", self._base_url)

    @property
    def name(self) -> str:
        return "ups"

    async def health_check(self) -> bool:
        """Check if UPS API is reachable by attempting token acquisition."""
        try:
            await self._ensure_token()
            return bool(self._access_token)
        except Exception:
            return False

    async def validate_address(self, address: AddressInput) -> ProviderResult:
        """Validate address via UPS Address Validation API (v1).

        Endpoint: POST /api/addressvalidation/v1/1
        The '1' path param = max candidates to return.
        """
        try:
            await self._ensure_token()

            payload = {
                "XAVRequest": {
                    "AddressKeyFormat": {
                        "AddressLine": [address.street],
                        "PoliticalDivision2": address.city,
                        "PoliticalDivision1": address.state,
                        "PostcodePrimaryLow": address.zip_code,
                        "CountryCode": address.country,
                    },
                },
            }

            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{self._base_url}/api/addressvalidation/v1/1",
                    json=payload,
                    headers=self._auth_headers(),
                )

            if resp.status_code != 200:
                return ProviderResult(
                    success=False,
                    data={"is_valid": False, "issues": [f"UPS API returned {resp.status_code}"]},
                    provider=self.name,
                    error=f"UPS Address Validation API error: HTTP {resp.status_code}",
                )

            body = resp.json()
            xav = body.get("XAVResponse", {})

            # Check for valid/ambiguous/invalid indicators
            is_valid = "ValidAddressIndicator" in xav
            is_ambiguous = "AmbiguousAddressIndicator" in xav

            if is_valid:
                candidate = xav.get("Candidate", [{}])
                if isinstance(candidate, list):
                    candidate = candidate[0] if candidate else {}
                addr_fmt = candidate.get("AddressKeyFormat", {})
                normalized = {
                    "street": " ".join(addr_fmt.get("AddressLine", [])),
                    "city": addr_fmt.get("PoliticalDivision2", address.city),
                    "state": addr_fmt.get("PoliticalDivision1", address.state),
                    "zip_code": addr_fmt.get("PostcodePrimaryLow", address.zip_code),
                    "country": addr_fmt.get("CountryCode", address.country),
                }
                return ProviderResult(
                    success=True,
                    data={
                        "is_valid": True,
                        "normalized_address": normalized,
                        "deliverable": True,
                        "address_type": addr_fmt.get("AddressClassification", {}).get(
                            "Description", "unknown"
                        ).lower(),
                    },
                    provider=self.name,
                )

            issues = []
            if is_ambiguous:
                issues.append("Address is ambiguous — multiple matches found")
            if "NoCandidatesIndicator" in xav:
                issues.append("No matching address found")

            return ProviderResult(
                success=False,
                data={"is_valid": False, "issues": issues or ["Address could not be validated"]},
                provider=self.name,
                error="; ".join(issues) if issues else "Validation failed",
            )

        except httpx.HTTPError as exc:
            logger.error("UPS address validation network error: %s", exc)
            return ProviderResult(
                success=False,
                data={"is_valid": False, "issues": ["UPS API unreachable"]},
                provider=self.name,
                error=f"Network error: {exc}",
            )

    async def get_quote_preview(self, shipment: QuotePreviewInput) -> ProviderResult:
        """Get rate estimates via UPS Rating API (v2401).

        Endpoint: POST /api/rating/v2401/Shop
        'Shop' request type returns rates for all available service levels.
        """
        try:
            await self._ensure_token()

            dim_weight = (
                shipment.length_in * shipment.width_in * shipment.height_in
            ) / 139
            billable_weight = max(shipment.weight_lbs, dim_weight)

            payload = {
                "RateRequest": {
                    "Request": {
                        "SubVersion": "2401",
                        "TransactionReference": {"CustomerContext": "ShipSmart quote preview"},
                    },
                    "Shipment": {
                        "Shipper": {
                            "Address": {
                                "PostalCode": shipment.origin_zip,
                                "CountryCode": "US",
                            },
                        },
                        "ShipTo": {
                            "Address": {
                                "PostalCode": shipment.destination_zip,
                                "CountryCode": "US",
                            },
                        },
                        "ShipFrom": {
                            "Address": {
                                "PostalCode": shipment.origin_zip,
                                "CountryCode": "US",
                            },
                        },
                        "Package": {
                            "PackagingType": {"Code": "02", "Description": "Customer Supplied"},
                            "Dimensions": {
                                "UnitOfMeasurement": {"Code": "IN"},
                                "Length": str(round(shipment.length_in)),
                                "Width": str(round(shipment.width_in)),
                                "Height": str(round(shipment.height_in)),
                            },
                            "PackageWeight": {
                                "UnitOfMeasurement": {"Code": "LBS"},
                                "Weight": str(round(shipment.weight_lbs, 1)),
                            },
                        },
                    },
                },
            }

            # Add account number if available (required for negotiated rates)
            if self._account_number:
                payload["RateRequest"]["Shipment"]["Shipper"]["ShipperNumber"] = (
                    self._account_number
                )

            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    f"{self._base_url}/api/rating/v2401/Shop",
                    json=payload,
                    headers=self._auth_headers(),
                )

            if resp.status_code != 200:
                return ProviderResult(
                    success=False,
                    data={"services": []},
                    provider=self.name,
                    error=f"UPS Rating API error: HTTP {resp.status_code}",
                )

            body = resp.json()
            rated_shipments = body.get("RateResponse", {}).get("RatedShipment", [])
            if not isinstance(rated_shipments, list):
                rated_shipments = [rated_shipments]

            services = []
            for rs in rated_shipments:
                service_code = rs.get("Service", {}).get("Code", "")
                service_name, est_days = _UPS_SERVICES.get(
                    service_code, (f"UPS Service {service_code}", 5)
                )
                total = rs.get("TotalCharges", {})
                price = float(total.get("MonetaryValue", "0"))

                # Use guaranteed days if available
                days_str = rs.get("GuaranteedDelivery", {}).get("BusinessDaysInTransit", "")
                if days_str:
                    est_days = int(days_str)

                services.append({
                    "service": service_name,
                    "carrier": "UPS",
                    "price_usd": round(price, 2),
                    "estimated_days": est_days,
                    "service_code": service_code,
                })

            services.sort(key=lambda s: s["price_usd"])

            return ProviderResult(
                success=True,
                data={
                    "billable_weight_lbs": round(billable_weight, 2),
                    "dim_weight_lbs": round(dim_weight, 2),
                    "actual_weight_lbs": shipment.weight_lbs,
                    "services": services,
                    "disclaimer": "UPS rate estimates — not a binding quote. "
                    "Final rates from Spring Boot /api/v1/quotes.",
                },
                provider=self.name,
            )

        except httpx.HTTPError as exc:
            logger.error("UPS rating network error: %s", exc)
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
                f"{self._base_url}/security/v1/oauth/token",
                data={"grant_type": "client_credentials"},
                auth=(self._client_id, self._client_secret),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        if resp.status_code != 200:
            self._access_token = ""
            raise RuntimeError(f"UPS OAuth2 token request failed: HTTP {resp.status_code}")

        token_data = resp.json()
        self._access_token = token_data.get("access_token", "")
        expires_in = int(token_data.get("expires_in", 3600))
        self._token_expires_at = time.time() + expires_in
        logger.info("UPS OAuth2 token acquired (expires_in=%ds)", expires_in)

    def _auth_headers(self) -> dict[str, str]:
        """Return authorization headers for UPS API requests."""
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
            "transId": "shipsmart-preview",
            "transactionSrc": "shipsmart",
        }
