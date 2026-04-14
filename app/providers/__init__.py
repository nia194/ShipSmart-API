"""
Provider abstractions and factory for external service integrations.

Usage:
    from app.providers import create_shipping_provider
    provider = create_shipping_provider()  # reads SHIPPING_PROVIDER from config

Behavior:
- SHIPPING_PROVIDER unset / "mock": returns MockShippingProvider with a
  prominent WARNING so operators know they are seeing fake data.
- SHIPPING_PROVIDER set to a real carrier (ups/fedex/dhl/usps): all required
  credentials must be present in env. If any are missing this raises
  ValueError at startup so misconfiguration is loud and immediate, not
  silently masked at request time.
"""

from __future__ import annotations

import logging

from app.core.config import settings
from app.providers.shipping_provider import ShippingProvider

logger = logging.getLogger(__name__)

# Provider name → module path, class name
_PROVIDER_REGISTRY: dict[str, tuple[str, str]] = {
    "mock": ("app.providers.mock_provider", "MockShippingProvider"),
    "ups": ("app.providers.ups_provider", "UPSProvider"),
    "fedex": ("app.providers.fedex_provider", "FedExProvider"),
    "dhl": ("app.providers.dhl_provider", "DHLProvider"),
    "usps": ("app.providers.usps_provider", "USPSProvider"),
}

# Per-provider list of (env_var_name, value) tuples that MUST be non-empty.
def _required_credentials(provider_name: str) -> list[tuple[str, str]]:
    if provider_name == "ups":
        return [
            ("UPS_CLIENT_ID", settings.ups_client_id),
            ("UPS_CLIENT_SECRET", settings.ups_client_secret),
            ("UPS_ACCOUNT_NUMBER", settings.ups_account_number),
        ]
    if provider_name == "fedex":
        return [
            ("FEDEX_CLIENT_ID", settings.fedex_client_id),
            ("FEDEX_CLIENT_SECRET", settings.fedex_client_secret),
            ("FEDEX_ACCOUNT_NUMBER", settings.fedex_account_number),
        ]
    if provider_name == "dhl":
        return [
            ("DHL_API_KEY", settings.dhl_api_key),
            ("DHL_API_SECRET", settings.dhl_api_secret),
        ]
    if provider_name == "usps":
        return [
            ("USPS_CLIENT_ID", settings.usps_client_id),
            ("USPS_CLIENT_SECRET", settings.usps_client_secret),
        ]
    return []


def _has_required_credentials(provider_name: str) -> bool:
    """True if every required credential for the provider is non-empty."""
    return all(value.strip() for _, value in _required_credentials(provider_name))


def _build_mock() -> ShippingProvider:
    from app.providers.mock_provider import MockShippingProvider
    return MockShippingProvider()


def create_shipping_provider() -> ShippingProvider:
    """Factory: create the configured shipping provider.

    Raises:
        ValueError: if a real carrier is selected but required credentials
        are missing. This is intentionally loud — silent fallback to mock
        masks misconfiguration.
    """
    provider_name = (settings.shipping_provider or "").lower().strip()

    if provider_name in ("", "mock"):
        logger.warning(
            "SHIPPING_PROVIDER=mock — using MockShippingProvider. "
            "All quote previews and address validations return FAKE data. "
            "Set SHIPPING_PROVIDER={ups,fedex,dhl,usps} with credentials for real carrier integration."
        )
        return _build_mock()

    if provider_name not in _PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown SHIPPING_PROVIDER={provider_name!r}. "
            f"Valid options: mock, ups, fedex, dhl, usps."
        )

    # Validate required credentials BEFORE attempting to instantiate.
    missing = [name for name, value in _required_credentials(provider_name) if not value.strip()]
    if missing:
        raise ValueError(
            f"SHIPPING_PROVIDER={provider_name} requires the following env vars "
            f"to be set: {', '.join(missing)}. "
            f"Either provide credentials or set SHIPPING_PROVIDER=mock."
        )

    module_path, class_name = _PROVIDER_REGISTRY[provider_name]
    try:
        import importlib
        module = importlib.import_module(module_path)
        provider_class = getattr(module, class_name)
        provider = provider_class()
    except Exception as exc:
        raise ValueError(
            f"Failed to instantiate provider {provider_name!r}: {exc}. "
            f"Note: real carrier providers are currently stubs — see "
            f"docs/provider-setup-ups-fedex-dhl-usps.md."
        ) from exc

    logger.info("Created shipping provider: %s", provider_name)
    return provider
