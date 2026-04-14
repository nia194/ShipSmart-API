"""Tests for provider factory, selection, and fallback behavior."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.providers import _has_required_credentials, create_shipping_provider
from app.providers.mock_provider import MockShippingProvider
from app.providers.shipping_provider import AddressInput
from app.tools.address_tools import ValidateAddressTool
from app.tools.base import ToolInput
from app.tools.quote_tools import GetQuotePreviewTool

# ── Factory selection tests ──────────────────────────────────────────────────


def test_factory_returns_mock_by_default():
    """Default SHIPPING_PROVIDER=mock returns MockShippingProvider."""
    with patch("app.providers.settings") as mock_settings:
        mock_settings.shipping_provider = "mock"
        provider = create_shipping_provider()
        assert isinstance(provider, MockShippingProvider)
        assert provider.name == "mock"


def test_factory_returns_mock_for_empty_string():
    """Empty SHIPPING_PROVIDER falls back to mock."""
    with patch("app.providers.settings") as mock_settings:
        mock_settings.shipping_provider = ""
        provider = create_shipping_provider()
        assert isinstance(provider, MockShippingProvider)


def test_factory_raises_for_unknown_provider():
    with patch("app.providers.settings") as mock_settings:
        mock_settings.shipping_provider = "carrier_pigeon"
        with pytest.raises(ValueError, match="Unknown SHIPPING_PROVIDER"):
            create_shipping_provider()


def test_factory_raises_when_ups_credentials_missing():
    with patch("app.providers.settings") as mock_settings:
        mock_settings.shipping_provider = "ups"
        mock_settings.ups_client_id = ""
        mock_settings.ups_client_secret = ""
        mock_settings.ups_account_number = ""
        mock_settings.ups_base_url = "https://onlinetools.ups.com"
        with pytest.raises(ValueError, match="UPS_CLIENT_ID"):
            create_shipping_provider()


def test_factory_raises_when_fedex_credentials_missing():
    with patch("app.providers.settings") as mock_settings:
        mock_settings.shipping_provider = "fedex"
        mock_settings.fedex_client_id = ""
        mock_settings.fedex_client_secret = ""
        mock_settings.fedex_account_number = ""
        mock_settings.fedex_base_url = "https://apis.fedex.com"
        with pytest.raises(ValueError, match="FEDEX_CLIENT_ID"):
            create_shipping_provider()


def test_factory_raises_when_dhl_credentials_missing():
    with patch("app.providers.settings") as mock_settings:
        mock_settings.shipping_provider = "dhl"
        mock_settings.dhl_api_key = ""
        mock_settings.dhl_api_secret = ""
        mock_settings.dhl_account_number = ""
        mock_settings.dhl_base_url = "https://express.api.dhl.com"
        with pytest.raises(ValueError, match="DHL_API_KEY"):
            create_shipping_provider()


def test_factory_raises_when_usps_credentials_missing():
    with patch("app.providers.settings") as mock_settings:
        mock_settings.shipping_provider = "usps"
        mock_settings.usps_client_id = ""
        mock_settings.usps_client_secret = ""
        mock_settings.usps_base_url = "https://api.usps.com"
        with pytest.raises(ValueError, match="USPS_CLIENT_ID"):
            create_shipping_provider()


# ── Credential check tests ──────────────────────────────────────────────────


def test_has_required_credentials_ups_present():
    with patch("app.providers.settings") as mock_settings:
        mock_settings.ups_client_id = "test-id"
        mock_settings.ups_client_secret = "test-secret"
        assert _has_required_credentials("ups") is True


def test_has_required_credentials_ups_missing():
    with patch("app.providers.settings") as mock_settings:
        mock_settings.ups_client_id = "test-id"
        mock_settings.ups_client_secret = ""
        assert _has_required_credentials("ups") is False


def test_has_required_credentials_fedex_present():
    with patch("app.providers.settings") as mock_settings:
        mock_settings.fedex_client_id = "test-id"
        mock_settings.fedex_client_secret = "test-secret"
        assert _has_required_credentials("fedex") is True


def test_has_required_credentials_dhl_present():
    with patch("app.providers.settings") as mock_settings:
        mock_settings.dhl_api_key = "test-key"
        mock_settings.dhl_api_secret = "test-secret"
        assert _has_required_credentials("dhl") is True


def test_has_required_credentials_usps_present():
    with patch("app.providers.settings") as mock_settings:
        mock_settings.usps_client_id = "test-id"
        mock_settings.usps_client_secret = "test-secret"
        assert _has_required_credentials("usps") is True


def test_has_required_credentials_unknown_provider():
    """Unknown provider name returns True (no checks to fail)."""
    assert _has_required_credentials("unknown") is True


# ── Provider name and interface tests ────────────────────────────────────────


def test_ups_provider_name():
    from app.providers.ups_provider import UPSProvider
    with patch("app.providers.ups_provider.settings") as mock_settings:
        mock_settings.ups_base_url = "https://onlinetools.ups.com"
        mock_settings.ups_client_id = "test"
        mock_settings.ups_client_secret = "test"
        mock_settings.ups_account_number = "123"
        p = UPSProvider()
        assert p.name == "ups"


def test_fedex_provider_name():
    from app.providers.fedex_provider import FedExProvider
    with patch("app.providers.fedex_provider.settings") as mock_settings:
        mock_settings.fedex_base_url = "https://apis.fedex.com"
        mock_settings.fedex_client_id = "test"
        mock_settings.fedex_client_secret = "test"
        mock_settings.fedex_account_number = "123"
        p = FedExProvider()
        assert p.name == "fedex"


def test_dhl_provider_name():
    from app.providers.dhl_provider import DHLProvider
    with patch("app.providers.dhl_provider.settings") as mock_settings:
        mock_settings.dhl_base_url = "https://express.api.dhl.com"
        mock_settings.dhl_api_key = "test"
        mock_settings.dhl_api_secret = "test"
        mock_settings.dhl_account_number = "123"
        p = DHLProvider()
        assert p.name == "dhl"


def test_usps_provider_name():
    from app.providers.usps_provider import USPSProvider
    with patch("app.providers.usps_provider.settings") as mock_settings:
        mock_settings.usps_base_url = "https://api.usps.com"
        mock_settings.usps_client_id = "test"
        mock_settings.usps_client_secret = "test"
        p = USPSProvider()
        assert p.name == "usps"


# ── DHL local address validation (no API call) ──────────────────────────────


@pytest.mark.asyncio
async def test_dhl_local_address_validation_valid():
    """DHL address validation is local-only — should work without API."""
    from app.providers.dhl_provider import DHLProvider
    with patch("app.providers.dhl_provider.settings") as mock_settings:
        mock_settings.dhl_base_url = "https://express.api.dhl.com"
        mock_settings.dhl_api_key = "test"
        mock_settings.dhl_api_secret = "test"
        mock_settings.dhl_account_number = "123"
        p = DHLProvider()
        result = await p.validate_address(AddressInput(
            street="456 Oak Ave",
            city="San Francisco",
            state="CA",
            zip_code="94102",
        ))
        assert result.success is True
        assert result.data["is_valid"] is True
        assert result.provider == "dhl"
        assert "_note" in result.data  # documents that it's local-only


@pytest.mark.asyncio
async def test_dhl_local_address_validation_invalid():
    from app.providers.dhl_provider import DHLProvider
    with patch("app.providers.dhl_provider.settings") as mock_settings:
        mock_settings.dhl_base_url = "https://express.api.dhl.com"
        mock_settings.dhl_api_key = "test"
        mock_settings.dhl_api_secret = "test"
        mock_settings.dhl_account_number = "123"
        p = DHLProvider()
        result = await p.validate_address(AddressInput(
            street="",
            city="",
            state="",
            zip_code="BADZIP",
        ))
        assert result.success is False
        assert result.data["is_valid"] is False
        assert len(result.data["issues"]) >= 3


# ── Tool compatibility with mock provider ────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_validate_address_via_factory_mock():
    """Tools work correctly with mock provider from factory."""
    provider = MockShippingProvider()
    tool = ValidateAddressTool(provider)
    result = await tool.execute(ToolInput(params={
        "street": "123 Main St",
        "city": "Los Angeles",
        "state": "CA",
        "zip_code": "90001",
    }))
    assert result.success is True
    assert result.data["is_valid"] is True
    assert result.metadata["provider"] == "mock"
    assert result.metadata["tool"] == "validate_address"


@pytest.mark.asyncio
async def test_tool_quote_preview_via_factory_mock():
    """Tools work correctly with mock provider from factory."""
    provider = MockShippingProvider()
    tool = GetQuotePreviewTool(provider)
    result = await tool.execute(ToolInput(params={
        "origin_zip": "90210",
        "destination_zip": "10001",
        "weight_lbs": 5.0,
        "length_in": 12.0,
        "width_in": 8.0,
        "height_in": 6.0,
    }))
    assert result.success is True
    assert len(result.data["services"]) == 3
    assert result.metadata["provider"] == "mock"
    assert result.metadata["tool"] == "get_quote_preview"


# ── Provider case insensitivity ──────────────────────────────────────────────


def test_factory_case_insensitive():
    """Provider name should be case-insensitive."""
    with patch("app.providers.settings") as mock_settings:
        mock_settings.shipping_provider = "MOCK"
        provider = create_shipping_provider()
        assert isinstance(provider, MockShippingProvider)


def test_factory_strips_whitespace():
    """Provider name should be trimmed."""
    with patch("app.providers.settings") as mock_settings:
        mock_settings.shipping_provider = "  mock  "
        provider = create_shipping_provider()
        assert isinstance(provider, MockShippingProvider)
