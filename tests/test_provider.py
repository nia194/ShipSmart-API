"""Tests for provider abstraction behavior."""

import pytest

from app.providers.mock_provider import MockShippingProvider
from app.providers.shipping_provider import AddressInput, QuotePreviewInput


@pytest.fixture
def provider():
    return MockShippingProvider()


@pytest.mark.asyncio
async def test_provider_name(provider):
    assert provider.name == "mock"


@pytest.mark.asyncio
async def test_provider_health_check(provider):
    assert await provider.health_check() is True


@pytest.mark.asyncio
async def test_validate_valid_address(provider):
    result = await provider.validate_address(AddressInput(
        street="456 Oak Ave",
        city="San Francisco",
        state="CA",
        zip_code="94102",
    ))
    assert result.success is True
    assert result.data["is_valid"] is True
    assert result.data["deliverable"] is True
    assert result.provider == "mock"


@pytest.mark.asyncio
async def test_validate_invalid_address(provider):
    result = await provider.validate_address(AddressInput(
        street="",
        city="",
        state="",
        zip_code="not-a-zip",
    ))
    assert result.success is False
    assert result.data["is_valid"] is False
    assert len(result.data["issues"]) >= 3


@pytest.mark.asyncio
async def test_quote_preview(provider):
    result = await provider.get_quote_preview(QuotePreviewInput(
        origin_zip="90210",
        destination_zip="10001",
        weight_lbs=10.0,
        length_in=12.0,
        width_in=10.0,
        height_in=8.0,
    ))
    assert result.success is True
    assert len(result.data["services"]) == 3
    # Services should be ordered cheapest to most expensive
    prices = [s["price_usd"] for s in result.data["services"]]
    assert prices == sorted(prices)
