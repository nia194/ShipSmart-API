"""Tests for individual tool execution."""

import pytest

from app.providers.mock_provider import MockShippingProvider
from app.tools.address_tools import ValidateAddressTool
from app.tools.base import ToolInput
from app.tools.quote_tools import GetQuotePreviewTool


@pytest.fixture
def provider():
    return MockShippingProvider()


@pytest.mark.asyncio
async def test_validate_address_valid(provider):
    tool = ValidateAddressTool(provider)
    result = await tool.execute(ToolInput(params={
        "street": "123 main st",
        "city": "los angeles",
        "state": "ca",
        "zip_code": "90210",
    }))
    assert result.success is True
    assert result.data["is_valid"] is True
    addr = result.data["normalized_address"]
    assert addr["city"] == "Los Angeles"
    assert addr["state"] == "CA"
    assert result.metadata["provider"] == "mock"


@pytest.mark.asyncio
async def test_validate_address_invalid_zip(provider):
    tool = ValidateAddressTool(provider)
    result = await tool.execute(ToolInput(params={
        "street": "123 main st",
        "city": "los angeles",
        "state": "ca",
        "zip_code": "BADZIP",
    }))
    assert result.success is False
    assert result.data["is_valid"] is False
    assert "Invalid zip code" in result.error


@pytest.mark.asyncio
async def test_validate_address_missing_street(provider):
    tool = ValidateAddressTool(provider)
    result = await tool.execute(ToolInput(params={
        "street": "",
        "city": "LA",
        "state": "CA",
        "zip_code": "90210",
    }))
    assert result.success is False
    assert "Street is required" in result.error


@pytest.mark.asyncio
async def test_validate_address_input_validation(provider):
    tool = ValidateAddressTool(provider)
    errors = tool.validate_input({})
    assert len(errors) >= 4  # street, city, state, zip_code all required


@pytest.mark.asyncio
async def test_quote_preview_basic(provider):
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
    services = result.data["services"]
    assert len(services) == 3
    assert all("price_usd" in s for s in services)
    assert all("estimated_days" in s for s in services)
    assert result.data["disclaimer"]


@pytest.mark.asyncio
async def test_quote_preview_dim_weight(provider):
    """Heavy dimensional package should have dim_weight > actual_weight."""
    tool = GetQuotePreviewTool(provider)
    result = await tool.execute(ToolInput(params={
        "origin_zip": "90210",
        "destination_zip": "10001",
        "weight_lbs": 1.0,  # light
        "length_in": 24.0,  # but large box
        "width_in": 24.0,
        "height_in": 24.0,
    }))
    assert result.success is True
    assert result.data["dim_weight_lbs"] > result.data["actual_weight_lbs"]


@pytest.mark.asyncio
async def test_quote_preview_input_validation(provider):
    tool = GetQuotePreviewTool(provider)
    errors = tool.validate_input({"origin_zip": "90210"})
    assert len(errors) >= 5  # missing destination_zip, weight, l, w, h
