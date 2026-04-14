"""Tests for tool registry."""

import pytest

from app.providers.mock_provider import MockShippingProvider
from app.tools.address_tools import ValidateAddressTool
from app.tools.quote_tools import GetQuotePreviewTool
from app.tools.registry import ToolRegistry


@pytest.fixture
def registry():
    provider = MockShippingProvider()
    reg = ToolRegistry()
    reg.register(ValidateAddressTool(provider))
    reg.register(GetQuotePreviewTool(provider))
    return reg


def test_register_and_count(registry):
    assert registry.count() == 2


def test_get_by_name(registry):
    tool = registry.get("validate_address")
    assert tool is not None
    assert tool.name == "validate_address"


def test_get_unknown_returns_none(registry):
    assert registry.get("nonexistent_tool") is None


def test_list_tools_sorted(registry):
    tools = registry.list_tools()
    names = [t.name for t in tools]
    assert names == sorted(names)


def test_list_schemas(registry):
    schemas = registry.list_schemas()
    assert len(schemas) == 2
    for schema in schemas:
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema


def test_duplicate_registration_raises():
    provider = MockShippingProvider()
    reg = ToolRegistry()
    reg.register(ValidateAddressTool(provider))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(ValidateAddressTool(provider))


def test_tool_schema_has_required_fields(registry):
    tool = registry.get("validate_address")
    schema = tool.schema()
    assert schema["name"] == "validate_address"
    assert len(schema["parameters"]) >= 4
    param_names = [p["name"] for p in schema["parameters"]]
    assert "street" in param_names
    assert "city" in param_names
