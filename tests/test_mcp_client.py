"""Tests for the remote MCP client + tool-registry shim (app/services/mcp_client.py).

These lock the HTTP contract the API depends on (ShipSmart-MCP's /tools/list +
/tools/call) and the graceful-degradation behavior when the MCP server is down —
all over an ``httpx.MockTransport``, so no live MCP service is required. The
canned transport mirrors the real wire shapes (see tests/conftest.py).
"""

from __future__ import annotations

import httpx

from app.services.mcp_client import (
    McpClient,
    ToolInput,
    _params_from_input_schema,
    create_remote_registry,
)
from tests.conftest import build_mcp_mock_transport


async def _registry(transport: httpx.MockTransport):
    return await create_remote_registry(base_url="http://mcp.test", transport=transport)


# ── Hydration / registry shape ───────────────────────────────────────────────


async def test_registry_hydrates_both_tools_with_schemas():
    registry = await _registry(build_mcp_mock_transport())
    try:
        assert registry.count() == 2
        assert {t.name for t in registry.list_tools()} == {
            "validate_address",
            "get_quote_preview",
        }
        # list_schemas exposes the old ToolRegistry shape consumers rely on.
        for schema in registry.list_schemas():
            assert {"name", "description", "parameters"} <= set(schema)
    finally:
        await registry.aclose()


async def test_remote_tool_execute_parses_data_and_metadata():
    registry = await _registry(build_mcp_mock_transport())
    try:
        tool = registry.get("get_quote_preview")
        out = await tool.execute(ToolInput(params={
            "origin_zip": "90210", "destination_zip": "10001",
            "weight_lbs": 5, "length_in": 12, "width_in": 8, "height_in": 6,
        }))
        assert out.success is True
        assert isinstance(out.data.get("services"), list) and out.data["services"]
        # metadata is reconstructed from the trailing "Metadata:" content block,
        # and the shim stamps transport=mcp.
        assert out.metadata["provider"] == "mock"
        assert out.metadata["transport"] == "mcp"
    finally:
        await registry.aclose()


async def test_remote_tool_validate_input_flags_missing_required():
    registry = await _registry(build_mcp_mock_transport())
    try:
        tool = registry.get("validate_address")
        errors = tool.validate_input({"street": "1 Main St"})  # city/state/zip missing
        assert any("city" in e for e in errors)
        assert len(errors) >= 3
    finally:
        await registry.aclose()


# ── Graceful degradation ─────────────────────────────────────────────────────


async def test_call_tool_unknown_tool_returns_failure_envelope():
    # MCP returns 404 for an unknown tool name; the client maps it to a
    # structured failure rather than raising.
    transport = build_mcp_mock_transport()
    client = McpClient(base_url="http://mcp.test", transport=transport)
    try:
        body = await client.call_tool("does_not_exist", {})
        assert body["success"] is False
        assert "does_not_exist" in body["error"]
        assert body["content"] == []
    finally:
        await client.aclose()


async def test_remote_tool_execute_survives_mcp_down():
    """If the MCP transport raises (server down), execute() returns a failed
    ToolOutput with an explanatory error — never an exception — so the advisor
    can still answer."""
    registry = await _registry(build_mcp_mock_transport())

    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    # Swap the underlying transport for one that always fails.
    tool = registry.get("validate_address")
    tool._client._client = httpx.AsyncClient(
        base_url="http://mcp.test", transport=httpx.MockTransport(dead)
    )
    try:
        out = await tool.execute(ToolInput(params={
            "street": "1 Main St", "city": "LA", "state": "CA", "zip_code": "90001",
        }))
        assert out.success is False
        assert "MCP server error" in out.error
    finally:
        await tool._client._client.aclose()


# ── Schema → parameter conversion ────────────────────────────────────────────


def test_params_from_input_schema_marks_required_and_types():
    params = _params_from_input_schema({
        "type": "object",
        "properties": {
            "zip_code": {"type": "string", "description": "ZIP"},
            "weight_lbs": {"type": "number", "description": "Weight"},
        },
        "required": ["zip_code"],
    })
    by_name = {p.name: p for p in params}
    assert by_name["zip_code"].required is True
    assert by_name["zip_code"].type == "string"
    assert by_name["weight_lbs"].required is False
    assert by_name["weight_lbs"].type == "number"


def test_mcp_client_rejects_empty_base_url():
    import pytest

    with pytest.raises(ValueError, match="non-empty base_url"):
        McpClient(base_url="")
