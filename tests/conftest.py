"""
Shared test fixtures for ShipSmart-API.

After the MCP migration, the tool layer lives in the standalone ShipSmart-MCP
service. Tests that previously registered in-process `ValidateAddressTool` /
`GetQuotePreviewTool` now build a `RemoteToolRegistry` backed by an
`httpx.MockTransport`. The transport serves canned `/tools/list` and
`/tools/call` responses that mirror what the real MCP would produce for the
mock shipping provider — keeping every downstream assert identical.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest_asyncio

from app.services.mcp_client import RemoteToolRegistry, create_remote_registry

_ZIP_PATTERN = re.compile(r"^\d{5}(-\d{4})?$")


# ── Canned /tools/list response ─────────────────────────────────────────────

_TOOLS_LIST = {
    "tools": [
        {
            "name": "validate_address",
            "description": (
                "Validate a shipping address and return a normalized version. "
                "Checks for required fields and format issues."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "street": {"type": "string", "description": "Street address line"},
                    "city": {"type": "string", "description": "City name"},
                    "state": {"type": "string", "description": "State code (e.g. CA, NY)"},
                    "zip_code": {"type": "string", "description": "ZIP code (e.g. 90210)"},
                    "country": {"type": "string", "description": "Country code (default US)"},
                },
                "required": ["street", "city", "state", "zip_code"],
            },
        },
        {
            "name": "get_quote_preview",
            "description": (
                "Preview shipping rates for a shipment based on origin, "
                "destination, weight, and dimensions."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "origin_zip": {"type": "string", "description": "Origin ZIP"},
                    "destination_zip": {"type": "string", "description": "Destination ZIP"},
                    "weight_lbs": {"type": "number", "description": "Weight in pounds"},
                    "length_in": {"type": "number", "description": "Length in inches"},
                    "width_in": {"type": "number", "description": "Width in inches"},
                    "height_in": {"type": "number", "description": "Height in inches"},
                },
                "required": [
                    "origin_zip",
                    "destination_zip",
                    "weight_lbs",
                    "length_in",
                    "width_in",
                    "height_in",
                ],
            },
        },
    ]
}


# ── Fake tool handlers — replicate MockShippingProvider shapes ──────────────

def _fake_validate_address(args: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    street = (args.get("street") or "").strip()
    city = (args.get("city") or "").strip()
    state = (args.get("state") or "").strip()
    zip_code = (args.get("zip_code") or "").strip()
    country = (args.get("country") or "US").upper()

    if not street:
        issues.append("Street is required")
    if not city:
        issues.append("City is required")
    if not state:
        issues.append("State is required")
    if not _ZIP_PATTERN.match(zip_code):
        issues.append(f"Invalid zip code format: {zip_code}")

    if issues:
        return {
            "success": False,
            "data": {"is_valid": False, "issues": issues},
            "error": "; ".join(issues),
            "metadata": {"provider": "mock", "tool": "validate_address"},
        }

    normalized = {
        "street": street.title(),
        "city": city.title(),
        "state": state.upper()[:2],
        "zip_code": zip_code,
        "country": country,
    }
    return {
        "success": True,
        "data": {
            "is_valid": True,
            "normalized_address": normalized,
            "deliverable": True,
            "address_type": "residential",
        },
        "error": None,
        "metadata": {"provider": "mock", "tool": "validate_address"},
    }


def _fake_get_quote_preview(args: dict[str, Any]) -> dict[str, Any]:
    weight_lbs = float(args.get("weight_lbs", 0))
    length_in = float(args.get("length_in", 0))
    width_in = float(args.get("width_in", 0))
    height_in = float(args.get("height_in", 0))

    dim_weight = (length_in * width_in * height_in) / 139
    billable_weight = max(weight_lbs, dim_weight)
    base_rate = 5.99
    weight_rate = billable_weight * 0.45
    ground_price = round(base_rate + weight_rate, 2)

    services = [
        {
            "service": "Ground",
            "carrier": "MockCarrier",
            "price_usd": ground_price,
            "estimated_days": 5,
        },
        {
            "service": "Express",
            "carrier": "MockCarrier",
            "price_usd": round(ground_price * 1.8, 2),
            "estimated_days": 2,
        },
        {
            "service": "Overnight",
            "carrier": "MockCarrier",
            "price_usd": round(ground_price * 3.2, 2),
            "estimated_days": 1,
        },
    ]

    return {
        "success": True,
        "data": {
            "billable_weight_lbs": round(billable_weight, 2),
            "dim_weight_lbs": round(dim_weight, 2),
            "actual_weight_lbs": weight_lbs,
            "services": services,
            "disclaimer": (
                "Preview only — not a binding quote. "
                "Final rates from Spring Boot /api/v1/quotes."
            ),
        },
        "error": None,
        "metadata": {"provider": "mock", "tool": "get_quote_preview"},
    }


_FAKE_TOOLS = {
    "validate_address": _fake_validate_address,
    "get_quote_preview": _fake_get_quote_preview,
}


def _build_mcp_call_response(result: dict[str, Any]) -> dict[str, Any]:
    """Wrap a fake-tool result in the MCP /tools/call response envelope."""
    content: list[dict[str, Any]] = []
    if result["success"]:
        content.append({"type": "text", "text": json.dumps(result["data"], indent=2)})
    else:
        content.append({"type": "text", "text": f"Error: {result['error']}"})
    if result.get("metadata"):
        content.append(
            {"type": "text", "text": f"Metadata: {json.dumps(result['metadata'], indent=2)}"}
        )
    return {
        "success": result["success"],
        "content": content,
        "error": result.get("error"),
    }


# ── MockTransport factory ───────────────────────────────────────────────────

def build_mcp_mock_transport() -> httpx.MockTransport:
    """An httpx.MockTransport that emulates the ShipSmart-MCP HTTP contract."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path

        if path == "/tools/list":
            return httpx.Response(200, json=_TOOLS_LIST)

        if path == "/tools/call":
            body = json.loads(request.content or b"{}")
            name = body.get("name")
            arguments = body.get("arguments", {})
            fn = _FAKE_TOOLS.get(name)
            if fn is None:
                return httpx.Response(
                    404, json={"detail": f"Tool not found: {name}"}
                )
            result = fn(arguments)
            return httpx.Response(200, json=_build_mcp_call_response(result))

        return httpx.Response(404, json={"detail": "not found"})

    return httpx.MockTransport(handler)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def mcp_tool_registry() -> AsyncIterator[RemoteToolRegistry]:
    """A `RemoteToolRegistry` hydrated from a MockTransport — no live MCP needed."""
    transport = build_mcp_mock_transport()
    registry = await create_remote_registry(
        base_url="http://mcp.test",
        api_key="",
        transport=transport,
    )
    try:
        yield registry
    finally:
        await registry.aclose()
