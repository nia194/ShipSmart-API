"""Agent tool-call policy validation (guardrails §5.4)."""

from __future__ import annotations

from app.schemas.typed_outputs import ToolCallPolicy
from app.security.tool_policy import (
    DEFAULT_TOOL_POLICIES,
    TOOL_DENIED_TAG,
    validate_tool_call,
)


def test_allows_known_read_tool_on_allowed_route():
    d = validate_tool_call("validate_address", route="/api/v1/agent/run", call_count=1)
    assert d.allowed and d.tags == []


def test_denies_unknown_tool():
    d = validate_tool_call("delete_everything", route="/api/v1/agent/run")
    assert not d.allowed and d.tags == [TOOL_DENIED_TAG]


def test_denies_wrong_route_and_call_cap():
    d1 = validate_tool_call("get_quote_preview", route="/api/v1/rag/query")
    assert not d1.allowed and d1.tags == [TOOL_DENIED_TAG]
    d2 = validate_tool_call("validate_address", route="/api/v1/agent/run", call_count=99)
    assert not d2.allowed and d2.tags == [TOOL_DENIED_TAG]


def test_denies_high_risk_without_confirmation():
    policies = {
        "book_shipment": ToolCallPolicy(
            tool_name="book_shipment", risk_tier="high", requires_confirmation=True
        )
    }
    denied = validate_tool_call("book_shipment", call_count=1, confirmed=False, policies=policies)
    assert not denied.allowed and denied.tags == [TOOL_DENIED_TAG]
    ok = validate_tool_call("book_shipment", call_count=1, confirmed=True, policies=policies)
    assert ok.allowed


def test_default_policies_cover_the_mcp_read_only_tools():
    assert set(DEFAULT_TOOL_POLICIES) == {
        "validate_address",
        "get_quote_preview",
        "calculate_dimensional_weight",
        "estimate_package_profile",
    }
    # Every governed tool stays read/quote tier — MCP serves no write tool.
    assert all(p.risk_tier in {"read", "quote"} for p in DEFAULT_TOOL_POLICIES.values())
