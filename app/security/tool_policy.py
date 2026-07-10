"""Agent tool-call policy (Governance & Guardrails §5.4).

The model reasons about which tool to call; **code decides whether that call is
allowed**. Every tool has a ``ToolCallPolicy`` (risk tier, allowed routes, call
cap, confirmation requirement — the F1 schema); the planner validates a proposed
call against it *before* execution. An unknown tool, a wrong route, too many
calls, or a high-risk call without confirmation is denied (``guardrail:tool_denied``)
— never executed. Read-only tools need no confirmation; today the whole tool
surface is read-only (see ShipSmart-MCP's enforced allowlist).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.typed_outputs import ToolCallPolicy

TOOL_DENIED_TAG = "guardrail:tool_denied"

# Default policies — keyed to the tools ShipSmart-MCP actually serves. A tool not
# listed here is unknown and denied (no tool invention).
DEFAULT_TOOL_POLICIES: dict[str, ToolCallPolicy] = {
    "validate_address": ToolCallPolicy(
        tool_name="validate_address",
        risk_tier="read",
        allowed_routes=[
            "/api/v1/agent/run",
            "/api/v1/advisor/shipping",
            "/api/v1/advisor/tracking",
            "/api/v1/orchestration/run",
        ],
        max_calls_per_request=3,
    ),
    "get_quote_preview": ToolCallPolicy(
        tool_name="get_quote_preview",
        risk_tier="quote",
        allowed_routes=[
            "/api/v1/agent/run",
            "/api/v1/advisor/shipping",
            "/api/v1/orchestration/run",
        ],
        max_calls_per_request=3,
    ),
    # Package-intelligence tools (Product Roadmap §11 — read-only, deterministic).
    "calculate_dimensional_weight": ToolCallPolicy(
        tool_name="calculate_dimensional_weight",
        risk_tier="read",
        allowed_routes=[
            "/api/v1/agent/run",
            "/api/v1/advisor/shipping",
            "/api/v1/orchestration/run",
        ],
        max_calls_per_request=3,
    ),
    "estimate_package_profile": ToolCallPolicy(
        tool_name="estimate_package_profile",
        risk_tier="read",
        allowed_routes=[
            "/api/v1/agent/run",
            "/api/v1/advisor/shipping",
            "/api/v1/orchestration/run",
        ],
        max_calls_per_request=3,
    ),
    "parse_address": ToolCallPolicy(
        tool_name="parse_address",
        risk_tier="read",
        allowed_routes=[
            "/api/v1/agent/run",
            "/api/v1/advisor/shipping",
            "/api/v1/advisor/tracking",
            "/api/v1/orchestration/run",
        ],
        max_calls_per_request=3,
    ),
    "check_restricted_items": ToolCallPolicy(
        tool_name="check_restricted_items",
        risk_tier="read",
        allowed_routes=[
            "/api/v1/agent/run",
            "/api/v1/advisor/shipping",
            "/api/v1/compliance/check",
            "/api/v1/orchestration/run",
        ],
        max_calls_per_request=3,
    ),
}


@dataclass
class ToolPolicyDecision:
    allowed: bool
    reason: str = ""
    tags: list[str] = field(default_factory=list)


def validate_tool_call(
    tool_name: str,
    *,
    route: str = "",
    call_count: int = 1,
    confirmed: bool = False,
    policies: dict[str, ToolCallPolicy] | None = None,
) -> ToolPolicyDecision:
    """Allow or deny a proposed tool call against its policy (before execution)."""
    registry = policies if policies is not None else DEFAULT_TOOL_POLICIES
    policy = registry.get(tool_name)
    if policy is None:
        return ToolPolicyDecision(False, f"unknown tool {tool_name!r}", [TOOL_DENIED_TAG])
    if policy.allowed_routes and route and route not in policy.allowed_routes:
        return ToolPolicyDecision(
            False, f"{tool_name}: route {route!r} not allowed", [TOOL_DENIED_TAG]
        )
    if call_count > policy.max_calls_per_request:
        return ToolPolicyDecision(
            False,
            f"{tool_name}: call {call_count} exceeds cap {policy.max_calls_per_request}",
            [TOOL_DENIED_TAG],
        )
    if policy.requires_confirmation and not confirmed:
        return ToolPolicyDecision(
            False, f"{tool_name}: requires user confirmation", [TOOL_DENIED_TAG]
        )
    return ToolPolicyDecision(True, "allowed")
