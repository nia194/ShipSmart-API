"""Isolated tests for the orchestration service (rule + LLM tool selection).

The route-level tests in test_orchestration.py exercise this end-to-end; these
pin the decision logic directly: deterministic regex selection, the LLM-assisted
slow path (with caching), and the AppError mapping in execute_tool. Tools are
served by the MockTransport-backed RemoteToolRegistry fixture (tests/conftest.py).
"""

from __future__ import annotations

import pytest

from app.core.errors import AppError
from app.services import orchestration_service as orch
from app.services.mcp_client import ToolOutput
from app.services.orchestration_service import (
    execute_tool,
    run_orchestration,
    select_tool,
    select_tool_with_llm,
)

_ADDR = {"street": "1 Main St", "city": "Los Angeles", "state": "CA", "zip_code": "90001"}
_QUOTE = {
    "origin_zip": "90210", "destination_zip": "10001",
    "weight_lbs": 5, "length_in": 12, "width_in": 8, "height_in": 6,
}


class _FakeLLM:
    """Minimal LLMClient stand-in: returns a queued reply, counts calls."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0
        self.provider_name = "fake"

    async def complete(self, messages, **kwargs) -> str:
        self.calls += 1
        return self.reply


@pytest.fixture(autouse=True)
def _clear_selection_cache():
    orch._tool_selection_cache.clear()
    yield
    orch._tool_selection_cache.clear()


# ── Rule-based selection ─────────────────────────────────────────────────────


async def test_select_tool_matches_address_and_quote_rules(mcp_tool_registry):
    assert select_tool("Please validate this address", mcp_tool_registry) == "validate_address"
    assert select_tool("What's the shipping rate?", mcp_tool_registry) == "get_quote_preview"
    # A general knowledge question matches no tool → direct-answer path.
    assert select_tool("What is dimensional weight?", mcp_tool_registry) is None


async def test_run_orchestration_rule_path_executes_tool(mcp_tool_registry):
    result = await run_orchestration(
        "validate this address", _ADDR, mcp_tool_registry,
    )
    assert result.type == "tool_result"
    assert result.tool_used == "validate_address"
    assert result.metadata["selection_method"] == "rule"


async def test_run_orchestration_no_match_returns_direct_answer(mcp_tool_registry):
    result = await run_orchestration(
        "Tell me about carriers", {}, mcp_tool_registry, llm_client=None,
    )
    assert result.type == "direct_answer"
    assert result.metadata["selection_method"] == "none"


# ── LLM-assisted selection (slow path) ───────────────────────────────────────


async def test_run_orchestration_llm_path_when_rules_miss(mcp_tool_registry):
    # A phrasing the regex can't catch; the LLM dispatcher picks the tool.
    llm = _FakeLLM("validate_address")
    result = await run_orchestration(
        "Can you check whether this place exists?", _ADDR, mcp_tool_registry, llm_client=llm,
    )
    assert result.tool_used == "validate_address"
    assert result.metadata["selection_method"] == "llm"
    assert llm.calls == 1


async def test_select_tool_with_llm_rejects_none_and_junk(mcp_tool_registry):
    assert await select_tool_with_llm("x", mcp_tool_registry, _FakeLLM("NONE")) is None
    assert await select_tool_with_llm("y", mcp_tool_registry, _FakeLLM("not_a_tool")) is None
    assert await select_tool_with_llm("z", mcp_tool_registry, None) is None


async def test_select_tool_with_llm_is_cached(mcp_tool_registry):
    llm = _FakeLLM("get_quote_preview")
    first = await select_tool_with_llm("same query", mcp_tool_registry, llm)
    second = await select_tool_with_llm("same query", mcp_tool_registry, llm)
    assert first == second == "get_quote_preview"
    assert llm.calls == 1  # second lookup served from cache, no extra LLM call


# ── execute_tool error mapping ───────────────────────────────────────────────


async def test_execute_tool_unknown_raises_404(mcp_tool_registry):
    with pytest.raises(AppError) as exc:
        await execute_tool("no_such_tool", {}, mcp_tool_registry)
    assert exc.value.status_code == 404


async def test_execute_tool_invalid_input_raises_422(mcp_tool_registry):
    with pytest.raises(AppError) as exc:
        await execute_tool("validate_address", {"street": "only"}, mcp_tool_registry)
    assert exc.value.status_code == 422


async def test_execute_tool_success_summarizes_quote(mcp_tool_registry):
    result = await execute_tool("get_quote_preview", _QUOTE, mcp_tool_registry)
    assert result.type == "tool_result"
    assert "Cheapest" in result.answer  # human-readable summary of cheapest service


# ── Summary formatting (pure) ────────────────────────────────────────────────


def test_summarize_tool_result_handles_failure():
    summary = orch._summarize_tool_result(
        "get_quote_preview", ToolOutput(success=False, error="provider down"),
    )
    assert "failed" in summary and "provider down" in summary
