"""
Orchestration service.
Receives a user query, decides whether a tool call is needed or a direct
RAG answer suffices, executes the appropriate path, and returns a structured result.

Decision logic is rule-based for now. LLM-assisted tool selection can be
added in a future phase by swapping the _select_tool implementation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.core.cache import TTLCache
from app.core.errors import AppError
from app.llm.client import LLMClient
from app.tools.base import ToolInput, ToolOutput
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Small cache for LLM-assisted tool selection so repeated identical queries
# don't pay the LLM cost twice.
_tool_selection_cache = TTLCache(default_ttl=600, max_size=128)

# Keyword patterns for rule-based tool selection.
# Each entry: (compiled regex, tool_name, param_extractor or None)
_TOOL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bvalidat\w*\s+.{0,10}address", re.IGNORECASE), "validate_address"),
    (re.compile(r"\baddress\s+.{0,10}valid", re.IGNORECASE), "validate_address"),
    (re.compile(r"\bquote\s+preview", re.IGNORECASE), "get_quote_preview"),
    (re.compile(r"\bshipping\s+(rate|cost|price|estimate)", re.IGNORECASE), "get_quote_preview"),
    (re.compile(r"\bestimate.{0,15}(delivery|shipping)", re.IGNORECASE), "get_quote_preview"),
]


@dataclass
class OrchestrationResult:
    """Result from the orchestration service."""

    type: str  # "tool_result", "direct_answer", "error"
    tool_used: str | None = None
    answer: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def select_tool(query: str, registry: ToolRegistry) -> str | None:
    """Rule-based tool selection from a natural-language query.

    Returns the tool name if a match is found, None for direct-answer path.
    """
    for pattern, tool_name in _TOOL_PATTERNS:
        if pattern.search(query) and registry.get(tool_name) is not None:
            return tool_name
    return None


async def select_tool_with_llm(
    query: str,
    registry: ToolRegistry,
    llm_client: LLMClient | None,
) -> str | None:
    """LLM-assisted tool selection used when the regex shortcut misses.

    Issues a tightly constrained prompt: the LLM must reply with EXACTLY one
    tool name from the registry, or the literal string 'NONE'. Anything else
    is treated as no-match. Cached by query string for cost control.
    """
    if llm_client is None:
        return None

    available = [s["name"] for s in registry.list_schemas()]
    if not available:
        return None

    cache_key = _tool_selection_cache.make_key("tool-select", query, tuple(sorted(available)))
    cached = _tool_selection_cache.get(cache_key)
    if cached is not None:
        return cached if cached != "__NONE__" else None

    tool_list = ", ".join(available)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict tool dispatcher for a shipping assistant. "
                "Reply with EXACTLY one tool name from the provided list, or the "
                "literal word NONE if no tool fits. No explanation, no punctuation, "
                "no other text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User query: {query}\n"
                f"Available tools: {tool_list}\n"
                "Which tool should be invoked? Reply with one of: "
                f"{tool_list}, NONE."
            ),
        },
    ]

    try:
        raw = await llm_client.complete(messages)
    except Exception as exc:
        logger.warning("LLM tool selection failed: %s", exc)
        return None

    candidate = (raw or "").strip().splitlines()[0].strip().strip(".:,")
    if candidate.upper() == "NONE" or candidate not in available:
        _tool_selection_cache.set(cache_key, "__NONE__")
        return None

    _tool_selection_cache.set(cache_key, candidate)
    logger.info("LLM-assisted tool selection chose: %s", candidate)
    return candidate


async def run_orchestration(
    query: str,
    params: dict[str, Any],
    registry: ToolRegistry,
    llm_client: LLMClient | None = None,
) -> OrchestrationResult:
    """Execute the orchestration flow for a query.

    1. Select a tool (rule-based) based on the query text.
    2. If a tool is selected and params are provided, execute it.
    3. If no tool matches, return a direct-answer indicator so the
       caller can fall back to RAG.

    Args:
        query: Natural-language user query.
        params: Explicit tool parameters (from the API request body).
        registry: The tool registry to search.

    Returns:
        OrchestrationResult with either tool output or direct-answer signal.
    """
    # Fast path: deterministic regex rules.
    tool_name = select_tool(query, registry)

    # Slow path: LLM-assisted selection when the regex misses and a client
    # is available. Keeps cost low because the LLM is only called for the
    # natural-language edge cases the rules can't catch.
    selection_method = "rule"
    if tool_name is None and llm_client is not None:
        tool_name = await select_tool_with_llm(query, registry, llm_client)
        if tool_name is not None:
            selection_method = "llm"

    if tool_name is None:
        return OrchestrationResult(
            type="direct_answer",
            answer="No tool matched this query. Use the RAG endpoint for general questions.",
            metadata={"reason": "no_tool_match", "selection_method": "none"},
        )

    result = await execute_tool(tool_name, params, registry)
    result.metadata = {**result.metadata, "selection_method": selection_method}
    return result


async def execute_tool(
    tool_name: str,
    params: dict[str, Any],
    registry: ToolRegistry,
) -> OrchestrationResult:
    """Look up and execute a specific tool by name.

    Args:
        tool_name: The tool to execute.
        params: Parameters to pass to the tool.
        registry: The tool registry.

    Returns:
        OrchestrationResult wrapping the tool output.

    Raises:
        AppError: If tool not found or input validation fails.
    """
    tool = registry.get(tool_name)
    if tool is None:
        raise AppError(status_code=404, message=f"Unknown tool: {tool_name}")

    # Validate input
    errors = tool.validate_input(params)
    if errors:
        raise AppError(status_code=422, message=f"Invalid tool input: {'; '.join(errors)}")

    logger.info("Executing tool=%s with %d params", tool_name, len(params))

    try:
        result: ToolOutput = await tool.execute(ToolInput(params=params))
    except Exception as exc:
        logger.error("Tool execution failed: tool=%s error=%s", tool_name, exc)
        raise AppError(
            status_code=502,
            message=f"Tool execution failed: {tool_name}",
        ) from exc

    answer = _summarize_tool_result(tool_name, result)

    return OrchestrationResult(
        type="tool_result",
        tool_used=tool_name,
        answer=answer,
        data=result.data,
        metadata=result.metadata,
    )


def _summarize_tool_result(tool_name: str, result: ToolOutput) -> str:
    """Generate a human-readable summary of a tool result."""
    if not result.success:
        return f"Tool '{tool_name}' failed: {result.error or 'unknown error'}"

    if tool_name == "validate_address":
        if result.data.get("is_valid"):
            addr = result.data.get("normalized_address", {})
            parts = [
                addr.get("street", ""),
                addr.get("city", ""),
                addr.get("state", ""),
                addr.get("zip_code", ""),
            ]
            formatted = ", ".join(p for p in parts if p)
            return f"Address is valid and deliverable: {formatted}"
        return f"Address validation failed: {result.error}"

    if tool_name == "get_quote_preview":
        services = result.data.get("services", [])
        if services:
            cheapest = min(services, key=lambda s: s["price_usd"])
            return (
                f"Found {len(services)} service options. "
                f"Cheapest: {cheapest['service']} at ${cheapest['price_usd']:.2f} "
                f"({cheapest['estimated_days']} days). "
                "Note: these are preview estimates only."
            )
        return "No service options available for this shipment."

    return f"Tool '{tool_name}' completed successfully."
