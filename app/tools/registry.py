"""
Tool registry.
Central place to register, discover, and look up tools.
The orchestration layer and LLM tool-selection logic use this registry.
"""

from __future__ import annotations

import logging

from app.tools.base import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for tool discovery and lookup."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises if name is already taken."""
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name. Returns None if not found."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """Return all registered tools, sorted by name."""
        return sorted(self._tools.values(), key=lambda t: t.name)

    def list_schemas(self) -> list[dict]:
        """Return JSON-serializable schemas for all tools."""
        return [t.schema() for t in self.list_tools()]

    def count(self) -> int:
        return len(self._tools)
