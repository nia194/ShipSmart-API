"""
Base tool interface.
Every tool defines its name, description, input/output schemas,
and an execute function. Tools are the only way the AI layer
interacts with external providers or performs actions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolInput:
    """Validated input to a tool, carrying the parsed parameters."""

    params: dict[str, Any]


@dataclass
class ToolOutput:
    """Structured result from a tool execution."""

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolParameter:
    """Describes one parameter of a tool's input schema."""

    name: str
    type: str  # "string", "number", "boolean"
    description: str
    required: bool = True


class Tool(ABC):
    """Abstract base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier (e.g. 'validate_address')."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""

    @property
    @abstractmethod
    def parameters(self) -> list[ToolParameter]:
        """Input parameter schema for this tool."""

    @abstractmethod
    async def execute(self, tool_input: ToolInput) -> ToolOutput:
        """Run the tool with validated input. Returns structured output."""

    def schema(self) -> dict:
        """Return a JSON-serializable schema for LLM tool selection."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                }
                for p in self.parameters
            ],
        }

    def validate_input(self, params: dict[str, Any]) -> list[str]:
        """Validate params against the parameter schema. Returns error messages."""
        errors: list[str] = []
        for param in self.parameters:
            if param.required and param.name not in params:
                errors.append(f"Missing required parameter: {param.name}")
        return errors
