"""
Remote MCP client + tool registry shim.

The tool layer used to live in-process under `app.tools`. After the migration,
tool definitions and execution live in the standalone ShipSmart-MCP service.
This module is a thin HTTP client that ducktypes the old Tool / ToolRegistry
interface so that `orchestration_service`, `shipping_advisor_service`, and
`tracking_advisor_service` don't need to change.

Contract (defined by ShipSmart-MCP):

    POST /tools/list           → { tools: [{ name, description, input_schema }] }
    POST /tools/call           → { success, content: [...], error? }
    Header X-MCP-Api-Key       → required when MCP server has MCP_API_KEY set
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ── Wire-compatible tool input/output types ─────────────────────────────────
# Re-homed from the old app/tools/base.py so consumers keep working after
# the local tools package is removed.


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
    """One parameter in a tool's input schema."""

    name: str
    type: str  # "string", "number", "boolean"
    description: str
    required: bool = True


# ── Low-level HTTP client ────────────────────────────────────────────────────


class McpClient:
    """Async HTTP client for the ShipSmart MCP server."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 20.0,
    ) -> None:
        if not base_url:
            raise ValueError("McpClient requires a non-empty base_url")
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["X-MCP-Api-Key"] = api_key
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=transport,
        )
        self._base_url = base_url

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_tools(self) -> list[dict[str, Any]]:
        resp = await self._client.post("/tools/list")
        resp.raise_for_status()
        body = resp.json()
        return list(body.get("tools", []))

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        resp = await self._client.post(
            "/tools/call",
            json={"name": name, "arguments": arguments},
        )
        if resp.status_code == 404:
            return {
                "success": False,
                "content": [],
                "error": f"Tool not found on MCP server: {name}",
            }
        resp.raise_for_status()
        return resp.json()


# ── Tool / Registry shim ─────────────────────────────────────────────────────


class RemoteTool:
    """Quacks like the old `app.tools.base.Tool` but delegates to an MCP server."""

    def __init__(
        self,
        client: McpClient,
        name: str,
        description: str,
        parameters: list[ToolParameter],
    ) -> None:
        self._client = client
        self._name = name
        self._description = description
        self._parameters = parameters

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> list[ToolParameter]:
        return list(self._parameters)

    def schema(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "description": self._description,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                }
                for p in self._parameters
            ],
        }

    def validate_input(self, params: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        for param in self._parameters:
            if param.required and param.name not in params:
                errors.append(f"Missing required parameter: {param.name}")
        return errors

    async def execute(self, tool_input: ToolInput) -> ToolOutput:
        try:
            response = await self._client.call_tool(self._name, tool_input.params)
        except httpx.HTTPError as exc:
            logger.error("MCP call_tool(%s) failed: %s", self._name, exc)
            return ToolOutput(
                success=False,
                error=f"MCP server error: {exc}",
                metadata={"tool": self._name, "transport": "mcp"},
            )

        success = bool(response.get("success"))
        error = response.get("error")
        data, metadata = _parse_content(response.get("content", []))
        metadata.setdefault("tool", self._name)
        metadata.setdefault("transport", "mcp")
        return ToolOutput(success=success, data=data, error=error, metadata=metadata)


class RemoteToolRegistry:
    """Drop-in replacement for the old `ToolRegistry`, hydrated from MCP."""

    def __init__(self, client: McpClient) -> None:
        self._client = client
        self._tools: dict[str, RemoteTool] = {}

    async def refresh(self) -> None:
        """Call /tools/list on the MCP server and rebuild the local index."""
        raw_tools = await self._client.list_tools()
        self._tools = {}
        for t in raw_tools:
            name = t["name"]
            params = _params_from_input_schema(t.get("input_schema", {}))
            self._tools[name] = RemoteTool(
                self._client,
                name=name,
                description=t.get("description", ""),
                parameters=params,
            )
        logger.info(
            "Hydrated RemoteToolRegistry with %d tools from MCP", len(self._tools)
        )

    def get(self, name: str):
        return self._tools.get(name)

    def list_tools(self) -> list[RemoteTool]:
        return sorted(self._tools.values(), key=lambda t: t.name)

    def list_schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.list_tools()]

    def count(self) -> int:
        return len(self._tools)

    async def aclose(self) -> None:
        await self._client.aclose()


# ── Helpers ──────────────────────────────────────────────────────────────────


def _params_from_input_schema(input_schema: dict[str, Any]) -> list[ToolParameter]:
    """Convert an MCP `input_schema` (JSON Schema object) to ToolParameter list."""
    properties = input_schema.get("properties", {}) or {}
    required = set(input_schema.get("required", []) or [])
    out: list[ToolParameter] = []
    for name, prop in properties.items():
        out.append(
            ToolParameter(
                name=name,
                type=prop.get("type", "string"),
                description=prop.get("description", ""),
                required=name in required,
            )
        )
    return out


def _parse_content(
    content: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Best-effort parse of MCP content blocks back into (data, metadata).

    The MCP server serializes `data` as JSON in the first text block and an
    optional `metadata` block as `"Metadata: {...}"` in a second text block.
    This mirrors `ShipSmart-MCP/app/main.py::call_tool`.
    """
    data: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for block in content:
        if block.get("type") != "text":
            continue
        text = block.get("text", "")
        if text.startswith("Metadata:"):
            try:
                metadata = json.loads(text.removeprefix("Metadata:").strip())
            except json.JSONDecodeError:
                continue
        elif text.startswith("Error:"):
            # Error branch — keep data empty, the top-level `error` field carries it.
            continue
        else:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = {"text": text}
    return data, metadata


async def create_remote_registry(
    base_url: str,
    api_key: str = "",
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> RemoteToolRegistry:
    """Build + hydrate a registry in one call. Raises if the MCP is unreachable."""
    client = McpClient(base_url=base_url, api_key=api_key, transport=transport)
    registry = RemoteToolRegistry(client)
    await registry.refresh()
    return registry
