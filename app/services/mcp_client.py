"""Back-compat shim — the MCP client moved to :mod:`app.integrations.mcp_client`.

Importing from ``app.services.mcp_client`` still works; new first-party code
imports from ``app.integrations.mcp_client`` (the outbound-integration layer).
"""

from app.integrations.mcp_client import (
    McpClient,
    RemoteTool,
    RemoteToolRegistry,
    ToolInput,
    ToolOutput,
    ToolParameter,
    _params_from_input_schema,
    _parse_content,
    create_remote_registry,
)

__all__ = [
    "McpClient",
    "RemoteTool",
    "RemoteToolRegistry",
    "ToolInput",
    "ToolOutput",
    "ToolParameter",
    "_params_from_input_schema",
    "_parse_content",
    "create_remote_registry",
]
