"""
MCP Server for ShipSmart Tools.

Exposes the ToolRegistry (validate_address, get_quote_preview) via MCP HTTP protocol.
This allows Claude Code and other MCP clients to discover and execute tools.

Supports both local (stdio) and HTTP transport modes.
- Local: For Claude Code running on the same machine
- HTTP: For remote clients, future Render deployment

Start:
    uvicorn app.mcp_server:app --port 8001 --host 0.0.0.0

Test locally:
    curl -X POST http://localhost:8001/tools/list
    curl -X POST http://localhost:8001/tools/call -H "Content-Type: application/json" \
        -d '{"name":"validate_address","arguments":{"street":"123 Main St","city":"NYC","state":"NY","zip_code":"10001"}}'
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.providers import create_shipping_provider
from app.tools.address_tools import ValidateAddressTool
from app.tools.quote_tools import GetQuotePreviewTool
from app.tools.registry import ToolRegistry
from app.tools.base import ToolInput, ToolOutput

logger = logging.getLogger(__name__)

# Initialize tool registry
_tool_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Get or create the tool registry."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()

        # Register tools with the default provider
        provider = create_shipping_provider()
        _tool_registry.register(ValidateAddressTool(provider))
        _tool_registry.register(GetQuotePreviewTool(provider))

        logger.info("Tool registry initialized with %d tools", _tool_registry.count())

    return _tool_registry


# FastAPI app for MCP HTTP transport
app = FastAPI(
    title="ShipSmart MCP Server",
    description="MCP server exposing ShipSmart tools (validate_address, get_quote_preview)",
    version="1.0.0",
)


# Request/response models for MCP
class MCPToolDefinition(BaseModel):
    """MCP tool definition (compatible with Claude API)."""
    name: str
    description: str
    input_schema: dict[str, Any]


class MCPToolListResponse(BaseModel):
    """Response for /tools/list."""
    tools: list[MCPToolDefinition]


class MCPToolCallRequest(BaseModel):
    """Request for /tools/call."""
    name: str
    arguments: dict[str, Any]


class MCPToolCallResponse(BaseModel):
    """Response for /tools/call."""
    success: bool
    content: list[dict[str, Any]]
    error: str | None = None


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "shipsmart-mcp-server",
        "tools": get_tool_registry().count(),
    }


# MCP Tools Listing
@app.post("/tools/list")
async def list_tools() -> MCPToolListResponse:
    """
    MCP tools/list endpoint.
    Returns schema for all available tools.

    Equivalent to MCP `tools/list` resource.
    """
    registry = get_tool_registry()
    tools = registry.list_tools()

    tool_definitions = []
    for tool in tools:
        # Convert tool schema to MCP-compatible input_schema
        schema = tool.schema()

        # Build JSON Schema for input
        properties = {}
        required = []

        for param in tool.parameters:
            param_schema: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            properties[param.name] = param_schema

            if param.required:
                required.append(param.name)

        input_schema = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        tool_definitions.append(
            MCPToolDefinition(
                name=schema["name"],
                description=schema["description"],
                input_schema=input_schema,
            )
        )

    logger.info("Listed %d tools", len(tool_definitions))
    return MCPToolListResponse(tools=tool_definitions)


# MCP Tool Execution
@app.post("/tools/call")
async def call_tool(request: MCPToolCallRequest) -> MCPToolCallResponse:
    """
    MCP tools/call endpoint.
    Executes a tool with the provided arguments.

    Equivalent to MCP `tools/call` resource.

    Example:
        POST /tools/call
        {
            "name": "validate_address",
            "arguments": {
                "street": "123 Main St",
                "city": "San Francisco",
                "state": "CA",
                "zip_code": "94105"
            }
        }
    """
    registry = get_tool_registry()
    tool = registry.get(request.name)

    if tool is None:
        logger.error("Tool not found: %s", request.name)
        raise HTTPException(status_code=404, detail=f"Tool not found: {request.name}")

    # Validate input
    errors = tool.validate_input(request.arguments)
    if errors:
        logger.warning("Validation failed for %s: %s", request.name, errors)
        return MCPToolCallResponse(
            success=False,
            content=[],
            error="; ".join(errors),
        )

    # Execute tool
    try:
        logger.info("Executing tool: %s with args: %s", request.name, request.arguments)

        tool_input = ToolInput(params=request.arguments)
        result: ToolOutput = await tool.execute(tool_input)

        # Format result for MCP (content is a list of blocks)
        content = []

        if result.success:
            content.append({
                "type": "text",
                "text": json.dumps(result.data, indent=2),
            })
        else:
            content.append({
                "type": "text",
                "text": f"Error: {result.error}",
            })

        # Add metadata if present
        if result.metadata:
            content.append({
                "type": "text",
                "text": f"Metadata: {json.dumps(result.metadata, indent=2)}",
            })

        logger.info("Tool execution completed: %s (success=%s)", request.name, result.success)

        return MCPToolCallResponse(
            success=result.success,
            content=content,
            error=result.error,
        )

    except Exception as e:
        logger.error("Tool execution failed: %s", e, exc_info=True)
        return MCPToolCallResponse(
            success=False,
            content=[],
            error=str(e),
        )


# Root endpoint for discovery
@app.get("/")
async def root():
    """Root endpoint for service discovery."""
    registry = get_tool_registry()
    return {
        "name": "ShipSmart MCP Server",
        "version": "1.0.0",
        "description": "MCP server for ShipSmart tools",
        "tools_count": registry.count(),
        "endpoints": {
            "health": "/health",
            "tools_list": "/tools/list (POST)",
            "tools_call": "/tools/call (POST)",
        },
    }


# Startup/shutdown
@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    registry = get_tool_registry()
    logger.info("MCP server starting with %d tools", registry.count())


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("MCP server shutting down")
