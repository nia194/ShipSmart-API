"""Tests for the LLM tool-calling primitive (Phase 1).

Covers the provider-agnostic ``ToolCall`` / ``ToolCallResult`` shapes, the
default ``NotImplementedError`` on non-native providers, and the deterministic
``ScriptedToolCallingClient`` replay used by the keyless agent e2e.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.llm.client import (
    EchoClient,
    ScriptedToolCallingClient,
    ToolCall,
    ToolCallResult,
    _to_anthropic_tools,
    build_provider_client,
)

# ── Non-native providers raise NotImplementedError ──────────────────────────


@pytest.mark.asyncio
async def test_echo_client_has_no_native_tool_calling():
    client = EchoClient()
    with pytest.raises(NotImplementedError):
        await client.complete_with_tools([{"role": "user", "content": "hi"}], [])


# ── Scripted client replays its sequence, then returns final ────────────────


@pytest.mark.asyncio
async def test_scripted_client_replays_sequence_then_final():
    turns = [
        ToolCallResult(
            kind="tool_calls",
            calls=[ToolCall(id="c1", name="retrieve_rag", arguments={"query": "x"})],
        ),
        ToolCallResult(
            kind="tool_calls",
            calls=[ToolCall(id="c2", name="validate_address", arguments={})],
        ),
        ToolCallResult(kind="final", text="all done"),
    ]
    client = ScriptedToolCallingClient(turns, final_text="exhausted")
    assert client.provider_name == "scripted"

    out1 = await client.complete_with_tools([], [])
    assert out1.kind == "tool_calls"
    assert out1.calls[0].name == "retrieve_rag"

    out2 = await client.complete_with_tools([], [])
    assert out2.calls[0].name == "validate_address"

    out3 = await client.complete_with_tools([], [])
    assert out3.kind == "final"
    assert out3.text == "all done"

    # Exhausted → always a final turn with the configured final text.
    out4 = await client.complete_with_tools([], [])
    assert out4.kind == "final"
    assert out4.text == "exhausted"


@pytest.mark.asyncio
async def test_scripted_client_complete_returns_final_text():
    client = ScriptedToolCallingClient([], final_text="canned")
    assert await client.complete([{"role": "user", "content": "q"}]) == "canned"


@pytest.mark.asyncio
async def test_scripted_client_default_script_is_canonical_sequence():
    client = ScriptedToolCallingClient()
    names = []
    for _ in range(3):
        out = await client.complete_with_tools([], [])
        assert out.kind == "tool_calls"
        names.append(out.calls[0].name)
    assert names == ["retrieve_rag", "validate_address", "get_quote_preview"]
    assert (await client.complete_with_tools([], [])).kind == "final"


# ── Factory selection (gated to non-production) ─────────────────────────────


def test_factory_builds_scripted_outside_production():
    with patch("app.llm.client.settings") as s:
        s.is_production = False
        client = build_provider_client("scripted")
        assert isinstance(client, ScriptedToolCallingClient)


def test_factory_refuses_scripted_in_production():
    with patch("app.llm.client.settings") as s:
        s.is_production = True
        assert build_provider_client("scripted") is None


# ── Anthropic tool-schema conversion ────────────────────────────────────────


def test_to_anthropic_tools_converts_registry_schema():
    tools = [
        {
            "name": "validate_address",
            "description": "Validate an address",
            "parameters": [
                {"name": "street", "type": "string", "description": "Street", "required": True},
                {"name": "country", "type": "string", "description": "Country", "required": False},
            ],
        },
    ]
    out = _to_anthropic_tools(tools)
    assert len(out) == 1
    schema = out[0]
    assert schema["name"] == "validate_address"
    assert schema["input_schema"]["type"] == "object"
    assert "street" in schema["input_schema"]["properties"]
    assert schema["input_schema"]["required"] == ["street"]
