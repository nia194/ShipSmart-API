"""Tests for the AgentService loop (Phase 2).

Mocks the reasoning client (via a scripted/fake ``complete_with_tools``) and the
MCP registry (the shared ``mcp_tool_registry`` fixture), following the suite's
existing style. The synthesis client is a real EchoClient so the grounded-answer
path runs without API keys.
"""

from __future__ import annotations

import pytest

from app.llm.client import EchoClient, ScriptedToolCallingClient, ToolCall, ToolCallResult
from app.llm.router import TASK_FALLBACK, TASK_REASONING, TASK_SYNTHESIS, LLMRouter
from app.rag.embeddings import LocalHashEmbedding
from app.rag.vector_store import InMemoryVectorStore, StoredChunk
from app.services.agent_service import run_agent

_KB_TEXT = "Power banks contain lithium batteries; ship ground, declare watt-hours."


# ── Fixtures / builders ──────────────────────────────────────────────────────


def _router(reasoning) -> LLMRouter:
    """Router whose reasoning client is the scripted/fake, synthesis is Echo."""
    echo = EchoClient()
    return LLMRouter(
        clients={
            TASK_REASONING: reasoning,
            TASK_SYNTHESIS: echo,
            TASK_FALLBACK: echo,
        },
        fallback=echo,
    )


def _embed() -> LocalHashEmbedding:
    return LocalHashEmbedding(dims=16)


async def _seeded_store(embed: LocalHashEmbedding) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    vec = (await embed.embed([_KB_TEXT]))[0]
    await store.add([
        StoredChunk(text=_KB_TEXT, source="hazmat", chunk_index=0, embedding=vec),
    ])
    return store


class _FakeReasoning(ScriptedToolCallingClient):
    """A scripted reasoning client (reuses the production stub's replay logic)."""


# ── single-tool ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_tool_then_final(mcp_tool_registry):
    embed = _embed()
    store = await _seeded_store(embed)
    reasoning = _FakeReasoning([
        ToolCallResult(
            kind="tool_calls",
            calls=[ToolCall(id="c1", name="retrieve_rag", arguments={"query": "power bank"})],
        ),
        ToolCallResult(kind="final", text="done"),
    ])
    res = await run_agent(
        "Any restrictions on power banks?", {},
        registry=mcp_tool_registry, llm_router=_router(reasoning),
        embedding_provider=embed, vector_store=store,
    )
    assert res.tools_used == ["retrieve_rag"]
    assert "agent:tool:retrieve_rag" in res.decisions
    assert res.answer  # Echo synthesized a grounded answer
    assert res.provider == "echo"
    assert len(res.steps) == 1


# ── multi-tool chain ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_multi_tool_chain(mcp_tool_registry):
    embed = _embed()
    store = await _seeded_store(embed)
    reasoning = _FakeReasoning([
        ToolCallResult(kind="tool_calls", calls=[
            ToolCall(id="c1", name="retrieve_rag", arguments={"query": "lithium"})]),
        ToolCallResult(kind="tool_calls", calls=[
            ToolCall(id="c2", name="validate_address", arguments={
                "street": "123 Main St", "city": "Beverly Hills",
                "state": "CA", "zip_code": "90210"})]),
        ToolCallResult(kind="tool_calls", calls=[
            ToolCall(id="c3", name="get_quote_preview", arguments={
                "origin_zip": "10001", "destination_zip": "90210", "weight_lbs": 5.0,
                "length_in": 10.0, "width_in": 8.0, "height_in": 6.0})]),
        ToolCallResult(kind="final", text="here is your plan"),
    ])
    res = await run_agent(
        "Ship a 5lb box, cheapest, is the address ok, power bank restrictions?", {},
        registry=mcp_tool_registry, llm_router=_router(reasoning),
        embedding_provider=embed, vector_store=store,
    )
    assert res.tools_used == ["retrieve_rag", "validate_address", "get_quote_preview"]
    assert len(res.steps) == 3
    # validate_address + get_quote_preview observations feed grounding (tool_results).
    assert res.answer


# ── step-cap ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_step_cap_forces_final_synthesis(mcp_tool_registry):
    embed = _embed()
    store = await _seeded_store(embed)
    # Never returns "final" → loop must stop at max_steps and synthesize.
    never_final = _FakeReasoning(
        [ToolCallResult(kind="tool_calls", calls=[
            ToolCall(id="c", name="retrieve_rag", arguments={"query": "x"})])] * 10,
        final_text="unused",
    )
    res = await run_agent(
        "loop forever", {},
        registry=mcp_tool_registry, llm_router=_router(never_final),
        embedding_provider=embed, vector_store=store,
        max_steps=3,
    )
    assert "agent:max_steps" in res.decisions
    assert sum(1 for d in res.decisions if d.startswith("agent:step")) == 3
    assert res.answer


# ── tool-error recovery ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_error_recovery(mcp_tool_registry):
    embed = _embed()
    store = await _seeded_store(embed)
    # First call has missing required params → execute_tool raises 422; the loop
    # must capture it as an observation and continue to a final answer.
    reasoning = _FakeReasoning([
        ToolCallResult(kind="tool_calls", calls=[
            ToolCall(id="c1", name="validate_address", arguments={})]),  # invalid
        ToolCallResult(kind="final", text="recovered"),
    ])
    res = await run_agent(
        "validate my address", {},
        registry=mcp_tool_registry, llm_router=_router(reasoning),
        embedding_provider=embed, vector_store=store,
    )
    assert res.tools_used == ["validate_address"]
    assert res.steps[0]["tool"] == "validate_address"
    assert "error" in res.steps[0]["observation"].lower()
    assert res.answer  # did not abort


# ── pure-RAG (single retrieve then answer) ──────────────────────────────────


@pytest.mark.asyncio
async def test_pure_rag(mcp_tool_registry):
    embed = _embed()
    store = await _seeded_store(embed)
    reasoning = _FakeReasoning([
        ToolCallResult(kind="tool_calls", calls=[
            ToolCall(id="c1", name="retrieve_rag",
                     arguments={"query": "power bank lithium"})]),
        ToolCallResult(kind="final", text="grounded"),
    ])
    res = await run_agent(
        "power bank restrictions?", {},
        registry=mcp_tool_registry, llm_router=_router(reasoning),
        embedding_provider=embed, vector_store=store,
    )
    assert res.tools_used == ["retrieve_rag"]
    assert res.sources  # retrieved chunk surfaced as a grounded source


# ── text-fallback (provider without native tool calling) ────────────────────


@pytest.mark.asyncio
async def test_text_fallback_when_no_native_tools(mcp_tool_registry):
    embed = _embed()
    store = await _seeded_store(embed)
    # EchoClient as the reasoning client → complete_with_tools raises
    # NotImplementedError → single-pass text fallback runs.
    res = await run_agent(
        "how do I ship a power bank?", {},
        registry=mcp_tool_registry, llm_router=_router(EchoClient()),
        embedding_provider=embed, vector_store=store,
    )
    assert "agent:fallback:text" in res.decisions
    assert res.answer
    assert res.provider == "echo"
