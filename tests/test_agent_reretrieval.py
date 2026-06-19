"""Tests for conditional, bounded re-retrieval in the agent loop (Phase 2).

The agent may call ``retrieve_rag`` more than once per run — with a different,
model-written query — ONLY after a prior retrieval returned weak coverage. The
loop bounds total retrievals (``agent_max_retrievals``), rejects identical-query
retries, and tags every decision so the path is auditable. A well-covered first
retrieval stays single-shot (unchanged from today).

Coverage is driven deterministically with a fixed-vector embedding: queries map
to orthogonal/aligned vectors so a retrieval is reliably weak or strong.
"""

from __future__ import annotations

import pytest

from app.llm.client import EchoClient, ScriptedToolCallingClient, ToolCall, ToolCallResult
from app.llm.router import TASK_FALLBACK, TASK_REASONING, TASK_SYNTHESIS, LLMRouter
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import InMemoryVectorStore, StoredChunk
from app.services.agent_service import run_agent

# Fixed 4-d basis vectors → fully controllable cosine similarity.
_LITHIUM = [1.0, 0.0, 0.0, 0.0]
_ELECTRONICS = [0.0, 1.0, 0.0, 0.0]
_ORTHOGONAL = [0.0, 0.0, 1.0, 0.0]   # aligns with no seeded chunk → weak coverage
_DEFAULT = [0.0, 0.0, 0.0, 1.0]      # any unmapped query → also weak


class _VecEmbedding(EmbeddingProvider):
    """Deterministic embedding: maps exact text → a chosen vector (else default)."""

    def __init__(self, mapping: dict[str, list[float]]):
        self._m = mapping

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(self._m.get(t, _DEFAULT)) for t in texts]

    @property
    def dimensions(self) -> int:
        return 4


# Queries used across the scripted reasoning turns.
Q_DRONE = "drone germany"
Q_LITHIUM = "lithium battery intl shipping"
Q_ELECTRONICS = "electronics import germany"
Q_DRONE_ALT = "drone to germany restrictions"

_MAPPING = {
    Q_DRONE: _ORTHOGONAL,        # weak (matches nothing)
    Q_DRONE_ALT: _DEFAULT,       # weak (matches nothing)
    Q_LITHIUM: _LITHIUM,         # strong (matches lithium chunk)
    Q_ELECTRONICS: _ELECTRONICS,  # strong (matches electronics chunk)
}


def _router(reasoning) -> LLMRouter:
    echo = EchoClient()
    return LLMRouter(
        clients={
            TASK_REASONING: reasoning,
            TASK_SYNTHESIS: echo,
            TASK_FALLBACK: echo,
        },
        fallback=echo,
    )


def _embed() -> _VecEmbedding:
    return _VecEmbedding(_MAPPING)


async def _seeded_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    await store.add([
        StoredChunk(text="lithium battery rules", source="hazmat",
                    chunk_index=0, embedding=_LITHIUM),
        StoredChunk(text="electronics import rules", source="customs",
                    chunk_index=1, embedding=_ELECTRONICS),
    ])
    return store


def _retrieve(call_query: str) -> ToolCallResult:
    return ToolCallResult(
        kind="tool_calls",
        calls=[ToolCall(id=call_query, name="retrieve_rag", arguments={"query": call_query})],
    )


def _retrieve_count(decisions: list[str]) -> int:
    return sum(
        1 for d in decisions
        if d.startswith("agent:retrieve:") and d.split(":")[-1].isdigit()
    )


# ── weak coverage triggers one justified re-retrieval ───────────────────────


@pytest.mark.asyncio
async def test_weak_coverage_triggers_reretrieval(mcp_tool_registry):
    reasoning = ScriptedToolCallingClient([
        _retrieve(Q_DRONE),       # weak → triggers re-retrieval
        _retrieve(Q_LITHIUM),     # different query, strong
        ToolCallResult(kind="final", text="done"),
    ])
    res = await run_agent(
        "sending a drone to Germany — any restrictions?", {},
        registry=mcp_tool_registry, llm_router=_router(reasoning),
        embedding_provider=_embed(), vector_store=await _seeded_store(),
    )
    assert "agent:retrieve:1" in res.decisions
    assert "agent:retrieve:2" in res.decisions
    assert "agent:retrieve:reformulate" in res.decisions
    # Last retrieval was strong → no honest-gap flag.
    assert "agent:retrieve:uncovered" not in res.decisions
    assert _retrieve_count(res.decisions) == 2


# ── identical-query retry is rejected (no degenerate loops) ─────────────────


@pytest.mark.asyncio
async def test_identical_query_retry_rejected(mcp_tool_registry):
    reasoning = ScriptedToolCallingClient([
        _retrieve(Q_DRONE),
        _retrieve(Q_DRONE),       # identical → rejected, not executed
        ToolCallResult(kind="final", text="done"),
    ])
    res = await run_agent(
        "drone?", {}, registry=mcp_tool_registry, llm_router=_router(reasoning),
        embedding_provider=_embed(), vector_store=await _seeded_store(),
    )
    assert "agent:retrieve:rejected" in res.decisions
    assert "agent:retrieve:1" in res.decisions
    assert "agent:retrieve:2" not in res.decisions  # second never executed
    assert _retrieve_count(res.decisions) == 1
    assert res.tools_used == ["retrieve_rag"]  # only the executed one counts
    # The rejection observation is fed back, not silently dropped.
    assert any("unchanged" in s["observation"].lower() for s in res.steps)


# ── agent_max_retrievals caps total retrievals ──────────────────────────────


@pytest.mark.asyncio
async def test_max_retrievals_cap_enforced(mcp_tool_registry):
    reasoning = ScriptedToolCallingClient([
        _retrieve(Q_DRONE),       # executed (1st, weak)
        _retrieve(Q_LITHIUM),     # cap=1 → capped, not executed
        ToolCallResult(kind="final", text="done"),
    ])
    res = await run_agent(
        "drone?", {}, registry=mcp_tool_registry, llm_router=_router(reasoning),
        embedding_provider=_embed(), vector_store=await _seeded_store(),
        max_retrievals=1,
    )
    assert "agent:retrieve:capped" in res.decisions
    assert "agent:retrieve:1" in res.decisions
    assert "agent:retrieve:2" not in res.decisions
    assert _retrieve_count(res.decisions) == 1
    assert any("limit reached" in s["observation"].lower() for s in res.steps)


# ── well-covered first retrieval stays single-shot (regression) ─────────────


@pytest.mark.asyncio
async def test_well_covered_stays_single_shot(mcp_tool_registry):
    reasoning = ScriptedToolCallingClient([
        _retrieve(Q_LITHIUM),     # strong on the first pass
        ToolCallResult(kind="final", text="done"),
    ])
    res = await run_agent(
        "lithium battery shipping?", {}, registry=mcp_tool_registry, llm_router=_router(reasoning),
        embedding_provider=_embed(), vector_store=await _seeded_store(),
    )
    assert res.tools_used == ["retrieve_rag"]
    assert "agent:tool:retrieve_rag" in res.decisions  # backward-compat tag
    assert "agent:retrieve:1" in res.decisions
    assert _retrieve_count(res.decisions) == 1
    # None of the agentic-only tags fire on the single-shot path.
    for tag in (
        "agent:retrieve:2", "agent:retrieve:reformulate",
        "agent:retrieve:rejected", "agent:retrieve:capped",
        "agent:retrieve:uncovered",
    ):
        assert tag not in res.decisions
    assert len(res.steps) == 1


# ── a sub-area still uncovered after retries is honestly flagged ────────────


@pytest.mark.asyncio
async def test_uncovered_after_retries_flags_gap(mcp_tool_registry):
    reasoning = ScriptedToolCallingClient([
        _retrieve(Q_DRONE),       # weak
        _retrieve(Q_DRONE_ALT),   # different, still weak
        ToolCallResult(kind="final", text="done"),
    ])
    res = await run_agent(
        "drone restrictions?", {}, registry=mcp_tool_registry, llm_router=_router(reasoning),
        embedding_provider=_embed(), vector_store=await _seeded_store(),
    )
    assert "agent:retrieve:reformulate" in res.decisions
    assert "agent:retrieve:uncovered" in res.decisions
    assert _retrieve_count(res.decisions) == 2
