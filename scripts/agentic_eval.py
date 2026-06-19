"""
Agentic re-retrieval comparison eval — does the loop earn its cost?

Runs a small set of HARD, compound shipping questions (the drone-to-Germany
class) through BOTH retrieval strategies and reports, per query:

  * coverage BEFORE  — the single broad retrieval a non-agentic agent would use
  * coverage AFTER   — the best of the agent's reformulated sub-area retrievals
  * grounding change — did the answer move from a partial refusal (weak coverage,
                       nothing clears the grounding threshold) to a grounded one?
  * added cost       — extra retrievals / agent steps the loop spent to get there

Everything is driven deterministically and KEYLESS via the
``ScriptedToolCallingClient`` (no API keys, no network): one script that
re-retrieves on weak coverage (agentic) and one that does not (single-shot).
Coverage is forced with fixed-vector embeddings so the numbers are reproducible.

Usage:
    python scripts/agentic_eval.py
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field

from app.llm.client import (
    EchoClient,
    ScriptedToolCallingClient,
    ToolCall,
    ToolCallResult,
)
from app.llm.router import TASK_FALLBACK, TASK_REASONING, TASK_SYNTHESIS, LLMRouter
from app.rag.embeddings import EmbeddingProvider
from app.rag.retrieval import retrieve_auto
from app.rag.vector_store import InMemoryVectorStore, StoredChunk
from app.services.agent_service import coverage_of, run_agent

# ── Deterministic coverage substrate ─────────────────────────────────────────
# One-hot basis vectors -> exact, reproducible cosine similarity. A query that
# matches a seeded chunk's vector scores 1.0 (covered); an orthogonal query
# scores 0.0 (weak — nothing clears the grounding threshold).

_LITHIUM = [1.0, 0.0, 0.0, 0.0]
_ELECTRONICS = [0.0, 1.0, 0.0, 0.0]
_WEAK = [0.0, 0.0, 1.0, 0.0]        # broad compound queries -> match nothing
_UNCOVERED = [0.0, 0.0, 0.0, 1.0]   # a sub-area with NO chunk in the KB


class _StubRegistry:
    """Keyless no-MCP registry: the eval exercises only ``retrieve_rag``, so the
    agent needs an empty tool list and never dispatches a remote tool."""

    def list_schemas(self) -> list[dict]:
        return []


class _VecEmbedding(EmbeddingProvider):
    """Maps exact query/chunk text -> a chosen vector (default = weak/orthogonal)."""

    def __init__(self, mapping: dict[str, list[float]]):
        self._m = mapping

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(self._m.get(t, _WEAK)) for t in texts]

    @property
    def dimensions(self) -> int:
        return 4


# Reformulated sub-area queries (what the agent writes after seeing weak coverage).
SUB_LITHIUM = "lithium battery international shipping limits"
SUB_ELECTRONICS = "electronics import rules Germany"
SUB_RADIO = "radio transmitter frequency license export"  # intentionally uncovered

_QUERY_VECTORS = {
    SUB_LITHIUM: _LITHIUM,
    SUB_ELECTRONICS: _ELECTRONICS,
    SUB_RADIO: _UNCOVERED,
    # control: a simple question whose broad query is already well covered
    "lithium battery shipping": _LITHIUM,
}


@dataclass
class HardQuery:
    label: str
    broad: str              # single-shot / first-pass query
    sub_queries: list[str]  # reformulations the agent issues on weak coverage
    hard: bool = True       # False = control (should stay single-shot)


_QUERIES = [
    HardQuery(
        label="drone -> Germany",
        broad="sending a drone to Germany, any restrictions?",
        sub_queries=[SUB_LITHIUM, SUB_ELECTRONICS],
    ),
    HardQuery(
        label="hoverboard -> Germany",
        broad="can I ship a hoverboard to Germany?",
        sub_queries=[SUB_LITHIUM, SUB_ELECTRONICS],
    ),
    HardQuery(
        label="radio transmitter abroad",
        broad="shipping a vintage radio transmitter overseas",
        sub_queries=[SUB_ELECTRONICS, SUB_RADIO],  # one sub-area stays uncovered
    ),
    HardQuery(
        label="lithium battery (control)",
        broad="lithium battery shipping",  # already strong on the first pass
        sub_queries=[],
        hard=False,
    ),
]

# Per-query bound: broad pass + up to 2 reformulated sub-areas.
_EVAL_MAX_RETRIEVALS = 3


@dataclass
class QueryReport:
    label: str
    hard: bool
    covered_before: bool
    top_before: float
    covered_after: bool
    top_after: float
    uncovered_subareas: int
    single_shot_retrievals: int
    agentic_retrievals: int
    single_shot_steps: int
    agentic_steps: int
    decisions: list[str] = field(default_factory=list)

    @property
    def grounding_improved(self) -> bool:
        # Partial refusal (weak) -> grounded (covered) for the same question.
        return self.covered_after and not self.covered_before

    @property
    def added_retrievals(self) -> int:
        return self.agentic_retrievals - self.single_shot_retrievals

    @property
    def added_steps(self) -> int:
        return self.agentic_steps - self.single_shot_steps


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


def _retrieve_turn(query: str) -> ToolCallResult:
    return ToolCallResult(
        kind="tool_calls",
        calls=[ToolCall(id=query, name="retrieve_rag", arguments={"query": query})],
    )


def _embedding() -> _VecEmbedding:
    mapping = dict(_QUERY_VECTORS)
    return _VecEmbedding(mapping)


async def _seeded_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    await store.add([
        StoredChunk(text="Lithium batteries: declare watt-hours; ground only.",
                    source="hazmat", chunk_index=0, embedding=_LITHIUM),
        StoredChunk(text="Germany import: CE marking and customs forms for electronics.",
                    source="customs-de", chunk_index=1, embedding=_ELECTRONICS),
    ])
    return store


def _count_retrievals(decisions: list[str]) -> int:
    return sum(
        1 for d in decisions
        if d.startswith("agent:retrieve:") and d.split(":")[-1].isdigit()
    )


async def _coverage(query: str, embed, store) -> tuple[bool, float]:
    results, _mode = await retrieve_auto(query, embed, store, top_k=3)
    cov = coverage_of(list(results))
    return cov.covered, cov.top_score


async def _eval_query(q: HardQuery) -> QueryReport:
    embed = _embedding()
    store = await _seeded_store()

    # Coverage the broad single-shot query would see.
    covered_before, top_before = await _coverage(q.broad, embed, store)

    # Best coverage across the agent's reformulated sub-areas (or the broad query
    # itself for the control, which is already strong).
    sub_coverages = [await _coverage(s, embed, store) for s in q.sub_queries]
    if sub_coverages:
        covered_after = any(c for c, _ in sub_coverages)
        top_after = max(t for _, t in sub_coverages)
        uncovered = sum(1 for c, _ in sub_coverages if not c)
    else:
        covered_after, top_after, uncovered = covered_before, top_before, 0

    # ── Single-shot path: one retrieval, then answer. ──
    single = await run_agent(
        q.broad, {}, registry=_StubRegistry(), llm_router=_router(
            ScriptedToolCallingClient([
                _retrieve_turn(q.broad), ToolCallResult(kind="final", text="answer"),
            ])
        ),
        embedding_provider=_embedding(), vector_store=await _seeded_store(),
        max_retrievals=1,
    )

    # ── Agentic path: broad -> reformulate on weak coverage -> sub-areas -> answer. ──
    if q.sub_queries:
        turns = [_retrieve_turn(q.broad)]
        turns += [_retrieve_turn(s) for s in q.sub_queries]
        turns.append(ToolCallResult(kind="final", text="answer"))
        agentic = await run_agent(
            q.broad, {},
            registry=_StubRegistry(),
            llm_router=_router(ScriptedToolCallingClient(turns)),
            embedding_provider=_embedding(), vector_store=await _seeded_store(),
            max_retrievals=_EVAL_MAX_RETRIEVALS,
        )
        agentic_retrievals = _count_retrievals(agentic.decisions)
        agentic_steps = len(agentic.steps)
        decisions = agentic.decisions
    else:
        # Control: no re-retrieval scripted — it stays single-shot.
        agentic_retrievals = _count_retrievals(single.decisions)
        agentic_steps = len(single.steps)
        decisions = single.decisions

    return QueryReport(
        label=q.label, hard=q.hard,
        covered_before=covered_before, top_before=top_before,
        covered_after=covered_after, top_after=top_after,
        uncovered_subareas=uncovered,
        single_shot_retrievals=_count_retrievals(single.decisions),
        agentic_retrievals=agentic_retrievals,
        single_shot_steps=len(single.steps), agentic_steps=agentic_steps,
        decisions=decisions,
    )


async def run_eval() -> list[QueryReport]:
    """Run every query through both strategies; return structured reports."""
    return [await _eval_query(q) for q in _QUERIES]


def _fmt_bool(b: bool) -> str:
    return "yes" if b else "NO "


def main() -> int:
    reports = asyncio.run(run_eval())

    print("Agentic re-retrieval comparison eval")
    print("=" * 82)
    print(f"{'query':<27}{'cover@1':>10}{'cover@N':>10}{'grounded':>13}"
          f"{'+retr':>7}{'+steps':>8}")
    print("-" * 82)
    for r in reports:
        before = f"{_fmt_bool(r.covered_before)} {r.top_before:.2f}"
        after = f"{_fmt_bool(r.covered_after)} {r.top_after:.2f}"
        improved = "refuse>grnd" if r.grounding_improved else ("same" if r.hard else "n/a")
        print(f"{r.label:<27}{before:>10}{after:>10}{improved:>13}"
              f"{r.added_retrievals:>7}{r.added_steps:>8}")
    print("-" * 82)

    hard = [r for r in reports if r.hard]
    improved = [r for r in hard if r.grounding_improved]
    added_retr = sum(r.added_retrievals for r in hard)
    added_steps = sum(r.added_steps for r in hard)
    uncovered = sum(r.uncovered_subareas for r in hard)

    print()
    print("Headline")
    print(f"  hard compound queries              : {len(hard)}")
    print(f"  grounding improved (refusal->grounded): "
          f"{len(improved)}/{len(hard)}")
    print(f"  added cost (extra retrievals)      : +{added_retr} retrievals, "
          f"+{added_steps} agent steps total")
    print(f"  honestly-flagged uncovered sub-areas: {uncovered}")
    control = [r for r in reports if not r.hard]
    stayed = all(r.added_retrievals == 0 for r in control)
    print(f"  control query stayed single-shot   : {_fmt_bool(stayed).strip()}")
    print()

    ok = len(improved) == len(hard) and len(hard) > 0 and stayed
    if ok:
        print(f"PASS - agentic re-retrieval lifted {len(improved)}/{len(hard)} hard "
              f"queries from a partial refusal to a grounded answer "
              f"for +{added_retr} retrievals.")
        return 0
    print("FAIL - agentic path did not improve grounding on all hard queries.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
