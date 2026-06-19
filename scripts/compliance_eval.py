"""
Compliance flow eval — does the audit trail tell the whole story, and does the
UC2 critic earn its cost?

Runs a few representative shipments through ``check_compliance`` and prints, per
case: the advisory verdict, the findings (flag / info / unverified), and the full
decision trail. It contrasts the critic OFF (deterministic only) vs ON
(model-in-the-loop) on a compound case (a drone with a lithium battery into
Brazil), showing the critic surface a destination-specific area the four fixed
areas under-cover — while an uncovered proposal stays an honest ``unverified``
finding, never a fabricated flag.

Everything is deterministic and KEYLESS: a fixed-vector embedding controls
coverage exactly, the ``ScriptedToolCallingClient`` stands in for a native
tool-calling model, and the summary runs through ``EchoClient``.

Usage:
    python scripts/compliance_eval.py
"""

from __future__ import annotations

import asyncio

from app.agents.compliance.models import Shipment
from app.agents.compliance.service import check_compliance
from app.core.audit import InMemoryAuditSink
from app.llm.client import (
    EchoClient,
    ScriptedToolCallingClient,
    ToolCall,
    ToolCallResult,
)
from app.llm.router import TASK_FALLBACK, TASK_REASONING, TASK_SYNTHESIS, LLMRouter
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import InMemoryVectorStore, StoredChunk

# ── Deterministic coverage substrate ─────────────────────────────────────────
# Map specific area-query *substrings* to one-hot vectors so coverage is exact and
# reproducible. Anything unmapped embeds to a weak/orthogonal vector (uncovered).

_COVERED = [1.0, 0.0]
_WEAK = [0.0, 1.0]


class _SubstringEmbedding(EmbeddingProvider):
    """Covered if the text contains any 'covered' keyword, else weak."""

    def __init__(self, covered_keywords: tuple[str, ...]):
        self._covered = covered_keywords

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            low = t.lower()
            out.append(_COVERED if any(k in low for k in self._covered) else _WEAK)
        return out

    @property
    def dimensions(self) -> int:
        return 2


def _router(reasoning=None) -> LLMRouter:
    echo = EchoClient()
    return LLMRouter(
        clients={
            TASK_REASONING: reasoning or echo,
            TASK_SYNTHESIS: echo,
            TASK_FALLBACK: echo,
        },
        fallback=echo,
    )


async def _store(*vectors: list[float]) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    await store.add([
        StoredChunk(text=f"compliance KB chunk {i}", source=f"compliance/doc-{i}.md",
                    chunk_index=i, embedding=list(v))
        for i, v in enumerate(vectors)
    ])
    return store


def _print(title: str, result) -> None:
    print(f"\n=== {title} ===")
    print(f"verdict: {result.verdict}   critique_rounds: {result.critique_rounds}   "
          f"provider: {result.provider}")
    print("findings:")
    for f in result.findings:
        print(f"  - [{f.status:<10}] {f.kind:<13} {f.area}")
    print("decision trail:")
    print("  " + " · ".join(result.decisions))


async def main() -> None:
    # Case 1 — well-grounded domestic shipment, no concerns → advisory.
    r1 = await check_compliance(
        Shipment("US", "US", declared_value_usd=20, description="a hardcover book"),
        llm_router=_router(),
        embedding_provider=_SubstringEmbedding(("compliance kb",)),  # everything covered
        vector_store=await _store(_COVERED),
    )
    _print("Case 1: domestic book (clean)", r1)

    # Case 2 — international power bank, empty KB → structural flags + honest gaps.
    r2 = await check_compliance(
        Shipment("US", "DE", declared_value_usd=0, description="20000mAh power bank"),
        llm_router=_router(),
        embedding_provider=_SubstringEmbedding(()),  # nothing covered
        vector_store=InMemoryVectorStore(),
    )
    _print("Case 2: intl power bank, empty KB (flags + unverified)", r2)

    # Case 3 — compound drone+lithium into Brazil. Critic OFF: the destination
    # drone rule is under-covered by the four fixed areas.
    drone = Shipment("US", "BR", declared_value_usd=600,
                     description="camera drone with lithium battery")
    covered_kb = _SubstringEmbedding(("lithium", "drone import"))
    r3_off = await check_compliance(
        drone, llm_router=_router(),
        embedding_provider=covered_kb,
        vector_store=await _store(_COVERED),
        critique_max_rounds=0,
    )
    _print("Case 3a: drone→BR, critic OFF", r3_off)

    # Case 3 — critic ON proposes the destination drone-import area; its query
    # mentions 'drone import' → grounded as a real finding the fixed pass missed.
    scripted = ScriptedToolCallingClient([
        ToolCallResult(kind="tool_calls", calls=[ToolCall(
            id="c1", name="propose_gaps",
            arguments={"areas": "destination_drone_import_rules",
                       "rationale": "drone into Brazil needs ANATEL homologation"})]),
    ])
    sink = InMemoryAuditSink()
    r3_on = await check_compliance(
        drone, llm_router=_router(scripted),
        embedding_provider=covered_kb,
        vector_store=await _store(_COVERED),
        critique_max_rounds=1,
        audit_sink=sink,
    )
    _print("Case 3b: drone→BR, critic ON (finds grounded gap)", r3_on)
    print(f"\naudit events emitted: {[e.event for e in sink.events]}")


if __name__ == "__main__":
    asyncio.run(main())
