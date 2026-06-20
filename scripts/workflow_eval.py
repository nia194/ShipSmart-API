"""
Workflow durability eval (UC4) — does kill-and-resume actually work?

Runs the multi-agent workflow keyless and exercises the human-in-the-loop
lifecycle end to end:

  1. An unverified high-risk shipment SUSPENDS for review (awaiting_review).
  2. A "process restart" is simulated with the SQLite checkpointer: a BRAND-NEW
     orchestrator + checkpointer on the same file loads the suspended workflow and
     RESUMES it — proving durability across instances.
  3. ``cleared`` continues to documentation and completes; ``blocked`` terminates.

Everything is deterministic and keyless (mock domain adapters, EchoClient, a
fixed-vector embedding, an empty KB so the high-risk areas are unverified). Prints
PASS/FAIL and exits non-zero on failure.

Usage:
    python scripts/workflow_eval.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from app.domain.adapters import default_providers
from app.llm.client import EchoClient
from app.llm.router import TASK_FALLBACK, TASK_REASONING, TASK_SYNTHESIS, LLMRouter
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import InMemoryVectorStore
from app.workflow.checkpointer import InMemoryCheckpointer, SqliteCheckpointer
from app.workflow.engine import StateMachineEngine
from app.workflow.orchestrator import DurableWorkflow, WorkflowDeps
from app.workflow.review_queue import InMemoryReviewQueue
from app.workflow.state import WorkflowState

_HIGH_RISK = frozenset({"lithium_battery", "import_restriction"})


class _FixedEmbedding(EmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    @property
    def dimensions(self) -> int:
        return 3


def _router() -> LLMRouter:
    echo = EchoClient()
    return LLMRouter(
        clients={TASK_REASONING: echo, TASK_SYNTHESIS: echo, TASK_FALLBACK: echo},
        fallback=echo,
    )


def _deps(*, checkpointer, review_queue) -> WorkflowDeps:
    return WorkflowDeps(
        providers=default_providers(),
        llm_router=_router(),
        embedding_provider=_FixedEmbedding(),
        vector_store=InMemoryVectorStore(),  # empty → high-risk areas unverified
        checkpointer=checkpointer,
        review_queue=review_queue,
        high_risk_areas=_HIGH_RISK,
    )


def _shipment(wid: str) -> WorkflowState:
    return WorkflowState(
        workflow_id=wid, origin_country="US", destination_country="BR",
        declared_value_usd=600.0, weight_lbs=3.0,
        description="camera drone with lithium battery",
    )


def _check(label: str, ok: bool) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return ok


async def main() -> int:
    ok = True

    # 1) Interrupt + in-memory blocked resume.
    cp, rq = InMemoryCheckpointer(), InMemoryReviewQueue()
    wf = DurableWorkflow(engine=StateMachineEngine(), deps=_deps(checkpointer=cp, review_queue=rq))
    suspended = await wf.process(_shipment("wf-mem"))
    print("\n# In-memory: interrupt then block")
    ok &= _check("process suspends (awaiting_review)", suspended.status == "awaiting_review")
    ok &= _check("high-risk area enqueued for review", rq.peek("wf-mem") is not None)
    blocked = await wf.resume(wf.load("wf-mem"), determination="blocked", note="prohibited")
    ok &= _check("blocked → status blocked, no docs",
                 blocked.status == "blocked" and not blocked.documents)

    # 2) Durable kill-and-resume with SQLite + cleared.
    print("\n# SQLite: kill & resume, then clear")
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "wf.db")
        cp1 = SqliteCheckpointer(path)
        wf1 = DurableWorkflow(
            engine=StateMachineEngine(),
            deps=_deps(checkpointer=cp1, review_queue=InMemoryReviewQueue()),
        )
        await wf1.process(_shipment("wf-sql"))

        # Simulate a restart: brand-new checkpointer + orchestrator on the same file.
        cp2 = SqliteCheckpointer(path)
        reloaded = cp2.load("wf-sql")
        ok &= _check("suspended state survives 'restart'",
                     reloaded is not None and reloaded.status == "awaiting_review")
        wf2 = DurableWorkflow(
            engine=StateMachineEngine(),
            deps=_deps(checkpointer=cp2, review_queue=InMemoryReviewQueue()),
        )
        cleared = await wf2.resume(reloaded, determination="cleared", note="reviewed")
        ok &= _check("cleared → completed with documents on fresh instance",
                     cleared.status == "completed" and bool(cleared.documents))
        ok &= _check("trail records determination + resume",
                     "workflow:review:determination" in cleared.decisions
                     and "workflow:resume" in cleared.decisions)

    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
