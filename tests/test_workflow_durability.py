"""UC4 tests — the durable human-in-the-loop interrupt/resume lifecycle.

Hermetic + keyless. An empty vector store makes the high-risk areas unverified
(triggering the interrupt); a seeded store covers them (no interrupt). The
load-bearing test resumes on a BRAND-NEW orchestrator instance sharing only the
checkpointer — proving durable resume survives a process restart.
"""

from __future__ import annotations

from app.core.audit import InMemoryAuditSink
from app.domain.adapters import default_providers
from app.llm.client import EchoClient
from app.llm.router import TASK_FALLBACK, TASK_REASONING, TASK_SYNTHESIS, LLMRouter
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import InMemoryVectorStore, StoredChunk
from app.workflow.checkpointer import InMemoryCheckpointer
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


def _deps(*, store, high_risk=_HIGH_RISK, checkpointer=None, review_queue=None, audit=None):
    return WorkflowDeps(
        providers=default_providers(),
        llm_router=_router(),
        embedding_provider=_FixedEmbedding(),
        vector_store=store,
        audit_sink=audit,
        checkpointer=checkpointer or InMemoryCheckpointer(),
        review_queue=review_queue or InMemoryReviewQueue(),
        high_risk_areas=high_risk,
    )


def _state(wid="wf-1") -> WorkflowState:
    return WorkflowState(
        workflow_id=wid, origin_country="US", destination_country="BR",
        declared_value_usd=600.0, weight_lbs=3.0,
        description="camera drone with lithium battery",
    )


async def _seeded_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    await store.add([
        StoredChunk(text="KB chunk", source="compliance/doc.md", chunk_index=0,
                    embedding=[1.0, 0.0, 0.0]),
    ])
    return store


# ── interrupt ─────────────────────────────────────────────────────────────────


async def test_interrupt_on_unverified_high_risk():
    cp, rq = InMemoryCheckpointer(), InMemoryReviewQueue()
    wf = DurableWorkflow(
        engine=StateMachineEngine(),
        deps=_deps(store=InMemoryVectorStore(), checkpointer=cp, review_queue=rq),
    )
    state = await wf.process(_state())

    assert state.status == "awaiting_review"
    assert set(state.pending_review_areas) <= _HIGH_RISK
    assert "lithium_battery" in state.pending_review_areas
    assert "workflow:interrupt:human_review" in state.decisions
    assert state.documents == []                       # suspended before documentation
    assert cp.load("wf-1").status == "awaiting_review"  # checkpointed
    assert rq.peek("wf-1") is not None                  # enqueued for review


async def test_no_interrupt_when_high_risk_set_empty():
    # Empty high-risk set ⇒ never interrupt (this is why Phase 2 tests still pass).
    wf = DurableWorkflow(
        engine=StateMachineEngine(),
        deps=_deps(store=InMemoryVectorStore(), high_risk=frozenset()),
    )
    state = await wf.process(_state())
    assert state.status == "completed"


async def test_no_interrupt_when_areas_covered():
    wf = DurableWorkflow(
        engine=StateMachineEngine(), deps=_deps(store=await _seeded_store()),
    )
    state = await wf.process(_state())
    assert state.status == "completed"
    assert state.documents  # reached documentation


# ── resume (load-bearing: new orchestrator instance, shared checkpointer) ──────


async def test_resume_cleared_completes_on_fresh_orchestrator_instance():
    cp, rq = InMemoryCheckpointer(), InMemoryReviewQueue()
    # Process to suspension on one instance.
    await DurableWorkflow(
        engine=StateMachineEngine(),
        deps=_deps(store=InMemoryVectorStore(), checkpointer=cp, review_queue=rq),
    ).process(_state())

    # Resume on a BRAND-NEW instance sharing only the checkpointer + queue.
    wf2 = DurableWorkflow(
        engine=StateMachineEngine(),
        deps=_deps(store=InMemoryVectorStore(), checkpointer=cp, review_queue=rq),
    )
    loaded = wf2.load("wf-1")
    resumed = await wf2.resume(loaded, determination="cleared", note="reviewed")

    assert resumed.status == "completed"
    assert resumed.officer_determination == "cleared"
    assert resumed.documents                                   # documentation ran on resume
    assert "workflow:review:determination" in resumed.decisions
    assert "workflow:resume" in resumed.decisions
    assert "workflow:complete" in resumed.decisions
    assert rq.peek("wf-1").status == "resolved"


async def test_resume_blocked_terminates():
    cp, rq = InMemoryCheckpointer(), InMemoryReviewQueue()
    await DurableWorkflow(
        engine=StateMachineEngine(),
        deps=_deps(store=InMemoryVectorStore(), checkpointer=cp, review_queue=rq),
    ).process(_state())

    wf2 = DurableWorkflow(
        engine=StateMachineEngine(),
        deps=_deps(store=InMemoryVectorStore(), checkpointer=cp, review_queue=rq),
    )
    resumed = await wf2.resume(wf2.load("wf-1"), determination="blocked", note="prohibited")

    assert resumed.status == "blocked"
    assert resumed.documents == []                  # no documentation when blocked
    assert "workflow:blocked" in resumed.decisions


async def test_human_determination_is_audited():
    cp, rq, audit = InMemoryCheckpointer(), InMemoryReviewQueue(), InMemoryAuditSink()
    wf = DurableWorkflow(
        engine=StateMachineEngine(),
        deps=_deps(store=InMemoryVectorStore(), checkpointer=cp, review_queue=rq, audit=audit),
    )
    await wf.process(_state())
    await wf.resume(wf.load("wf-1"), determination="cleared", note="ok")

    events = {e.event for e in audit.events}
    assert "workflow:interrupt:human_review" in events
    human = [e for e in audit.events if e.event == "workflow:review:determination"]
    assert human and human[0].actor == "human" and human[0].actor_name == "officer"
