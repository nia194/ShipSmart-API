"""
Workflow orchestrator (UC3 + UC4) — the multi-agent stage graph + durability.

``DurableWorkflow`` wires the specialist agents into the fixed stage sequence and
runs them through the injected ``WorkflowEngine``:

    classify → (landed-cost ‖ routing) → compliance (+UC2 critic)
             → [interrupt if unverified high-risk] → documentation

Landed-cost and routing run in parallel; everything else is sequential. Control
flow lives here in plain, auditable Python — never in a model.

UC4 (durability + human-in-the-loop): after compliance, if a **high-risk area is
unverified**, the workflow checkpoints its state, enqueues a review, tags
``workflow:interrupt:human_review``, and suspends (``awaiting_review``). An officer
later resolves it via :meth:`resume`, which loads the checkpoint, injects the
human determination *in code*, records a human audit event, and either terminates
(``blocked``) or continues to documentation (``cleared``). Resume works on a fresh
orchestrator instance — the checkpointer is the only shared state — so it survives
a process restart.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.audit import AuditEvent, AuditSink
from app.domain.adapters import DomainProviders
from app.llm.router import LLMRouter
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import VectorStore
from app.workflow.checkpointer import InMemoryCheckpointer, WorkflowCheckpointer
from app.workflow.engine import WorkflowEngine
from app.workflow.nodes import (
    classification_node,
    compliance_node,
    documentation_node,
    landed_cost_node,
    routing_node,
)
from app.workflow.review_queue import (
    Determination,
    InMemoryReviewQueue,
    ReviewItem,
    ReviewQueue,
)
from app.workflow.state import WorkflowState, _now


@dataclass(frozen=True)
class WorkflowDeps:
    """Everything the workflow needs, injected (ports + LLM/RAG + durability)."""

    providers: DomainProviders
    llm_router: LLMRouter
    embedding_provider: EmbeddingProvider
    vector_store: VectorStore
    audit_sink: AuditSink | None = None
    compliance_critique_max_rounds: int | None = None
    # Additive explicit-compliance feature switch. False ⇒ skip the compliance
    # stage entirely; the workflow runs the rest of the graph straight through.
    # The high-risk HITL interrupt cannot fire when this is off (it depends on the
    # explicit pass's findings) — intentional.
    compliance_explicit_enabled: bool = True
    # UC4. ``high_risk_areas`` empty ⇒ never interrupt (the Phase 2 behavior).
    checkpointer: WorkflowCheckpointer | None = None
    review_queue: ReviewQueue | None = None
    high_risk_areas: frozenset[str] = frozenset()


class DurableWorkflow:
    """Sequences the specialist agents into the workflow graph, with UC4 HITL."""

    def __init__(self, *, engine: WorkflowEngine, deps: WorkflowDeps) -> None:
        self._engine = engine
        self._audit_sink = deps.audit_sink
        self._checkpointer = deps.checkpointer or InMemoryCheckpointer()
        self._review_queue = deps.review_queue or InMemoryReviewQueue()
        self._high_risk_areas = deps.high_risk_areas
        self._compliance_explicit_enabled = deps.compliance_explicit_enabled
        self._classify = classification_node(deps.providers.classification)
        self._landed_cost = landed_cost_node(deps.providers.duty)
        self._routing = routing_node(deps.providers.carrier)
        self._compliance = compliance_node(
            llm_router=deps.llm_router,
            embedding_provider=deps.embedding_provider,
            vector_store=deps.vector_store,
            audit_sink=deps.audit_sink,
            critique_max_rounds=deps.compliance_critique_max_rounds,
        )
        self._documentation = documentation_node(deps.providers.doc_renderer)

    def load(self, workflow_id: str) -> WorkflowState | None:
        """Return the persisted state for a workflow (or None if unknown)."""
        return self._checkpointer.load(workflow_id)

    async def process(self, state: WorkflowState) -> WorkflowState:
        """Run the graph; suspend for human review on an unverified high-risk gap."""
        state.status = "running"
        state.decisions.append("workflow:start")

        state = await self._engine.run_step(state, self._classify)
        state = await self._engine.run_parallel(state, [self._landed_cost, self._routing])

        if self._compliance_explicit_enabled:
            state = await self._engine.run_step(state, self._compliance)

            # ── UC4 interrupt: unverified high-risk area → human review ────────
            gaps = self._high_risk_gaps(state)
            if gaps:
                return self._interrupt(state, gaps)
        else:
            # Explicit compliance feature off: skip the hard pass (and therefore the
            # HITL interrupt). The normal flow's lightweight checks still applied
            # upstream; record the deliberate skip on the trail.
            state.decisions.append("workflow:compliance:explicit_skipped")

        state = await self._engine.run_step(state, self._documentation)
        return self._finish(state)

    async def resume(
        self, state: WorkflowState, *, determination: Determination, note: str = "",
    ) -> WorkflowState:
        """Resume a suspended workflow with the officer's determination.

        ``blocked`` terminates the workflow; ``cleared`` continues to documentation
        and completes. The determination is injected in code (the model never
        authored it), recorded as a human audit event, and checkpointed.
        """
        state.officer_determination = determination
        state.officer_note = note
        state.decisions.append("workflow:review:determination")
        self._emit(
            "workflow:review:determination", actor="human", actor_name="officer",
            state=state, payload={"determination": determination, "note": note},
        )
        self._review_queue.resolve(state.workflow_id, determination, note)
        state.decisions.append("workflow:resume")

        if determination == "blocked":
            state.status = "blocked"
            state.decisions.append("workflow:blocked")
            state.updated_at = _now()
            self._checkpointer.save(state)
            return state

        state = await self._engine.run_step(state, self._documentation)
        return self._finish(state)

    # ── internals ─────────────────────────────────────────────────────────────

    def _high_risk_gaps(self, state: WorkflowState) -> list[str]:
        if not self._high_risk_areas or state.compliance is None:
            return []
        return sorted(set(state.compliance.unverified_areas) & self._high_risk_areas)

    def _interrupt(self, state: WorkflowState, gaps: list[str]) -> WorkflowState:
        state.status = "awaiting_review"
        state.pending_review_areas = gaps
        state.decisions.append("workflow:interrupt:human_review")
        question = (
            f"Unverified high-risk area(s) for shipment "
            f"{state.origin_country}->{state.destination_country}: "
            f"{', '.join(gaps)}. Human review required."
        )
        self._review_queue.enqueue(
            ReviewItem(workflow_id=state.workflow_id, question=question, high_risk_areas=gaps)
        )
        self._emit(
            "workflow:interrupt:human_review", actor="agent", actor_name="workflow",
            state=state, payload={"high_risk_areas": gaps},
        )
        state.updated_at = _now()
        self._checkpointer.save(state)
        return state

    def _finish(self, state: WorkflowState) -> WorkflowState:
        state.status = "completed"
        state.decisions.append("workflow:complete")
        state.updated_at = _now()
        self._checkpointer.save(state)
        return state

    def _emit(
        self, event: str, *, actor: str, actor_name: str,
        state: WorkflowState, payload: dict,
    ) -> None:
        if self._audit_sink is None:
            return
        try:
            self._audit_sink.emit(
                AuditEvent(
                    event=event, actor=actor, actor_name=actor_name,
                    workflow_id=state.workflow_id, request_id=state.request_id,
                    payload=payload,
                )
            )
        except Exception:  # noqa: BLE001 - auditing is best-effort, never blocks
            pass
