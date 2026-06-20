"""
Workflow nodes (UC3) — thin adapters from ``WorkflowState`` to the agents.

Each builder captures the dependencies a stage needs and returns a ``NodeFn``
(``state -> state``). Nodes own the workflow-level decision-tag vocabulary
(``workflow:*``) and the state read/writes; the agents own the computation. Nodes
mutate the state they are handed and return it — the engine hands parallel nodes
independent deep copies, so in-place mutation is safe.
"""

from __future__ import annotations

from app.agents import (
    classification_agent,
    documentation_agent,
    landed_cost_agent,
    routing_agent,
)
from app.agents.compliance import Shipment, check_compliance
from app.core.audit import AuditSink
from app.domain.ports import (
    CarrierProvider,
    ClassificationProvider,
    DocRenderer,
    DutyRateProvider,
)
from app.llm.router import LLMRouter
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import VectorStore
from app.workflow.engine import NodeFn
from app.workflow.state import ComplianceSummary, WorkflowState, _now

_LOW_CONFIDENCE = 0.5


def classification_node(provider: ClassificationProvider) -> NodeFn:
    async def _node(state: WorkflowState) -> WorkflowState:
        candidates, chosen = classification_agent.classify(
            state.description, provider=provider,
        )
        state.hs_candidates = candidates
        state.hs_code = chosen.hs_code
        state.hs_title = chosen.title
        state.decisions.append(f"workflow:classify:{chosen.hs_code}")
        if chosen.confidence < _LOW_CONFIDENCE:
            state.decisions.append("workflow:classify:low_confidence")
        state.updated_at = _now()
        return state

    return _node


def landed_cost_node(provider: DutyRateProvider) -> NodeFn:
    async def _node(state: WorkflowState) -> WorkflowState:
        quote = landed_cost_agent.estimate(
            state.hs_code, state.origin_country, state.destination_country,
            state.declared_value_usd, provider=provider,
        )
        state.landed_cost = quote
        state.decisions.append("workflow:landed_cost:computed")
        if quote.trade_note:
            state.decisions.append("workflow:landed_cost:trade_preference")
        state.updated_at = _now()
        return state

    return _node


def routing_node(provider: CarrierProvider) -> NodeFn:
    async def _node(state: WorkflowState) -> WorkflowState:
        quotes, recommended = routing_agent.recommend(
            state.origin_country, state.destination_country, state.weight_lbs,
            provider=provider,
        )
        state.carrier_quotes = quotes
        state.recommended_carrier = recommended
        if recommended is not None:
            state.decisions.append(
                f"workflow:routing:{recommended.carrier}:{recommended.service}"
            )
        else:
            state.decisions.append("workflow:routing:no_quotes")
        state.updated_at = _now()
        return state

    return _node


def compliance_node(
    *,
    llm_router: LLMRouter,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    audit_sink: AuditSink | None,
    critique_max_rounds: int | None,
) -> NodeFn:
    """Wrap the Phase 1 compliance flow as a workflow stage (incl. the UC2 critic)."""

    async def _node(state: WorkflowState) -> WorkflowState:
        result = await check_compliance(
            Shipment(
                origin_country=state.origin_country,
                destination_country=state.destination_country,
                declared_value_usd=state.declared_value_usd,
                weight_lbs=state.weight_lbs,
                description=state.description,
                category=state.category,
            ),
            llm_router=llm_router,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            audit_sink=audit_sink,
            critique_max_rounds=critique_max_rounds,
            request_id=state.request_id,
        )
        state.compliance = ComplianceSummary(
            verdict=result.verdict,
            summary=result.summary,
            flagged_areas=[f.area for f in result.findings if f.status == "flag"],
            unverified_areas=[f.area for f in result.findings if f.status == "unverified"],
            critique_rounds=result.critique_rounds,
            provider=result.provider,
        )
        # Fold the compliance/critique decision trail into the workflow trail.
        state.decisions.extend(result.decisions)
        state.updated_at = _now()
        return state

    return _node


def documentation_node(renderer: DocRenderer) -> NodeFn:
    async def _node(state: WorkflowState) -> WorkflowState:
        docs = documentation_agent.generate(
            {
                "origin": state.origin_country,
                "destination": state.destination_country,
                "value_usd": state.declared_value_usd,
                "weight_lbs": state.weight_lbs,
                "description": state.description,
                "hs_code": state.hs_code,
                "international": state.international,
            },
            renderer=renderer,
        )
        state.documents = docs
        state.decisions.append(f"workflow:docs:generated:{len(docs)}")
        state.updated_at = _now()
        return state

    return _node
