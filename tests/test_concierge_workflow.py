"""Concierge → multi-agent workflow bridge tests (gated, keyless).

Proves the "compliance-workflows ON" path drives the existing DurableWorkflow for
international shipments — and, crucially, that it stays OFF (domestic / flags off /
no workflow supplied) so today's behavior is unchanged."""

from __future__ import annotations

import app.agents.concierge.service as svc
from app.agents.concierge.models import ConversationState
from app.agents.concierge.service import _should_run_workflow, run_concierge
from app.llm.client import EchoClient
from app.llm.router import TASK_FALLBACK, TASK_REASONING, TASK_SYNTHESIS, LLMRouter
from app.rag.embeddings import LocalHashEmbedding
from app.rag.vector_store import InMemoryVectorStore
from app.workflow.checkpointer import InMemoryCheckpointer
from app.workflow.factory import build_workflow
from app.workflow.review_queue import InMemoryReviewQueue


def _deps():
    echo = EchoClient()
    return {
        "llm_router": LLMRouter(
            clients={TASK_REASONING: echo, TASK_SYNTHESIS: echo, TASK_FALLBACK: echo},
            fallback=echo,
        ),
        "embedding_provider": LocalHashEmbedding(dims=16),
        "vector_store": InMemoryVectorStore(),
    }


def _build_wf(deps):
    return build_workflow(
        llm_router=deps["llm_router"],
        embedding_provider=deps["embedding_provider"],
        vector_store=deps["vector_store"],
        checkpointer=InMemoryCheckpointer(),
        review_queue=InMemoryReviewQueue(),
    )


def test_should_run_workflow_gating(monkeypatch):
    monkeypatch.setattr(svc.settings, "shipping_scope", "worldwide", raising=False)
    assert _should_run_workflow("compliance", {"destination_country": "BR"}) is True
    assert _should_run_workflow("quote", {"destination_country": "DE"}) is True
    assert _should_run_workflow(
        "quote", {"destination_country": "US", "origin_country": "US"},
    ) is False
    assert _should_run_workflow("tracking", {"destination_country": "BR"}) is False
    assert _should_run_workflow("compliance", {}) is False  # no destination country
    monkeypatch.setattr(svc.settings, "shipping_scope", "domestic", raising=False)
    assert _should_run_workflow("compliance", {"destination_country": "BR"}) is False


async def test_bridge_suspends_on_high_risk_unverified(monkeypatch):
    # Default high-risk areas + an empty RAG ⇒ unverified high-risk ⇒ human review.
    monkeypatch.setattr(svc.settings, "shipping_scope", "worldwide", raising=False)
    monkeypatch.setattr(
        svc.settings, "workflow_high_risk_areas", "lithium_battery,import_restriction",
        raising=False,
    )
    deps = _deps()
    wf = _build_wf(deps)
    state = ConversationState(
        slots={
            "destination_country": "BR",
            "description": "drone with lithium battery",
            "weight_lbs": 5.0,
        },
        intent="compliance",
    )
    res = await run_concierge(
        "is my drone with a lithium battery allowed to Brazil?", state, workflow=wf, **deps,
    )
    assert res.dispatched_to == "workflow"
    assert any(d.startswith("workflow:") for d in res.decisions)
    assert "review" in res.reply.lower()


async def test_bridge_completes_when_no_high_risk(monkeypatch):
    monkeypatch.setattr(svc.settings, "shipping_scope", "worldwide", raising=False)
    monkeypatch.setattr(svc.settings, "workflow_high_risk_areas", "", raising=False)
    deps = _deps()
    wf = _build_wf(deps)  # built AFTER monkeypatch ⇒ no interrupt
    state = ConversationState(
        slots={"destination_country": "DE", "description": "books", "weight_lbs": 3.0},
        intent="compliance",
    )
    res = await run_concierge("are books allowed into Germany?", state, workflow=wf, **deps)
    assert res.dispatched_to == "workflow"
    assert "workflow:complete" in res.decisions
    assert "Done" in res.reply


async def test_off_state_no_workflow_falls_back_to_compliance(monkeypatch):
    # The route supplies workflow=None when flags are off; an international compliance
    # shipment then takes the existing UC2 path — never the workflow.
    monkeypatch.setattr(svc.settings, "shipping_scope", "worldwide", raising=False)
    deps = _deps()
    state = ConversationState(
        slots={"destination_country": "BR", "description": "laptop"}, intent="compliance",
    )
    res = await run_concierge("is this compliant?", state, **deps)  # no workflow passed
    assert res.dispatched_to == "compliance"
    assert not any(d.startswith("workflow:") for d in res.decisions)
