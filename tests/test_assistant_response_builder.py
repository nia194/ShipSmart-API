"""Concierge → typed AssistantResponse mapper tests (Product Roadmap P1 wiring)."""

from __future__ import annotations

from app.agents.concierge.assistant_response import (
    build_assistant_response,
    build_from_shipping_advice,
)
from app.schemas.advisor import DecisionPath, ShippingAdvisorResponse
from app.schemas.concierge import ConciergeResponse, ConciergeState


def _response(**over) -> ConciergeResponse:
    base = dict(
        reply="Here's what I found.",
        state=ConciergeState(intent="quote", status="ready"),
        provider="openai",
        sources=[],
        decisions=[],
        clarification=None,
    )
    base.update(over)
    return ConciergeResponse(**base)


def test_maps_concierge_intent_to_roadmap_vocabulary():
    assert build_assistant_response(_response(state=ConciergeState(intent="quote"))).intent == (
        "quote_search"
    )
    assert build_assistant_response(
        _response(state=ConciergeState(intent="compliance"))
    ).intent == "policy_question"
    assert build_assistant_response(
        _response(state=ConciergeState(intent="tracking"))
    ).intent == "tracking_question"
    assert build_assistant_response(
        _response(state=ConciergeState(intent="advice"))
    ).intent == "recommendation"
    assert build_assistant_response(
        _response(state=ConciergeState(intent=None))
    ).intent == "general_question"


def test_apply_policy_is_none_for_a_conversational_turn():
    # No form patch is emitted here, so the deterministic gate never auto-mutates.
    assert build_assistant_response(_response()).apply_policy == "none"


def test_clarification_becomes_ask_followup_with_missing_info_result():
    resp = _response(
        state=ConciergeState(intent="quote", status="gathering"),
        clarification="What's the destination ZIP?",
    )
    a = build_assistant_response(resp)
    assert a.type == "ask_followup"
    assert a.next_question is not None and "ZIP" in a.next_question.text
    assert a.result is not None and a.result.type == "missing_info"


def test_compliance_turn_is_a_sourced_policy_answer():
    resp = _response(
        state=ConciergeState(intent="compliance", status="ready"),
        reply="Lithium batteries are dangerous goods.",
        sources=[{"source": "compliance/lithium-batteries-dangerous-goods.md", "score": 0.9}],
    )
    a = build_assistant_response(resp)
    assert a.result is not None and a.result.type == "policy_answer"
    assert a.result.sources and a.result.sources[0].source.startswith("compliance/")
    assert a.sources[0].score == 0.9


def test_audit_and_confidence_are_rule_derived():
    a = build_assistant_response(_response())
    assert a.audit is not None and a.audit.provider == "openai"
    assert a.audit.selection_method == "concierge"
    assert 0.0 <= a.confidence <= 1.0 and a.schema_version == "1"


# ── advisor surface mapper ────────────────────────────────────────────────────
def test_shipping_advice_maps_to_a_sourced_advisory_answer():
    advice = ShippingAdvisorResponse(
        answer="Ground is cheapest for a non-urgent box.",
        reasoning_summary="compared price + transit",
        tools_used=["get_quote_preview"],
        sources=[{"source": "policies/carrier-comparison.md", "score": 0.8}],
        context_used=True,
        decision_path=DecisionPath(provider="openai", tags=["agent:tool"]),
    )
    a = build_from_shipping_advice(advice)
    assert a.intent == "recommendation" and a.apply_policy == "none"  # advisory never mutates
    assert a.result is not None and a.result.type == "policy_answer"
    assert a.result.sources[0].source.startswith("policies/")
    assert a.tool_calls[0].name == "get_quote_preview"
    assert a.audit is not None and a.audit.selection_method == "advisor"
