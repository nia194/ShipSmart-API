"""Build the typed AssistantResponse from the concierge result (Product Roadmap P1 wiring).

The concierge already decides everything the structured contract needs — a reply,
an intent, a clarification, dispatched worker, sources, and the decision path.
This maps that (deterministically) into the typed ``AssistantResponse`` the
product renders, so the frontend can stop reverse-engineering structure from
prose. Pure + rule-derived: intent comes from the classifier, apply_policy from
the policy table (``decide_apply_policy``) — never LLM self-report.

Emission is gated by ``settings.assistant_contract_v1`` at the call site; this
builder is always safe to call (additive, side-effect-free).
"""

from __future__ import annotations

from app.agents.concierge.apply_policy import decide_apply_policy
from app.schemas.advisor import ShippingAdvisorResponse
from app.schemas.concierge import ConciergeResponse
from app.schemas.typed_outputs import (
    AssistantAudit,
    AssistantResponse,
    MissingInfoResult,
    NextQuestion,
    PolicyAnswerResult,
    SourceCitation,
    ToolCallTrace,
)

# Concierge intent → the roadmap §6 AssistantIntent vocabulary.
_INTENT_MAP = {
    "quote": "quote_search",
    "compliance": "policy_question",
    "tracking": "tracking_question",
    "advice": "recommendation",
}


def _map_intent(concierge_intent: str | None) -> str | None:
    if concierge_intent is None:
        return "general_question"
    return _INTENT_MAP.get(concierge_intent, "general_question")


def _sources(raw: list[dict]) -> list[SourceCitation]:
    out: list[SourceCitation] = []
    for s in raw or []:
        if isinstance(s, dict) and s.get("source"):
            out.append(
                SourceCitation(
                    source=str(s["source"]),
                    chunk_index=s.get("chunk_index"),
                    score=s.get("score"),
                )
            )
    return out


def build_assistant_response(response: ConciergeResponse) -> AssistantResponse:
    """Deterministically map a concierge response to the typed AssistantResponse."""
    intent = _map_intent(response.state.intent)
    gathering = bool(response.clarification) or response.state.status == "gathering"
    citations = _sources(response.sources)

    # Rule-derived confidence: a classified intent is more trustworthy than a
    # fallthrough; a still-gathering turn is inherently less certain.
    confidence = 0.7 if response.state.intent else 0.3
    if gathering:
        confidence = min(confidence, 0.4)

    # apply_policy is never a mutation here — the concierge turn carries no form
    # patch — so field_paths is empty and the gate resolves to "none".
    apply_policy = decide_apply_policy(intent=intent, field_paths=[], confidence=confidence)

    next_question = (
        NextQuestion(field="clarification", text=response.clarification)
        if response.clarification
        else None
    )

    # Typed result: an in-progress turn asks; a compliance turn is a sourced
    # policy answer; anything else is a plain answer (no typed card).
    result = None
    if gathering:
        result = MissingInfoResult(next_question=response.clarification or "")
    elif response.state.intent == "compliance":
        result = PolicyAnswerResult(answer=response.reply, sources=citations)

    return AssistantResponse(
        type="ask_followup" if response.clarification else "answer",
        message=response.reply,
        sources=citations,
        intent=intent,
        apply_policy=apply_policy,
        confidence=confidence,
        next_question=next_question,
        result=result,
        audit=AssistantAudit(provider=response.provider, selection_method="concierge"),
    )


def build_from_shipping_advice(response: ShippingAdvisorResponse) -> AssistantResponse:
    """Map a shipping-advice response to the typed AssistantResponse (advisor surface).

    Shipping advice is advisory (never mutates the form): apply_policy is always
    ``none``, and the grounded answer becomes a sourced ``policy_answer`` result.
    """
    citations = _sources(response.sources)
    provider = response.decision_path.provider if response.decision_path else ""
    return AssistantResponse(
        type="answer",
        message=response.answer,
        sources=citations,
        intent="recommendation",
        apply_policy="none",
        confidence=0.7 if response.context_used else 0.5,
        result=PolicyAnswerResult(answer=response.answer, sources=citations),
        tool_calls=[ToolCallTrace(name=t) for t in response.tools_used],
        audit=AssistantAudit(provider=provider, selection_method="advisor"),
    )
