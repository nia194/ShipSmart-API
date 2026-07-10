"""Deterministic apply-policy tests (Product Roadmap §6)."""

from __future__ import annotations

from app.agents.concierge.apply_policy import (
    decide_apply_policy,
    is_risky_field,
)
from app.schemas.typed_outputs import AssistantResponse, ShippingOptionResult


def test_high_confidence_plain_fields_auto_apply():
    assert (
        decide_apply_policy(
            intent="form_fill", field_paths=["origin.city", "destination.city"], confidence=0.95
        )
        == "auto"
    )


def test_medium_confidence_confirms():
    assert (
        decide_apply_policy(intent="form_fill", field_paths=["package.weight"], confidence=0.6)
        == "confirm"
    )


def test_low_confidence_does_not_mutate():
    assert (
        decide_apply_policy(intent="form_fill", field_paths=["origin.city"], confidence=0.2)
        == "none"
    )


def test_risky_field_always_confirms_even_at_high_confidence():
    for path in ("shipment.declared_value_usd", "package.hazmat", "customs_category", "hs_code"):
        assert (
            decide_apply_policy(intent="form_fill", field_paths=[path], confidence=0.99)
            == "confirm"
        ), path
    assert is_risky_field("shipment.declared_value") and not is_risky_field("origin.city")


def test_advisory_intents_never_mutate():
    for intent in ("recommendation", "policy_question", "compare_options", "tracking_question"):
        assert (
            decide_apply_policy(intent=intent, field_paths=["origin.city"], confidence=0.99)
            == "none"
        ), intent


def test_empty_patch_is_none():
    assert decide_apply_policy(intent="form_fill", field_paths=[], confidence=1.0) == "none"


# ── the additive contract stays valid + round-trips ───────────────────────────
def test_assistant_response_carries_the_roadmap_fields():
    resp = AssistantResponse(
        type="answer",
        message="Cheapest option is FedEx Ground.",
        intent="quote_search",
        apply_policy="none",
        confidence=0.9,
        result=ShippingOptionResult(
            label="Cheapest",
            quote_id="Q-100",
            carrier="FedEx",
            service_name="Ground",
            price_usd=42.50,
            transit_days=3,
        ),
    )
    assert resp.schema_version == "1"
    assert resp.result.type == "shipping_option" and resp.result.quote_id == "Q-100"
    # discriminated union round-trips from a dict (what the validator parses)
    reparsed = AssistantResponse.model_validate(resp.model_dump())
    assert reparsed.result.label == "Cheapest"


def test_old_f1_shape_still_validates():
    # Additive: a pre-roadmap response with none of the new fields is still valid.
    resp = AssistantResponse(type="refusal", message="no")
    assert resp.intent is None and resp.apply_policy == "none" and resp.result is None
