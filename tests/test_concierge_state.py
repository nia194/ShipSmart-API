"""Pure-reducer + extraction tests for the Conversational Concierge."""

from __future__ import annotations

from app.agents.concierge.extract import extract_deterministic
from app.agents.concierge.models import ConversationState
from app.agents.concierge.state import fold_turn, missing_required


def test_empty_never_overwrites_nonempty():
    s = fold_turn(
        ConversationState(slots={"origin": "Atlanta, GA"}),
        {"origin": "", "weight_lbs": 12.0},
    )
    assert s.slots["origin"] == "Atlanta, GA"
    assert s.slots["weight_lbs"] == 12.0


def test_newest_nonempty_wins():
    s = fold_turn(
        ConversationState(slots={"destination": "Seattle, WA"}),
        {"destination": "Boston, MA"},
    )
    assert s.slots["destination"] == "Boston, MA"


def test_equivalent_restatement_is_noop():
    s = fold_turn(ConversationState(slots={"origin": "Atlanta, GA"}), {"origin": "atlanta"})
    assert s.slots["origin"] == "Atlanta, GA"  # city head matches → kept


def test_extract_route_weight_intent():
    intent, slots = extract_deterministic("ship from Atlanta to Seattle, 12 lb")
    assert slots["origin"].lower().startswith("atlanta")
    assert slots["destination"].lower().startswith("seattle")
    assert slots["weight_lbs"] == 12.0
    assert intent == "quote"


def test_extract_compliance_intent_and_country():
    intent, slots = extract_deterministic("is my shipment to Brazil compliant with customs?")
    assert intent == "compliance"
    assert slots["destination_country"] == "BR"
    # "my shipment" must NOT be mis-parsed as an origin
    assert "origin" not in slots


def test_missing_required_for_quote():
    assert "weight_lbs" in missing_required({"origin": "A", "destination": "B"}, "quote")
    assert missing_required({"origin": "A", "destination": "B", "weight_lbs": 5}, "quote") == []
