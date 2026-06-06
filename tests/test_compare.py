"""Tests for the /api/v1/compare endpoint + compare_service.

Under the hermetic test profile the LLM is the EchoClient, whose reply is not
valid compare JSON — so the service takes its deterministic fallback path. That
makes the winner_id and comparison numbers fully predictable (they are rule-based
by design, "H": the LLM only ever writes prose), which is exactly what we assert.
No real LLM key is required.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.llm.client import create_llm_client
from app.llm.router import create_llm_router
from app.main import app
from app.schemas.compare import CompareOption, CompareRequest, ShipmentContext
from app.services import compare_service
from app.services.compare_service import (
    _build_fallback_dimensions,
    _clean_service_name,
    generate_compare_response,
)

# Two real-fact options: opt1 is cheaper + slower, opt2 is pricier + faster + guaranteed.
_OPT_CHEAP = CompareOption(
    id="ups-ground", carrier="UPS", service_name="UPS Ground", carrier_type="public",
    price_usd=10.0, arrival_date="2026-06-10", arrival_label="Wed, Jun 10",
    transit_days=5, guaranteed=False,
)
_OPT_FAST = CompareOption(
    id="fedex-2day", carrier="FedEx", service_name="FedEx 2Day", carrier_type="public",
    price_usd=25.0, arrival_date="2026-06-07", arrival_label="Sun, Jun 7",
    transit_days=2, guaranteed=True,
)


def _request(priority: str = "ontime") -> CompareRequest:
    return CompareRequest(
        shipment=ShipmentContext(
            item_description="ceramic mugs", origin_zip="90210",
            destination_zip="10001", deadline_date="2026-06-09", weight_lb=4.0,
        ),
        option_ids=[_OPT_CHEAP.id, _OPT_FAST.id],
        options=[_OPT_CHEAP, _OPT_FAST],
        selected_priority=priority,
    )


@pytest.fixture(autouse=True)
def _clear_compare_cache():
    compare_service._compare_cache.cache.clear()
    yield
    compare_service._compare_cache.cache.clear()


# ── Service level: deterministic fallback scenarios ──────────────────────────


async def test_generate_compare_response_all_four_scenarios_with_rule_winners():
    resp = await generate_compare_response(_request(), _request().options, create_llm_client())

    assert set(resp.scenarios) == {"ontime", "damage", "price", "speed"}
    # Winners are deterministic from the quote facts (not the LLM):
    assert resp.scenarios["price"].winner_id == "ups-ground"   # cheapest
    assert resp.scenarios["speed"].winner_id == "fedex-2day"   # fastest
    assert resp.scenarios["ontime"].winner_id == "fedex-2day"  # only guaranteed option
    # EchoClient isn't valid JSON → fallback narrative, but winners stay rule-based.
    assert resp.decision_path.answer == "fallback"
    assert "winner:rule" in resp.decision_path.tags
    assert resp.shipment_summary.startswith("ceramic mugs")
    assert "90210 → 10001" in resp.shipment_summary


async def test_compare_response_is_cached_on_second_call():
    req = _request()
    first = await generate_compare_response(req, req.options, create_llm_client())
    second = await generate_compare_response(req, req.options, create_llm_client())
    assert second is first  # identical request served from the in-memory cache


# ── Route level ──────────────────────────────────────────────────────────────


def test_compare_endpoint_returns_scenarios():
    app.state.llm_router = create_llm_router()
    client = TestClient(app)
    body = _request(priority="price").model_dump()
    resp = client.post("/api/v1/compare", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert set(data["scenarios"]) == {"ontime", "damage", "price", "speed"}
    assert data["scenarios"]["price"]["winner_id"] == "ups-ground"
    assert data["decision_path"]["tags"]


def test_compare_endpoint_rejects_single_option():
    app.state.llm_router = create_llm_router()
    client = TestClient(app)
    body = _request().model_dump()
    body["options"] = body["options"][:1]
    body["option_ids"] = body["option_ids"][:1]
    resp = client.post("/api/v1/compare", json=body)
    assert resp.status_code == 422  # min_length=2 enforced by the schema


# ── Pure helpers ─────────────────────────────────────────────────────────────


def test_clean_service_name_strips_redundant_carrier_prefix():
    assert _clean_service_name("FedEx", "FedEx Express Saver") == "Express Saver"
    assert _clean_service_name("USPS", "Priority Mail") == "Priority Mail"  # no prefix


def test_fallback_dimensions_pick_price_and_speed_winners():
    dims = {d.dimension: d for d in _build_fallback_dimensions([_OPT_CHEAP, _OPT_FAST])}
    assert dims["Price"].winner_id == "ups-ground"
    assert dims["Speed"].winner_id == "fedex-2day"
    assert dims["Reliability"].winner_id == "fedex-2day"  # exactly one guaranteed
