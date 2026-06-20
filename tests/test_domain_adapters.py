"""Tests for the domain mock adapters (UC3) — pure, deterministic, keyless.

Classification, duty/landed-cost, carrier quoting, and document rendering are all
deterministic mock implementations of the ports, so they are asserted exactly.
"""

from __future__ import annotations

import pytest

from app.domain.adapters import default_providers
from app.domain.adapters.mock_carrier import MockCarrierAdapter
from app.domain.adapters.mock_classification import MockClassificationAdapter
from app.domain.adapters.mock_doc_renderer import MockDocRenderer
from app.domain.adapters.mock_duty import MockDutyRateAdapter
from app.domain.ports import (
    CarrierProvider,
    ClassificationProvider,
    DocRenderer,
    DutyRateProvider,
)

# ── default_providers satisfies the port contracts (runtime_checkable) ─────────


def test_default_providers_satisfy_ports():
    p = default_providers()
    assert isinstance(p.classification, ClassificationProvider)
    assert isinstance(p.duty, DutyRateProvider)
    assert isinstance(p.carrier, CarrierProvider)
    assert isinstance(p.doc_renderer, DocRenderer)


# ── classification ────────────────────────────────────────────────────────────


def test_classification_matches_known_goods():
    cands = MockClassificationAdapter().candidates("a 20000mAh power bank")
    assert cands[0].hs_code == "8507.60"
    assert cands[0].confidence > 0.9


def test_classification_unknown_returns_fallback():
    cands = MockClassificationAdapter().candidates("an inscrutable widget")
    assert len(cands) == 1
    assert cands[0].hs_code == "9999.99"
    assert cands[0].confidence < 0.5


def test_classification_ranks_by_confidence():
    cands = MockClassificationAdapter().candidates("drone with a camera")
    assert [c.confidence for c in cands] == sorted(
        (c.confidence for c in cands), reverse=True
    )


# ── duty / landed cost ────────────────────────────────────────────────────────


def test_duty_electronics_zero_duty_with_vat():
    q = MockDutyRateAdapter().rate("8507.60", "US", "DE", 1000.0)
    assert q.duty_pct == 0.0
    assert q.duty_usd == 0.0
    assert q.tax_label == "VAT"
    assert q.tax_usd == pytest.approx(190.0)        # 19% of (1000 + 0)
    assert q.total_landed_usd == pytest.approx(1190.0)


def test_duty_apparel_has_duty_and_tax():
    q = MockDutyRateAdapter().rate("6109.10", "CN", "DE", 1000.0)
    assert q.duty_pct == pytest.approx(0.12)
    assert q.duty_usd == pytest.approx(120.0)
    assert q.tax_usd == pytest.approx(212.8)        # 19% of (1000 + 120)
    assert q.total_landed_usd == pytest.approx(1332.8)


def test_duty_usmca_trade_preference_zeroes_duty():
    q = MockDutyRateAdapter().rate("6109.10", "US", "CA", 1000.0)
    assert q.duty_pct == 0.0                         # USMCA US -> CA
    assert q.trade_note
    assert q.tax_label == "GST"
    assert q.total_landed_usd == pytest.approx(1050.0)  # value + 0 duty + 5% GST


def test_duty_unknown_chapter_uses_default():
    q = MockDutyRateAdapter().rate("9999.99", "CN", "US", 500.0)
    assert q.duty_pct == pytest.approx(0.035)


# ── carrier ───────────────────────────────────────────────────────────────────


def test_carrier_domestic_quotes():
    quotes = MockCarrierAdapter().quotes("US", "US", 5.0)
    assert len(quotes) == 4
    economy = next(q for q in quotes if q.service == "Economy")
    assert economy.price_usd == pytest.approx(9.25)   # 6.50 + 0.55*5
    assert economy.estimated_days == 6


def test_carrier_international_surcharge_and_eta():
    economy = next(
        q for q in MockCarrierAdapter().quotes("US", "DE", 5.0) if q.service == "Economy"
    )
    assert economy.price_usd == pytest.approx(12.95)  # 9.25 * 1.4
    assert economy.estimated_days == 12


# ── doc renderer ──────────────────────────────────────────────────────────────


def _ctx(international: bool) -> dict:
    return {
        "origin": "US", "destination": "DE" if international else "US",
        "value_usd": 1000, "weight_lbs": 5, "description": "power bank",
        "hs_code": "8507.60", "international": international,
    }


def test_doc_renderer_domestic_packing_list_only():
    docs = MockDocRenderer().render(_ctx(international=False))
    assert [d.doc_type for d in docs] == ["packing_list"]


def test_doc_renderer_international_adds_customs_docs():
    docs = MockDocRenderer().render(_ctx(international=True))
    types = {d.doc_type for d in docs}
    assert types == {"packing_list", "commercial_invoice", "customs_declaration_cn23"}
    invoice = next(d for d in docs if d.doc_type == "commercial_invoice")
    assert invoice.fields["hs_code"] == "8507.60"
