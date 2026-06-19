"""Tests for the pure structural compliance rules (no LLM, no retrieval).

These rules read only the Shipment fields and are fully deterministic, so they
are asserted exactly — including the decision tags they emit.
"""

from __future__ import annotations

from app.agents.compliance.models import Shipment
from app.agents.compliance.structural import (
    RULE_COMMERCIAL_INVOICE,
    RULE_CUSTOMS_VALUE_MISSING,
    RULE_DANGEROUS_GOODS,
    RULE_RESTRICTED_ITEM,
    run_structural_checks,
)


def _check(shipment: Shipment, threshold: float = 2500.0):
    return run_structural_checks(shipment, value_threshold_usd=threshold)


# ── international is derived, never trusted from the client ────────────────────


def test_international_is_derived_from_countries():
    assert Shipment("US", "DE").international is True
    assert Shipment("US", "US").international is False
    assert Shipment("us", "US").international is False  # case-insensitive


# ── rule 1: international + no declared value → customs value missing ──────────


def test_customs_value_missing_flagged_for_international_without_value():
    findings, decisions = _check(Shipment("US", "DE", declared_value_usd=0))
    areas = {f.area for f in findings}
    assert RULE_CUSTOMS_VALUE_MISSING in areas
    assert f"compliance:structural:{RULE_CUSTOMS_VALUE_MISSING}" in decisions
    assert all(f.status == "flag" and f.kind == "structural" for f in findings)


def test_customs_value_not_flagged_domestic():
    findings, _ = _check(Shipment("US", "US", declared_value_usd=0))
    assert RULE_CUSTOMS_VALUE_MISSING not in {f.area for f in findings}


# ── rule 2: battery / power-bank keyword → dangerous goods ────────────────────


def test_dangerous_goods_flagged_on_battery_keyword():
    findings, decisions = _check(Shipment("US", "US", description="A 20000mAh power bank"))
    assert RULE_DANGEROUS_GOODS in {f.area for f in findings}
    assert f"compliance:structural:{RULE_DANGEROUS_GOODS}" in decisions


def test_dangerous_goods_not_flagged_without_keyword():
    findings, _ = _check(Shipment("US", "US", description="cotton t-shirts"))
    assert RULE_DANGEROUS_GOODS not in {f.area for f in findings}


# ── rule 3: high value on international → commercial invoice ───────────────────


def test_commercial_invoice_flagged_at_threshold():
    findings, _ = _check(Shipment("US", "DE", declared_value_usd=2500.0))
    assert RULE_COMMERCIAL_INVOICE in {f.area for f in findings}


def test_commercial_invoice_not_flagged_below_threshold_or_domestic():
    below, _ = _check(Shipment("US", "DE", declared_value_usd=100.0))
    assert RULE_COMMERCIAL_INVOICE not in {f.area for f in below}
    domestic, _ = _check(Shipment("US", "US", declared_value_usd=9999.0))
    assert RULE_COMMERCIAL_INVOICE not in {f.area for f in domestic}


# ── rule 4: restricted keyword → restriction review ───────────────────────────


def test_restricted_item_flagged_on_keyword():
    findings, decisions = _check(
        Shipment("US", "DE", description="a bottle of wine", declared_value_usd=50)
    )
    assert RULE_RESTRICTED_ITEM in {f.area for f in findings}
    assert f"compliance:structural:{RULE_RESTRICTED_ITEM}" in decisions


def test_clean_domestic_shipment_has_no_structural_flags():
    findings, decisions = _check(
        Shipment("US", "US", description="a hardcover book", declared_value_usd=20)
    )
    assert findings == []
    assert decisions == []
