"""
Structural compliance checks — pure, deterministic, no model, no retrieval.

These rules read only the :class:`~app.agents.compliance.models.Shipment` fields
and emit ``flag`` findings for structural facts that are true by construction
(e.g. an international shipment with no declared value cannot clear customs). They
run BEFORE any retrieval and never call an LLM, so they are fully testable and
reproducible. Grounded guidance for WHY each flag matters comes from the
investigation stage; these rules just establish the facts.

Each rule emits a decision tag ``compliance:structural:{rule}`` so the branch is
visible in the audit trail.
"""

from __future__ import annotations

from app.agents.compliance.models import Finding, Shipment

# Keyword sets are intentionally small and lexical. Detection is substring-based
# over a lowercased description — deterministic and dependency-free. Borderline /
# destination-specific cases (e.g. drones into a particular country) are LEFT for
# the grounded investigation + UC2 critic rather than hard-coded here.
_BATTERY_KEYWORDS = (
    "lithium", "li-ion", "li ion", "battery", "batteries", "power bank",
    "powerbank", "power-bank",
)
_RESTRICTED_KEYWORDS = (
    "weapon", "firearm", "ammunition", "explosive", "flammable", "aerosol",
    "hazmat", "hazardous", "alcohol", "wine", "liquor", "spirits", "tobacco",
    "cigarette", "cannabis", "cbd", "currency", "cash bills",
)

# Stable rule identifiers (used in finding.area and the decision tag).
RULE_CUSTOMS_VALUE_MISSING = "customs_value_missing"
RULE_DANGEROUS_GOODS = "dangerous_goods_declaration"
RULE_COMMERCIAL_INVOICE = "commercial_invoice"
RULE_RESTRICTED_ITEM = "restricted_item"


def _matches(description: str, keywords: tuple[str, ...]) -> str | None:
    """Return the first keyword found in ``description`` (lowercased), else None."""
    low = (description or "").lower()
    return next((kw for kw in keywords if kw in low), None)


def run_structural_checks(
    shipment: Shipment, *, value_threshold_usd: float,
) -> tuple[list[Finding], list[str]]:
    """Apply the deterministic structural rules.

    Returns ``(findings, decisions)``. Findings are all ``flag``/``structural``;
    decisions carry one ``compliance:structural:{rule}`` tag per fired rule.
    """
    findings: list[Finding] = []
    decisions: list[str] = []

    def flag(rule: str, detail: str) -> None:
        findings.append(
            Finding(area=rule, status="flag", kind="structural", detail=detail)
        )
        decisions.append(f"compliance:structural:{rule}")

    intl = shipment.international

    # 1) International with no declared value → cannot complete a customs declaration.
    if intl and shipment.declared_value_usd <= 0:
        flag(
            RULE_CUSTOMS_VALUE_MISSING,
            "International shipment is missing a positive declared value; a customs "
            "declaration requires a stated value for duty/tax assessment.",
        )

    # 2) Battery / power-bank keyword → dangerous-goods declaration likely required.
    battery_kw = _matches(shipment.description, _BATTERY_KEYWORDS)
    if battery_kw:
        flag(
            RULE_DANGEROUS_GOODS,
            f"Description mentions '{battery_kw}'. Lithium cells are regulated "
            "dangerous goods and may require a UN-numbered declaration, the Class 9 "
            "lithium mark, and carrier/state-of-charge handling.",
        )

    # 3) High declared value on an international shipment → commercial invoice.
    if intl and shipment.declared_value_usd >= value_threshold_usd:
        flag(
            RULE_COMMERCIAL_INVOICE,
            f"Declared value ${shipment.declared_value_usd:,.2f} meets the "
            f"${value_threshold_usd:,.0f} threshold; a commercial invoice "
            "(and possibly export filing) is typically required.",
        )

    # 4) Restricted/prohibited keyword → flag for restriction review.
    restricted_kw = _matches(shipment.description, _RESTRICTED_KEYWORDS)
    if restricted_kw:
        flag(
            RULE_RESTRICTED_ITEM,
            f"Description mentions '{restricted_kw}', which is commonly restricted "
            "or prohibited by carriers and/or destinations; verify eligibility "
            "before shipping.",
        )

    return findings, decisions
