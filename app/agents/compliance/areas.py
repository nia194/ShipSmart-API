"""
Compliance decomposition — the fixed investigation areas (deterministic).

A compliance check decomposes into four fixed areas, each grounded independently
through the shared :func:`app.rag.grounding.retrieve_area` primitive. The areas
are fixed so the deterministic path is reproducible; the UC2 critic may propose
ADDITIONAL areas on top (see ``critic.py``), but these four always run.

Queries are built deterministically from the shipment (description + destination)
so retrieval targets the relevant corpus without any LLM in the control flow.
"""

from __future__ import annotations

from app.agents.compliance.models import Shipment

# Fixed investigation areas. Order is stable for a reproducible decision trail.
AREA_LITHIUM_BATTERY = "lithium_battery"
AREA_CUSTOMS_DOCS = "customs_docs"
AREA_IMPORT_RESTRICTION = "import_restriction"
AREA_VALUE_THRESHOLD = "value_threshold"

FIXED_AREAS = (
    AREA_LITHIUM_BATTERY,
    AREA_CUSTOMS_DOCS,
    AREA_IMPORT_RESTRICTION,
    AREA_VALUE_THRESHOLD,
)


def _dest(shipment: Shipment) -> str:
    return (shipment.destination_country or "").strip().upper()


def decompose(shipment: Shipment) -> list[tuple[str, str]]:
    """Return ``[(area, query)]`` for the four fixed areas, enriched per shipment.

    The description is folded into each query so lexical/semantic retrieval can
    latch onto the specific goods; the destination is added to the import and
    value-threshold areas because those rules are country-specific.
    """
    desc = (shipment.description or "").strip()
    dest = _dest(shipment)
    desc_suffix = f" for: {desc}" if desc else ""

    return [
        (
            AREA_LITHIUM_BATTERY,
            f"lithium battery power bank dangerous goods shipping rules{desc_suffix}",
        ),
        (
            AREA_CUSTOMS_DOCS,
            f"customs documentation commercial invoice CN22 CN23 requirements{desc_suffix}",
        ),
        (
            AREA_IMPORT_RESTRICTION,
            f"import restrictions prohibited items destination {dest}{desc_suffix}".strip(),
        ),
        (
            AREA_VALUE_THRESHOLD,
            f"de minimis duty tax value threshold destination {dest}".strip(),
        ),
    ]
