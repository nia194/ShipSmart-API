"""
Mock document renderer (UC3) — deterministic, keyless.

Fills the standard customs/shipping documents from an already-computed context
dict (kept dict-shaped so the renderer stays decoupled from ``WorkflowState``).
Implements ``DocRenderer``. International shipments get the customs forms; every
shipment gets a packing list. Deterministic strings only — no model, no I/O.
"""

from __future__ import annotations

from app.domain.models import GeneratedDoc


def _s(context: dict, key: str, default: str = "") -> str:
    value = context.get(key, default)
    return "" if value is None else str(value)


class MockDocRenderer:
    """Deterministic renderer for commercial invoice / packing list / CN23."""

    def render(self, context: dict) -> list[GeneratedDoc]:
        origin = _s(context, "origin")
        destination = _s(context, "destination")
        description = _s(context, "description", "goods")
        value = _s(context, "value_usd", "0")
        weight = _s(context, "weight_lbs", "0")
        hs_code = _s(context, "hs_code", "n/a")
        international = bool(context.get("international"))

        docs: list[GeneratedDoc] = [
            GeneratedDoc(
                doc_type="packing_list",
                title="Packing List",
                fields={
                    "origin": origin,
                    "destination": destination,
                    "contents": description,
                    "weight_lbs": weight,
                },
            ),
        ]
        if international:
            docs.append(
                GeneratedDoc(
                    doc_type="commercial_invoice",
                    title="Commercial Invoice",
                    fields={
                        "ship_from": origin,
                        "ship_to": destination,
                        "description": description,
                        "hs_code": hs_code,
                        "declared_value_usd": value,
                        "incoterm": "DAP",
                    },
                )
            )
            docs.append(
                GeneratedDoc(
                    doc_type="customs_declaration_cn23",
                    title="CN23 Customs Declaration",
                    fields={
                        "origin_country": origin,
                        "destination_country": destination,
                        "contents": description,
                        "hs_code": hs_code,
                        "declared_value_usd": value,
                        "category": "merchandise",
                    },
                )
            )
        return docs
