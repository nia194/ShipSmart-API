"""
Domain models — frozen, typed shapes that cross the port boundary (UC3).

These are the values the specialist agents and the workflow exchange with the
swappable domain providers (classification, duty, carrier, doc rendering). They
are **frozen** (immutable) and plain Pydantic so they are both safe to pass
around and JSON-serializable — the latter matters for the Phase 3 durable
checkpointer, which serializes the whole ``WorkflowState`` (these included).

Nothing here knows about FastAPI, the workflow engine, or any concrete adapter —
just data.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class HsCandidate(_Frozen):
    """A candidate Harmonized System classification for a goods description."""

    hs_code: str
    title: str
    confidence: float


class DutyQuote(_Frozen):
    """Landed-cost estimate for one HS code into one destination.

    All money in USD. ``duty`` is the customs duty; ``tax`` is the destination's
    import tax (VAT/GST); ``total_landed_usd`` = value + duty + tax. ``trade_note``
    records any trade-agreement adjustment applied (advisory, mock data).
    """

    hs_code: str
    destination: str
    value_usd: float
    duty_pct: float
    duty_usd: float
    tax_label: str
    tax_pct: float
    tax_usd: float
    total_landed_usd: float
    trade_note: str = ""


class CarrierQuote(_Frozen):
    """A single carrier service option (mirrors the MCP get_quote_preview shape)."""

    carrier: str
    service: str
    price_usd: float
    estimated_days: int


class GeneratedDoc(_Frozen):
    """A deterministically rendered shipping/customs document."""

    doc_type: str
    title: str
    fields: dict[str, str]
