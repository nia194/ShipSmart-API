"""
Workflow state (UC3) — the single typed object that flows through the graph.

``WorkflowState`` is a mutable Pydantic model carrying the input shipment, each
stage's result, and the cumulative decision trail. It is JSON-serializable end to
end (domain results are frozen Pydantic models) so the Phase 3 durable
checkpointer can persist and restore it without any extra mapping.

``international`` is derived (origin != destination), never trusted from a client
— the same rule the compliance flow uses.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.domain.models import CarrierQuote, DutyQuote, GeneratedDoc, HsCandidate

# Lifecycle. Phase 2 runs pending → running → completed; "awaiting_review" /
# "blocked" are reached only once the Phase 3 interrupt/resume is wired.
WorkflowStatus = Literal[
    "pending", "running", "completed", "awaiting_review", "blocked", "failed",
]


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


class ComplianceSummary(BaseModel):
    """The compliance stage's result, distilled for the workflow state.

    ``flagged_areas`` / ``unverified_areas`` are kept so the Phase 3 interrupt
    predicate (unverified high-risk) can read them without re-running compliance.
    """

    verdict: str
    summary: str
    flagged_areas: list[str] = Field(default_factory=list)
    unverified_areas: list[str] = Field(default_factory=list)
    critique_rounds: int = 0
    provider: str = ""


class WorkflowState(BaseModel):
    """The durable, typed state threaded through every workflow node."""

    workflow_id: str
    status: WorkflowStatus = "pending"
    request_id: str = ""

    # ── Input shipment ────────────────────────────────────────────────────────
    origin_country: str
    destination_country: str
    declared_value_usd: float = 0.0
    weight_lbs: float = 0.0
    description: str = ""
    category: str | None = None

    # ── Stage results (None / empty until the stage runs) ─────────────────────
    hs_candidates: list[HsCandidate] = Field(default_factory=list)
    hs_code: str = ""
    hs_title: str = ""
    landed_cost: DutyQuote | None = None
    carrier_quotes: list[CarrierQuote] = Field(default_factory=list)
    recommended_carrier: CarrierQuote | None = None
    compliance: ComplianceSummary | None = None
    documents: list[GeneratedDoc] = Field(default_factory=list)

    # ── Human-in-the-loop (UC4) ───────────────────────────────────────────────
    # Set at the interrupt: the high-risk areas that could not be verified and so
    # require a human determination. Officer fields are set on resume.
    pending_review_areas: list[str] = Field(default_factory=list)
    officer_determination: str = ""  # "" | "cleared" | "blocked"
    officer_note: str = ""

    # ── Trail ─────────────────────────────────────────────────────────────────
    decisions: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)

    @property
    def international(self) -> bool:
        return (
            self.origin_country.strip().upper()
            != self.destination_country.strip().upper()
        )
