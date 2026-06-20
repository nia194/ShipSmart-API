"""
Request/response schemas for the workflow endpoint (UC3).

The request is a shipment to process end to end; the response exposes the
finished ``WorkflowState`` — each stage's result plus the full decision trail.
Domain results (HS candidates, landed cost, carrier quotes, documents) are the
frozen domain models, surfaced directly. ``international`` is derived server-side.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.domain.models import CarrierQuote, DutyQuote, GeneratedDoc, HsCandidate
from app.workflow.state import ComplianceSummary, WorkflowState


class WorkflowProcessRequest(BaseModel):
    """A shipment to run through the multi-agent workflow."""

    origin_country: str = Field(
        ..., min_length=2, max_length=2,
        description="Origin country as an ISO-3166 alpha-2 code (e.g. US).",
    )
    destination_country: str = Field(
        ..., min_length=2, max_length=2,
        description="Destination country as an ISO-3166 alpha-2 code (e.g. DE).",
    )
    declared_value_usd: float = Field(default=0.0, ge=0)
    weight_lbs: float = Field(default=0.0, ge=0)
    description: str = Field(default="", max_length=2000)
    category: str | None = Field(default=None, max_length=120)


class WorkflowReviewRequest(BaseModel):
    """An officer's determination for a workflow awaiting human review (UC4)."""

    determination: Literal["cleared", "blocked"] = Field(
        ..., description="cleared → continue to documentation; blocked → terminate.",
    )
    note: str = Field(default="", max_length=2000, description="Reviewer note (audited).")


class WorkflowResponse(BaseModel):
    """The finished (or current) workflow state, flattened for the API."""

    workflow_id: str
    status: str
    hs_code: str = ""
    hs_title: str = ""
    hs_candidates: list[HsCandidate] = Field(default_factory=list)
    landed_cost: DutyQuote | None = None
    carrier_quotes: list[CarrierQuote] = Field(default_factory=list)
    recommended_carrier: CarrierQuote | None = None
    compliance: ComplianceSummary | None = None
    documents: list[GeneratedDoc] = Field(default_factory=list)
    pending_review_areas: list[str] = Field(default_factory=list)
    officer_determination: str = ""
    officer_note: str = ""
    decisions: list[str] = Field(default_factory=list)

    @classmethod
    def from_state(cls, state: WorkflowState) -> WorkflowResponse:
        return cls(
            workflow_id=state.workflow_id,
            status=state.status,
            hs_code=state.hs_code,
            hs_title=state.hs_title,
            hs_candidates=state.hs_candidates,
            landed_cost=state.landed_cost,
            carrier_quotes=state.carrier_quotes,
            recommended_carrier=state.recommended_carrier,
            compliance=state.compliance,
            documents=state.documents,
            pending_review_areas=state.pending_review_areas,
            officer_determination=state.officer_determination,
            officer_note=state.officer_note,
            decisions=state.decisions,
        )
