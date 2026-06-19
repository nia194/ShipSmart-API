"""
Request and response schemas for the compliance endpoint (UC2).

The API takes a shipment description and returns an ADVISORY review: an
advisory verdict, the individual findings (flags / grounded info / honest
unverified gaps), a grounded summary, the cited sources, and the full decision
trail. ``international`` is never accepted from the client — it is derived from
the origin/destination countries server-side.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ComplianceRequest(BaseModel):
    """A shipment to review for compliance concerns."""

    origin_country: str = Field(
        ..., min_length=2, max_length=2,
        description="Origin country as an ISO-3166 alpha-2 code (e.g. US).",
    )
    destination_country: str = Field(
        ..., min_length=2, max_length=2,
        description="Destination country as an ISO-3166 alpha-2 code (e.g. DE).",
    )
    declared_value_usd: float = Field(
        default=0.0, ge=0,
        description="Declared customs value in USD (0 if unknown).",
    )
    weight_lbs: float = Field(
        default=0.0, ge=0, description="Total shipment weight in pounds.",
    )
    description: str = Field(
        default="", max_length=2000,
        description="Free-text description of the goods (drives keyword + area analysis).",
    )
    category: str | None = Field(
        default=None, max_length=120, description="Optional goods category.",
    )


class ComplianceFinding(BaseModel):
    """One compliance observation with its supporting sources."""

    area: str
    status: str = Field(description="flag | info | unverified")
    kind: str = Field(description="structural | investigation | critic")
    detail: str
    sources: list[dict] = Field(default_factory=list)


class ComplianceResponse(BaseModel):
    """The advisory compliance review plus its full reasoning trace."""

    verdict: str = Field(description="action_required | review_recommended | advisory")
    summary: str
    findings: list[ComplianceFinding] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    critique_rounds: int = 0
    provider: str = ""
