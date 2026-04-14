"""
Request and response schemas for advisor endpoints.
"""

from typing import Any

from pydantic import BaseModel, Field


class ShippingAdvisorRequest(BaseModel):
    """Request for shipping advice."""

    query: str = Field(..., min_length=1, max_length=2000)
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional context: origin_zip, destination_zip, weight_lbs, dimensions, etc.",
    )


class ShippingAdvisorResponse(BaseModel):
    """Response with shipping advice."""

    answer: str
    reasoning_summary: str
    tools_used: list[str]
    sources: list[dict]
    context_used: bool


class TrackingAdvisorRequest(BaseModel):
    """Request for tracking/delivery guidance."""

    issue: str = Field(..., min_length=1, max_length=2000)
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional context: address fields, tracking info, etc.",
    )


class TrackingAdvisorResponse(BaseModel):
    """Response with tracking guidance."""

    guidance: str
    issue_summary: str
    tools_used: list[str]
    sources: list[dict]
    next_steps: list[str]


class ServiceOption(BaseModel):
    """A single shipping service option."""

    service_name: str
    price_usd: float
    estimated_days: int
    recommendation_type: str
    explanation: str
    score: float


class RecommendationRequest(BaseModel):
    """Request for quote recommendations."""

    services: list[dict[str, Any]] = Field(
        ...,
        description="Services from quote preview: list of {service, price_usd, estimated_days}",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional context: fragile, urgent, budget_preference, etc.",
    )


class RecommendationResponse(BaseModel):
    """Response with service recommendations."""

    primary_recommendation: ServiceOption
    alternatives: list[ServiceOption]
    summary: str
    metadata: dict[str, Any]
