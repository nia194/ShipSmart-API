"""
Request and response schemas for the compare endpoint.

Real facts only: price, carrier, service_name, arrival_date, transit_days, guaranteed.
No fabricated insurance caps, claim rates, or on-time rates.
"""

from typing import Any

from pydantic import BaseModel, Field


class ShipmentContext(BaseModel):
    """Shipment details (real facts only)."""

    item_description: str = Field(..., description="What is being shipped")
    origin_zip: str = Field(..., description="Origin ZIP code")
    destination_zip: str = Field(..., description="Destination ZIP code")
    deadline_date: str = Field(..., description="User's target delivery date (YYYY-MM-DD)")
    weight_lb: float = Field(..., gt=0, description="Total weight in pounds")
    declared_value_usd: float = Field(default=100, ge=0, description="Declared value")


class CompareOption(BaseModel):
    """A shipping option with real quote facts only."""

    id: str = Field(..., description="Service ID (e.g., 'ups-2da_xxx')")
    carrier: str = Field(..., description="Carrier name (UPS, FedEx, DHL, USPS)")
    service_name: str = Field(..., description="Service tier name (e.g., 'UPS 2nd Day Air')")
    carrier_type: str = Field(..., description="'public' or 'private'")
    price_usd: float = Field(..., ge=0, description="Total price (real)")
    arrival_date: str = Field(..., description="Estimated arrival date (YYYY-MM-DD)")
    arrival_label: str = Field(..., description="Human-readable label (e.g., 'Fri, Dec 19')")
    transit_days: int = Field(..., gt=0, description="Days in transit")
    guaranteed: bool = Field(default=False, description="Guaranteed delivery?")


class CompareRequest(BaseModel):
    """Request to compare 2-3 shipping options."""

    shipment: ShipmentContext
    option_ids: list[str] = Field(
        ..., min_items=2, max_items=3, description="2-3 option IDs (for caching)"
    )
    options: list[CompareOption] = Field(
        ..., min_items=2, max_items=3, description="Full real option data"
    )
    selected_priority: str = Field(
        default="ontime",
        description="User's selected shipping priority (ontime, damage, price, speed)",
    )


class Verdict(BaseModel):
    """A recommendation for a given priority."""

    purpose: str = Field(..., description="e.g., 'Best for on-time delivery'")
    pick_name: str = Field(..., description="Clean service name of the winner")
    reason: str = Field(..., description="1-2 sentences with evidence + concession")
    context_note: str = Field(
        default="",
        description="What about this shipment drives the recommendation (1 sentence)",
    )
    override_note: str = Field(
        default="",
        description="When the user should reasonably pick a different option (1 sentence)",
    )


class OptionInsight(BaseModel):
    """Per-option decision insight — positioning, use-case guidance."""

    option_id: str
    role_label: str = Field(
        ...,
        description="Short role framing: 'Best for urgency', 'Budget pick', etc. 2-5 words.",
    )
    strength: str = Field(
        ..., description="1 sentence: strongest factual advantage, with numbers"
    )
    consideration: str = Field(
        ..., description="1 sentence: honest limitation or cost, with numbers"
    )
    choose_when: str = Field(
        ..., description="1 sentence: when this option is the smart pick"
    )
    skip_when: str = Field(
        ..., description="1 sentence: when the user should look at a different option"
    )
    card_tag: str = Field(..., description="4-8 word factual tag for the card UI")


class ComparisonDimension(BaseModel):
    """One row of the attribute-by-attribute comparison table."""

    dimension: str = Field(..., description="e.g., 'Price', 'Delivery speed', 'Guarantee'")
    values: dict[str, str] = Field(
        ..., description="option_id -> value string for this dimension"
    )
    winner_id: str = Field(
        default="",
        description="option_id that wins on this dimension, or empty if tied",
    )
    note: str = Field(
        default="",
        description="Short note about this dimension for the selected priority (optional)",
    )


class DecisionFactors(BaseModel):
    """Shipment-specific decision intelligence."""

    primary_driver: str = Field(
        ...,
        description="1 sentence: the single most important factor for this shipment + priority",
    )
    key_tradeoff: str = Field(
        ...,
        description="1 sentence: the core tension the user is navigating between these options",
    )
    what_would_change: str = Field(
        ...,
        description="1 sentence: what would flip the recommendation to a different option",
    )


class Scenario(BaseModel):
    """A complete comparison scenario for one priority."""

    winner_id: str = Field(..., description="ID of the winning option")
    verdict: Verdict
    option_insights: list[OptionInsight] = Field(
        ..., description="Per-option decision insights"
    )
    comparison_dimensions: list[ComparisonDimension] = Field(
        ..., description="Attribute-by-attribute comparison rows"
    )
    decision_summary: str = Field(
        default="",
        description="2-3 sentences: how these options differ for this priority, with numbers",
    )
    decision_factors: DecisionFactors | None = Field(
        default=None,
        description="Shipment-specific decision intelligence",
    )


class CompareResponse(BaseModel):
    """Response with verdicts, insights, and comparison dimensions."""

    shipment_summary: str = Field(
        ..., description="One-line summary of shipment + route + deadline"
    )
    scenarios: dict[str, Scenario] = Field(
        ...,
        description="Four scenario keys: ontime, damage, price, speed. All precomputed.",
    )
