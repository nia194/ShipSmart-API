"""
Recommendation Service.
Takes quote preview results and generates scored recommendations with explanations.

Combines deterministic scoring with LLM reasoning to classify options as:
- cheapest
- fastest
- best value / balanced
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

from app.core.cache import recommendation_cache
from app.llm.client import LLMClient
from app.services.java_client import JavaApiClient

logger = logging.getLogger(__name__)


class RecommendationType(StrEnum):
    """Types of recommendations."""

    CHEAPEST = "cheapest"
    FASTEST = "fastest"
    BEST_VALUE = "best_value"
    BALANCED = "balanced"


@dataclass
class ServiceRecommendation:
    """A single service recommendation."""

    service_name: str
    price_usd: float
    estimated_days: int
    recommendation_type: RecommendationType
    explanation: str
    score: float = 0.0


@dataclass
class QuoteRecommendations:
    """Structured recommendations for multiple shipping options."""

    primary_recommendation: ServiceRecommendation
    alternatives: list[ServiceRecommendation] = field(default_factory=list)
    summary: str = ""
    metadata: dict = field(default_factory=dict)


async def generate_recommendations(
    services: list[dict],
    context: dict | None = None,
    llm_client: LLMClient | None = None,
    java_client: JavaApiClient | None = None,
    auth_token: str | None = None,
) -> QuoteRecommendations:
    """Generate ranked recommendations from quote preview services.

    Args:
        services: List of service dicts with service, price_usd, estimated_days.
        context: Optional context (weight, fragility, urgency, etc.)
        llm_client: For LLM-generated explanations.
        java_client: Optional Java API client. If `services` is empty AND
            `context` carries `shipment_request_id`, the recommendations
            will be hydrated from the Java side.
        auth_token: Optional JWT forwarded to Java when fetching quotes.

    Returns:
        QuoteRecommendations with primary recommendation and alternatives.
    """
    # If services not supplied, try to hydrate from Java by shipment_request_id.
    if not services and context and java_client and context.get("shipment_request_id"):
        fetched = await java_client.get_quotes(
            shipment_request_id=str(context["shipment_request_id"]),
            auth_token=auth_token,
        )
        if fetched:
            services = fetched
            logger.info(
                "Hydrated %d services from Java for shipment_request_id=%s",
                len(services), context.get("shipment_request_id"),
            )

    # Check cache (key includes both services and context for correctness)
    cache_key = recommendation_cache.make_key(services, context)
    cached = recommendation_cache.get(cache_key)
    if cached is not None:
        logger.debug("Recommendation cache hit")
        return cached

    if not services:
        return QuoteRecommendations(
            primary_recommendation=ServiceRecommendation(
                service_name="N/A",
                price_usd=0.0,
                estimated_days=0,
                recommendation_type=RecommendationType.BALANCED,
                explanation="No services available.",
            ),
            summary="No shipping services available for this shipment.",
        )

    # Step 1: Score each service
    scored = []
    for service in services:
        service_name = service.get("service", "Unknown")
        price = service.get("price_usd", 0.0)
        days = service.get("estimated_days", 0)

        # Determine recommendation type
        rec_type = _classify_service(service_name, price, days, services)

        # Generate explanation
        explanation = _explain_service(service_name, price, days, context, rec_type)

        # Compute score (higher is better)
        score = _score_service(price, days, services, rec_type)

        scored.append({
            "service": service_name,
            "price": price,
            "days": days,
            "type": rec_type,
            "explanation": explanation,
            "score": score,
        })

    # Step 2: Sort by score
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Step 3: Build recommendations
    primary_dict = scored[0]
    primary = ServiceRecommendation(
        service_name=primary_dict["service"],
        price_usd=primary_dict["price"],
        estimated_days=primary_dict["days"],
        recommendation_type=primary_dict["type"],
        explanation=primary_dict["explanation"],
        score=primary_dict["score"],
    )

    alternatives = [
        ServiceRecommendation(
            service_name=s["service"],
            price_usd=s["price"],
            estimated_days=s["days"],
            recommendation_type=s["type"],
            explanation=s["explanation"],
            score=s["score"],
        )
        for s in scored[1:]
    ]

    summary = _generate_summary(primary, alternatives, llm_client)

    result = QuoteRecommendations(
        primary_recommendation=primary,
        alternatives=alternatives,
        summary=summary,
        metadata={
            "num_options": len(services),
            "primary_type": primary.recommendation_type.value,
        },
    )

    recommendation_cache.set(cache_key, result)
    return result


def _classify_service(
    service_name: str, price: float, days: int, all_services: list[dict],
) -> RecommendationType:
    """Classify a service as cheapest, fastest, or balanced."""
    prices = [s.get("price_usd", 0.0) for s in all_services]
    days_list = [s.get("estimated_days", 0) for s in all_services]

    is_cheapest = price == min(prices) if prices else False
    is_fastest = days == min(days_list) if days_list else False

    if is_cheapest and is_fastest:
        return RecommendationType.BEST_VALUE
    if is_cheapest:
        return RecommendationType.CHEAPEST
    if is_fastest:
        return RecommendationType.FASTEST
    return RecommendationType.BALANCED


def _score_service(
    price: float, days: int, all_services: list[dict], rec_type: RecommendationType,
) -> float:
    """Score a service. Higher score = better recommendation."""
    if not all_services:
        return 0.0

    prices = [s.get("price_usd", float("inf")) for s in all_services]
    days_list = [s.get("estimated_days", float("inf")) for s in all_services]

    min_price = min(prices) if prices else float("inf")
    max_price = max(prices) if prices else 0.0
    min_days = min(days_list) if days_list else float("inf")
    max_days = max(days_list) if days_list else 0.0

    # Normalize price and days to 0-1 scale (lower is better)
    price_range = max_price - min_price
    days_range = max_days - min_days
    price_score = 1.0 - ((price - min_price) / price_range if price_range > 0 else 0.0)
    days_score = 1.0 - ((days - min_days) / days_range if days_range > 0 else 0.0)

    # Weight by recommendation type
    if rec_type == RecommendationType.CHEAPEST:
        return price_score * 1.5 + days_score * 0.5
    if rec_type == RecommendationType.FASTEST:
        return days_score * 1.5 + price_score * 0.5
    if rec_type == RecommendationType.BEST_VALUE:
        return (price_score + days_score) * 1.2
    return (price_score + days_score) / 2  # Balanced


def _explain_service(
    service_name: str,
    price: float,
    days: int,
    context: dict | None,
    rec_type: RecommendationType | None = None,
) -> str:
    """Generate an explanation for a service option."""
    base = f"{service_name} at ${price:.2f}"
    delivery = f"{days} day{'s' if days != 1 else ''}"

    # Context-specific explanations
    if context and context.get("fragile"):
        if days <= 1:
            return f"{base} ({delivery}) — fast handling, good for fragile items"
        return f"{base} ({delivery}) — consider faster option for fragile goods"

    if context and context.get("urgent"):
        if days <= 1:
            return f"{base} ({delivery}) — meets urgent delivery requirement"
        return f"{base} ({delivery}) — may not meet urgent deadline"

    # Type-specific explanations when no special context
    if rec_type == RecommendationType.CHEAPEST:
        return f"{base} ({delivery}) — lowest price option"
    if rec_type == RecommendationType.FASTEST:
        return f"{base} ({delivery}) — fastest delivery"
    if rec_type == RecommendationType.BEST_VALUE:
        return f"{base} ({delivery}) — best combination of price and speed"

    return f"{base} ({delivery})"


def _generate_summary(
    primary: ServiceRecommendation,
    alternatives: list[ServiceRecommendation],
    llm_client: LLMClient | None,
) -> str:
    """Generate a summary explanation of the recommendation."""
    summary = f"Recommended: {primary.service_name} at ${primary.price_usd:.2f} "
    summary += f"({primary.estimated_days} days). {primary.explanation}."

    if alternatives:
        summary += " Alternative options: "
        alt_names = ", ".join([a.service_name for a in alternatives[:2]])
        summary += alt_names + "."

    return summary
