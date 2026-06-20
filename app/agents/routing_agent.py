"""
Routing agent (UC3) — get carrier options and recommend one.

Deterministic: asks the injected ``CarrierProvider`` for quotes and picks the
recommended service — cheapest price, with faster transit as a deterministic
tie-breaker. No LLM. The workflow node wraps this and emits the
``workflow:routing:*`` decision tags.
"""

from __future__ import annotations

from app.domain.models import CarrierQuote
from app.domain.ports import CarrierProvider


def recommend(
    origin: str, destination: str, weight_lbs: float, *, provider: CarrierProvider,
) -> tuple[list[CarrierQuote], CarrierQuote | None]:
    """Return ``(quotes, recommended)``; recommended is cheapest then fastest.

    ``recommended`` is None only when the provider returns no quotes.
    """
    quotes = provider.quotes(origin, destination, weight_lbs)
    if not quotes:
        return [], None
    recommended = min(quotes, key=lambda q: (q.price_usd, q.estimated_days))
    return quotes, recommended
