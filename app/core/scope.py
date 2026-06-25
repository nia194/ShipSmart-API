"""Shipping-scope policy guard (platform policy enforcement).

Single, pure home for the "worldwide vs domestic" decision. In ``worldwide``
scope every function here is a no-op — today's behavior is untouched. In
``domestic`` scope a shipment may only move within ``settings.home_country``;
anything crossing that border is rejected at the API edge (422) or, for the
conversational concierge, surfaced as a graceful domestic-only reply.

Kept dependency-light (config + AppError) so routes, the concierge, and tests can
all reuse one definition of "is this shipment in scope?".
"""

from __future__ import annotations

from app.core.config import settings
from app.core.errors import AppError


def violates_domestic_scope(origin_country: str, destination_country: str) -> str | None:
    """Return the first country that breaks the domestic home border, else ``None``.

    A no-op (always ``None``) in ``worldwide`` scope. Empty country values are
    ignored — they default to the home country downstream, so they never violate.
    """
    if not settings.is_domestic_scope:
        return None
    home = settings.home_country
    for country in (origin_country, destination_country):
        normalized = (country or "").strip().upper()
        if normalized and normalized != home:
            return normalized
    return None


def enforce_scope(origin_country: str, destination_country: str) -> None:
    """Reject a cross-border shipment when the deployment is domestic-only (422).

    No-op in ``worldwide`` scope. Call at the API edge before doing any work.
    """
    offending = violates_domestic_scope(origin_country, destination_country)
    if offending is not None:
        raise AppError(
            status_code=422,
            message=(
                f"This deployment ships domestically only (within "
                f"{settings.home_country}); destination/origin {offending} "
                f"is not supported."
            ),
        )
