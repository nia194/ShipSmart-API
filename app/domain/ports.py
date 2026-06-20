"""
Domain ports (UC3) — the swappable boundary to every external domain.

Each specialist agent depends on one of these ``Protocol``s, never on a concrete
adapter. The default wiring (``bootstrap.py`` / the route) injects the
deterministic ``Mock*`` adapters in ``app/domain/adapters/``; a real backend (an
HS database, a duty engine, a live carrier API via MCP) is a future adapter that
implements the same Protocol — a swap, not an architecture change.

Ports are ``runtime_checkable`` so wiring can assert an injected object satisfies
the contract (mirrors ``app.core.audit.AuditSink``).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.domain.models import CarrierQuote, DutyQuote, GeneratedDoc, HsCandidate


@runtime_checkable
class ClassificationProvider(Protocol):
    """Map a free-text goods description to ranked HS-code candidates."""

    def candidates(self, description: str) -> list[HsCandidate]: ...


@runtime_checkable
class DutyRateProvider(Protocol):
    """Estimate landed cost (duty + import tax) for an HS code into a destination.

    ``origin`` is included (a refinement over the design doc's signature) because
    trade-agreement preferences such as USMCA are bilateral — duty depends on
    where the goods ship FROM, not just the destination.
    """

    def rate(
        self, hs_code: str, origin: str, destination: str, value_usd: float,
    ) -> DutyQuote: ...


@runtime_checkable
class CarrierProvider(Protocol):
    """Return carrier service options for a lane and weight."""

    def quotes(
        self, origin: str, destination: str, weight_lbs: float,
    ) -> list[CarrierQuote]: ...


@runtime_checkable
class DocRenderer(Protocol):
    """Render the customs/shipping documents for a finished workflow.

    Takes the already-computed stage fields (kept dict-shaped so the renderer
    stays decoupled from ``WorkflowState``).
    """

    def render(self, context: dict) -> list[GeneratedDoc]: ...
