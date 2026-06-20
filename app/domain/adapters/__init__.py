"""
Domain adapters (UC3) — concrete, swappable implementations of the ports.

The defaults are the deterministic, keyless ``Mock*`` adapters. ``DomainProviders``
bundles one adapter per port so wiring (``bootstrap.py`` / the workflow route)
passes a single typed object; ``default_providers()`` builds the all-mock bundle.
Swapping any port to a real backend is a one-line change here.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.adapters.mock_carrier import MCPCarrierAdapter, MockCarrierAdapter
from app.domain.adapters.mock_classification import MockClassificationAdapter
from app.domain.adapters.mock_doc_renderer import MockDocRenderer
from app.domain.adapters.mock_duty import MockDutyRateAdapter
from app.domain.ports import (
    CarrierProvider,
    ClassificationProvider,
    DocRenderer,
    DutyRateProvider,
)


@dataclass(frozen=True)
class DomainProviders:
    """One adapter per domain port — the swappable boundary, bundled."""

    classification: ClassificationProvider
    duty: DutyRateProvider
    carrier: CarrierProvider
    doc_renderer: DocRenderer


def default_providers() -> DomainProviders:
    """Build the all-mock provider bundle (deterministic, keyless)."""
    return DomainProviders(
        classification=MockClassificationAdapter(),
        duty=MockDutyRateAdapter(),
        carrier=MockCarrierAdapter(),
        doc_renderer=MockDocRenderer(),
    )


__all__ = [
    "DomainProviders",
    "MCPCarrierAdapter",
    "MockCarrierAdapter",
    "MockClassificationAdapter",
    "MockDocRenderer",
    "MockDutyRateAdapter",
    "default_providers",
]
