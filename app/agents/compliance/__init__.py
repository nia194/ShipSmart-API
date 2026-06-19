"""
Compliance agent (UC2).

A deterministic compliance flow — structural rules + grounded, coverage-gated
investigation of fixed areas — with an OPTIONAL model-in-the-loop critic that
proposes gaps a single pass may have missed. Uncovered areas surface as honest
``unverified`` findings, never fabricated flags. Advisory only.

Public surface:
  * :func:`check_compliance` — run the flow end to end.
  * :class:`Shipment` / :class:`ComplianceResult` / :class:`Finding` — domain types.
"""

from __future__ import annotations

from app.agents.compliance.models import (
    ComplianceResult,
    Finding,
    GapProposal,
    Shipment,
)
from app.agents.compliance.service import check_compliance

__all__ = [
    "ComplianceResult",
    "Finding",
    "GapProposal",
    "Shipment",
    "check_compliance",
]
