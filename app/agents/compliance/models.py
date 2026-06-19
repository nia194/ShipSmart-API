"""
Compliance domain models (UC2).

Pure, framework-free dataclasses shared by the compliance agent's stages
(structural checks → grounded investigation → optional critic → summary). The
API request/response shapes live in ``app.schemas.compliance``; these are the
internal domain types the service reasons over.

Vocabulary (kept deliberately small so the audit trail reads cleanly):
  * :class:`Shipment` — the normalized shipment under review. ``international`` is
    DERIVED in code (origin != destination), never trusted from the client.
  * :class:`Finding` — one observation. ``status`` is ``flag`` (a concern to act
    on), ``info`` (grounded guidance, no concern), or ``unverified`` (an area the
    knowledge base could not cover — the honest gap, NEVER a fabricated flag).
  * :class:`GapProposal` — an area the UC2 critic proposes to investigate.
  * :class:`ComplianceResult` — the full outcome: advisory verdict + findings +
    grounded summary + the decision trail + the sources that grounded it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# Advisory verdicts — never "compliant" / "cleared to ship". This endpoint
# assists a human reviewer; it does not make a customs/legal determination.
Verdict = Literal["action_required", "review_recommended", "advisory"]

# One observation's status. The three-way split is the heart of the design:
# uncovered areas surface as ``unverified``, so the system never invents a
# clearance or a violation it cannot ground.
FindingStatus = Literal["flag", "info", "unverified"]

# Where a finding came from — for the audit trail and response grouping.
FindingKind = Literal["structural", "investigation", "critic"]


@dataclass(frozen=True)
class Shipment:
    """Normalized shipment under compliance review.

    Countries are ISO-3166 alpha-2 codes (e.g. ``US``, ``DE``). ``international``
    is computed from origin/destination, not accepted from the caller, so the
    structural checks stay deterministic and unspoofable.
    """

    origin_country: str
    destination_country: str
    declared_value_usd: float = 0.0
    weight_lbs: float = 0.0
    description: str = ""
    category: str | None = None

    @property
    def international(self) -> bool:
        return self.origin_country.strip().upper() != self.destination_country.strip().upper()


@dataclass(frozen=True)
class Finding:
    """One compliance observation, with the evidence that supports it."""

    area: str
    status: FindingStatus
    kind: FindingKind
    detail: str
    sources: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class GapProposal:
    """An investigation area proposed by the UC2 critic (model-in-the-loop)."""

    area: str
    rationale: str = ""


@dataclass
class ComplianceResult:
    """The compliance check's full, advisory outcome (maps onto the route response)."""

    verdict: Verdict
    summary: str
    findings: list[Finding] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    critique_rounds: int = 0
    provider: str = ""
