"""
UC2 critic — the only model-in-the-loop step of the compliance flow.

After the deterministic structural + fixed-area investigation, the critic asks a
reasoning model: *what did a single pass miss?* It proposes additional, specific
investigation areas via a native tool call (``propose_gaps``). The service then
GROUNDS each proposal through the same ``retrieve_area`` primitive — so the model
can only ever direct attention, never assert a conclusion. An uncovered proposal
becomes an honest ``unverified`` finding, never a fabricated flag.

Honest degradation:
  * Providers with native tool calling emit a ``propose_gaps`` tool call; we
    parse + validate + cap it.
  * Providers without native tool calling (e.g. the keyless ``echo`` default)
    raise ``NotImplementedError`` from ``complete_with_tools`` → deterministic
    no-op (zero gaps). The deterministic compliance path is unaffected.

This is genuinely "agentic" in the narrow, honest sense: a model reasons in the
loop and changes what gets investigated. The surrounding analysis is not — it is
deterministic RAG.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.agents.compliance.models import Finding, GapProposal, Shipment
from app.llm.router import TASK_REASONING, LLMRouter

logger = logging.getLogger(__name__)

# Slugs are normalized to this charset so they compose cleanly into decision tags
# and area names regardless of how the model phrases them.
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_MAX_SLUG_LEN = 48

# Tool the critic uses to return structured gaps. ``areas`` is a single
# semicolon-separated string (not an array) so it round-trips through the existing
# registry-to-provider tool-schema converter without needing JSON-Schema ``items``.
PROPOSE_GAPS_SCHEMA: dict[str, Any] = {
    "name": "propose_gaps",
    "description": (
        "Propose specific compliance sub-areas the initial analysis may have "
        "missed for THIS shipment, so they can be checked against the knowledge "
        "base. Only propose areas justified by the shipment details and the "
        "evidence already gathered. Propose nothing if coverage looks complete."
    ),
    "parameters": [
        {
            "name": "areas",
            "type": "string",
            "required": True,
            "description": (
                "Semicolon-separated concise sub-area slugs to investigate, e.g. "
                "'destination_drone_import_rules; battery_watt_hour_limit'. "
                "Empty string if there are no gaps."
            ),
        },
        {
            "name": "rationale",
            "type": "string",
            "required": False,
            "description": "Brief reason these areas matter for this shipment.",
        },
    ],
}

_CRITIC_SYSTEM_PROMPT = (
    "You are a compliance critic for a shipping platform. A first, deterministic "
    "pass has already run structural checks and investigated four fixed areas "
    "(lithium batteries, customs docs, import restrictions, value thresholds). "
    "Your ONLY job is to spot SPECIFIC, shipment-relevant sub-areas that pass may "
    "have missed — especially destination-specific or item-specific rules — and "
    "return them via the propose_gaps tool so they can be verified against the "
    "knowledge base. Do not assert conclusions or invent rules; only name areas "
    "to investigate. If the existing coverage looks complete, propose nothing."
)


def _slugify(raw: str) -> str:
    slug = _SLUG_RE.sub("_", (raw or "").strip().lower()).strip("_")
    return slug[:_MAX_SLUG_LEN]


def _parse_areas(areas_value: str, rationale: str) -> list[GapProposal]:
    """Split the model's semicolon-separated ``areas`` into validated proposals."""
    proposals: list[GapProposal] = []
    seen: set[str] = set()
    for piece in (areas_value or "").split(";"):
        slug = _slugify(piece)
        if not slug or slug in seen:
            continue
        seen.add(slug)
        proposals.append(GapProposal(area=slug, rationale=rationale.strip()))
    return proposals


def _build_user_message(
    shipment: Shipment, findings: list[Finding], already_investigated: set[str],
) -> str:
    """Render the shipment + evidence-so-far the critic reasons over."""
    lines = [
        "Shipment under review:",
        f"- route: {shipment.origin_country} -> {shipment.destination_country} "
        f"(international={shipment.international})",
        f"- declared_value_usd: {shipment.declared_value_usd}",
        f"- weight_lbs: {shipment.weight_lbs}",
        f"- description: {shipment.description or '(none)'}",
        f"- category: {shipment.category or '(none)'}",
        "",
        "Already investigated areas: " + (", ".join(sorted(already_investigated)) or "(none)"),
        "",
        "Findings so far:",
    ]
    if findings:
        lines += [f"- [{f.status}] {f.area}: {f.detail}" for f in findings]
    else:
        lines.append("- (none)")
    lines += [
        "",
        "Call propose_gaps with any specific sub-areas still worth checking for "
        "this shipment (especially destination- or item-specific rules). Propose "
        "nothing if coverage is complete.",
    ]
    return "\n".join(lines)


async def propose_gaps(
    shipment: Shipment,
    findings: list[Finding],
    *,
    already_investigated: set[str],
    llm_router: LLMRouter,
    max_gap_areas: int,
) -> list[GapProposal]:
    """Ask the reasoning model for missed areas (bounded, validated, deduped).

    Returns at most ``max_gap_areas`` proposals, excluding areas already
    investigated. Returns ``[]`` deterministically when the provider has no native
    tool calling or on any provider error (the critic is best-effort and must
    never break the deterministic analysis).
    """
    if max_gap_areas <= 0:
        return []

    reasoning = llm_router.for_task(TASK_REASONING)
    messages = [
        {"role": "system", "content": _CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(shipment, findings, already_investigated)},
    ]

    try:
        out = await reasoning.complete_with_tools(messages, [PROPOSE_GAPS_SCHEMA])
    except NotImplementedError:
        logger.debug("Critic no-op: provider %s has no native tool calling",
                     reasoning.provider_name)
        return []
    except Exception as exc:  # noqa: BLE001 - critic is best-effort; never abort the request
        logger.warning("Critic propose_gaps failed (%s) — proceeding with zero gaps", exc)
        return []

    proposals: list[GapProposal] = []
    for call in getattr(out, "calls", []) or []:
        if call.name != "propose_gaps":
            continue
        args = call.arguments or {}
        proposals.extend(
            _parse_areas(str(args.get("areas") or ""), str(args.get("rationale") or ""))
        )

    # Drop anything already investigated; cap the blast radius.
    deduped: list[GapProposal] = []
    seen = {a.lower() for a in already_investigated}
    for p in proposals:
        if p.area in seen:
            continue
        seen.add(p.area)
        deduped.append(p)
        if len(deduped) >= max_gap_areas:
            break
    return deduped
