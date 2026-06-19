"""
Compliance service (UC2) — the compliance flow's composition of stages.

Pipeline (the deterministic spine, with one optional model-in-the-loop step):

  1. ``compliance:plan`` — start the trail.
  2. Structural checks (pure, deterministic) — establish facts → ``flag`` findings.
  3. Fixed-area investigation — ground each of the four areas via the shared
     :func:`app.rag.grounding.retrieve_area` primitive. Covered → ``info``;
     uncovered → an honest ``unverified`` finding.
  4. UC2 critic (optional, ``COMPLIANCE_CRITIQUE_MAX_ROUNDS`` > 0) — a model
     proposes additional areas; each is GROUNDED the same way. An uncovered
     proposal becomes ``unverified``, NEVER a fabricated flag.
  5. Advisory summary — grounded synthesis through the same guardrailed assembler
     + synthesis failover chain the RAG/agent paths use. Advisory only.
  6. Verdict — derived from the findings; never "compliant"/"cleared".

Every branch appends a namespaced decision tag, and (when an audit sink is wired)
the verdict is emitted as an :class:`~app.core.audit.AuditEvent` — the emergent,
replayable trail the platform treats as a first-class feature.
"""

from __future__ import annotations

import logging

from app.agents.compliance.areas import decompose
from app.agents.compliance.critic import propose_gaps
from app.agents.compliance.models import (
    ComplianceResult,
    Finding,
    Shipment,
    Verdict,
)
from app.agents.compliance.structural import run_structural_checks
from app.core.audit import AuditEvent, AuditSink
from app.core.config import settings
from app.llm.guardrails import SAFE_REFUSAL, assemble
from app.llm.prompts import COMPLIANCE_SYSTEM_PROMPT
from app.llm.router import TASK_SYNTHESIS, LLMRouter
from app.rag.embeddings import EmbeddingProvider
from app.rag.grounding import AreaRetrieval, retrieve_area
from app.rag.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)

_DETAIL_SNIPPET_CHARS = 160

_UNVERIFIED_DETAIL = (
    "The knowledge base did not return grounded coverage for this area; human "
    "review or additional information is needed before relying on it."
)


def _source_dict(r: SearchResult) -> dict:
    return {"source": r.source, "chunk_index": r.chunk_index, "score": round(r.score, 4)}


def _finding_from_area(
    ar: AreaRetrieval, *, kind: str, rationale: str = "",
) -> Finding:
    """Turn a grounded area outcome into a finding (info if covered, else unverified)."""
    sources = [_source_dict(r) for r in ar.results]
    if ar.covered:
        snippet = ar.results[0].text.strip().replace("\n", " ")[:_DETAIL_SNIPPET_CHARS]
        detail = f"Knowledge-base coverage found: {snippet}…"
        if rationale:
            detail = f"{rationale.strip()} — {detail}"
        return Finding(area=ar.area, status="info", kind=kind, detail=detail, sources=sources)
    detail = f"{rationale.strip()} — {_UNVERIFIED_DETAIL}" if rationale else _UNVERIFIED_DETAIL
    return Finding(area=ar.area, status="unverified", kind=kind, detail=detail, sources=sources)


def _verdict(findings: list[Finding]) -> Verdict:
    """Advisory verdict from findings. Precedence: flags > gaps > advisory."""
    if any(f.status == "flag" for f in findings):
        return "action_required"
    if any(f.status == "unverified" for f in findings):
        return "review_recommended"
    return "advisory"


def _gap_query(shipment: Shipment, area: str) -> str:
    dest = (shipment.destination_country or "").strip().upper()
    topic = area.replace("_", " ")
    desc = (shipment.description or "").strip()
    return f"{topic} {desc} destination {dest}".strip()


def _summary_request(shipment: Shipment, findings: list[Finding]) -> str:
    lines = [
        f"Compliance review for a shipment from {shipment.origin_country} to "
        f"{shipment.destination_country} (international={shipment.international}).",
        f"Declared value: ${shipment.declared_value_usd:,.2f}; "
        f"weight: {shipment.weight_lbs} lb; description: {shipment.description or '(none)'}.",
        "",
        "Findings:",
    ]
    lines += [f"- [{f.status}/{f.kind}] {f.area}: {f.detail}" for f in findings] or ["- (none)"]
    lines += ["", "Write the advisory summary for the reviewer per your rules."]
    return "\n".join(lines)


async def check_compliance(
    shipment: Shipment,
    *,
    llm_router: LLMRouter,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    audit_sink: AuditSink | None = None,
    critique_max_rounds: int | None = None,
    request_id: str = "",
) -> ComplianceResult:
    """Run the compliance flow end to end and return an advisory result."""
    decisions: list[str] = ["compliance:plan"]
    findings: list[Finding] = []
    chunks: dict[tuple[str, int], SearchResult] = {}

    # 2) Structural checks — pure, deterministic facts.
    structural_findings, structural_decisions = run_structural_checks(
        shipment, value_threshold_usd=settings.compliance_value_threshold_usd,
    )
    findings.extend(structural_findings)
    decisions.extend(structural_decisions)

    # 3) Fixed-area grounded investigation.
    investigated: set[str] = set()

    async def investigate(area: str, query: str, *, kind: str, rationale: str = "") -> None:
        decisions.append(f"compliance:investigate:{area}")
        ar = await retrieve_area(
            area, query,
            embedding_provider=embedding_provider, vector_store=vector_store,
            request_id=request_id,
        )
        decisions.extend(ar.decisions)
        investigated.add(area)
        for r in ar.results:
            chunks[(r.source, r.chunk_index)] = r
        findings.append(_finding_from_area(ar, kind=kind, rationale=rationale))

    for area, query in decompose(shipment):
        await investigate(area, query, kind="investigation")

    # 4) UC2 critic — optional model-in-the-loop gap proposal, then grounding.
    rounds_cap = (
        settings.compliance_critique_max_rounds
        if critique_max_rounds is None else critique_max_rounds
    )
    critique_rounds = 0
    round_n = 0
    while round_n < rounds_cap:
        round_n += 1
        decisions.append(f"critique:round:{round_n}")
        proposals = await propose_gaps(
            shipment, findings,
            already_investigated=investigated, llm_router=llm_router,
            max_gap_areas=settings.compliance_max_gap_areas,
        )
        if not proposals:
            decisions.append("critique:complete")
            break
        new_added = 0
        for p in proposals:
            if p.area in investigated:
                decisions.append("critique:rejected")
                continue
            decisions.append(f"critique:gap:{p.area}")
            await investigate(
                p.area, _gap_query(shipment, p.area), kind="critic", rationale=p.rationale,
            )
            new_added += 1
        critique_rounds += 1
        if new_added == 0:
            decisions.append("critique:complete")
            break
    else:
        if rounds_cap > 0:
            decisions.append("critique:capped")

    # 5) Advisory summary — grounded synthesis (reuses the shared assembler + chain).
    summary, provider, summary_sources = await _summarize(
        shipment, findings, list(chunks.values()),
        llm_router=llm_router, decisions=decisions, request_id=request_id,
    )

    # 6) Verdict + audit.
    verdict = _verdict(findings)
    decisions.append(f"compliance:verdict:{verdict}")
    _emit(audit_sink, verdict, findings, critique_rounds, request_id)

    return ComplianceResult(
        verdict=verdict,
        summary=summary,
        findings=findings,
        sources=summary_sources,
        decisions=decisions,
        critique_rounds=critique_rounds,
        provider=provider,
    )


async def _summarize(
    shipment: Shipment,
    findings: list[Finding],
    contexts: list[SearchResult],
    *,
    llm_router: LLMRouter,
    decisions: list[str],
    request_id: str,
) -> tuple[str, str, list[dict]]:
    """Ground the findings into an advisory summary via the synthesis chain."""
    assembled = assemble(
        system_prompt=COMPLIANCE_SYSTEM_PROMPT,
        user_text=_summary_request(shipment, findings),
        contexts=contexts,
        request_id=request_id,
    )
    decisions.extend(assembled.decisions)
    if assembled.blocked:
        decisions.append("guardrail:blocked")
        return (assembled.refusal or SAFE_REFUSAL), "guardrail", []

    res = await llm_router.execute(TASK_SYNTHESIS, assembled.messages, request_id=request_id)
    decisions.append("compliance:summarized")
    sources = [_source_dict(r) for r in assembled.kept_sources]
    return res.text, res.provider, sources


def _emit(
    audit_sink: AuditSink | None,
    verdict: str,
    findings: list[Finding],
    critique_rounds: int,
    request_id: str,
) -> None:
    """Best-effort audit emission (never raises — auditing must not break a request)."""
    if audit_sink is None:
        return
    try:
        audit_sink.emit(
            AuditEvent(
                event=f"compliance:verdict:{verdict}",
                actor="agent",
                actor_name="compliance",
                request_id=request_id,
                payload={
                    "verdict": verdict,
                    "flags": sum(1 for f in findings if f.status == "flag"),
                    "unverified": sum(1 for f in findings if f.status == "unverified"),
                    "critique_rounds": critique_rounds,
                },
            )
        )
    except Exception:  # noqa: BLE001 - auditing is best-effort
        logger.debug("Audit emit failed for compliance verdict", exc_info=True)
