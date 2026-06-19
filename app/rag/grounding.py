"""
Shared grounding primitive — the reasoning engine, reused by both doors.

This is the single home of grounded, coverage-gated retrieval: the one definition
of "covered" (``coverage_of`` over the deterministic ``_covered`` threshold) and
the bounded, conditional, non-degenerate single-area retrieval (``retrieve_area``)
that the agent loop (``app.services.agent_service``) and the compliance flow share.

There is NO LLM in this layer's control flow — models reason ABOVE it (the agent's
tool-calling loop, the compliance critic). Retrieval quality is read into an
observable :class:`CoverageSignal` so callers can decide whether to re-investigate.

``CoverageSignal`` / ``coverage_of`` were relocated here from
``app.services.agent_service`` (re-exported there for back-compat); ``_covered``
(the grounding threshold) is reused from ``app.rag.iterative`` — never redefined.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from app.core.config import settings
from app.rag.embeddings import EmbeddingProvider
from app.rag.iterative import _covered
from app.rag.retrieval import retrieve_auto
from app.rag.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)

# Deterministic reformulation: (area, original_query, coverage) -> a new query.
# Supplied by the caller; never an LLM call (the model reasons above this layer).
ReformulateFn = Callable[[str, str, "CoverageSignal"], str]


@dataclass
class CoverageSignal:
    """Quality signal a caller observes after a retrieval to decide whether the
    result is strong enough or a re-retrieval is warranted.

    Generic on purpose (any grounded investigation surfaces the same shape): it
    carries the highest similarity among retrieved chunks, whether anything cleared
    the deterministic RAG layer's grounding threshold, and how many chunks came
    back. ``covered`` reuses the pure layer's grounding notion (``_covered``) so
    every component agrees on what "covered" means.
    """

    top_score: float
    covered: bool
    chunk_count: int

    def as_line(self) -> str:
        return (
            f"coverage: top_score={self.top_score:.3f} "
            f"covered={'true' if self.covered else 'false'} "
            f"chunks={self.chunk_count}"
        )


def coverage_of(results: list[SearchResult]) -> CoverageSignal:
    """Read (do not change) the deterministic layer's per-chunk scores + grounding
    threshold into an observable coverage signal."""
    top = max((float(getattr(r, "score", 0.0) or 0.0) for r in results), default=0.0)
    return CoverageSignal(
        top_score=top, covered=_covered(results), chunk_count=len(results),
    )


@dataclass(frozen=True)
class AreaRetrieval:
    """Outcome of grounding one investigation area (bounded, deterministic)."""

    area: str
    query: str
    results: list[SearchResult]
    coverage: CoverageSignal
    covered: bool
    attempts: int
    decisions: list[str] = field(default_factory=list)


async def retrieve_area(
    area: str,
    query: str,
    *,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    max_retrievals: int | None = None,
    reformulate: ReformulateFn | None = None,
    request_id: str = "",
) -> AreaRetrieval:
    """Ground one area with bounded, conditional, non-degenerate re-retrieval.

    One pass via ``retrieve_auto``; if coverage is weak AND a deterministic
    ``reformulate`` is supplied AND the bound allows, retry with a DIFFERENT
    normalized query (identical queries are rejected — never loop). Stops as soon
    as coverage clears or the bound is hit. ``reformulate=None`` ⇒ a single pass.
    No LLM here — the control flow is deterministic; the model reasons above it.
    """
    bound = max(1, max_retrievals if max_retrievals is not None else settings.agent_max_retrievals)
    decisions: list[str] = []
    seen: set[str] = set()
    accumulated: dict[tuple[str, int], SearchResult] = {}
    current = query
    attempts = 0

    while attempts < bound:
        norm = current.strip().lower()
        if norm in seen:
            decisions.append(f"{area}:retrieve:rejected")
            break
        seen.add(norm)
        attempts += 1
        results, _mode = await retrieve_auto(
            current, embedding_provider, vector_store, top_k=settings.rag_top_k,
        )
        for r in results:
            accumulated[(r.source, r.chunk_index)] = r
        decisions.append(f"{area}:retrieve:{attempts}")
        if coverage_of(results).covered:
            decisions.append(f"{area}:covered")
            break
        if reformulate is None or attempts >= bound:
            break
        current = reformulate(area, query, coverage_of(results))
        decisions.append(f"{area}:reformulate")

    merged = sorted(accumulated.values(), key=lambda r: r.score, reverse=True)[: settings.rag_top_k]
    coverage = coverage_of(merged)
    if not coverage.covered:
        decisions.append(f"{area}:uncovered")
    return AreaRetrieval(
        area=area, query=query, results=merged, coverage=coverage,
        covered=coverage.covered, attempts=attempts, decisions=decisions,
    )
