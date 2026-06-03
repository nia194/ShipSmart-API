"""
Context-window budgeting + token/length limits (B).

Before every LLM call we estimate prompt tokens and ensure
``prompt_tokens + max_output_tokens <= LLM_MAX_CONTEXT_TOKENS``. When the
retrieved context pushes us over, we deterministically drop the lowest-scoring
chunks first until it fits; if even the fixed scaffolding (system prompt +
query) plus the requested output cannot fit, we raise ContextLengthError
(terminal) rather than firing an over-long request that the provider would
reject anyway.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from app.llm.errors import ContextLengthError

logger = logging.getLogger(__name__)


class _Scored(Protocol):
    """Anything with retrieval text + score (e.g. rag.vector_store.SearchResult)."""

    text: str
    score: float


@lru_cache(maxsize=1)
def _encoder():
    """Return a tiktoken encoder if installed, else None (cached)."""
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - defensive
        return None


def estimate_tokens(text: str) -> int:
    """Estimate token count: tiktoken when available, else a conservative
    chars/4 heuristic (rounded up). The heuristic intentionally over-counts a
    little so the budget check stays safe when tiktoken is absent."""
    if not text:
        return 0
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text))
    return math.ceil(len(text) / 4)


def clamp_temperature(value: float, ceiling: float = 0.3) -> float:
    """Clamp temperature into [0, ceiling]. ShipSmart is a grounded advisor, so
    advisor/synthesis temperature never exceeds 0.3 even if a per-task override
    asks for more."""
    return max(0.0, min(float(value), ceiling))


def parse_float_or(raw: str | float | None, default: float) -> float:
    """Parse an optional override (empty/blank/None → default)."""
    if raw is None:
        return default
    text = str(raw).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def parse_int_or(raw: str | int | None, default: int) -> int:
    if raw is None:
        return default
    text = str(raw).strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        return default


@dataclass
class BudgetReport:
    """Outcome of fitting retrieved context to the token budget."""

    kept: list  # list[_Scored] — chunks that fit, in original order
    dropped: int
    prompt_tokens: int  # estimated tokens of fixed_text + kept chunks


def fit_to_budget(
    fixed_text: str,
    chunks: list,
    *,
    max_context_tokens: int,
    max_output_tokens: int,
) -> BudgetReport:
    """Fit retrieved ``chunks`` under the context budget.

    ``fixed_text`` is the non-droppable prompt scaffolding (system instruction +
    user query + fences). We greedily keep the highest-scoring chunks while
    ``fixed + kept + max_output <= budget``; lowest-scoring chunks are dropped
    first. Kept chunks are returned in their original order for stable prompts.

    Raises:
        ContextLengthError: if the fixed scaffolding + requested output already
        exceed the budget (no amount of trimming can help).
    """
    base = estimate_tokens(fixed_text)
    if base + max_output_tokens > max_context_tokens:
        raise ContextLengthError(
            detail=(
                f"prompt scaffolding {base} + output {max_output_tokens} tokens "
                f"exceeds context budget {max_context_tokens}"
            )
        )

    # Greedily keep highest-scoring chunks that still fit.
    ordered = sorted(enumerate(chunks), key=lambda p: getattr(p[1], "score", 0.0), reverse=True)
    used = base
    kept_idx: set[int] = set()
    for orig_i, chunk in ordered:
        t = estimate_tokens(getattr(chunk, "text", "") or "")
        if used + t + max_output_tokens <= max_context_tokens:
            kept_idx.add(orig_i)
            used += t

    kept = [c for i, c in enumerate(chunks) if i in kept_idx]
    dropped = len(chunks) - len(kept)
    if dropped:
        logger.info(
            "Context budget: kept %d/%d chunks (%d tokens, budget=%d, output=%d)",
            len(kept), len(chunks), used, max_context_tokens, max_output_tokens,
        )
    return BudgetReport(kept=kept, dropped=dropped, prompt_tokens=used)
