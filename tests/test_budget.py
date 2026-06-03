"""Tests for the context-window budgeter (B)."""

from __future__ import annotations

import pytest

from app.llm.budget import (
    BudgetReport,
    clamp_temperature,
    estimate_tokens,
    fit_to_budget,
    parse_float_or,
    parse_int_or,
)
from app.llm.errors import ContextLengthError
from app.rag.vector_store import SearchResult


def _chunk(text: str, score: float) -> SearchResult:
    return SearchResult(text=text, source="s", chunk_index=0, score=score)


# ── token estimate + helpers ─────────────────────────────────────────────────


def test_estimate_tokens_empty_is_zero():
    assert estimate_tokens("") == 0


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("a" * 400) >= 80  # ~100 via chars/4 heuristic (no tiktoken)
    assert estimate_tokens("short") > 0


def test_clamp_temperature():
    assert clamp_temperature(0.9) == 0.3
    assert clamp_temperature(0.2) == 0.2
    assert clamp_temperature(-1.0) == 0.0
    assert clamp_temperature(0.9, ceiling=0.5) == 0.5


def test_parse_overrides():
    assert parse_float_or("", 0.3) == 0.3
    assert parse_float_or(None, 0.3) == 0.3
    assert parse_float_or("0.7", 0.3) == 0.7
    assert parse_float_or("nope", 0.3) == 0.3
    assert parse_int_or("", 1024) == 1024
    assert parse_int_or("512", 1024) == 512


# ── fit_to_budget: fits / trims-then-fits / raises ───────────────────────────


def test_budget_all_chunks_fit():
    chunks = [_chunk("x" * 40, 0.9), _chunk("y" * 40, 0.5)]
    report = fit_to_budget(
        "system + query", chunks, max_context_tokens=8000, max_output_tokens=1024
    )
    assert isinstance(report, BudgetReport)
    assert report.dropped == 0
    assert len(report.kept) == 2


def test_budget_trims_lowest_scoring_first():
    # original order deliberately differs from score order
    mid = _chunk("m" * 400, 0.5)
    hi = _chunk("h" * 400, 0.9)
    lo = _chunk("l" * 400, 0.1)
    report = fit_to_budget(
        "fixed" * 8,  # ~10 tokens
        [mid, hi, lo],
        max_context_tokens=220,
        max_output_tokens=10,
    )
    assert report.dropped == 1
    kept_text = [c.text[0] for c in report.kept]
    assert "l" not in kept_text  # lowest score dropped
    assert kept_text == ["m", "h"]  # survivors preserved in ORIGINAL order


def test_budget_raises_when_scaffolding_alone_too_big():
    with pytest.raises(ContextLengthError):
        fit_to_budget(
            "x" * 400,  # ~100 tokens of fixed scaffolding
            [_chunk("y" * 40, 0.9)],
            max_context_tokens=50,
            max_output_tokens=10,
        )


def test_budget_keeps_zero_chunks_but_does_not_raise_if_base_fits():
    # base fits, but no single chunk fits alongside the requested output → kept empty
    report = fit_to_budget(
        "tiny",
        [_chunk("z" * 4000, 0.9)],
        max_context_tokens=60,
        max_output_tokens=50,
    )
    assert report.kept == []
    assert report.dropped == 1
