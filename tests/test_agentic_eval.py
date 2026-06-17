"""Test for the agentic re-retrieval comparison eval (Phase 3).

Asserts the eval script runs end to end (keyless, deterministic) and that the
agentic loop earns its cost: every HARD compound query improves from a partial
refusal (weak coverage) to a grounded answer, while the control query stays
single-shot. Imports the eval module directly so the assertions read its
structured reports rather than parsing stdout.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_EVAL_PATH = Path(__file__).resolve().parents[1] / "scripts" / "agentic_eval.py"


def _load_eval():
    spec = importlib.util.spec_from_file_location("agentic_eval", _EVAL_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the module's dataclasses can resolve their own
    # annotations (PEP 563) via sys.modules during class creation.
    sys.modules["agentic_eval"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_eval_runs_and_agentic_improves_hard_queries():
    ev = _load_eval()
    reports = await ev.run_eval()

    assert reports, "eval produced no reports"
    hard = [r for r in reports if r.hard]
    control = [r for r in reports if not r.hard]
    assert hard and control

    # Every hard compound query: weak on the single broad pass, grounded after
    # the agent reformulates into covered sub-areas — at a real, bounded cost.
    for r in hard:
        assert r.covered_before is False, f"{r.label}: broad query unexpectedly covered"
        assert r.covered_after is True, f"{r.label}: sub-areas still uncovered"
        assert r.grounding_improved, f"{r.label}: no refusal->grounded improvement"
        assert r.added_retrievals >= 1, f"{r.label}: agentic added no retrievals"
        assert "agent:retrieve:reformulate" in r.decisions

    # The control query is already covered → it must NOT spend extra retrievals.
    for r in control:
        assert r.covered_before is True
        assert r.added_retrievals == 0


def test_eval_main_reports_pass():
    ev = _load_eval()
    assert ev.main() == 0
