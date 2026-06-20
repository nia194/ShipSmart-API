"""Tests for the UC4 workflow checkpointers (in-memory + SQLite).

Both adapters round-trip ``WorkflowState`` through JSON, so save→load reproduces
the state exactly. The SQLite adapter additionally persists across instances —
the durable "kill & resume" property.
"""

from __future__ import annotations

from app.workflow.checkpointer import (
    InMemoryCheckpointer,
    SqliteCheckpointer,
    create_checkpointer,
)
from app.workflow.state import ComplianceSummary, WorkflowState


def _state(workflow_id: str = "wf-1") -> WorkflowState:
    return WorkflowState(
        workflow_id=workflow_id,
        origin_country="US", destination_country="DE",
        declared_value_usd=1000.0, weight_lbs=5.0, description="power bank",
        hs_code="8507.60", hs_title="Lithium-ion accumulators",
        status="awaiting_review", pending_review_areas=["lithium_battery"],
        compliance=ComplianceSummary(
            verdict="action_required", summary="...",
            unverified_areas=["lithium_battery"], flagged_areas=["dangerous_goods_declaration"],
        ),
        decisions=["workflow:start", "workflow:interrupt:human_review"],
    )


# ── in-memory ─────────────────────────────────────────────────────────────────


def test_inmemory_round_trip_reproduces_state():
    cp = InMemoryCheckpointer()
    original = _state()
    cp.save(original)
    loaded = cp.load("wf-1")
    assert loaded is not None
    assert loaded.model_dump_json() == original.model_dump_json()


def test_inmemory_load_missing_returns_none():
    assert InMemoryCheckpointer().load("nope") is None


def test_inmemory_returns_independent_copy():
    cp = InMemoryCheckpointer()
    cp.save(_state())
    loaded = cp.load("wf-1")
    loaded.decisions.append("mutated")
    # The stored copy is unaffected by mutating a loaded instance.
    assert "mutated" not in cp.load("wf-1").decisions


# ── sqlite (durable) ──────────────────────────────────────────────────────────


def test_sqlite_round_trip(tmp_path):
    cp = SqliteCheckpointer(str(tmp_path / "wf.db"))
    original = _state()
    cp.save(original)
    loaded = cp.load("wf-1")
    assert loaded is not None
    assert loaded.model_dump_json() == original.model_dump_json()


def test_sqlite_persists_across_instances(tmp_path):
    path = str(tmp_path / "wf.db")
    SqliteCheckpointer(path).save(_state("wf-restart"))
    # A brand-new checkpointer on the same file sees the saved workflow.
    reopened = SqliteCheckpointer(path)
    loaded = reopened.load("wf-restart")
    assert loaded is not None and loaded.status == "awaiting_review"


def test_sqlite_load_missing_returns_none(tmp_path):
    assert SqliteCheckpointer(str(tmp_path / "wf.db")).load("nope") is None


# ── factory ───────────────────────────────────────────────────────────────────


def test_factory_selects_adapter(tmp_path):
    assert isinstance(create_checkpointer(False, ""), InMemoryCheckpointer)
    assert isinstance(
        create_checkpointer(True, str(tmp_path / "wf.db")), SqliteCheckpointer
    )
