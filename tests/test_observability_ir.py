"""Observability + incident response: guardrail metrics, kill-switches, admin (F9)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core import guardrail_metrics as gm
from app.core.ai_events import InMemoryAIEventSink, build_ai_event
from app.core.config import settings
from app.core.kill_switch import (
    KILLSWITCH_TAG,
    KillSwitchRegistry,
    registry,
    require_feature,
)
from app.main import app

client = TestClient(app)

SECRET = "test-secret"


def _event(tags: list[str]):
    return build_ai_event(route="/x", decisions=tags, secret=SECRET)


# ── guardrail metrics (§11) ───────────────────────────────────────────────────
def test_collect_counts_by_tag_prefix_including_dynamic_tails():
    events = [
        _event(["guardrail:blocked_injection"]),
        _event(["guardrail:injection:high"]),          # dynamic tail still counts
        _event(["guardrail:misuse_refused"]),
        _event(["guardrail:structured_output_invalid"]),
        _event(["guardrail:quarantined_chunk", "budget:exceeded"]),
        _event(["agent:plan"]),                        # not a guardrail event
    ]
    m = gm.collect(events)
    assert m.total_events == 6
    assert m.count("injection_blocks") == 2
    assert m.count("refusals") == 1
    assert m.count("structured_output_invalid") == 1
    assert m.count("quarantined_chunks") == 1
    assert m.count("budget_exceeded") == 1
    assert m.rate("injection_blocks") == pytest.approx(2 / 6)


def test_thresholds_alert_only_on_breach_and_enough_volume():
    quiet = gm.collect([_event(["agent:plan"])] * 30)
    assert gm.check_thresholds(quiet) == []

    # 3/30 = 10% structured-output-invalid, breaching the 2% ceiling.
    noisy = gm.collect(
        [_event(["guardrail:structured_output_invalid"])] * 3 + [_event(["agent:plan"])] * 27
    )
    alerts = gm.check_thresholds(noisy)
    assert len(alerts) == 1 and "structured_output_invalid" in alerts[0]

    # Same breach rate but under the volume floor -> no alert (too small to page on).
    small = gm.collect([_event(["guardrail:structured_output_invalid"])] * 2)
    assert gm.check_thresholds(small) == []


# ── kill-switch registry (§12) ────────────────────────────────────────────────
def test_registry_flip_is_audited_and_queryable():
    sink = InMemoryAIEventSink()
    reg = KillSwitchRegistry(sink=sink)
    assert reg.is_enabled("concierge")

    state = reg.set_enabled("concierge", False, actor="test", reason="incident drill")
    assert not reg.is_enabled("concierge") and not state.enabled
    assert reg.snapshot()["concierge"] is False

    tags = [t for e in sink.events for t in e.decisions]
    assert f"{KILLSWITCH_TAG}:concierge:off" in tags

    reg.set_enabled("concierge", True, actor="test", reason="restored")
    assert reg.is_enabled("concierge")
    assert any(t == f"{KILLSWITCH_TAG}:concierge:on" for e in sink.events for t in e.decisions)


def test_registry_rejects_unknown_features():
    reg = KillSwitchRegistry(sink=InMemoryAIEventSink())
    with pytest.raises(ValueError):
        reg.is_enabled("guardrails")  # protection is not killable
    with pytest.raises(ValueError):
        reg.set_enabled("nope", False, actor="t", reason="")
    with pytest.raises(ValueError):
        require_feature("nope")


# ── admin endpoint (fail-closed auth) ─────────────────────────────────────────
def test_admin_endpoint_does_not_exist_without_a_token():
    assert settings.admin_api_token == ""  # default
    assert client.get("/api/v1/admin/ai-controls").status_code == 404


def test_admin_endpoint_authenticates_and_flips(monkeypatch):
    monkeypatch.setattr(settings, "admin_api_token", "secret-token")

    assert client.get("/api/v1/admin/ai-controls").status_code == 403  # missing token
    bad = client.get("/api/v1/admin/ai-controls", headers={"X-Admin-Token": "wrong"})
    assert bad.status_code == 403

    ok = client.get("/api/v1/admin/ai-controls", headers={"X-Admin-Token": "secret-token"})
    assert ok.status_code == 200 and ok.json()["features"]["agent"] is True

    try:
        flip = client.post(
            "/api/v1/admin/ai-controls",
            json={"feature": "agent", "enabled": False, "reason": "drill"},
            headers={"X-Admin-Token": "secret-token"},
        )
        assert flip.status_code == 200 and flip.json()["features"]["agent"] is False

        unknown = client.post(
            "/api/v1/admin/ai-controls",
            json={"feature": "guardrails", "enabled": False, "reason": ""},
            headers={"X-Admin-Token": "secret-token"},
        )
        assert unknown.status_code == 422
    finally:
        registry.set_enabled("agent", True, actor="test", reason="restore")


# ── a kill actually takes a route offline ─────────────────────────────────────
def test_killed_feature_404s_its_routes():
    body = {"query": "quote LA to Tokyo"}
    try:
        registry.set_enabled("agent", False, actor="test", reason="drill")
        r = client.post("/api/v1/agent/run", json=body)
        assert r.status_code == 404
    finally:
        registry.set_enabled("agent", True, actor="test", reason="restore")
    # restored: no longer 404 by the kill-switch (503 tool-registry is fine here)
    r2 = client.post("/api/v1/agent/run", json=body)
    assert r2.status_code != 404
