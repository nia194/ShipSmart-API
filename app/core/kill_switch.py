"""Runtime AI kill-switches (Governance & Guardrails §12 — incident response).

When an AI feature misbehaves in production (an injection wave, a runaway agent
loop, a poisoned corpus), the incident response must not be "redeploy with a new
env var" — that is minutes-to-hours of exposure. This registry lets an operator
disable a specific AI FEATURE at runtime through the admin endpoint, while the
guardrails themselves stay on (guardrails are never killable — the switch turns
off capability, never protection).

Every flip is recorded as an append-only AIEvent (pseudonymized, redacted), so a
kill/restore is attributable and auditable after the incident. The env-level
``*_enabled`` settings remain the deploy-time baseline; a runtime kill overrides
an enabled feature until re-enabled or restart.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime

from app.core.ai_events import AIEventSink, create_ai_event_sink, record_ai_event
from app.core.config import settings
from app.core.errors import AppError

# The features an operator may kill at runtime. Deliberately NOT here: the
# guardrail/security modules — protection cannot be switched off by this path.
KILLABLE_FEATURES = ("agent", "concierge", "workflow", "compliance", "rag")

KILLSWITCH_TAG = "guardrail:killswitch"  # + ":{feature}:{on|off}" detail suffix


@dataclass(frozen=True)
class FeatureState:
    feature: str
    enabled: bool
    actor: str
    reason: str
    changed_at: str


class KillSwitchRegistry:
    """Thread-safe runtime feature switches with an audited flip path."""

    def __init__(self, sink: AIEventSink | None = None) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, FeatureState] = {
            f: FeatureState(f, True, "boot", "default", datetime.now(UTC).isoformat())
            for f in KILLABLE_FEATURES
        }
        self._sink = sink

    def _audit_sink(self) -> AIEventSink:
        if self._sink is None:
            self._sink = create_ai_event_sink(settings.ai_event_sink)
        return self._sink

    def is_enabled(self, feature: str) -> bool:
        state = self._states.get(feature)
        if state is None:
            raise ValueError(f"unknown killable feature {feature!r} (not in {KILLABLE_FEATURES})")
        return state.enabled

    def set_enabled(self, feature: str, enabled: bool, *, actor: str, reason: str) -> FeatureState:
        if feature not in self._states:
            raise ValueError(f"unknown killable feature {feature!r} (not in {KILLABLE_FEATURES})")
        with self._lock:
            state = FeatureState(
                feature, enabled, actor, reason, datetime.now(UTC).isoformat()
            )
            self._states[feature] = state
        # Best-effort append-only audit of the flip (never blocks the flip itself).
        try:
            record_ai_event(
                self._audit_sink(),
                secret=settings.pseudonym_secret,
                route="/api/v1/admin/ai-controls",
                intent="killswitch",
                decisions=[f"{KILLSWITCH_TAG}:{feature}:{'on' if enabled else 'off'}"],
            )
        except Exception:  # noqa: BLE001 - auditing must not break incident response
            pass
        return state

    def snapshot(self) -> dict[str, bool]:
        return {f: s.enabled for f, s in sorted(self._states.items())}

    def states(self) -> list[FeatureState]:
        return [self._states[f] for f in sorted(self._states)]


# Process-wide registry the routers and the admin endpoint share.
registry = KillSwitchRegistry()


def require_feature(feature: str):
    """FastAPI dependency: 404 the whole router when ``feature`` is killed.

    Matches the env-flag convention (disabled features 404 rather than 403) so a
    kill is indistinguishable from the feature not existing.
    """
    if feature not in KILLABLE_FEATURES:  # fail at import/wiring time, not request time
        raise ValueError(f"unknown killable feature {feature!r}")

    def _dep() -> None:
        if not registry.is_enabled(feature):
            raise AppError(status_code=404, message=f"{feature} endpoint is disabled")

    return _dep
