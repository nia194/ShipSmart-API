"""Guardrail metrics (Governance & Guardrails §11 — observability).

Turns the audit stream (AIEvents' ``decisions`` + ``guardrail_events`` tags) into
the small set of rates the runbooks alert on. Counting is prefix-based over the
same canonical tag vocabulary the evals join on, so a metric here and an eval
case in ShipSmart-Test are two views of one control.

Pure + deterministic: aggregation over an event list (the SQL twin lives in
ShipSmart-Infra as the ``ai_guardrail_daily`` view over ``ai_audit_log``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.ai_event import AIEvent

# Metric -> the tag prefixes that count toward it (prefix match: a dynamic
# detail suffix like ":injection:high" still counts toward its control).
METRIC_TAG_PREFIXES: dict[str, tuple[str, ...]] = {
    "injection_blocks": ("guardrail:blocked_injection", "guardrail:injection"),
    "refusals": ("guardrail:refused", "guardrail:misuse_refused"),
    "structured_output_invalid": ("guardrail:structured_output_invalid",),
    "quarantined_chunks": ("guardrail:quarantined_chunk",),
    "tool_denials": ("guardrail:tool_denied",),
    "state_integrity_failures": ("guardrail:state_unsigned",),
    "budget_exceeded": ("budget:exceeded",),
    "killswitch_flips": ("guardrail:killswitch",),
}


@dataclass(frozen=True)
class GuardrailMetrics:
    total_events: int = 0
    counts: dict[str, int] = field(default_factory=dict)

    def count(self, metric: str) -> int:
        return self.counts.get(metric, 0)

    def rate(self, metric: str) -> float:
        return self.count(metric) / self.total_events if self.total_events else 0.0


def _tags_of(event: AIEvent) -> list[str]:
    return [*event.decisions, *event.guardrail_events]


def collect(events: list[AIEvent]) -> GuardrailMetrics:
    """Aggregate the §11 guardrail counters over a window of AIEvents."""
    counts = dict.fromkeys(METRIC_TAG_PREFIXES, 0)
    for event in events:
        tags = _tags_of(event)
        for metric, prefixes in METRIC_TAG_PREFIXES.items():
            if any(t == p or t.startswith(p + ":") for t in tags for p in prefixes):
                counts[metric] += 1
    return GuardrailMetrics(total_events=len(events), counts=counts)


# Alert thresholds (rates over the window). structured_output_invalid > 2% is the
# §11 canonical alert; an injection-block spike suggests an attack wave (page).
DEFAULT_THRESHOLDS: dict[str, float] = {
    "structured_output_invalid": 0.02,
    "injection_blocks": 0.05,
    "state_integrity_failures": 0.02,
    "tool_denials": 0.05,
}

MIN_EVENTS_FOR_ALERTS = 20  # below this the window is too small to alert on


def check_thresholds(
    metrics: GuardrailMetrics, thresholds: dict[str, float] | None = None
) -> list[str]:
    """Human-readable alerts for every metric whose rate breaches its threshold."""
    if metrics.total_events < MIN_EVENTS_FOR_ALERTS:
        return []
    alerts = []
    for metric, ceiling in (thresholds or DEFAULT_THRESHOLDS).items():
        rate = metrics.rate(metric)
        if rate > ceiling:
            alerts.append(
                f"{metric} rate {rate:.1%} exceeds {ceiling:.1%} "
                f"({metrics.count(metric)}/{metrics.total_events} events)"
            )
    return alerts
