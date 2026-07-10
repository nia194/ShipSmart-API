"""The unified AI event (Governance & Guardrails Control System §5.8 + Appendix).

One record per model/tool call. Identity is already pseudonymized
(``session_id_hash``) and all free text is PII-redacted before an event is built
(see ``app.core.ai_events.build_ai_event``). The correlation keys mirror the eval
``EvalTrace`` (evals §3.4) so production events can be sampled and replayed as
eval runs (the Layer-6 online loop).

``prompt_version`` / ``schema_version`` / ``embedding_version`` are the reproducibility
fields flagged as "to-add in Phase 2" by both design docs' addenda.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Bump on any system-prompt / output-schema change so an event can be reproduced
# from its exact configuration (evals §7.3 / guardrails §7.3).
PROMPT_VERSION = "v1"
SCHEMA_VERSION = "v1"


class AIEvent(BaseModel):
    """A single, append-only, PII-safe record of a model/tool call."""

    request_id: str = ""
    session_id_hash: str | None = None       # pseudonymized at write time (§6.1)
    route: str = ""
    intent: str = ""
    provider: str = ""
    model: str = ""
    prompt_version: str = PROMPT_VERSION
    schema_version: str = SCHEMA_VERSION
    embedding_version: str = ""
    decisions: list[str] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    guardrail_events: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    token_count: int = 0
    cost_estimate_usd: float = 0.0
    # Explicit user feedback (§6.6 / Layer-6 online loop). PII-redacted at build
    # time; empty for every non-feedback event. Additive — correlation keys above
    # are what the EvalTrace contract pins.
    feedback_comment: str = ""
