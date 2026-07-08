"""Output validator (Governance & Guardrails Control System §5.3).

Every model response is validated against its schema; on invalid output we retry
**once** with a correction, then fail **safely** to a structured refusal — model
output becomes typed data the product can render or reject, never trusted prose.

Emits ``guardrail:structured_output_invalid`` on the fallback path so the failure
is visible in ``decision_path`` and countable for the §11 ``structured_output_invalid_rate``
alert.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from pydantic import BaseModel, ValidationError

from app.schemas.typed_outputs import AssistantResponse

# A corrector re-prompts the model with the raw output + the validation error and
# returns a corrected raw string. None => no retry (deterministic/keyless callers).
Corrector = Callable[[str, str], str]

STRUCTURED_OUTPUT_INVALID_TAG = "guardrail:structured_output_invalid"
STRUCTURED_OUTPUT_RETRY_TAG = "guardrail:structured_output_retry"


def _coerce(raw: str | dict) -> dict:
    if isinstance(raw, dict):
        return raw
    return json.loads(raw)


def validate_model_output(
    raw: str | dict,
    model: type[BaseModel] = AssistantResponse,
    *,
    corrector: Corrector | None = None,
    decisions: list[str] | None = None,
) -> BaseModel:
    """Validate ``raw`` against ``model``; retry once via ``corrector``; else safe-fallback.

    ``decisions`` (if given) is appended in place with the guardrail tag taken.
    Returns a valid model instance — never raises on bad model output.
    """
    tags = decisions if decisions is not None else []
    try:
        return model.model_validate(_coerce(raw))
    except (ValidationError, json.JSONDecodeError, TypeError) as first_err:
        if corrector is not None:
            try:
                raw_str = raw if isinstance(raw, str) else json.dumps(raw)
                corrected = corrector(raw_str, str(first_err))
                result = model.model_validate(_coerce(corrected))
                tags.append(STRUCTURED_OUTPUT_RETRY_TAG)
                return result
            except (ValidationError, json.JSONDecodeError, TypeError):
                pass
        tags.append(STRUCTURED_OUTPUT_INVALID_TAG)
        return safe_fallback()


def safe_fallback(
    message: str = "I couldn't produce a valid response, so I'm not guessing.",
) -> AssistantResponse:
    """The structured refusal a caller renders when model output can't be trusted."""
    return AssistantResponse(type="refusal", message=message, risk_tier="read")
