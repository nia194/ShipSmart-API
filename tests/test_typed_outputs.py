"""Typed model-output schemas + the output validator (guardrails §5.3)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.llm.output_validator import (
    STRUCTURED_OUTPUT_INVALID_TAG,
    STRUCTURED_OUTPUT_RETRY_TAG,
    safe_fallback,
    validate_model_output,
)
from app.schemas.typed_outputs import (
    AssistantResponse,
    FieldPatch,
    FormPatchProposal,
    ToolCallPolicy,
)


def test_assistant_response_valid_shapes():
    r = AssistantResponse(type="answer", message="hi", risk_tier="read")
    assert r.type == "answer" and r.sources == [] and r.actions == []
    r2 = AssistantResponse(
        type="form_patch",
        form_patch=FormPatchProposal(
            patches=[FieldPatch(field_path="origin.city", new_value="Atlanta")]
        ),
    )
    assert r2.form_patch.patches[0].field_path == "origin.city"


def test_invalid_enums_and_ranges_rejected():
    with pytest.raises(ValidationError):
        AssistantResponse(type="not_a_type")  # bad ResponseType literal
    with pytest.raises(ValidationError):
        FieldPatch(field_path="x", new_value=1, confidence=2.0)  # out of 0..1
    with pytest.raises(ValidationError):
        ToolCallPolicy(tool_name="t", risk_tier="nuclear")  # bad RiskTier


def test_validator_passes_valid_output():
    out = validate_model_output({"type": "answer", "message": "ok"})
    assert isinstance(out, AssistantResponse) and out.type == "answer"


def test_validator_fails_safely_without_corrector():
    tags: list[str] = []
    out = validate_model_output("not json at all", decisions=tags)
    assert out.type == "refusal"
    assert STRUCTURED_OUTPUT_INVALID_TAG in tags


def test_validator_retries_once_then_succeeds():
    tags: list[str] = []
    out = validate_model_output(
        "{bad json}",
        corrector=lambda _raw, _err: '{"type": "answer", "message": "fixed"}',
        decisions=tags,
    )
    assert out.type == "answer" and out.message == "fixed"
    assert STRUCTURED_OUTPUT_RETRY_TAG in tags


def test_validator_falls_back_when_retry_also_invalid():
    tags: list[str] = []
    out = validate_model_output(
        "{bad}",
        corrector=lambda _raw, _err: "{still bad}",
        decisions=tags,
    )
    assert out.type == "refusal"
    assert STRUCTURED_OUTPUT_INVALID_TAG in tags


def test_safe_fallback_is_a_refusal():
    assert safe_fallback().type == "refusal"
