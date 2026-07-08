"""Typed model-output schemas (Governance & Guardrails Control System §5.3 + Appendix).

The AI boundary must return **typed data the product can render or reject** — never
trusted prose that drives a UI mutation or a backend action. These are the strict
schemas a model response is validated against (see ``app/llm/output_validator.py``):

- ``AssistantResponse`` — the top-level envelope every model turn conforms to.
- ``FieldPatch`` / ``FormPatchProposal`` — proposed form changes (never applied from raw prose).
- ``ToolCallPolicy`` — the per-tool risk/authorization contract the planner validates against.
- ``Refusal`` — a structured refusal (also expressible as ``AssistantResponse(type="refusal")``).

Additive and standalone in F1: the schemas + validator exist and are contract-checked
against the Web TypeScript types; wiring them into every live route is a later product phase.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RiskTier = Literal["read", "quote", "write", "high"]
PatchSource = Literal["user_text", "tool_result", "quote_data"]
ResponseType = Literal["answer", "form_patch", "ask_followup", "refusal"]


class SourceCitation(BaseModel):
    """A grounded citation — a policy/restriction answer must carry at least one."""

    source: str
    chunk_index: int | None = None
    score: float | None = None


class Action(BaseModel):
    """A typed action the assistant proposes — executed only after policy + confirmation."""

    name: str
    risk_tier: RiskTier = "read"
    params: dict[str, Any] = Field(default_factory=dict)
    requires_confirmation: bool = False


class FieldPatch(BaseModel):
    """One proposed form-field change. Never mutates a field from raw prose."""

    field_path: str = Field(..., examples=["origin.city"])
    new_value: Any
    old_value: Any | None = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reason: str = ""
    source: PatchSource = "tool_result"
    requires_confirmation: bool = False


class FormPatchProposal(BaseModel):
    """A set of proposed field patches the form can apply or reject per-field."""

    patches: list[FieldPatch] = Field(default_factory=list)


class Refusal(BaseModel):
    """A structured refusal — a safe user-facing message plus its machine reason + tag."""

    reason: str
    safe_message: str = "I can't help with that request."
    tag: str = "guardrail:refused"


class ToolCallPolicy(BaseModel):
    """Per-tool risk + authorization contract the agent planner validates against."""

    tool_name: str
    version: str = "v1"
    risk_tier: RiskTier = "read"
    allowed_routes: list[str] = Field(default_factory=list)
    max_calls_per_request: int = Field(3, ge=0)
    requires_confirmation: bool = False


class AssistantResponse(BaseModel):
    """The typed envelope every model turn conforms to — the product renders this, not prose."""

    type: ResponseType
    message: str = ""
    sources: list[SourceCitation] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    form_patch: FormPatchProposal | None = None
    risk_tier: RiskTier = "read"
    requires_confirmation: bool = False
