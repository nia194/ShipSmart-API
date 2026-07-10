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

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

RiskTier = Literal["read", "quote", "write", "high"]
PatchSource = Literal["user_text", "tool_result", "quote_data"]
ResponseType = Literal["answer", "form_patch", "ask_followup", "refusal"]

# Product Roadmap §6 vocabulary (the structured assistant contract).
AssistantIntent = Literal[
    "form_fill", "form_edit", "quote_search", "recommendation", "compare_options",
    "policy_question", "package_help", "tracking_question", "general_question",
]
ApplyPolicy = Literal["auto", "confirm", "none"]
ResultLabel = Literal["Cheapest", "Fastest", "Best value", "Safest"]


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


# ── Typed result union (Product Roadmap §6) — the product renders a card per type ──
class NextQuestion(BaseModel):
    """The single next field to ask for, with the exact prompt text."""

    field: str
    text: str


class ShippingOptionResult(BaseModel):
    """A ranked quote card — the number is always Java's, keyed by ``quote_id``."""

    type: Literal["shipping_option"] = "shipping_option"
    label: ResultLabel
    quote_id: str
    carrier: str
    service_name: str
    price_usd: float
    transit_days: int
    estimated_delivery_date: str | None = None
    reason: str = ""
    badges: list[str] = Field(default_factory=list)


class ComparisonResult(BaseModel):
    type: Literal["comparison"] = "comparison"
    options: list[str] = Field(default_factory=list)
    summary: str = ""


class MissingInfoResult(BaseModel):
    type: Literal["missing_info"] = "missing_info"
    missing_fields: list[str] = Field(default_factory=list)
    next_question: str = ""


class PolicyAnswerResult(BaseModel):
    type: Literal["policy_answer"] = "policy_answer"
    answer: str = ""
    sources: list[SourceCitation] = Field(default_factory=list)


AssistantResult = Annotated[
    ShippingOptionResult | ComparisonResult | MissingInfoResult | PolicyAnswerResult,
    Field(discriminator="type"),
]


class ToolCallTrace(BaseModel):
    """UI-visible tool-call transparency (name + args SHAPE, never raw args)."""

    name: str
    args_shape: list[str] = Field(default_factory=list)
    status: str = "ok"
    latency_ms: float = 0.0


class AssistantAudit(BaseModel):
    """Which model/provider produced this turn + how it was selected."""

    model: str = ""
    provider: str = ""
    selection_method: str = ""
    latency_ms: float = 0.0


class AssistantResponse(BaseModel):
    """The typed envelope every model turn conforms to — the product renders this, not prose.

    The Product Roadmap §6 fields (intent, apply_policy, confidence, missing_fields,
    next_question, typed result, tool_calls, audit) are ADDITIVE: the F1 fields keep
    their meaning, old clients keep working, and ``apply_policy``/``confidence`` are
    decided by deterministic policy code, never LLM self-report.
    """

    type: ResponseType
    message: str = ""
    sources: list[SourceCitation] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    form_patch: FormPatchProposal | None = None
    risk_tier: RiskTier = "read"
    requires_confirmation: bool = False
    # ── Product Roadmap §6 additions (additive) ──
    schema_version: str = "1"
    intent: AssistantIntent | None = None
    apply_policy: ApplyPolicy = "none"
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    missing_fields: list[str] = Field(default_factory=list)
    next_question: NextQuestion | None = None
    result: AssistantResult | None = None
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    audit: AssistantAudit | None = None
