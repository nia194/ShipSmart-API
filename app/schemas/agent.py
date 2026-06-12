"""
Request and response schemas for the agent (Concierge) endpoint.
"""

from typing import Any

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    """A single free-text concierge request, with optional structured context."""

    query: str = Field(..., min_length=1, max_length=2000)
    context: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional context the agent can use for tools: origin_zip, "
            "destination_zip, weight_lbs, dimensions, address fields, etc."
        ),
    )


class AgentStep(BaseModel):
    """One tool-execution step in the agent's trace."""

    step: int
    tool: str
    observation: str


class AgentResponse(BaseModel):
    """The agent's grounded answer plus its reasoning trace (debuggable)."""

    answer: str
    steps: list[AgentStep] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    provider: str = ""
