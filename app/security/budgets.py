"""Per-session budgets (Governance & Guardrails §5.2).

slowapi already rate-limits per IP; this adds a per-session cost ceiling —
LLM calls, tool calls, and tokens — so a single conversation can't run the model
in a loop. In-memory / per-process (a durable store is a later swap). Emits
``budget:exceeded``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BUDGET_EXCEEDED_TAG = "budget:exceeded"


@dataclass(frozen=True)
class Budget:
    max_llm_calls: int = 20
    max_tool_calls: int = 30
    max_tokens: int = 100_000


@dataclass
class SessionBudget:
    """Running counters for one session against a :class:`Budget`."""

    budget: Budget = field(default_factory=Budget)
    llm_calls: int = 0
    tool_calls: int = 0
    tokens: int = 0

    def exceeded(self) -> bool:
        b = self.budget
        return (
            self.llm_calls > b.max_llm_calls
            or self.tool_calls > b.max_tool_calls
            or self.tokens > b.max_tokens
        )

    def consume(self, *, llm_calls: int = 0, tool_calls: int = 0, tokens: int = 0) -> list[str]:
        """Add usage; return ``[budget:exceeded]`` once any ceiling is crossed."""
        self.llm_calls += llm_calls
        self.tool_calls += tool_calls
        self.tokens += tokens
        return [BUDGET_EXCEEDED_TAG] if self.exceeded() else []
