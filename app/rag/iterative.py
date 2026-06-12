"""
Iterative RAG (G) — bounded plan → retrieve → assess → (tools) → ground+answer.

Used by the RAG/advisor paths when RAG_MODE=iterative; RAG_MODE=normal keeps the
single-shot pipeline. The loop is cost-bounded (RAG_ITERATIVE_MAX_STEPS), reuses
hybrid retrieval (F), respects the context budget + guardrails via the assembler
(B/C/D), answers through the router's failover chain (A), and tags every decision
(E). It NEVER blocks the response on trace logging.

This layer is DETERMINISTIC — there is no LLM in its control flow (the model is
used only for the final grounded answer). The genuinely model-driven loop lives
in ``app/services/agent_service.py``; this module is honest "iterative retrieval
with grounding," which is why it is named ``iterative`` rather than "agentic".

Design choices for determinism + testability:
  * the retriever is injected (``RetrieverFn``) so tests drive coverage without a
    real store/embeddings; ``make_retriever`` wraps ``retrieve_auto`` for prod.
  * reformulation + coverage are deterministic heuristics (no hidden LLM calls in
    the control flow); the LLM is used only for the final grounded answer.
  * when retrieval never covers the question, we refuse deterministically rather
    than letting the model guess (grounding, D).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from app.core.config import settings
from app.llm.client import LLMClient
from app.llm.guardrails import SAFE_REFUSAL, assemble
from app.llm.prompts import SYSTEM_PROMPT
from app.llm.router import TASK_SYNTHESIS, LLMRouter
from app.rag.embeddings import EmbeddingProvider
from app.rag.retrieval import retrieve_auto
from app.rag.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)

RetrieverFn = Callable[[str, int], Awaitable[list[SearchResult]]]

_UNCOVERED_REFUSAL = (
    "I don't have enough information in the ShipSmart knowledge base to answer "
    "that confidently. Try rephrasing, or ask about carriers, rates, packaging, "
    "or delivery issues."
)


@dataclass
class IterativeRagResult:
    answer: str
    sources: list[SearchResult] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    provider: str = ""
    steps: int = 0
    tools_used: list[str] = field(default_factory=list)
    blocked: bool = False
    answer_source: str = "llm"  # rule | llm | fallback


def make_retriever(
    embedding_provider: EmbeddingProvider, vector_store: VectorStore,
) -> RetrieverFn:
    """Default retriever: hybrid-or-dense per config (F)."""
    async def _retrieve(q: str, k: int) -> list[SearchResult]:
        results, _mode = await retrieve_auto(q, embedding_provider, vector_store, top_k=k)
        return results
    return _retrieve


def _covered(results: list[SearchResult]) -> bool:
    """Coverage heuristic: at least one retrieved chunk with positive score."""
    return any(getattr(r, "score", 0.0) > 0.0 for r in results)


def _reformulate(original: str, step: int) -> str:
    """Deterministic query reformulation when coverage is insufficient."""
    suffixes = ["shipping carrier delivery options", "rates packaging policy guidance"]
    return f"{original} {suffixes[min(step, len(suffixes) - 1)]}"


async def _maybe_escalate_tools(
    query: str, context: dict | None, tool_registry,
) -> tuple[list[str], str]:
    """Call MCP tools for ground truth when the query needs it and params exist.

    Returns (tools_used, tool_results_text). Best-effort: any failure is swallowed
    so the loop still answers from retrieved context.
    """
    if not context or tool_registry is None:
        return [], ""
    from app.services.orchestration_service import execute_tool

    tools_used: list[str] = []
    parts: list[str] = []
    quote_keys = ("origin_zip", "destination_zip", "weight_lbs")
    addr_keys = ("street", "city", "state", "zip_code")
    try:
        if all(k in context for k in quote_keys) and tool_registry.get("get_quote_preview"):
            res = await execute_tool("get_quote_preview", {
                "origin_zip": context["origin_zip"],
                "destination_zip": context["destination_zip"],
                "weight_lbs": context.get("weight_lbs", 1.0),
                "length_in": context.get("length_in", 10.0),
                "width_in": context.get("width_in", 8.0),
                "height_in": context.get("height_in", 6.0),
            }, tool_registry)
            tools_used.append("get_quote_preview")
            parts.append(f"Quote Preview: {json.dumps(res.data)}")
        if all(k in context for k in addr_keys) and tool_registry.get("validate_address"):
            res = await execute_tool("validate_address", {
                k: context[k] for k in addr_keys
            }, tool_registry)
            tools_used.append("validate_address")
            parts.append(f"Address Validation: {json.dumps(res.data)}")
    except Exception as exc:  # noqa: BLE001 - tools are best-effort in the loop
        logger.warning("Iterative-RAG tool escalation failed: %s", exc)
    return tools_used, "\n\n".join(parts)


async def iterative_rag(
    query: str,
    *,
    retriever: RetrieverFn,
    llm_router: LLMRouter | None = None,
    llm_client: LLMClient | None = None,
    tool_registry=None,
    context: dict | None = None,
    task: str = TASK_SYNTHESIS,
    top_k: int | None = None,
    max_steps: int | None = None,
    request_id: str = "",
) -> IterativeRagResult:
    """Run the bounded iterative retrieval loop and return a grounded answer."""
    top_k = top_k or settings.rag_top_k
    max_steps = max_steps or getattr(settings, "rag_iterative_max_steps", 3)
    decisions: list[str] = ["iterative:plan"]

    # PLAN → RETRIEVE → ASSESS loop (bounded).
    accumulated: dict[tuple[str, int], SearchResult] = {}
    current_q = query
    steps = 0
    while steps < max_steps:
        steps += 1
        results = await retriever(current_q, top_k)
        for r in results:
            accumulated[(r.source, r.chunk_index)] = r
        decisions.append(f"iterative:step{steps}:retrieved:{len(results)}")
        if _covered(results):
            break
        if steps < max_steps:
            current_q = _reformulate(query, steps - 1)
            decisions.append("iterative:reformulate")

    chunks = sorted(accumulated.values(), key=lambda r: r.score, reverse=True)[:top_k]

    # Optional tool escalation for ground truth.
    tools_used, tool_text = await _maybe_escalate_tools(query, context, tool_registry)
    if tools_used:
        decisions.append("iterative:tools:" + ",".join(tools_used))

    # REFUSE deterministically when nothing covered the question (D).
    if not chunks and not tool_text:
        decisions.append("iterative:uncovered_refusal")
        return IterativeRagResult(
            answer=_UNCOVERED_REFUSAL, sources=[], decisions=decisions,
            provider="rule", steps=steps, tools_used=tools_used, answer_source="rule",
        )

    # GROUND + ANSWER through the guardrailed assembler + router chain.
    assembled = assemble(
        system_prompt=SYSTEM_PROMPT, user_text=query,
        contexts=chunks, tool_results=tool_text, request_id=request_id,
    )
    decisions.extend(assembled.decisions)
    if assembled.blocked:
        decisions.append("guardrail:blocked")
        return IterativeRagResult(
            answer=assembled.refusal or SAFE_REFUSAL, sources=[], decisions=decisions,
            provider="guardrail", steps=steps, tools_used=tools_used,
            blocked=True, answer_source="rule",
        )

    provider, failed_over = "none", False
    if llm_router is not None:
        res = await llm_router.execute(task, assembled.messages, request_id=request_id)
        answer, provider, failed_over = res.text, res.provider, res.failed_over
    elif llm_client is not None:
        answer = await llm_client.complete(assembled.messages)
        provider = getattr(llm_client, "provider_name", "")
    else:
        answer = _UNCOVERED_REFUSAL
        provider = "none"

    answer_source = (
        "rule" if provider in ("", "none")
        else "fallback" if (provider == "echo" or failed_over)
        else "llm"
    )
    return IterativeRagResult(
        answer=answer, sources=assembled.kept_sources, decisions=decisions,
        provider=provider, steps=steps, tools_used=tools_used, answer_source=answer_source,
    )
