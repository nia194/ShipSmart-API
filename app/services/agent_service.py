"""
Agent service — the ShipSmart Concierge loop (model-driven reason → act → observe).

The reasoning client (``TASK_REASONING``) drives a bounded tool-calling loop over
the MCP read-only tools (``validate_address`` / ``get_quote_preview``) plus a
``retrieve_rag`` tool that wraps the existing single-shot RAG retrieval. Each step
the model decides which tool to call; observations are fed back until it stops or
the step cap is hit. The final, grounded answer is always produced through the
guardrailed assembler + the synthesis failover chain (``router.execute``), exactly
like the RAG/iterative paths — so grounding and failover are reused, not
re-invented.

Design notes:
  * The loop's control flow is the model's; the deterministic RAG layer stays pure
    (no LLM in its control flow) — retrieval is one pass per ``retrieve_rag`` call.
  * Providers without native tool calling raise ``NotImplementedError`` from
    ``complete_with_tools``; we catch it once and fall back to the existing
    single-pass text selection (``select_tool_with_llm``) — the keyless default.
  * MCP tools are dispatched through ``execute_tool`` so their input validation
    (422) and 502 handling are reused; a tool error becomes an observation the
    model can recover from rather than aborting the request.
  * Day-1 is read-only end to end: no persistence, no ``propose_shipment``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.core.errors import AppError
from app.integrations.mcp_client import RemoteToolRegistry as ToolRegistry
from app.llm.budget import estimate_tokens
from app.llm.client import ToolCallResult
from app.llm.guardrails import SAFE_REFUSAL, assemble
from app.llm.prompts import ADVISOR_SYSTEM_PROMPT
from app.llm.router import TASK_REASONING, TASK_SYNTHESIS, LLMRouter
from app.rag.embeddings import EmbeddingProvider
from app.rag.grounding import CoverageSignal, coverage_of
from app.rag.retrieval import retrieve_auto
from app.rag.vector_store import SearchResult, VectorStore
from app.services.orchestration_service import execute_tool, select_tool_with_llm

logger = logging.getLogger(__name__)


# In-API pseudo-tool: the model invokes retrieval under its own reasoning; the
# deterministic single-shot RAG sits underneath as the retrieval primitive.
RETRIEVE_RAG_SCHEMA: dict[str, Any] = {
    "name": "retrieve_rag",
    "description": (
        "Search the ShipSmart knowledge base for shipping policies, carrier "
        "info, packaging guidance, and restrictions (e.g. hazmat / lithium "
        "batteries / power banks). Returns grounded text chunks with scores."
    ),
    "parameters": [
        {
            "name": "query",
            "type": "string",
            "description": "What to search the knowledge base for.",
            "required": True,
        },
    ],
}

_AGENT_SYSTEM_PROMPT = (
    "You are ShipSmart's concierge agent. You help users with shipping tasks by "
    "planning and calling read-only tools, then giving a grounded answer.\n"
    "- Use `retrieve_rag` to look up shipping policies, restrictions (e.g. hazmat "
    "/ lithium batteries / power banks), carriers, rates, and packaging guidance.\n"
    "- Use `validate_address` to confirm an address is deliverable.\n"
    "- Use `get_quote_preview` to estimate shipping cost and transit time.\n"
    "Call tools as needed, one step at a time, reacting to each result. When you "
    "have enough information, stop calling tools and give your final answer.\n"
    "Each `retrieve_rag` result starts with a coverage signal "
    "(top_score, covered, chunk_count). If coverage is weak (covered=false or a "
    "low top_score), the knowledge base did not cover your question well: "
    "reformulate with a DIFFERENT, more specific query and retrieve again. For a "
    "compound question (e.g. shipping a drone abroad spans lithium batteries, "
    "electronics export, and the destination country's import rules), decompose "
    "it and retrieve each sub-area separately. Never repeat an identical query. "
    "When coverage is strong, proceed — do not retrieve again needlessly. If a "
    "sub-area is still uncovered after retrying, say so honestly in your answer "
    "rather than guessing."
)

_OBSERVATION_MAX_CHARS = 280

# Fed back as the observation when a retrieval is skipped, so the model gets a
# clear next-step instruction instead of a silently dropped tool call.
_DEGENERATE_RETRY_MSG = (
    "Retrieval skipped: this query is unchanged from a prior search this run. "
    "Reformulate with a different, more specific query or proceed to your answer."
)
_RETRIEVAL_CAP_MSG = (
    "Retrieval limit reached for this run (agent_max_retrievals). No further "
    "searches are allowed — synthesize your answer from the evidence gathered, "
    "and honestly flag any sub-area you could not cover."
)


@dataclass
class AgentResult:
    """Outcome of an agent run, mapped 1:1 onto the route response."""

    answer: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    provider: str = ""


# ── Helpers ──────────────────────────────────────────────────────────────────


def _format_user(query: str, context: dict | None) -> str:
    if context:
        return f"{query}\n\nContext: {json.dumps(context)}"
    return query


def _truncate(text: str, limit: int = _OBSERVATION_MAX_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _format_rag_observation(
    results: list[SearchResult], coverage: CoverageSignal,
) -> str:
    """Render the retrieval observation, leading with the coverage signal so the
    model can reason over result quality before reading the chunks."""
    header = coverage.as_line()
    if not results:
        return f"{header}\nNo relevant documents found in the ShipSmart knowledge base."
    body = "\n".join(f"[{r.source} score={r.score:.3f}] {r.text}" for r in results)
    return f"{header}\n{body}"


def _content_text(content: Any) -> str:
    """Flatten a message ``content`` (str or list of blocks) for token estimation."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, dict):
            parts.append(
                str(block.get("text") or block.get("content") or block.get("input") or "")
            )
        else:
            parts.append(str(block))
    return " ".join(parts)


def _trim_messages(
    messages: list[dict[str, Any]], max_context_tokens: int, max_output_tokens: int,
) -> list[dict[str, Any]]:
    """Trim the growing message list to the context budget before each LLM call.

    Keeps the system message (index 0) and the most recent message; drops the
    oldest middle messages first until ``prompt + reserved output`` fits. A long
    loop therefore can't overflow the context window.
    """
    def total() -> int:
        return sum(estimate_tokens(_content_text(m.get("content", ""))) for m in messages)

    while len(messages) > 2 and total() + max_output_tokens > max_context_tokens:
        del messages[1]
    return messages


def _assistant_turn(out: ToolCallResult) -> dict[str, Any]:
    """Echo the model's tool_use intent back into the history (native tool use)."""
    content: list[dict[str, Any]] = []
    if out.text:
        content.append({"type": "text", "text": out.text})
    for call in out.calls:
        content.append(
            {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
        )
    return {"role": "assistant", "content": content}


async def _dispatch(
    call,
    *,
    registry: ToolRegistry,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
) -> tuple[str, list[SearchResult], CoverageSignal | None]:
    """Run one tool call. Returns (observation_text, rag_chunks, coverage).

    ``retrieve_rag`` → existing single-shot RAG (one pass; chunks returned for
    grounding, plus an observable coverage signal). MCP tools → ``execute_tool``
    (input validation + 502 handling); an ``AppError`` becomes a recoverable
    observation rather than aborting, and ``coverage`` is None (not a retrieval).
    """
    name = call.name
    args = call.arguments or {}

    if name == "retrieve_rag":
        query = str(args.get("query") or "").strip()
        results, _mode = await retrieve_auto(
            query, embedding_provider, vector_store, top_k=settings.rag_top_k,
        )
        results = list(results)
        coverage = coverage_of(results)
        return _format_rag_observation(results, coverage), results, coverage

    try:
        res = await execute_tool(name, args, registry)
    except AppError as exc:
        # Hallucinated args / unknown tool / 502 → feed the error back as an
        # observation so the model can self-correct on the next step.
        return f"Tool '{name}' error: {exc.message}", [], None
    return res.answer, [], None


# ── Public entry point ───────────────────────────────────────────────────────


async def run_agent(
    query: str,
    context: dict | None,
    *,
    registry: ToolRegistry,
    llm_router: LLMRouter,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    max_steps: int | None = None,
    max_retrievals: int | None = None,
    request_id: str = "",
) -> AgentResult:
    """Run the model-driven agent loop and return a grounded answer.

    Retrieval is conditionally agentic: the model sees each ``retrieve_rag``
    result's coverage signal and may retrieve AGAIN with a different query when
    coverage is weak. Re-retrieval is bounded by ``agent_max_retrievals`` and
    guarded against identical-query loops, so a well-covered first retrieval
    stays single-shot (today's behavior).
    """
    max_steps = max_steps or settings.agent_max_steps
    max_retrievals = max_retrievals or settings.agent_max_retrievals
    reasoning = llm_router.for_task(TASK_REASONING)
    tools = [*registry.list_schemas(), RETRIEVE_RAG_SCHEMA]

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": _format_user(query, context)},
    ]

    decisions: list[str] = ["agent:plan"]
    steps_trace: list[dict[str, Any]] = []
    tools_used: list[str] = []
    rag_chunks: dict[tuple[str, int], SearchResult] = {}
    tool_text_parts: list[str] = []

    # Re-retrieval bookkeeping (conditional, bounded, non-degenerate).
    retrievals = 0
    retrieval_queries: set[str] = set()      # normalized queries already searched
    retrieval_coverages: list[bool] = []     # covered? per executed retrieval

    step = 0
    while step < max_steps:
        step += 1
        decisions.append(f"agent:step{step}")
        _trim_messages(messages, settings.llm_max_context_tokens, settings.llm_max_tokens)

        try:
            out = await reasoning.complete_with_tools(messages, tools)
        except NotImplementedError:
            # Provider has no native tool calling → single-pass text fallback.
            decisions.append("agent:fallback:text")
            return await _text_fallback(
                query, context,
                registry=registry, llm_router=llm_router,
                embedding_provider=embedding_provider, vector_store=vector_store,
                decisions=decisions, request_id=request_id,
            )

        if out.kind == "final":
            break

        messages.append(_assistant_turn(out))
        result_blocks: list[dict[str, Any]] = []
        for call in out.calls:
            if call.name == "retrieve_rag":
                # Conditional, bounded re-retrieval. Guard degenerate retries and
                # the per-run cap BEFORE touching the store; both feed a clear
                # observation back so the model can recover or proceed.
                norm = str((call.arguments or {}).get("query") or "").strip().lower()
                if norm and norm in retrieval_queries:
                    decisions.append("agent:retrieve:rejected")
                    observation = _DEGENERATE_RETRY_MSG
                elif retrievals >= max_retrievals:
                    decisions.append("agent:retrieve:capped")
                    observation = _RETRIEVAL_CAP_MSG
                else:
                    decisions.append("agent:tool:retrieve_rag")
                    tools_used.append("retrieve_rag")
                    # A retry after any prior weak-coverage retrieval is a
                    # justified reformulation (e.g. decomposing a compound query).
                    if retrievals >= 1 and not all(retrieval_coverages):
                        decisions.append("agent:retrieve:reformulate")
                    retrievals += 1
                    decisions.append(f"agent:retrieve:{retrievals}")
                    if norm:
                        retrieval_queries.add(norm)
                    observation, chunks, coverage = await _dispatch(
                        call, registry=registry,
                        embedding_provider=embedding_provider, vector_store=vector_store,
                    )
                    retrieval_coverages.append(bool(coverage and coverage.covered))
                    for r in chunks:
                        rag_chunks[(r.source, r.chunk_index)] = r
                steps_trace.append(
                    {"step": step, "tool": call.name, "observation": _truncate(observation)}
                )
                result_blocks.append(
                    {"type": "tool_result", "tool_use_id": call.id, "content": observation}
                )
                continue

            decisions.append(f"agent:tool:{call.name}")
            tools_used.append(call.name)
            observation, chunks, _coverage = await _dispatch(
                call, registry=registry,
                embedding_provider=embedding_provider, vector_store=vector_store,
            )
            steps_trace.append(
                {"step": step, "tool": call.name, "observation": _truncate(observation)}
            )
            if chunks:
                for r in chunks:
                    rag_chunks[(r.source, r.chunk_index)] = r
            else:
                tool_text_parts.append(f"{call.name}: {observation}")
            result_blocks.append(
                {"type": "tool_result", "tool_use_id": call.id, "content": observation}
            )
        messages.append({"role": "user", "content": result_blocks})
    else:
        # Loop exhausted without a final turn → force one synthesis pass below.
        decisions.append("agent:max_steps")

    # Honest gap: a sub-area the agent retried but still could not cover.
    if len(retrieval_coverages) >= 2 and not retrieval_coverages[-1]:
        decisions.append("agent:retrieve:uncovered")

    return await _synthesize(
        query,
        chunks=list(rag_chunks.values()),
        tool_text="\n\n".join(tool_text_parts),
        steps_trace=steps_trace, tools_used=tools_used,
        decisions=decisions, llm_router=llm_router, request_id=request_id,
    )


async def _synthesize(
    query: str,
    *,
    chunks: list[SearchResult],
    tool_text: str,
    steps_trace: list[dict[str, Any]],
    tools_used: list[str],
    decisions: list[str],
    llm_router: LLMRouter,
    request_id: str,
) -> AgentResult:
    """Ground the gathered evidence and answer via the synthesis failover chain."""
    assembled = assemble(
        system_prompt=ADVISOR_SYSTEM_PROMPT,
        user_text=query,
        contexts=chunks,
        tool_results=tool_text,
        request_id=request_id,
    )
    decisions.extend(assembled.decisions)
    if assembled.blocked:
        decisions.append("guardrail:blocked")
        return AgentResult(
            answer=assembled.refusal or SAFE_REFUSAL,
            steps=steps_trace, tools_used=tools_used, sources=[],
            decisions=decisions, provider="guardrail",
        )

    res = await llm_router.execute(TASK_SYNTHESIS, assembled.messages, request_id=request_id)
    sources = [
        {"source": r.source, "chunk_index": r.chunk_index, "score": round(r.score, 4)}
        for r in assembled.kept_sources
    ]
    return AgentResult(
        answer=res.text, steps=steps_trace, tools_used=tools_used,
        sources=sources, decisions=decisions, provider=res.provider,
    )


async def _text_fallback(
    query: str,
    context: dict | None,
    *,
    registry: ToolRegistry,
    llm_router: LLMRouter,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    decisions: list[str],
    request_id: str,
) -> AgentResult:
    """Single-pass path for providers without native tool calling.

    Reuses the existing text-based tool selection (``select_tool_with_llm``) — if
    a tool is chosen and the context supplies its params, run it — then does one
    RAG pass and grounds the answer through the synthesis chain.
    """
    reasoning = llm_router.for_task(TASK_REASONING)
    context = context or {}
    tools_used: list[str] = []
    tool_text_parts: list[str] = []
    steps_trace: list[dict[str, Any]] = []

    tool_name = await select_tool_with_llm(query, registry, reasoning)
    if tool_name:
        decisions.append(f"agent:tool:{tool_name}")
        tools_used.append(tool_name)
        tool = registry.get(tool_name)
        params = (
            {p.name: context[p.name] for p in tool.parameters if p.name in context}
            if tool else {}
        )
        try:
            res = await execute_tool(tool_name, params, registry)
            tool_text_parts.append(f"{tool_name}: {res.answer}")
            steps_trace.append({"step": 1, "tool": tool_name, "observation": _truncate(res.answer)})
        except AppError as exc:
            tool_text_parts.append(f"{tool_name} error: {exc.message}")
            steps_trace.append(
                {"step": 1, "tool": tool_name, "observation": _truncate(exc.message)}
            )

    results, _mode = await retrieve_auto(
        query, embedding_provider, vector_store, top_k=settings.rag_top_k,
    )
    return await _synthesize(
        query,
        chunks=results, tool_text="\n\n".join(tool_text_parts),
        steps_trace=steps_trace, tools_used=tools_used,
        decisions=decisions, llm_router=llm_router, request_id=request_id,
    )
