"""
Shipping Advisor Service.
Answers shipping-related questions by combining RAG context, tool execution,
and LLM reasoning.

Flow:
1. Retrieve relevant RAG context
2. Determine if tools are needed (address validation, quote preview)
3. Execute tools if needed
4. Pass context + tool results to LLM for reasoned advice
5. Return structured answer
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from app.integrations.mcp_client import RemoteToolRegistry as ToolRegistry
from app.llm.client import LLMClient
from app.llm.guardrails import SAFE_REFUSAL, assemble
from app.llm.prompts import ADVISOR_SYSTEM_PROMPT
from app.llm.reply_context import render_reference_block
from app.llm.router import TASK_REASONING, LLMRouter
from app.rag.embeddings import EmbeddingProvider
from app.rag.retrieval import retrieve
from app.rag.vector_store import SearchResult, VectorStore
from app.services.orchestration_service import execute_tool

logger = logging.getLogger(__name__)


@dataclass
class ShippingAdvice:
    """Structured shipping advice response."""

    answer: str
    reasoning_summary: str
    tools_used: list[str]
    sources: list[dict]
    context_used: bool
    decision_path: dict | None = None


async def get_shipping_advice(
    query: str,
    context: dict | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
    llm_client: LLMClient | None = None,
    tool_registry: ToolRegistry | None = None,
    llm_router: LLMRouter | None = None,
    task: str = TASK_REASONING,
    reply_to: object | None = None,
    recent_history: list | None = None,
    request_id: str = "",
) -> ShippingAdvice:
    """Generate shipping advice by combining RAG, tools, and LLM.

    Args:
        query: User's shipping question.
        context: Optional context dict (origin_zip, destination_zip, weight_lbs, etc.)
        embedding_provider: For RAG retrieval.
        vector_store: For RAG retrieval.
        llm_client: For generating advice.
        tool_registry: For tool selection/execution.

    Returns:
        ShippingAdvice with answer, reasoning, tools used, and sources.
    """
    tools_used: list[str] = []
    rag_sources: list[SearchResult] = []
    tool_results: list[str] = []

    # Step 1: Retrieve RAG context
    context_used = False
    if embedding_provider and vector_store:
        rag_sources = await retrieve(
            query, embedding_provider, vector_store, top_k=5,
        )
        context_used = len(rag_sources) > 0
        logger.info("Retrieved %d RAG sources for shipping query", len(rag_sources))

    # Step 2: Determine if tools are needed
    if context and tool_registry:
        # If origin/dest/weight/dimensions provided → get quote preview
        if all(k in context for k in ["origin_zip", "destination_zip", "weight_lbs"]):
            try:
                tool_result = await execute_tool(
                    "get_quote_preview",
                    {
                        "origin_zip": context["origin_zip"],
                        "destination_zip": context["destination_zip"],
                        "weight_lbs": context.get("weight_lbs", 1.0),
                        "length_in": context.get("length_in", 10.0),
                        "width_in": context.get("width_in", 8.0),
                        "height_in": context.get("height_in", 6.0),
                    },
                    tool_registry,
                )
                tools_used.append("get_quote_preview")
                tool_results.append(f"Quote Preview: {json.dumps(tool_result.data)}")
                logger.info("Executed get_quote_preview tool")
            except Exception as exc:
                logger.warning("Tool execution failed: %s", exc)

        # If address provided → validate
        if all(k in context for k in ["street", "city", "state", "zip_code"]):
            try:
                tool_result = await execute_tool(
                    "validate_address",
                    {
                        "street": context["street"],
                        "city": context["city"],
                        "state": context["state"],
                        "zip_code": context["zip_code"],
                    },
                    tool_registry,
                )
                tools_used.append("validate_address")
                tool_results.append(f"Address Validation: {json.dumps(tool_result.data)}")
                logger.info("Executed validate_address tool")
            except Exception as exc:
                logger.warning("Tool execution failed: %s", exc)

    # Step 3: Assemble a fenced/grounded/guardrailed prompt (C/D). The optional reply-to
    # reference is bounded + fence-stripped here; the live tool results above stay
    # authoritative (see _REPLY_CONTEXT_RULES in guardrails).
    tool_text = "\n\n".join(tool_results) if tool_results else ""
    reference_block = render_reference_block(reply_to, recent_history)
    assembled = assemble(
        system_prompt=ADVISOR_SYSTEM_PROMPT,
        user_text=query,
        contexts=rag_sources,
        tool_results=tool_text,
        reference_block=reference_block,
        request_id=request_id,
    )
    tags = list(assembled.decisions)
    if tools_used:
        tags.append("tools:rule")  # tool triggering is deterministic (rule-based)
    if reference_block:
        tags.append("advisor:reply_to")

    sources = [
        {"source": s.source, "chunk_index": s.chunk_index, "score": round(s.score, 3)}
        for s in rag_sources
    ]

    # Step 4: Guardrail block short-circuits before any LLM call.
    if assembled.blocked:
        refusal = assembled.refusal or SAFE_REFUSAL
        return ShippingAdvice(
            answer=refusal, reasoning_summary=refusal, tools_used=tools_used,
            sources=sources, context_used=context_used,
            decision_path={
                "mode": "normal", "retrieval": "dense", "answer": "rule",
                "provider": "guardrail", "tags": tags,
            },
        )

    # Step 5: Get LLM response (via failover chain when a router is provided).
    provider, failed_over = "none", False
    if llm_router is not None:
        res = await llm_router.execute(task, assembled.messages, request_id=request_id)
        answer, provider, failed_over = res.text, res.provider, res.failed_over
    elif llm_client:
        answer = await llm_client.complete(assembled.messages)
        provider = getattr(llm_client, "provider_name", "")
    else:
        answer = "No LLM configured. Could not generate shipping advice."

    sentences = answer.split(".")
    reasoning_summary = (sentences[0] + ".") if sentences and sentences[0] else answer[:100]
    src = (
        "rule" if provider in ("", "none")
        else "fallback" if (provider == "echo" or failed_over)
        else "llm"
    )

    return ShippingAdvice(
        answer=answer,
        reasoning_summary=reasoning_summary,
        tools_used=tools_used,
        sources=sources,
        context_used=context_used,
        decision_path={
            "mode": "normal", "retrieval": "dense", "answer": src,
            "provider": provider, "tags": tags,
        },
    )


