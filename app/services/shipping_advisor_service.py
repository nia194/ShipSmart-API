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

from app.llm.client import LLMClient
from app.llm.prompts import build_advisor_prompt
from app.rag.embeddings import EmbeddingProvider
from app.rag.retrieval import retrieve
from app.rag.vector_store import SearchResult, VectorStore
from app.services.orchestration_service import execute_tool
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class ShippingAdvice:
    """Structured shipping advice response."""

    answer: str
    reasoning_summary: str
    tools_used: list[str]
    sources: list[dict]
    context_used: bool


async def get_shipping_advice(
    query: str,
    context: dict | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
    llm_client: LLMClient | None = None,
    tool_registry: ToolRegistry | None = None,
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

    # Step 3: Build prompt for LLM
    context_text = ""
    if rag_sources:
        context_text = "\n\n".join([s.text for s in rag_sources])

    tool_text = ""
    if tool_results:
        tool_text = "\n\n".join(tool_results)

    prompt = build_advisor_prompt(query, context_text, tool_text)

    # Step 4: Get LLM response
    answer = ""
    reasoning_summary = ""
    if llm_client:
        answer = await llm_client.complete(prompt)
        # Extract first sentence as reasoning summary
        sentences = answer.split(".")
        reasoning_summary = (sentences[0] + ".") if sentences else answer[:100]
    else:
        answer = "No LLM configured. Could not generate shipping advice."
        reasoning_summary = answer

    return ShippingAdvice(
        answer=answer,
        reasoning_summary=reasoning_summary,
        tools_used=tools_used,
        sources=[
            {"source": s.source, "chunk_index": s.chunk_index, "score": round(s.score, 3)}
            for s in rag_sources
        ],
        context_used=context_used,
    )


