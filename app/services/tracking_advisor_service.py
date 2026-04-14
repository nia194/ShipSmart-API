"""
Tracking / Delivery Guidance Assistant Service.
Provides guidance on delivery issues, tracking problems, and address-related concerns
by combining RAG context and optional tools.

Flow:
1. Retrieve relevant RAG context (delivery, tracking, common issues)
2. Optionally validate address if provided
3. Pass context + optional tool results to LLM
4. Return structured guidance
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from app.llm.client import LLMClient
from app.rag.embeddings import EmbeddingProvider
from app.rag.retrieval import retrieve
from app.rag.vector_store import SearchResult, VectorStore
from app.services.orchestration_service import execute_tool
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class TrackingGuidance:
    """Structured tracking guidance response."""

    guidance: str
    issue_summary: str
    tools_used: list[str]
    sources: list[dict]
    next_steps: list[str]


async def get_tracking_guidance(
    issue: str,
    context: dict | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
    llm_client: LLMClient | None = None,
    tool_registry: ToolRegistry | None = None,
) -> TrackingGuidance:
    """Generate tracking/delivery guidance by combining RAG and optional tools.

    Args:
        issue: User's tracking or delivery issue.
        context: Optional context (tracking_number, address fields, etc.)
        embedding_provider: For RAG retrieval.
        vector_store: For RAG retrieval.
        llm_client: For generating guidance.
        tool_registry: For optional tool execution.

    Returns:
        TrackingGuidance with advice, issue summary, tools used, sources.
    """
    tools_used: list[str] = []
    rag_sources: list[SearchResult] = []
    tool_data = ""

    # Step 1: Retrieve RAG context (focus on delivery/tracking/issues)
    if embedding_provider and vector_store:
        # Augment query with delivery/tracking keywords for better retrieval
        enriched_query = f"{issue} delivery tracking shipping issue"
        rag_sources = await retrieve(
            enriched_query, embedding_provider, vector_store, top_k=5,
        )
        logger.info("Retrieved %d RAG sources for tracking issue", len(rag_sources))

    # Step 2: Optionally validate address if provided
    if context and tool_registry and all(
        k in context for k in ["street", "city", "state", "zip_code"]
    ):
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
            tool_data = f"Address Check: {json.dumps(tool_result.data)}"
            logger.info("Executed validate_address tool for delivery guidance")
        except Exception as exc:
            logger.warning("Tool execution failed: %s", exc)

    # Step 3: Build prompt for LLM
    context_text = ""
    if rag_sources:
        context_text = "\n\n".join([s.text for s in rag_sources])

    prompt = _build_guidance_prompt(issue, context_text, tool_data)

    # Step 4: Get LLM response
    guidance = ""
    issue_summary = ""
    next_steps: list[str] = []

    if llm_client:
        guidance = await llm_client.complete(prompt)
        # Extract first sentence as summary
        sentences = guidance.split(".")
        issue_summary = (sentences[0] + ".") if sentences else guidance[:100]
        # Extract potential next steps from guidance
        next_steps = _extract_next_steps(guidance)
    else:
        guidance = "No LLM configured. Could not generate tracking guidance."
        issue_summary = guidance

    return TrackingGuidance(
        guidance=guidance,
        issue_summary=issue_summary,
        tools_used=tools_used,
        sources=[
            {"source": s.source, "chunk_index": s.chunk_index, "score": round(s.score, 3)}
            for s in rag_sources
        ],
        next_steps=next_steps,
    )


def _build_guidance_prompt(issue: str, context: str, tool_data: str) -> list[dict]:
    """Build a chat-style prompt for tracking guidance."""
    system_msg = (
        "You are a helpful shipping and delivery guidance expert for ShipSmart. "
        "Provide practical, actionable guidance for shipping and delivery issues. "
        "Be empathetic and clear. If the issue requires contacting a carrier, say so. "
        "Always base advice on the provided context."
    )

    user_parts = [f"Issue: {issue}"]
    if context:
        user_parts.append(f"Relevant information:\n{context}")
    if tool_data:
        user_parts.append(f"Tool results:\n{tool_data}")

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def _extract_next_steps(guidance: str) -> list[str]:
    """Extract suggested next steps from guidance text."""
    steps = []
    lines = guidance.split("\n")
    for line in lines:
        line = line.strip()
        # Look for numbered steps or bullet points
        if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
            # Clean up the line
            step = line.lstrip("0123456789.-•) ").strip()
            if step:
                steps.append(step)
    return steps[:3]  # Return up to 3 steps
