"""
Prompt templates for RAG queries and advisor flows.
Separates system instruction, retrieved context, and user query.

Design principles:
  - Ground answers in retrieved context
  - Cite sources when possible
  - Refuse to guess when context is insufficient
  - Keep answers concise and practical

All prompt construction is routed through ``app.llm.guardrails.assemble`` so
role-separation, fencing, soft/hard guardrails (C) and grounding (D) are applied
uniformly. These functions are thin back-compat wrappers that return only the
messages; the advisor/RAG services call ``assemble`` directly so they can also
honor a hard-guardrail block and emit decision-path tags.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.llm.guardrails import assemble


@dataclass
class _PlainContext:
    """Adapter so plain context strings flow through the chunk-aware assembler."""

    text: str
    source: str = "kb"
    score: float = 0.0

SYSTEM_PROMPT = (
    "You are a shipping expert assistant for ShipSmart. "
    "Your role is to help users with shipping decisions, carrier comparisons, "
    "packaging advice, and delivery issues.\n\n"
    "Rules:\n"
    "1. ONLY answer based on the provided context. If the context does not "
    "contain enough information, say so honestly. Do NOT make up facts, "
    "prices, or policies.\n"
    "2. When the context contains relevant information, use it directly in "
    "your answer. Reference the source topic when helpful (e.g., "
    '"According to UPS\'s service levels...").\n'
    "3. Keep answers concise and practical. Lead with the most useful "
    "information.\n"
    "4. When comparing options, use clear structure (bullet points or brief "
    "comparisons).\n"
    "5. If the user asks about a specific carrier or service, prioritize "
    "that information.\n"
    "6. For pricing questions, always note that prices are estimates and "
    "vary by account, volume, and current surcharges.\n"
    "7. If the user's question is outside shipping/logistics, politely "
    "redirect to shipping topics."
)

ADVISOR_SYSTEM_PROMPT = (
    "You are ShipSmart's shipping advisor. You combine retrieved knowledge, "
    "tool results, and shipping expertise to give actionable advice.\n\n"
    "Rules:\n"
    "1. Base your advice ONLY on the provided context and tool results. "
    "Do not invent rates, transit times, or policies.\n"
    "2. If tool results include quote previews, reference the specific "
    "prices and service levels.\n"
    "3. If tool results include address validation, mention whether the "
    "address was confirmed valid.\n"
    "4. When recommending a shipping option, explain WHY it is the best "
    "fit (cost, speed, reliability).\n"
    "5. Keep advice concise — 2-4 paragraphs maximum.\n"
    "6. If you lack sufficient information to give good advice, say so "
    "and suggest what additional information would help."
)


COMPLIANCE_SYSTEM_PROMPT = (
    "You are ShipSmart's compliance assistant. You summarize a shipment "
    "compliance review for a human reviewer.\n\n"
    "Rules:\n"
    "1. This is ADVISORY ONLY. NEVER state or imply that a shipment is "
    '"compliant", "cleared", "approved", or "legal to ship". You assist a human '
    "decision; you do not make a customs or legal determination.\n"
    "2. Base the summary ONLY on the provided findings, retrieved knowledge-base "
    "context, and tool results. Do not invent rules, thresholds, or document "
    "names.\n"
    "3. Lead with any flags (concerns to act on), then grounded guidance. Call "
    "out every area marked 'unverified' explicitly as needing human review or "
    "more information — never paper over a gap.\n"
    "4. Cite the source name of any knowledge-base chunk you rely on.\n"
    "5. Keep it concise and practical — a reviewer should grasp the situation and "
    "the open items in a few sentences."
)


def build_rag_prompt(query: str, context_chunks: list[str]) -> list[dict[str, str]]:
    """Build a fenced/grounded chat prompt for a RAG query (back-compat wrapper).

    Delegates to the guardrail assembler. Uses neutralize (not block) mode so it
    always returns valid messages; the RAG service calls ``assemble`` directly
    when it needs blocking + decision tags.
    """
    contexts = [_PlainContext(text=c) for c in context_chunks]
    return assemble(
        system_prompt=SYSTEM_PROMPT,
        user_text=query,
        contexts=contexts,
        block_on_injection=False,
    ).messages


def build_advisor_prompt(
    query: str,
    context: str,
    tool_results: str,
) -> list[dict[str, str]]:
    """Build a fenced/grounded advisor prompt (back-compat wrapper).

    Delegates to the guardrail assembler (neutralize mode); the advisor service
    calls ``assemble`` directly for blocking + decision tags.
    """
    contexts = [_PlainContext(text=context)] if context else []
    return assemble(
        system_prompt=ADVISOR_SYSTEM_PROMPT,
        user_text=query,
        contexts=contexts,
        tool_results=tool_results,
        block_on_injection=False,
    ).messages
