"""
Prompt templates for RAG queries and advisor flows.
Separates system instruction, retrieved context, and user query.

Design principles:
  - Ground answers in retrieved context
  - Cite sources when possible
  - Refuse to guess when context is insufficient
  - Keep answers concise and practical
"""

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


def build_rag_prompt(query: str, context_chunks: list[str]) -> list[dict[str, str]]:
    """Build a chat-style message list for a RAG query.

    Args:
        query: The user's question.
        context_chunks: Retrieved text chunks as context.

    Returns:
        List of message dicts suitable for an LLM chat API.
    """
    if context_chunks:
        context_block = "\n\n---\n\n".join(context_chunks)
        user_content = (
            f"Context (from ShipSmart knowledge base):\n{context_block}\n\n"
            f"Question: {query}\n\n"
            "Answer based on the context above. If the context doesn't "
            "cover this topic, say so."
        )
    else:
        user_content = (
            f"Question: {query}\n\n"
            "No context was retrieved from the knowledge base. "
            "Answer only if you are confident, otherwise say you don't "
            "have enough information."
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_advisor_prompt(
    query: str,
    context: str,
    tool_results: str,
) -> list[dict[str, str]]:
    """Build a chat-style prompt for the shipping advisor.

    Used by the shipping_advisor_service when combining RAG context
    with tool results for comprehensive advice.

    Args:
        query: The user's shipping question.
        context: Retrieved RAG context text.
        tool_results: Formatted tool execution results.

    Returns:
        List of message dicts suitable for an LLM chat API.
    """
    user_parts = [f"Question: {query}"]

    if context:
        user_parts.append(f"Retrieved context:\n{context}")
    if tool_results:
        user_parts.append(f"Tool results:\n{tool_results}")

    if not context and not tool_results:
        user_parts.append(
            "No context or tool results available. Answer only if "
            "confident, otherwise say you need more information."
        )

    return [
        {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]
