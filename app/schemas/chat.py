"""Shared chat primitives used across the advisor + concierge request schemas."""

from pydantic import BaseModel, Field


class ReplyMessage(BaseModel):
    """One chat message referenced by a reply (the replied-to message, or a recent turn).

    Used to resolve what a follow-up question refers to ("the cheaper one") WITHOUT the
    user repeating context. It is advisory reference text only — never authoritative over
    the live shipment/quote/tool context (see ``app.llm.reply_context``). Bounded again
    server-side before it reaches the prompt.
    """

    role: str = Field("user", max_length=16, description='"user" | "assistant"')
    text: str = Field("", max_length=4000)
