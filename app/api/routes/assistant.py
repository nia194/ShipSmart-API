"""Streaming assistant endpoint (Product Roadmap P3 — perceived speed via SSE).

`POST /api/v1/assistant/stream` streams a grounded answer as Server-Sent Events:
the RAG context is retrieved (fast), then the LLM synthesis is streamed token by
token through `LLMRouter.stream` so the first token reaches the client as soon as
the model starts generating — the "first token < 1s" win. The stream closes with
a single typed-envelope event carrying the full `AssistantResponse`, so the client
gets both progressive text AND the structured contract it renders.

Gated by `assistant_contract_v1` (streaming emits the typed contract). Guardrail
blocks short-circuit to a streamed safe refusal and never call the LLM.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.errors import AppError
from app.core.rate_limit import limiter
from app.llm.guardrails import SAFE_REFUSAL, assemble
from app.llm.prompts import SYSTEM_PROMPT
from app.llm.router import TASK_SYNTHESIS, LLMRouter
from app.rag.retrieval import retrieve_auto
from app.schemas.typed_outputs import (
    AssistantResponse,
    PolicyAnswerResult,
    SourceCitation,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assistant", tags=["assistant"])


class AssistantStreamRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


def _sse(event: dict) -> str:
    """One Server-Sent Event frame."""
    return f"data: {json.dumps(event)}\n\n"


@router.post("/stream")
@limiter.limit(settings.rate_limit_advisor)
async def stream_assistant(body: AssistantStreamRequest, request: Request) -> StreamingResponse:
    """Stream a grounded answer as SSE, closing with the typed AssistantResponse."""
    if not settings.assistant_contract_v1:
        raise AppError(status_code=404, message="Assistant streaming is disabled")

    llm_router: LLMRouter | None = getattr(request.app.state, "llm_router", None)
    rag = getattr(request.app.state, "rag", None)
    if llm_router is None or rag is None:
        raise AppError(status_code=503, message="LLM router / RAG pipeline is not initialized")

    query = body.query
    request_id = getattr(request.state, "request_id", "")

    async def events() -> AsyncIterator[str]:
        results, _mode = await retrieve_auto(
            query, rag["embedding_provider"], rag["vector_store"], top_k=3
        )
        assembled = assemble(
            system_prompt=SYSTEM_PROMPT, user_text=query, contexts=results, request_id=request_id
        )

        # Guardrail block → streamed safe refusal, no LLM call.
        if assembled.blocked:
            refusal = assembled.refusal or SAFE_REFUSAL
            yield _sse({"delta": refusal})
            envelope = AssistantResponse(type="refusal", message=refusal, apply_policy="none")
            yield _sse({"done": True, "assistant": envelope.model_dump()})
            return

        chunks: list[str] = []
        try:
            async for delta in llm_router.stream(
                TASK_SYNTHESIS, assembled.messages, request_id=request_id
            ):
                chunks.append(delta)
                yield _sse({"delta": delta})
        except Exception as exc:  # noqa: BLE001 - end the stream gracefully
            logger.warning("assistant stream error rid=%s: %s", request_id, exc)
            yield _sse({"error": "stream interrupted"})
            return

        answer = "".join(chunks)
        citations = [
            SourceCitation(source=r.source, chunk_index=r.chunk_index, score=round(r.score, 4))
            for r in results
        ]
        envelope = AssistantResponse(
            type="answer",
            message=answer,
            sources=citations,
            intent="policy_question",
            apply_policy="none",
            confidence=0.7,
            result=PolicyAnswerResult(answer=answer, sources=citations),
        )
        yield _sse({"done": True, "assistant": envelope.model_dump()})

    return StreamingResponse(events(), media_type="text/event-stream")
