"""
RAG service — orchestrates retrieval and LLM completion.

Retrieval → fenced/grounded/guardrailed prompt assembly (C/D) → LLM via the
router's failover chain (A) with a decision-path tag (E). Dense-only retrieval
and an empty fallback chain reproduce today's single-shot behavior.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from app.core.config import settings
from app.llm.client import LLMClient
from app.llm.guardrails import SAFE_REFUSAL, assemble
from app.llm.prompts import SYSTEM_PROMPT
from app.llm.router import TASK_SYNTHESIS, LLMRouter
from app.rag.embeddings import EmbeddingProvider
from app.rag.retrieval import retrieve_auto
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


async def _maybe_log_trace(vector_store, query: str, ag, request_id: str) -> None:
    """Best-effort iterative-RAG trace into rag_query_log (G). Never blocks/raises.

    Only attempts a write when RAG_QUERY_LOG=true and the vector store exposes a
    Postgres pool (pgvector). Any failure is swallowed — tracing must never break
    the response.
    """
    if not getattr(settings, "rag_query_log", False):
        return
    pool = getattr(vector_store, "_pool", None)
    if pool is None:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO rag_query_log "
                "(request_id, query, plan_json, retrieved_chunk_ids, decision_path) "
                "VALUES ($1, $2, $3::jsonb, $4::bigint[], $5::text[])",
                request_id or None, query,
                json.dumps({"steps": ag.steps, "tools": ag.tools_used}),
                [], list(ag.decisions),
            )
    except Exception as exc:  # noqa: BLE001 - tracing is best-effort
        logger.warning("rag_query_log trace insert failed (ignored): %s", exc)


@dataclass
class RAGResult:
    answer: str
    sources: list[dict]
    metadata: dict


def _answer_source(provider: str, failed_over: bool) -> str:
    """Provenance tag for the prose (E)."""
    if provider == "echo" or failed_over:
        return "fallback"
    return "llm"


async def rag_query(
    query: str,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    llm_client: LLMClient,
    top_k: int = 3,
    *,
    llm_router: LLMRouter | None = None,
    task: str = TASK_SYNTHESIS,
    request_id: str = "",
) -> RAGResult:
    """Execute a RAG query: retrieve context, assemble a guarded prompt, answer.

    When ``llm_router`` is provided the call goes through its failover chain;
    otherwise the single ``llm_client`` is used (today's behavior). A guardrail
    block short-circuits with a safe refusal and never calls the LLM.

    RAG_MODE=iterative runs the bounded iterative loop (G); RAG_MODE=normal
    (default) is the single-shot path below.
    """
    # "agentic" is a deprecated alias for "iterative" (legacy .env compatibility).
    if getattr(settings, "rag_mode", "normal") in ("iterative", "agentic"):
        from app.rag.iterative import iterative_rag, make_retriever

        ag = await iterative_rag(
            query,
            retriever=make_retriever(embedding_provider, vector_store),
            llm_router=llm_router, llm_client=llm_client, task=task,
            top_k=top_k, request_id=request_id,
        )
        await _maybe_log_trace(vector_store, query, ag, request_id)
        sources = [
            {"source": r.source, "chunk_index": r.chunk_index, "score": round(r.score, 4)}
            for r in ag.sources
        ]
        return RAGResult(
            answer=ag.answer,
            sources=sources,
            metadata={
                "chunks_retrieved": len(ag.sources),
                "store_size": vector_store.count(),
                "steps": ag.steps,
                "tools_used": ag.tools_used,
                "decision_path": {
                    "mode": "iterative",
                    "retrieval": "hybrid" if getattr(settings, "rag_hybrid", False) else "dense",
                    "answer": ag.answer_source, "provider": ag.provider,
                    "tags": ag.decisions,
                },
            },
        )

    results, retrieval_mode = await retrieve_auto(
        query, embedding_provider, vector_store, top_k=top_k,
    )

    assembled = assemble(
        system_prompt=SYSTEM_PROMPT,
        user_text=query,
        contexts=results,
        request_id=request_id,
    )

    if assembled.blocked:
        logger.info("RAG query blocked by guardrails (rid=%s)", request_id)
        return RAGResult(
            answer=assembled.refusal or SAFE_REFUSAL,
            sources=[],
            metadata={
                "chunks_retrieved": len(results),
                "store_size": vector_store.count(),
                "decision_path": {
                    "mode": "normal", "retrieval": retrieval_mode, "answer": "rule",
                    "provider": "guardrail", "tags": assembled.decisions,
                },
            },
        )

    if llm_router is not None:
        exec_res = await llm_router.execute(task, assembled.messages, request_id=request_id)
        answer, provider, failed_over = exec_res.text, exec_res.provider, exec_res.failed_over
    else:
        answer = await llm_client.complete(assembled.messages)
        provider, failed_over = getattr(llm_client, "provider_name", ""), False

    used = assembled.kept_sources
    sources = [
        {"source": r.source, "chunk_index": r.chunk_index, "score": round(r.score, 4)}
        for r in used
    ]
    logger.info(
        "RAG query completed: %d sources, provider=%s, answer length=%d",
        len(sources), provider, len(answer),
    )

    return RAGResult(
        answer=answer,
        sources=sources,
        metadata={
            "chunks_retrieved": len(results),
            "store_size": vector_store.count(),
            "decision_path": {
                "mode": "normal", "retrieval": retrieval_mode,
                "answer": _answer_source(provider, failed_over),
                "provider": provider, "tags": assembled.decisions,
            },
        },
    )
