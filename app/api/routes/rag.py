"""RAG query and ingestion endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.errors import AppError
from app.llm.router import TASK_SYNTHESIS, LLMRouter
from app.rag.ingestion import ingest_documents, load_documents
from app.services.rag_service import rag_query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


# ── Schemas ──────────────────────────────────────────────────────────────────

class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)


class SourceInfo(BaseModel):
    source: str
    chunk_index: int
    score: float


class RAGQueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    metadata: dict


class IngestResponse(BaseModel):
    chunks_ingested: int
    total_chunks: int


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/query", response_model=RAGQueryResponse)
async def query_rag(body: RAGQueryRequest, request: Request) -> RAGQueryResponse:
    """Query the RAG pipeline: retrieve context and generate an answer."""
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        raise AppError(status_code=503, message="RAG pipeline is not initialized")

    llm_router: LLMRouter | None = getattr(request.app.state, "llm_router", None)
    synth_client = (
        llm_router.for_task(TASK_SYNTHESIS) if llm_router else rag["llm_client"]
    )

    result = await rag_query(
        query=body.query,
        embedding_provider=rag["embedding_provider"],
        vector_store=rag["vector_store"],
        llm_client=synth_client,
        top_k=settings.rag_top_k,
    )

    return RAGQueryResponse(
        answer=result.answer,
        sources=[
            SourceInfo(source=s["source"], chunk_index=s["chunk_index"], score=s["score"])
            for s in result.sources
        ],
        metadata=result.metadata,
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_rag(request: Request) -> IngestResponse:
    """Ingest documents from the configured documents directory."""
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        raise AppError(status_code=503, message="RAG pipeline is not initialized")

    documents = load_documents(settings.rag_documents_path)
    if not documents:
        raise AppError(status_code=404, message="No documents found in configured path")

    count = await ingest_documents(
        documents=documents,
        embedding_provider=rag["embedding_provider"],
        vector_store=rag["vector_store"],
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )

    return IngestResponse(
        chunks_ingested=count,
        total_chunks=rag["vector_store"].count(),
    )
