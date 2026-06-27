"""
Composition root.

Builds and wires every long-lived singleton from configuration — the shared HTTP
client, embedding provider, vector store (connect + first-boot ingest), the
task-based LLM router, the RAG component bundle, the audit sink, and the remote
MCP tool registry — and exposes them on ``app.state``. ``app.main`` delegates its
lifespan here so wiring lives in exactly one place and adapters can be swapped per
environment without touching route or service code.

Behavior is identical to the previous inline ``app.main`` lifespan (same order,
logs, and ``app.state`` keys); the additions are ``app.state.audit_sink`` (P0),
``app.state.domain`` (the UC3 mock domain providers, P2), and
``app.state.workflow_checkpointer`` + ``app.state.review_queue`` (UC4, P3).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app.conversations.store import create_conversation_store
from app.core.audit import create_audit_sink
from app.core.config import settings
from app.core.logging import get_logger
from app.domain.adapters import default_providers
from app.integrations.mcp_client import create_remote_registry
from app.llm.router import TASK_SYNTHESIS, create_llm_router
from app.rag.embeddings import LocalHashEmbedding, create_embedding_provider
from app.rag.ingestion import ingest_documents, load_documents
from app.rag.vector_store import VectorStore, create_vector_store
from app.workflow.checkpointer import create_checkpointer
from app.workflow.review_queue import InMemoryReviewQueue

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle (composition root)."""
    logger.info(
        "Starting %s v%s in '%s' mode",
        settings.app_name, settings.app_version, settings.app_env,
    )

    # Shared HTTP client for calling the Java API and external services
    app.state.http_client = httpx.AsyncClient(
        base_url=settings.internal_java_api_url,
        timeout=30.0,
    )
    logger.info("Java API base URL: %s", settings.internal_java_api_url)

    # RAG pipeline components + task-based LLM router
    embedding_provider = create_embedding_provider()
    if isinstance(embedding_provider, LocalHashEmbedding):
        logger.warning(
            "EMBEDDING_PROVIDER is unset — using LocalHashEmbedding. "
            "Retrieval will be lexical/non-semantic and unsuitable for production. "
            "Set EMBEDDING_PROVIDER=openai + OPENAI_API_KEY for real semantic search."
        )

    vector_store: VectorStore = create_vector_store()
    logger.info(
        "Vector store backend: %s (%s)",
        settings.vector_store_type, type(vector_store).__name__,
    )

    # Connect persistent backends and optionally seed documents.
    # PGVectorStore is detected duck-typed to keep asyncpg an optional dep
    # for users running with VECTOR_STORE_TYPE=memory.
    if hasattr(vector_store, "connect") and hasattr(vector_store, "disconnect"):
        await vector_store.connect()  # type: ignore[attr-defined]
        if settings.rag_auto_ingest:
            existing = 0
            try:
                existing = await vector_store.count_async()  # type: ignore[attr-defined]
            except Exception as exc:
                logger.warning("Vector store count_async failed: %s", exc)
            if existing == 0:
                logger.info("Persistent vector store empty — auto-ingesting documents")
                docs = load_documents(settings.rag_documents_path)
                if docs:
                    await ingest_documents(
                        documents=docs,
                        embedding_provider=embedding_provider,
                        vector_store=vector_store,
                        chunk_size=settings.rag_chunk_size,
                        chunk_overlap=settings.rag_chunk_overlap,
                    )
            else:
                logger.info(
                    "Persistent vector store already has %d chunks — skipping auto-ingest",
                    existing,
                )

    llm_router = create_llm_router()
    app.state.llm_router = llm_router
    # Back-compat: existing callers still read rag["llm_client"]. Point it
    # at the synthesis client (RAG q&a is the historical use of this slot).
    app.state.rag = {
        "embedding_provider": embedding_provider,
        "vector_store": vector_store,
        "llm_client": llm_router.for_task(TASK_SYNTHESIS),
    }
    logger.info("LLM router initialized: %s", llm_router.describe())
    logger.info("RAG pipeline initialized (embedding=%s)",
                type(embedding_provider).__name__)

    # Audit sink — emergent auditability made first-class (see app.core.audit).
    app.state.audit_sink = create_audit_sink(settings.audit_sink)
    logger.info("Audit sink: %s", type(app.state.audit_sink).__name__)

    # Domain providers (UC3) — the swappable mock adapters behind the ports
    # (classification, duty, carrier, doc rendering). Deterministic + keyless.
    app.state.domain = default_providers()
    logger.info("Domain providers wired (mock adapters)")

    # Workflow durability + human-in-the-loop (UC4) — process-lifetime singletons
    # so a suspended workflow can be resumed across requests / restarts.
    app.state.workflow_checkpointer = create_checkpointer(
        settings.workflow_durable, settings.workflow_checkpoint_path,
    )
    app.state.review_queue = InMemoryReviewQueue()
    logger.info(
        "Workflow durability wired: %s",
        type(app.state.workflow_checkpointer).__name__,
    )
    if not settings.compliance_explicit_enabled and settings.workflow_enabled:
        logger.warning(
            "COMPLIANCE_EXPLICIT_ENABLED is off while WORKFLOW_ENABLED is on: the "
            "explicit compliance pass and its high-risk human-review interrupt "
            "(e.g. %s) will be SKIPPED. Only lightweight guardrail/RAG checks apply.",
            settings.workflow_high_risk_areas,
        )

    # Conversation memory (concierge recall) — swappable store behind a port.
    # In-memory by default (keyless); Postgres when CONVERSATION_STORE=postgres.
    # A connect() failure degrades gracefully to "no recall" rather than crashing.
    app.state.conversation_store = create_conversation_store(
        settings.conversation_store, settings.database_url,
    )
    if app.state.conversation_store is not None and hasattr(
        app.state.conversation_store, "connect",
    ):
        try:
            await app.state.conversation_store.connect()
        except Exception as exc:
            logger.error(
                "Conversation store connect failed (%s); concierge recall disabled: %s",
                settings.conversation_store, exc,
            )
            app.state.conversation_store = None
    logger.info(
        "Conversation store: %s",
        type(app.state.conversation_store).__name__
        if app.state.conversation_store else "disabled",
    )

    # Remote tool registry — hydrated from the standalone ShipSmart-MCP
    # service. If SHIPSMART_MCP_URL is not configured, the advisor and
    # orchestration routes will return 503 until it is set.
    tool_registry = None
    if settings.shipsmart_mcp_url:
        try:
            tool_registry = await create_remote_registry(
                base_url=settings.shipsmart_mcp_url,
                api_key=settings.shipsmart_mcp_api_key,
            )
            logger.info(
                "Remote tool registry hydrated from MCP %s (%d tools)",
                settings.shipsmart_mcp_url, tool_registry.count(),
            )
        except Exception as exc:
            logger.error(
                "Failed to hydrate remote tool registry from %s: %s. "
                "Advisor/orchestration routes will return 503.",
                settings.shipsmart_mcp_url, exc,
            )
    else:
        logger.warning(
            "SHIPSMART_MCP_URL is not set — advisor/orchestration routes "
            "will return 503 until it is configured."
        )
    app.state.tool_registry = tool_registry

    yield

    if hasattr(vector_store, "disconnect"):
        try:
            await vector_store.disconnect()  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("Vector store disconnect failed: %s", exc)
    conversation_store = getattr(app.state, "conversation_store", None)
    if conversation_store is not None and hasattr(conversation_store, "disconnect"):
        try:
            await conversation_store.disconnect()
        except Exception as exc:
            logger.warning("Conversation store disconnect failed: %s", exc)
    if tool_registry is not None:
        try:
            await tool_registry.aclose()
        except Exception as exc:
            logger.warning("Remote tool registry close failed: %s", exc)
    await app.state.http_client.aclose()
    logger.info("Shutting down %s", settings.app_name)
