"""
ShipSmart — FastAPI Python Service
Entry point for the AI/orchestration service.

Render start command:
  uvicorn app.main:app --host 0.0.0.0 --port $PORT

Local dev:
  uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.routes import advisor, compare, health, info, orchestration, rag
from app.core.config import settings
from app.core.errors import register_error_handlers
from app.core.logging import get_logger, setup_logging
from app.core.middleware import RequestLoggingMiddleware
from app.core.rate_limit import limiter
from app.llm.router import TASK_SYNTHESIS, create_llm_router
from app.providers import create_shipping_provider
from app.rag.embeddings import LocalHashEmbedding, create_embedding_provider
from app.rag.ingestion import ingest_documents, load_documents
from app.rag.vector_store import VectorStore, create_vector_store
from app.tools.address_tools import ValidateAddressTool
from app.tools.quote_tools import GetQuotePreviewTool
from app.tools.registry import ToolRegistry

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
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

    # Tool registry and provider (factory reads SHIPPING_PROVIDER from config)
    shipping_provider = create_shipping_provider()
    tool_registry = ToolRegistry()
    tool_registry.register(ValidateAddressTool(shipping_provider))
    tool_registry.register(GetQuotePreviewTool(shipping_provider))
    app.state.tool_registry = tool_registry
    logger.info(
        "Tool registry initialized: %d tools, provider=%s",
        tool_registry.count(), shipping_provider.name,
    )

    yield

    if hasattr(vector_store, "disconnect"):
        try:
            await vector_store.disconnect()  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("Vector store disconnect failed: %s", exc)
    await app.state.http_client.aclose()
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title="ShipSmart Python API",
    description="AI/orchestration service for the ShipSmart shipping platform.",
    version=settings.app_version,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    lifespan=lifespan,
)

# ── Rate limiter ─────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Error handlers ───────────────────────────────────────────────────────────
register_error_handlers(app)

# ── Middleware (order matters — last added runs first) ───────────────────────
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ───────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(info.router, prefix="/api/v1")
app.include_router(orchestration.router, prefix="/api/v1")
app.include_router(rag.router, prefix="/api/v1")
app.include_router(advisor.router, prefix="/api/v1")
app.include_router(compare.router, prefix="/api")


# ── Root ─────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {"service": settings.app_name, "version": settings.app_version}
