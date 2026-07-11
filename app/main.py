"""
ShipSmart — FastAPI Python Service
Entry point for the AI/orchestration service.

Render start command:
  uvicorn app.main:app --host 0.0.0.0 --port $PORT

Local dev:
  uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Startup/shutdown wiring lives in app.bootstrap (the composition root); this
module only constructs the FastAPI app, middleware, error handlers, and routes.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.routes import (
    admin,
    advisor,
    agent,
    assistant,
    compare,
    compliance,
    concierge,
    feedback,
    health,
    info,
    orchestration,
    rag,
    workflow,
)
from app.bootstrap import lifespan
from app.core.config import settings
from app.core.errors import register_error_handlers
from app.core.logging import setup_logging
from app.core.middleware import RequestLoggingMiddleware
from app.core.rate_limit import limiter

setup_logging()


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
app.include_router(agent.router, prefix="/api/v1")
app.include_router(compliance.router, prefix="/api/v1")
app.include_router(concierge.router, prefix="/api/v1")
app.include_router(workflow.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")
app.include_router(assistant.router, prefix="/api/v1")
app.include_router(feedback.router, prefix="/api/v1")
app.include_router(compare.router, prefix="/api")


# ── Root ─────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {"service": settings.app_name, "version": settings.app_version}
