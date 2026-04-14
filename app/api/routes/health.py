"""Health and readiness check routes."""

from datetime import UTC, datetime

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: str


class ReadyResponse(BaseModel):
    status: str


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Liveness check. Used by Render health checks."""
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        timestamp=datetime.now(tz=UTC).isoformat(),
    )


@router.get("/ready", response_model=ReadyResponse, tags=["health"])
async def ready() -> ReadyResponse:
    """Readiness check. Returns 200 when the service can accept traffic.

    Future: check DB connections, LLM client availability, etc.
    """
    return ReadyResponse(status="ready")
