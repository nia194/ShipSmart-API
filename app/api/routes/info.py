"""Service info route."""

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter()


class InfoResponse(BaseModel):
    service: str
    version: str
    env: str
    llm_provider: str
    rag_provider: str
    embedding_provider: str


@router.get("/info", response_model=InfoResponse, tags=["info"])
async def info() -> InfoResponse:
    """Return service metadata. Does not expose secrets."""
    return InfoResponse(
        service=settings.app_name,
        version=settings.app_version,
        env=settings.app_env,
        llm_provider=settings.llm_provider or "(not configured)",
        rag_provider=settings.rag_provider or "(not configured)",
        embedding_provider=settings.embedding_provider or "(not configured)",
    )
