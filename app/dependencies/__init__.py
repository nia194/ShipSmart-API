"""
FastAPI dependency injection providers.
Use these with Depends() in route handlers.
"""

from fastapi import Request

from app.core.config import Settings, settings


def get_settings() -> Settings:
    """Return the application settings singleton."""
    return settings


async def get_http_client(request: Request):  # noqa: ANN201
    """Return the shared httpx.AsyncClient from app state.

    Usage in routes:
        async def my_route(client = Depends(get_http_client)):
            resp = await client.get(...)
    """
    return request.app.state.http_client
