"""
Thin async client for the internal Java (Spring Boot) API.

Wraps the shared `httpx.AsyncClient` created in `app.main` lifespan so we
do not open a second connection pool. All methods degrade gracefully:
they log + return None on failure rather than raising, because Python-side
advisors should still produce *some* answer if Java is unreachable.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class JavaApiClient:
    """Lightweight wrapper around the shared httpx client."""

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._client = http_client

    async def get_quotes(
        self, shipment_request_id: str, auth_token: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Fetch service quotes for a shipment from the Java API.

        Returns the parsed `services` array or None on failure.
        """
        headers: dict[str, str] = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        try:
            resp = await self._client.get(
                "/api/v1/quotes",
                params={"shipmentRequestId": shipment_request_id},
                headers=headers,
            )
        except httpx.HTTPError as exc:
            logger.warning("JavaApiClient.get_quotes network error: %s", exc)
            return None

        if resp.status_code != 200:
            logger.warning(
                "JavaApiClient.get_quotes HTTP %d: %s",
                resp.status_code, resp.text[:200],
            )
            return None
        try:
            body = resp.json()
        except ValueError:
            logger.warning("JavaApiClient.get_quotes returned non-JSON body")
            return None
        services = body.get("services") if isinstance(body, dict) else None
        return services if isinstance(services, list) else None

    async def get_saved_options(
        self, auth_token: str,
    ) -> list[dict[str, Any]] | None:
        """Fetch the authenticated user's saved shipping options from Java.

        Requires a JWT (the Java side enforces auth on this route).
        """
        if not auth_token:
            return None
        try:
            resp = await self._client.get(
                "/api/v1/saved-options",
                headers={"Authorization": f"Bearer {auth_token}"},
            )
        except httpx.HTTPError as exc:
            logger.warning("JavaApiClient.get_saved_options network error: %s", exc)
            return None

        if resp.status_code != 200:
            logger.warning(
                "JavaApiClient.get_saved_options HTTP %d: %s",
                resp.status_code, resp.text[:200],
            )
            return None
        try:
            body = resp.json()
        except ValueError:
            return None
        if isinstance(body, list):
            return body
        if isinstance(body, dict) and isinstance(body.get("options"), list):
            return body["options"]
        return None
