"""Tests for JavaApiClient — the thin async wrapper over the shared httpx client.

The client degrades gracefully: every failure mode (network error, non-200,
non-JSON, wrong shape) must return None rather than raise, because the Python
advisors should still answer when the Java service is unreachable. A
``httpx.MockTransport`` stands in for the live Spring Boot API so no service is
required.
"""

from __future__ import annotations

import httpx

from app.services.java_client import JavaApiClient


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="http://java.test", transport=httpx.MockTransport(handler)
    )


# ── get_quotes ───────────────────────────────────────────────────────────────


async def test_get_quotes_returns_services_on_200():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["sid"] = request.url.params.get("shipmentRequestId")
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"services": [{"service": "Ground"}]})

    async with _client(handler) as http:
        services = await JavaApiClient(http).get_quotes("ship-1", auth_token="tok")

    assert services == [{"service": "Ground"}]
    assert captured["path"] == "/api/v1/quotes"
    assert captured["sid"] == "ship-1"
    assert captured["auth"] == "Bearer tok"  # token forwarded as a bearer header


async def test_get_quotes_returns_none_on_non_200():
    async with _client(lambda r: httpx.Response(500, text="boom")) as http:
        assert await JavaApiClient(http).get_quotes("ship-1") is None


async def test_get_quotes_returns_none_on_network_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with _client(handler) as http:
        assert await JavaApiClient(http).get_quotes("ship-1") is None


async def test_get_quotes_returns_none_when_services_missing_or_wrong_type():
    async with _client(lambda r: httpx.Response(200, json={"other": 1})) as http:
        assert await JavaApiClient(http).get_quotes("ship-1") is None
    async with _client(lambda r: httpx.Response(200, json={"services": "nope"})) as http:
        assert await JavaApiClient(http).get_quotes("ship-1") is None


# ── get_saved_options ────────────────────────────────────────────────────────


async def test_get_saved_options_requires_token():
    # Empty token short-circuits to None WITHOUT making a request.
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request should be made without a token")

    async with _client(handler) as http:
        assert await JavaApiClient(http).get_saved_options("") is None


async def test_get_saved_options_accepts_list_or_wrapped_object():
    async with _client(lambda r: httpx.Response(200, json=[{"id": "a"}])) as http:
        assert await JavaApiClient(http).get_saved_options("tok") == [{"id": "a"}]
    async with _client(lambda r: httpx.Response(200, json={"options": [{"id": "b"}]})) as http:
        assert await JavaApiClient(http).get_saved_options("tok") == [{"id": "b"}]


async def test_get_saved_options_returns_none_on_non_200():
    async with _client(lambda r: httpx.Response(401, text="unauthorized")) as http:
        assert await JavaApiClient(http).get_saved_options("tok") is None
