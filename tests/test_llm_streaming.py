"""LLM streaming primitive tests (Product Roadmap P3 — perceived speed). Keyless."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from app.llm.client import EchoClient, LLMClient
from app.llm.errors import ProviderOutageError
from app.llm.router import LLMRouter


async def _collect(agen: AsyncIterator[str]) -> list[str]:
    return [chunk async for chunk in agen]


# ── client-level streaming ────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_echo_client_streams_word_by_word_and_reassembles():
    echo = EchoClient()
    messages = [{"role": "user", "content": "Question: how do I ship a box?"}]
    deltas = await _collect(echo.stream(messages))
    assert len(deltas) > 1  # actually chunked, not one blob
    assert "".join(deltas) == await echo.complete(messages)  # lossless


@pytest.mark.asyncio
async def test_base_default_stream_yields_one_chunk():
    class OneShot(LLMClient):
        @property
        def provider_name(self) -> str:
            return "oneshot"

        async def complete(self, messages):
            return "hello world"

    deltas = await _collect(OneShot().stream([{"role": "user", "content": "x"}]))
    assert deltas == ["hello world"]


# ── router streaming + failover ───────────────────────────────────────────────
class _Boom(LLMClient):
    @property
    def provider_name(self) -> str:
        return "boom"

    async def complete(self, messages):
        raise ConnectionError("provider down")  # retryable → classified as outage

    async def stream(self, messages):
        raise ConnectionError("provider down")
        yield ""  # unreachable; makes this an async generator


@pytest.mark.asyncio
async def test_router_streams_from_the_task_client():
    echo = EchoClient()
    router = LLMRouter(clients={"synthesis": echo}, fallback=echo)
    deltas = await _collect(router.stream("synthesis", [{"role": "user", "content": "hi"}]))
    assert "".join(deltas) == await echo.complete([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_router_fails_over_before_first_token():
    echo = EchoClient()
    router = LLMRouter(
        clients={"synthesis": _Boom()},
        fallback=echo,
        chains={"synthesis": [_Boom(), echo]},
    )
    deltas = await _collect(router.stream("synthesis", [{"role": "user", "content": "hi"}]))
    assert "".join(deltas)  # recovered via the echo fallback, non-empty


@pytest.mark.asyncio
async def test_router_raises_when_whole_chain_fails():
    router = LLMRouter(
        clients={"synthesis": _Boom()}, fallback=_Boom(), chains={"synthesis": [_Boom()]}
    )
    with pytest.raises(ProviderOutageError):
        await _collect(router.stream("synthesis", [{"role": "user", "content": "hi"}]))
