"""Tests for the LLM error taxonomy (A) and the router failover chain."""

from __future__ import annotations

import json

import pytest

import app.llm.router as router_mod
from app.llm.client import EchoClient, LLMClient
from app.llm.errors import (
    AuthError,
    ContentFilterError,
    ContextLengthError,
    MalformedResponseError,
    ProviderOutageError,
    ProviderTimeoutError,
    RateLimitError,
    classify_provider_error,
)
from app.llm.router import TASK_SYNTHESIS, ExecuteResult, LLMRouter

# ── Fake clients ─────────────────────────────────────────────────────────────


class _Raises(LLMClient):
    """Client that raises a fixed exception every call."""

    def __init__(self, name: str, exc: Exception) -> None:
        self._name = name
        self._exc = exc
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return self._name

    async def complete(self, messages):
        self.calls += 1
        raise self._exc


class _Answers(LLMClient):
    def __init__(self, name: str, text: str = "ok") -> None:
        self._name = name
        self._text = text
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return self._name

    async def complete(self, messages):
        self.calls += 1
        return self._text


def _status_exc(code: int, message: str = "") -> Exception:
    exc = RuntimeError(message or f"HTTP {code}")
    exc.status_code = code  # type: ignore[attr-defined]
    return exc


def _router(chain: list[LLMClient]) -> LLMRouter:
    return LLMRouter(
        clients={TASK_SYNTHESIS: chain[0]},
        fallback=EchoClient(),
        chains={TASK_SYNTHESIS: chain} if len(chain) > 1 else {},
    )


# ── Taxonomy classification ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "exc,expected,retryable",
    [
        (_status_exc(429), RateLimitError, True),
        (_status_exc(503), ProviderOutageError, True),
        (_status_exc(500), ProviderOutageError, True),
        (_status_exc(401), AuthError, False),
        (_status_exc(403), AuthError, False),
        (_status_exc(504), ProviderTimeoutError, True),
        (TimeoutError("slow"), ProviderTimeoutError, True),
        (json.JSONDecodeError("x", "y", 0), MalformedResponseError, True),
        (_status_exc(400, "maximum context length is 8192 tokens"), ContextLengthError, False),
        (_status_exc(400, "blocked by content filter"), ContentFilterError, False),
        (ValueError("totally unknown"), ProviderOutageError, True),
    ],
)
def test_classify_provider_error(exc, expected, retryable):
    err = classify_provider_error(exc, "openai")
    assert isinstance(err, expected)
    assert err.retryable is retryable
    assert err.provider == "openai"


def test_classify_passthrough_keeps_existing_llm_error():
    original = RateLimitError(provider="gemini")
    assert classify_provider_error(original, "openai") is original
    assert original.provider == "gemini"  # provider not overwritten


# ── Router execute: single client (today's behavior) ─────────────────────────


@pytest.mark.asyncio
async def test_single_client_success_no_failover():
    good = _Answers("openai", "hello")
    result = await _router([good]).execute(TASK_SYNTHESIS, [{"role": "user", "content": "hi"}])
    assert isinstance(result, ExecuteResult)
    assert result.text == "hello"
    assert result.provider == "openai"
    assert result.failed_over is False
    assert good.calls == 1  # called exactly once — no retries with empty chain


@pytest.mark.asyncio
async def test_single_client_retryable_error_propagates_without_retry():
    bad = _Raises("openai", _status_exc(429))
    with pytest.raises(RateLimitError):
        await _router([bad]).execute(TASK_SYNTHESIS, [{"role": "user", "content": "hi"}])
    assert bad.calls == 1  # no retry, no echo fallback — byte-for-byte today


# ── Router execute: failover chain ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_retryable_retries_then_fails_over(monkeypatch):
    monkeypatch.setattr(router_mod.settings, "llm_retry_max_attempts", 2, raising=False)
    flaky = _Raises("openai", _status_exc(503))
    good = _Answers("gemini", "recovered")
    result = await _router([flaky, good]).execute(
        TASK_SYNTHESIS, [{"role": "user", "content": "hi"}]
    )
    assert result.provider == "gemini"
    assert result.failed_over is True
    assert flaky.calls == 2  # retried up to max_attempts before failing over
    assert good.calls == 1


@pytest.mark.asyncio
async def test_terminal_error_fails_fast_no_failover(monkeypatch):
    monkeypatch.setattr(router_mod.settings, "llm_retry_max_attempts", 3, raising=False)
    bad = _Raises("openai", _status_exc(401))
    never = _Answers("gemini")
    with pytest.raises(AuthError):
        await _router([bad, never]).execute(TASK_SYNTHESIS, [{"role": "user", "content": "hi"}])
    assert bad.calls == 1  # terminal → no retry
    assert never.calls == 0  # terminal → no failover


@pytest.mark.asyncio
async def test_chain_terminates_in_echo(monkeypatch):
    monkeypatch.setattr(router_mod.settings, "llm_retry_max_attempts", 1, raising=False)
    bad = _Raises("openai", _status_exc(503))
    echo = EchoClient()
    result = await _router([bad, echo]).execute(
        TASK_SYNTHESIS, [{"role": "user", "content": "Question: hi"}]
    )
    assert result.provider == "echo"
    assert result.failed_over is True
    assert bad.calls == 1


def test_describe_chains_shape():
    router = _router([_Answers("openai"), EchoClient()])
    chains = router.describe_chains()
    assert chains[TASK_SYNTHESIS] == ["openai", "echo"]
