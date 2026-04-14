"""Tests for task-based LLM routing."""

from __future__ import annotations

from unittest.mock import patch

from app.llm.client import EchoClient, GeminiClient, OpenAIClient
from app.llm.router import (
    TASK_FALLBACK,
    TASK_REASONING,
    TASK_SYNTHESIS,
    create_llm_router,
)


def _patch_settings(**overrides):
    """Patch app.llm.client.settings AND app.llm.router.settings together."""
    defaults = dict(
        llm_provider="",
        llm_provider_reasoning="",
        llm_provider_synthesis="",
        llm_provider_fallback="echo",
        openai_api_key="",
        openai_model="gpt-4o-mini",
        gemini_api_key="",
        gemini_model="gemini-2.0-flash",
        llama_base_url="http://localhost:11434",
        llama_model="llama3.2",
        llm_timeout=30,
        llm_max_tokens=1024,
        llm_temperature=0.3,
    )
    defaults.update(overrides)

    class _S:
        pass

    s = _S()
    for k, v in defaults.items():
        setattr(s, k, v)
    return patch("app.llm.client.settings", s), patch("app.llm.router.settings", s)


def test_router_defaults_to_echo_when_nothing_configured():
    p1, p2 = _patch_settings()
    with p1, p2:
        router = create_llm_router()
        assert isinstance(router.for_task(TASK_REASONING), EchoClient)
        assert isinstance(router.for_task(TASK_SYNTHESIS), EchoClient)
        assert isinstance(router.for_task(TASK_FALLBACK), EchoClient)


def test_router_routes_reasoning_to_openai_and_synthesis_to_gemini():
    p1, p2 = _patch_settings(
        llm_provider_reasoning="openai",
        llm_provider_synthesis="gemini",
        openai_api_key="sk-test",
        gemini_api_key="g-test",
    )
    with p1, p2:
        router = create_llm_router()
        assert isinstance(router.for_task(TASK_REASONING), OpenAIClient)
        assert isinstance(router.for_task(TASK_SYNTHESIS), GeminiClient)
        desc = router.describe()
        assert desc[TASK_REASONING] == "openai"
        assert desc[TASK_SYNTHESIS] == "gemini"


def test_task_inherits_legacy_llm_provider_when_unset():
    p1, p2 = _patch_settings(
        llm_provider="openai",
        openai_api_key="sk-test",
    )
    with p1, p2:
        router = create_llm_router()
        # Both tasks inherit from legacy LLM_PROVIDER
        assert isinstance(router.for_task(TASK_REASONING), OpenAIClient)
        assert isinstance(router.for_task(TASK_SYNTHESIS), OpenAIClient)


def test_missing_key_falls_back_to_fallback_provider():
    # reasoning=openai but no key → should fall back to echo (the configured fallback)
    p1, p2 = _patch_settings(
        llm_provider_reasoning="openai",
        llm_provider_synthesis="gemini",
        openai_api_key="",         # missing
        gemini_api_key="g-test",
        llm_provider_fallback="echo",
    )
    with p1, p2:
        router = create_llm_router()
        assert isinstance(router.for_task(TASK_REASONING), EchoClient)
        assert isinstance(router.for_task(TASK_SYNTHESIS), GeminiClient)


def test_unknown_task_returns_fallback():
    p1, p2 = _patch_settings()
    with p1, p2:
        router = create_llm_router()
        client = router.for_task("does-not-exist")
        assert isinstance(client, EchoClient)


def test_router_never_crashes_on_bad_fallback_name():
    p1, p2 = _patch_settings(llm_provider_fallback="not-a-real-provider")
    with p1, p2:
        router = create_llm_router()
        # Falls through to EchoClient
        assert isinstance(router.fallback, EchoClient)


def test_describe_shape():
    p1, p2 = _patch_settings(
        llm_provider_reasoning="openai",
        openai_api_key="sk-test",
    )
    with p1, p2:
        router = create_llm_router()
        desc = router.describe()
        assert set(desc.keys()) >= {TASK_REASONING, TASK_SYNTHESIS, "fallback"}
