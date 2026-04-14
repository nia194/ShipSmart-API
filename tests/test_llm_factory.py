"""Tests for LLM client factory, provider selection, fallback, and prompts."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.llm.client import (
    EchoClient,
    GeminiClient,
    LlamaClient,
    OpenAIClient,
    create_llm_client,
)
from app.llm.prompts import build_advisor_prompt, build_rag_prompt

# ── Factory selection tests ──────────────────────────────────────────────────


def test_factory_returns_echo_by_default():
    """Empty LLM_PROVIDER returns EchoClient."""
    with patch("app.llm.client.settings") as s:
        s.llm_provider = ""
        client = create_llm_client()
        assert isinstance(client, EchoClient)
        assert client.provider_name == "echo"


def test_factory_returns_echo_for_unknown_provider():
    """Unknown provider name falls back to EchoClient."""
    with patch("app.llm.client.settings") as s:
        s.llm_provider = "skynet"
        client = create_llm_client()
        assert isinstance(client, EchoClient)


def test_factory_openai_missing_key_falls_back():
    """OPENAI without API key falls back to EchoClient."""
    with patch("app.llm.client.settings") as s:
        s.llm_provider = "openai"
        s.openai_api_key = ""
        client = create_llm_client()
        assert isinstance(client, EchoClient)


def test_factory_openai_with_key():
    """OPENAI with API key creates OpenAIClient."""
    with patch("app.llm.client.settings") as s:
        s.llm_provider = "openai"
        s.openai_api_key = "sk-test-key"
        s.openai_model = "gpt-4o-mini"
        s.llm_timeout = 30
        s.llm_max_tokens = 1024
        s.llm_temperature = 0.3
        client = create_llm_client()
        assert isinstance(client, OpenAIClient)
        assert client.provider_name == "openai"


def test_factory_gemini_missing_key_falls_back():
    """GEMINI without API key falls back to EchoClient."""
    with patch("app.llm.client.settings") as s:
        s.llm_provider = "gemini"
        s.gemini_api_key = ""
        client = create_llm_client()
        assert isinstance(client, EchoClient)


def test_factory_gemini_with_key():
    """GEMINI with API key creates GeminiClient."""
    with patch("app.llm.client.settings") as s:
        s.llm_provider = "gemini"
        s.gemini_api_key = "test-gemini-key"
        s.gemini_model = "gemini-2.0-flash"
        s.llm_timeout = 30
        s.llm_max_tokens = 1024
        s.llm_temperature = 0.3
        client = create_llm_client()
        assert isinstance(client, GeminiClient)
        assert client.provider_name == "gemini"


def test_factory_llama_no_key_required():
    """LLAMA does not require an API key (local model)."""
    with patch("app.llm.client.settings") as s:
        s.llm_provider = "llama"
        s.llama_base_url = "http://localhost:11434"
        s.llama_model = "llama3.2"
        s.llm_timeout = 60
        s.llm_max_tokens = 1024
        s.llm_temperature = 0.3
        client = create_llm_client()
        assert isinstance(client, LlamaClient)
        assert client.provider_name == "llama"


def test_factory_case_insensitive():
    """Provider name is case-insensitive."""
    with patch("app.llm.client.settings") as s:
        s.llm_provider = "OPENAI"
        s.openai_api_key = "sk-test"
        s.openai_model = "gpt-4o-mini"
        s.llm_timeout = 30
        s.llm_max_tokens = 1024
        s.llm_temperature = 0.3
        client = create_llm_client()
        assert isinstance(client, OpenAIClient)


def test_factory_strips_whitespace():
    """Provider name is trimmed."""
    with patch("app.llm.client.settings") as s:
        s.llm_provider = "  openai  "
        s.openai_api_key = "sk-test"
        s.openai_model = "gpt-4o-mini"
        s.llm_timeout = 30
        s.llm_max_tokens = 1024
        s.llm_temperature = 0.3
        client = create_llm_client()
        assert isinstance(client, OpenAIClient)


# ── EchoClient behavior ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_echo_client_returns_context():
    """EchoClient should include context snippets in response."""
    client = EchoClient()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Context: UPS Ground takes 1-5 days.\n\nQuestion: How long?"},
    ]
    response = await client.complete(messages)
    assert "shipping information" in response.lower() or "documents" in response.lower()
    assert len(response) > 0


@pytest.mark.asyncio
async def test_echo_client_handles_empty_messages():
    """EchoClient handles empty message list gracefully."""
    client = EchoClient()
    response = await client.complete([])
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.asyncio
async def test_echo_client_provider_name():
    client = EchoClient()
    assert client.provider_name == "echo"


# ── Prompt building ──────────────────────────────────────────────────────────


def test_build_rag_prompt_with_context():
    """RAG prompt includes context chunks."""
    messages = build_rag_prompt("What is DIM weight?", ["Chunk 1", "Chunk 2"])
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Chunk 1" in messages[1]["content"]
    assert "Chunk 2" in messages[1]["content"]
    assert "DIM weight" in messages[1]["content"]


def test_build_rag_prompt_without_context():
    """RAG prompt handles empty context gracefully."""
    messages = build_rag_prompt("How fast is UPS Ground?", [])
    assert len(messages) == 2
    assert "No context was retrieved" in messages[1]["content"]


def test_build_rag_prompt_grounding_instruction():
    """System prompt should instruct grounding and anti-hallucination."""
    messages = build_rag_prompt("test", ["chunk"])
    system = messages[0]["content"]
    assert "ONLY" in system or "only" in system.lower()
    assert "context" in system.lower()


def test_build_advisor_prompt_with_tools():
    """Advisor prompt includes tool results."""
    messages = build_advisor_prompt(
        query="How much to ship to NYC?",
        context="UPS Ground is 1-5 days.",
        tool_results="Quote Preview: Ground $12.50",
    )
    assert len(messages) == 2
    assert "Quote Preview" in messages[1]["content"]
    assert "UPS Ground" in messages[1]["content"]


def test_build_advisor_prompt_no_context():
    """Advisor prompt handles missing context and tools."""
    messages = build_advisor_prompt(query="Help", context="", tool_results="")
    assert "need more information" in messages[1]["content"].lower() or \
           "no context" in messages[1]["content"].lower()


# ── Gemini message conversion ────────────────────────────────────────────────


def test_gemini_message_conversion():
    """Gemini contents format conversion works correctly."""
    from app.llm.client import _messages_to_gemini_contents

    messages = [
        {"role": "system", "content": "You are a helper."},
        {"role": "user", "content": "What is DIM weight?"},
        {"role": "assistant", "content": "DIM weight is..."},
        {"role": "user", "content": "Thanks, tell me more."},
    ]
    contents = _messages_to_gemini_contents(messages)

    # System message should be merged into first user message
    assert len(contents) == 3
    assert contents[0]["role"] == "user"
    assert "You are a helper." in contents[0]["parts"][0]["text"]
    assert "DIM weight" in contents[0]["parts"][0]["text"]
    assert contents[1]["role"] == "model"  # assistant → model
    assert contents[2]["role"] == "user"


def test_gemini_message_conversion_no_system():
    """Gemini conversion works without system message."""
    from app.llm.client import _messages_to_gemini_contents

    messages = [{"role": "user", "content": "Hello"}]
    contents = _messages_to_gemini_contents(messages)
    assert len(contents) == 1
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "Hello"
