"""
LLM client abstraction.
Multi-provider support with config-driven selection and graceful fallback.

Supported providers:
  - openai: OpenAI Chat Completions API (GPT-4o, GPT-4o-mini, etc.)
  - anthropic: Anthropic Messages API (Claude Sonnet, etc.)
  - gemini: Google Gemini API (Gemini 2.0 Flash, etc.)
  - llama: Local Llama via Ollama (OpenAI-compatible endpoint)
  - "" (empty): EchoClient placeholder (no external calls)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from app.core.config import settings
from app.core.errors import AppError

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract interface for LLM completion."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier for this provider (e.g. 'openai', 'gemini', 'echo')."""

    @abstractmethod
    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Send messages to the LLM and return the text response."""


# ── OpenAI ───────────────────────────────────────────────────────────────────


class OpenAIClient(LLMClient):
    """OpenAI Chat Completions API client.

    Production-ready with configurable timeout, temperature, and max tokens.
    Uses the openai AsyncOpenAI client with built-in retry support.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout: int = 30,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=2,
        )
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        logger.info("OpenAIClient initialized (model=%s, timeout=%ds)", model, timeout)

    @property
    def provider_name(self) -> str:
        return "openai"

    async def complete(self, messages: list[dict[str, str]]) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("OpenAI API error: %s", e)
            raise AppError(
                status_code=502, message="LLM provider returned an error"
            ) from e


# ── Google Gemini ────────────────────────────────────────────────────────────


class GeminiClient(LLMClient):
    """Google Gemini API client.

    Uses the Gemini REST API via httpx. Converts the standard
    chat message format to Gemini's contents format.

    API docs: https://ai.google.dev/gemini-api/docs
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        timeout: int = 30,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._temperature = temperature
        logger.info("GeminiClient initialized (model=%s, timeout=%ds)", model, timeout)

    @property
    def provider_name(self) -> str:
        return "gemini"

    async def complete(self, messages: list[dict[str, str]]) -> str:
        import httpx

        # Convert chat messages to Gemini contents format
        contents = _messages_to_gemini_contents(messages)

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent"
        )

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self._temperature,
                "maxOutputTokens": self._max_tokens,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    url,
                    json=payload,
                    params={"key": self._api_key},
                    headers={"Content-Type": "application/json"},
                )

            if resp.status_code != 200:
                logger.error("Gemini API error: HTTP %d — %s", resp.status_code, resp.text[:200])
                raise AppError(
                    status_code=502, message="Gemini API returned an error"
                )

            body = resp.json()
            candidates = body.get("candidates", [])
            if not candidates:
                return ""

            parts = candidates[0].get("content", {}).get("parts", [])
            return parts[0].get("text", "") if parts else ""

        except httpx.HTTPError as exc:
            logger.error("Gemini API network error: %s", exc)
            raise AppError(
                status_code=502, message="Gemini API network error"
            ) from exc


def _messages_to_gemini_contents(messages: list[dict[str, str]]) -> list[dict]:
    """Convert OpenAI-style messages to Gemini contents format.

    Gemini uses 'user' and 'model' roles (not 'assistant').
    System messages are prepended to the first user message.
    """
    system_text = ""
    contents: list[dict] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_text = content
            continue

        gemini_role = "model" if role == "assistant" else "user"

        # Prepend system text to first user message
        if system_text and gemini_role == "user":
            content = f"{system_text}\n\n{content}"
            system_text = ""

        contents.append({
            "role": gemini_role,
            "parts": [{"text": content}],
        })

    return contents


# ── Anthropic / Claude ───────────────────────────────────────────────────────


class AnthropicClient(LLMClient):
    """Anthropic Messages API client.

    Uses the official `anthropic` SDK's AsyncAnthropic. Converts the
    OpenAI-style chat messages list into Anthropic's `system=` + `messages=`
    shape (Anthropic does not accept a 'system' role inside messages).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5",
        timeout: int = 30,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ):
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=2)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        logger.info("AnthropicClient initialized (model=%s, timeout=%ds)", model, timeout)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    async def complete(self, messages: list[dict[str, str]]) -> str:
        try:
            system_parts: list[str] = []
            chat_messages: list[dict[str, str]] = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    if content:
                        system_parts.append(content)
                    continue
                # Anthropic expects 'user' or 'assistant'
                anth_role = "assistant" if role == "assistant" else "user"
                chat_messages.append({"role": anth_role, "content": content})

            if not chat_messages:
                # Anthropic requires at least one user message
                chat_messages = [{"role": "user", "content": "(no user message)"}]

            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system="\n\n".join(system_parts) if system_parts else None,
                messages=chat_messages,
            )

            # Concatenate any text blocks in the response
            parts: list[str] = []
            for block in getattr(response, "content", []) or []:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            return "".join(parts)
        except Exception as e:
            logger.error("Anthropic API error: %s", e)
            raise AppError(
                status_code=502, message="Anthropic API returned an error"
            ) from e


# ── Llama (via Ollama) ───────────────────────────────────────────────────────


class LlamaClient(LLMClient):
    """Local Llama client via Ollama's OpenAI-compatible API.

    Ollama exposes an OpenAI-compatible endpoint at /v1/chat/completions.
    This allows running local models without external API calls.

    Requires: Ollama running locally (https://ollama.com)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: int = 60,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ):
        from openai import AsyncOpenAI

        # Ollama provides an OpenAI-compatible endpoint
        self._client = AsyncOpenAI(
            api_key="ollama",  # Ollama doesn't require a real key
            base_url=f"{base_url}/v1",
            timeout=timeout,
            max_retries=1,
        )
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        logger.info("LlamaClient initialized (model=%s, base_url=%s)", model, base_url)

    @property
    def provider_name(self) -> str:
        return "llama"

    async def complete(self, messages: list[dict[str, str]]) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("Llama/Ollama API error: %s", e)
            raise AppError(
                status_code=502, message="Local LLM provider returned an error"
            ) from e


# ── Echo (fallback) ──────────────────────────────────────────────────────────


class EchoClient(LLMClient):
    """Placeholder LLM client that returns retrieved context as-is.

    Used when no LLM provider is configured. Returns a message explaining
    that no LLM is available, along with any context that was provided.
    """

    def __init__(self) -> None:
        logger.warning(
            "Using EchoClient — no LLM provider configured. "
            "Set LLM_PROVIDER=openai and OPENAI_API_KEY for real completions."
        )

    @property
    def provider_name(self) -> str:
        return "echo"

    async def complete(self, messages: list[dict[str, str]]) -> str:
        # Extract the user message and any context from the prompt
        user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )

        # Try to separate context from question
        parts = user_msg.split("\n\n")
        context_parts = [
            p for p in parts
            if p.startswith("Relevant") or p.startswith("Context:")
        ]
        question_parts = [
            p for p in parts
            if p.startswith("Question:") or p.startswith("Issue:")
        ]

        response = "Based on available shipping information:\n\n"
        if context_parts:
            snippet = context_parts[0][:500].strip()
            response += snippet + "\n\n"
        if question_parts:
            response += f"Your question: {question_parts[0]}\n\n"

        response += (
            "Note: This response is based on retrieved documents only. "
            "AI-powered answers will provide more detailed, personalized guidance."
        )
        return response


# ── Factory ──────────────────────────────────────────────────────────────────


def build_provider_client(provider: str) -> LLMClient | None:
    """Build a single provider client by name.

    Returns None if the provider name is unknown, credentials are missing,
    or instantiation raises. Callers (factory + router) decide how to
    fall back. Provider name is normalised case-insensitively.
    """
    provider = (provider or "").lower().strip()
    if not provider:
        return None
    if provider == "echo":
        return EchoClient()

    try:
        if provider == "openai":
            if not settings.openai_api_key:
                logger.warning(
                    "Provider 'openai' requested but OPENAI_API_KEY is not set"
                )
                return None
            return OpenAIClient(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                timeout=settings.llm_timeout,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
            )
        if provider == "gemini":
            if not settings.gemini_api_key:
                logger.warning(
                    "Provider 'gemini' requested but GEMINI_API_KEY is not set"
                )
                return None
            return GeminiClient(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                timeout=settings.llm_timeout,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
            )
        if provider == "anthropic":
            if not settings.anthropic_api_key:
                logger.warning(
                    "Provider 'anthropic' requested but ANTHROPIC_API_KEY is not set"
                )
                return None
            return AnthropicClient(
                api_key=settings.anthropic_api_key,
                model=settings.anthropic_model,
                timeout=settings.llm_timeout,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
            )
        if provider == "llama":
            return LlamaClient(
                base_url=settings.llama_base_url,
                model=settings.llama_model,
                timeout=settings.llm_timeout,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
            )
        logger.warning("Unknown LLM provider=%r", provider)
        return None
    except Exception as exc:
        logger.warning("Failed to create LLM client for provider=%s: %s", provider, exc)
        return None


def create_llm_client() -> LLMClient:
    """Factory: create the legacy single LLM client from LLM_PROVIDER.

    Kept for back-compat. Task-based routing should use
    `app.llm.router.create_llm_router()` instead.
    """
    provider = settings.llm_provider.lower().strip()

    if not provider:
        return EchoClient()

    try:
        if provider == "openai":
            if not settings.openai_api_key:
                logger.warning(
                    "LLM_PROVIDER=openai but OPENAI_API_KEY is not set — "
                    "falling back to EchoClient"
                )
                return EchoClient()
            return OpenAIClient(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                timeout=settings.llm_timeout,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
            )

        if provider == "gemini":
            if not settings.gemini_api_key:
                logger.warning(
                    "LLM_PROVIDER=gemini but GEMINI_API_KEY is not set — "
                    "falling back to EchoClient"
                )
                return EchoClient()
            return GeminiClient(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                timeout=settings.llm_timeout,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
            )

        if provider == "anthropic":
            if not settings.anthropic_api_key:
                logger.warning(
                    "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set — "
                    "falling back to EchoClient"
                )
                return EchoClient()
            return AnthropicClient(
                api_key=settings.anthropic_api_key,
                model=settings.anthropic_model,
                timeout=settings.llm_timeout,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
            )

        if provider == "llama":
            return LlamaClient(
                base_url=settings.llama_base_url,
                model=settings.llama_model,
                timeout=settings.llm_timeout,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
            )

        logger.warning(
            "Unknown LLM_PROVIDER=%r — falling back to EchoClient", provider
        )
        return EchoClient()

    except Exception as exc:
        logger.warning(
            "Failed to create LLM client for provider=%s: %s — "
            "falling back to EchoClient",
            provider, exc,
        )
        return EchoClient()
