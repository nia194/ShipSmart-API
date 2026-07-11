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
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.llm.errors import classify_provider_error

logger = logging.getLogger(__name__)


# ── Tool-calling primitives ──────────────────────────────────────────────────
# Minimal, provider-agnostic shapes for the agent loop (app/services/agent_service).
# Only providers with native function calling implement complete_with_tools; the
# rest keep the default NotImplementedError so the agent falls back to the
# text-based single-pass selection (select_tool_with_llm).


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallResult:
    """Outcome of one ``complete_with_tools`` turn.

    ``kind`` is ``"final"`` when the model is done (``text`` holds the answer) or
    ``"tool_calls"`` when it wants tools run (``calls`` holds them; ``text`` may
    carry any accompanying narration).
    """

    kind: str  # "final" | "tool_calls"
    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)


class LLMClient(ABC):
    """Abstract interface for LLM completion."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier for this provider (e.g. 'openai', 'gemini', 'echo')."""

    @abstractmethod
    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Send messages to the LLM and return the text response."""

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Yield the completion incrementally as text deltas.

        Default: providers without native streaming produce the whole answer in
        one chunk (so callers get a uniform interface and degrade gracefully).
        Streaming providers override this to yield token deltas as they arrive.
        """
        yield await self.complete(messages)

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCallResult:
        """Run one tool-calling turn and return a :class:`ToolCallResult`.

        Default: providers without native function calling raise
        ``NotImplementedError`` so the agent loop can fall back to the existing
        text-based tool selection. Override in providers that support native
        tool use (e.g. AnthropicClient).
        """
        raise NotImplementedError(
            f"{self.provider_name} has no native tool calling"
        )


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
            raise classify_provider_error(e, self.provider_name) from e

    async def stream(  # pragma: no cover - needs keys
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception as e:
            logger.error("OpenAI streaming error: %s", e)
            raise classify_provider_error(e, self.provider_name) from e


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
                # Send the API key as a header, NOT a `?key=` URL param — a query-string
                # key leaks into access/httpx logs and proxies in plaintext.
                resp = await client.post(
                    url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self._api_key,
                    },
                )

            if resp.status_code != 200:
                logger.error("Gemini API error: HTTP %d — %s", resp.status_code, resp.text[:200])
                http_err = RuntimeError(f"Gemini HTTP {resp.status_code}: {resp.text[:200]}")
                http_err.status_code = resp.status_code  # type: ignore[attr-defined]
                raise classify_provider_error(http_err, self.provider_name)

            body = resp.json()
            candidates = body.get("candidates", [])
            if not candidates:
                return ""

            parts = candidates[0].get("content", {}).get("parts", [])
            return parts[0].get("text", "") if parts else ""

        except httpx.HTTPError as exc:
            logger.error("Gemini API network error: %s", exc)
            raise classify_provider_error(exc, self.provider_name) from exc


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
            raise classify_provider_error(e, self.provider_name) from e

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCallResult:
        """Native Anthropic tool use (``tool_use`` blocks).

        Message ``content`` may be a plain string or a list of Anthropic content
        blocks (the agent loop appends ``tool_use`` / ``tool_result`` blocks for
        prior turns); both are passed through. Returns a ``tool_calls`` result
        when the model emits ``tool_use`` blocks, otherwise a ``final`` result.
        """
        try:
            system_parts: list[str] = []
            chat_messages: list[dict[str, Any]] = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    if content:
                        system_parts.append(
                            content if isinstance(content, str) else str(content)
                        )
                    continue
                anth_role = "assistant" if role == "assistant" else "user"
                chat_messages.append({"role": anth_role, "content": content})

            if not chat_messages:
                chat_messages = [{"role": "user", "content": "(no user message)"}]

            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system="\n\n".join(system_parts) if system_parts else None,
                messages=chat_messages,
                tools=_to_anthropic_tools(tools),
            )

            text_parts: list[str] = []
            calls: list[ToolCall] = []
            for block in getattr(response, "content", []) or []:
                btype = getattr(block, "type", None)
                if btype == "tool_use":
                    calls.append(
                        ToolCall(
                            id=getattr(block, "id", ""),
                            name=getattr(block, "name", ""),
                            arguments=dict(getattr(block, "input", {}) or {}),
                        )
                    )
                else:
                    text = getattr(block, "text", None)
                    if text:
                        text_parts.append(text)

            if calls:
                return ToolCallResult(
                    kind="tool_calls", text="".join(text_parts), calls=calls,
                )
            return ToolCallResult(kind="final", text="".join(text_parts))
        except Exception as e:
            logger.error("Anthropic tool-calling error: %s", e)
            raise classify_provider_error(e, self.provider_name) from e


def _to_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert registry tool schemas to Anthropic's ``tools`` shape.

    Registry schema (``RemoteTool.schema()`` / the agent's local tools):
        {"name", "description", "parameters": [{"name", "type", "description",
         "required"}]}
    Anthropic expects ``input_schema`` as a JSON-Schema object.
    """
    out: list[dict[str, Any]] = []
    for tool in tools:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param in tool.get("parameters", []):
            properties[param["name"]] = {
                "type": param.get("type", "string"),
                "description": param.get("description", ""),
            }
            if param.get("required"):
                required.append(param["name"])
        out.append({
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        })
    return out


# ── Scripted tool-calling stub (keyless, deterministic) ──────────────────────


class ScriptedToolCallingClient(LLMClient):
    """Deterministic, keyless stand-in for a native tool-calling provider.

    Modeled on ShipSmart-MCP's mock shipping provider: it replays a fixed list of
    ``ToolCallResult`` turns so the agent loop can run end-to-end with no API keys
    (the local stack runs keyless, where the reasoning client would otherwise be
    EchoClient — which has no native tool calling). Each ``complete_with_tools``
    call returns the next scripted turn; once the script is exhausted it returns a
    ``final`` turn. ``complete`` returns the configured final text.

    Selected via ``LLM_PROVIDER_REASONING=scripted`` and gated to non-production.
    """

    # Canonical sequence for the concierge use case (§3.3 of the plan):
    # retrieve a policy → validate the address → preview a quote → answer.
    _DEFAULT_FINAL_TEXT = (
        "Based on the gathered information, here is your shipping summary."
    )

    def __init__(
        self,
        turns: list[ToolCallResult] | None = None,
        *,
        final_text: str = _DEFAULT_FINAL_TEXT,
    ) -> None:
        self._turns = list(turns) if turns is not None else _default_script()
        self._final_text = final_text
        self._i = 0
        logger.info(
            "ScriptedToolCallingClient initialized (%d scripted turns)",
            len(self._turns),
        )

    @property
    def provider_name(self) -> str:
        return "scripted"

    async def complete(self, messages: list[dict[str, str]]) -> str:
        return self._final_text

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCallResult:
        if self._i < len(self._turns):
            turn = self._turns[self._i]
            self._i += 1
            return turn
        return ToolCallResult(kind="final", text=self._final_text)


def _default_script() -> list[ToolCallResult]:
    """The canonical concierge tool sequence the keyless e2e exercises."""
    return [
        ToolCallResult(
            kind="tool_calls",
            calls=[ToolCall(
                id="call_1", name="retrieve_rag",
                arguments={"query": "power bank lithium battery shipping restrictions"},
            )],
        ),
        ToolCallResult(
            kind="tool_calls",
            calls=[ToolCall(
                id="call_2", name="validate_address",
                arguments={
                    "street": "123 Main St", "city": "Beverly Hills",
                    "state": "CA", "zip_code": "90210",
                },
            )],
        ),
        ToolCallResult(
            kind="tool_calls",
            calls=[ToolCall(
                id="call_3", name="get_quote_preview",
                arguments={
                    "origin_zip": "10001", "destination_zip": "90210",
                    "weight_lbs": 5.0, "length_in": 10.0,
                    "width_in": 8.0, "height_in": 6.0,
                },
            )],
        ),
        ToolCallResult(kind="final", text=ScriptedToolCallingClient._DEFAULT_FINAL_TEXT),
    ]


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
            raise classify_provider_error(e, self.provider_name) from e


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

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Deterministic word-by-word streaming — keyless, so the SSE path is testable."""
        text = await self.complete(messages)
        for index, word in enumerate(text.split(" ")):
            yield word if index == 0 else " " + word


# ── Factory ──────────────────────────────────────────────────────────────────


def build_provider_client(
    provider: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMClient | None:
    """Build a single provider client by name.

    Returns None if the provider name is unknown, credentials are missing,
    or instantiation raises. Callers (factory + router) decide how to
    fall back. Provider name is normalised case-insensitively.

    Optional per-task overrides (model / temperature / max_tokens) default to
    the global LLM_* settings when None, so existing callers are unaffected.
    """
    provider = (provider or "").lower().strip()
    if not provider:
        return None
    if provider == "echo":
        return EchoClient()
    if provider == "scripted":
        # Deterministic keyless tool-calling stub for the agent loop. Gated to
        # non-production so it can never be selected in a live deployment.
        if settings.is_production:
            logger.warning(
                "Provider 'scripted' requested in production — refusing; "
                "falling back."
            )
            return None
        return ScriptedToolCallingClient()

    temp = settings.llm_temperature if temperature is None else temperature
    mt = max_tokens or settings.llm_max_tokens

    try:
        if provider == "openai":
            if not settings.openai_api_key:
                logger.warning(
                    "Provider 'openai' requested but OPENAI_API_KEY is not set"
                )
                return None
            return OpenAIClient(
                api_key=settings.openai_api_key,
                model=model or settings.openai_model,
                timeout=settings.llm_timeout,
                max_tokens=mt,
                temperature=temp,
            )
        if provider == "gemini":
            if not settings.gemini_api_key:
                logger.warning(
                    "Provider 'gemini' requested but GEMINI_API_KEY is not set"
                )
                return None
            return GeminiClient(
                api_key=settings.gemini_api_key,
                model=model or settings.gemini_model,
                timeout=settings.llm_timeout,
                max_tokens=mt,
                temperature=temp,
            )
        if provider == "anthropic":
            if not settings.anthropic_api_key:
                logger.warning(
                    "Provider 'anthropic' requested but ANTHROPIC_API_KEY is not set"
                )
                return None
            return AnthropicClient(
                api_key=settings.anthropic_api_key,
                model=model or settings.anthropic_model,
                timeout=settings.llm_timeout,
                max_tokens=mt,
                temperature=temp,
            )
        if provider == "llama":
            return LlamaClient(
                base_url=settings.llama_base_url,
                model=model or settings.llama_model,
                timeout=settings.llm_timeout,
                max_tokens=mt,
                temperature=temp,
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
