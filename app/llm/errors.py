"""
LLM error taxonomy (A).

Each provider client maps its raw SDK/HTTP exceptions into one of these typed
errors via :func:`classify_provider_error`. The router's execute chain then uses
the ``retryable`` flag to decide failover:

  retryable / failover  → RateLimitError, ProviderOutageError,
                          ProviderTimeoutError, MalformedResponseError
  terminal / fail-fast  → AuthError, ContextLengthError, ContentFilterError

All subclass :class:`app.core.errors.AppError`, so the existing global handler
renders them with the right HTTP status without leaking provider internals.
"""

from __future__ import annotations

import json

from app.core.errors import AppError


class LLMError(AppError):
    """Base for classified LLM provider failures.

    ``retryable`` drives failover: a retryable error may be retried on the same
    provider and/or failed over to the next provider in the chain; a terminal
    error fails fast (retrying/failing over would not help and may burn cost or
    repeat a policy violation).
    """

    retryable: bool = False
    kind: str = "llm_error"

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        status_code: int = 502,
        detail: str | None = None,
    ) -> None:
        self.provider = provider
        super().__init__(status_code=status_code, message=message, detail=detail)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.kind}(provider={self.provider!r}, status={self.status_code})"


# ── Retryable / failover ─────────────────────────────────────────────────────


class RateLimitError(LLMError):
    retryable = True
    kind = "rate_limit"

    def __init__(self, *, provider: str = "", detail: str | None = None) -> None:
        super().__init__(
            "LLM provider rate limit exceeded",
            provider=provider, status_code=429, detail=detail,
        )


class ProviderOutageError(LLMError):
    retryable = True
    kind = "provider_outage"

    def __init__(self, *, provider: str = "", detail: str | None = None) -> None:
        super().__init__(
            "LLM provider is unavailable",
            provider=provider, status_code=502, detail=detail,
        )


class ProviderTimeoutError(LLMError):
    retryable = True
    kind = "timeout"

    def __init__(self, *, provider: str = "", detail: str | None = None) -> None:
        super().__init__(
            "LLM provider timed out",
            provider=provider, status_code=504, detail=detail,
        )


class MalformedResponseError(LLMError):
    """Provider replied but the output could not be parsed (e.g. invalid JSON
    for a structured-output call). Retryable: a re-roll often succeeds."""

    retryable = True
    kind = "malformed_response"

    def __init__(self, *, provider: str = "", detail: str | None = None) -> None:
        super().__init__(
            "LLM provider returned a malformed response",
            provider=provider, status_code=502, detail=detail,
        )


# ── Terminal / fail-fast ─────────────────────────────────────────────────────


class AuthError(LLMError):
    retryable = False
    kind = "auth"

    def __init__(self, *, provider: str = "", detail: str | None = None) -> None:
        # Surface as 502: a provider auth failure is a server-side misconfig,
        # not the API caller's fault — don't leak it as a client 401.
        super().__init__(
            "LLM provider authentication failed",
            provider=provider, status_code=502, detail=detail,
        )


class ContextLengthError(LLMError):
    """Prompt + requested output exceed the model's context window."""

    retryable = False
    kind = "context_length"

    def __init__(self, *, provider: str = "", detail: str | None = None) -> None:
        super().__init__(
            "Request exceeds the model context window",
            provider=provider, status_code=400, detail=detail,
        )


class ContentFilterError(LLMError):
    retryable = False
    kind = "content_filter"

    def __init__(self, *, provider: str = "", detail: str | None = None) -> None:
        super().__init__(
            "Request was blocked by the provider content filter",
            provider=provider, status_code=422, detail=detail,
        )


# ── Classification ───────────────────────────────────────────────────────────

_CONTEXT_HINTS = (
    "context length", "context_length", "maximum context", "max_tokens",
    "too many tokens", "reduce the length", "string too long",
)
_CONTENT_HINTS = (
    "content filter", "content_filter", "content_policy", "content management policy",
    "safety", "responsibleai", "flagged",
)


def _status_of(exc: Exception) -> int | None:
    """Best-effort HTTP status from an SDK/HTTP exception."""
    code = getattr(exc, "status_code", None)
    if code is None:
        resp = getattr(exc, "response", None)
        code = getattr(resp, "status_code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def classify_provider_error(exc: Exception, provider: str) -> LLMError:
    """Map a raw provider exception into the typed taxonomy.

    Inspects (in order): an already-classified LLMError, timeouts, JSON-decode
    failures, HTTP status code, then the exception class name / message. Unknown
    failures default to a retryable ProviderOutageError so a configured fallback
    chain can degrade gracefully; with no chain the error simply propagates.
    """
    if isinstance(exc, LLMError):
        if not exc.provider:
            exc.provider = provider
        return exc

    name = type(exc).__name__.lower()
    msg = str(exc).lower()

    # Timeouts (stdlib, httpx.TimeoutException, openai.APITimeoutError, ...)
    if isinstance(exc, TimeoutError) or "timeout" in name or "timed out" in msg:
        return ProviderTimeoutError(provider=provider, detail=str(exc))

    # Structured-output / JSON parse failures
    if isinstance(exc, json.JSONDecodeError):
        return MalformedResponseError(provider=provider, detail=str(exc))

    status = _status_of(exc)
    if status is not None:
        if status == 429:
            return RateLimitError(provider=provider, detail=str(exc))
        if status in (401, 403):
            return AuthError(provider=provider, detail=str(exc))
        if status in (408, 504):
            return ProviderTimeoutError(provider=provider, detail=str(exc))
        if status >= 500:
            return ProviderOutageError(provider=provider, detail=str(exc))
        if status in (400, 413, 422):
            if any(h in msg for h in _CONTEXT_HINTS):
                return ContextLengthError(provider=provider, detail=str(exc))
            if any(h in msg for h in _CONTENT_HINTS):
                return ContentFilterError(provider=provider, detail=str(exc))
            # Other 4xx from the provider: treat as outage-ish but terminal-safe.
            return ProviderOutageError(provider=provider, detail=str(exc))

    # Class-name heuristics for SDKs that don't expose a status here.
    if "ratelimit" in name:
        return RateLimitError(provider=provider, detail=str(exc))
    if "authentication" in name or "permissiondenied" in name:
        return AuthError(provider=provider, detail=str(exc))
    if any(h in msg for h in _CONTENT_HINTS):
        return ContentFilterError(provider=provider, detail=str(exc))
    if any(h in msg for h in _CONTEXT_HINTS):
        return ContextLengthError(provider=provider, detail=str(exc))

    # Unknown → retryable outage (degrade via chain when one is configured).
    return ProviderOutageError(provider=provider, detail=str(exc))
