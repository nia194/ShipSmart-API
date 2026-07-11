"""
Task-based LLM routing.

Maps a logical task (reasoning, synthesis, ...) to an underlying LLMClient.
Each task is configured independently via env vars so different parts of the
system can use different providers/models — e.g. OpenAI for advisor reasoning
and Gemini for RAG synthesis — without changing service code.

Design notes (interview talking points):
  - Config-driven, not heuristic. The mapping is explicit in env vars.
  - Independent fallback per task: if a task's provider can't be built we
    fall through to LLM_PROVIDER_FALLBACK, then to EchoClient. The app never
    crashes because of LLM config.
  - The router is built once at startup and reused for every request, so
    there is no per-request provider construction cost.
  - The legacy `create_llm_client()` factory is preserved untouched; the
    router simply layers on top of `build_provider_client`.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.core.config import settings
from app.llm.budget import clamp_temperature, parse_float_or, parse_int_or
from app.llm.client import EchoClient, LLMClient, build_provider_client
from app.llm.errors import LLMError, ProviderOutageError, classify_provider_error

logger = logging.getLogger(__name__)


# Canonical task names. Services pass these strings to `for_task`.
TASK_REASONING = "reasoning"
TASK_SYNTHESIS = "synthesis"
TASK_FALLBACK = "fallback"

KNOWN_TASKS = (TASK_REASONING, TASK_SYNTHESIS, TASK_FALLBACK)


@dataclass
class ExecuteResult:
    """Outcome of an LLMRouter.execute() call.

    ``provider`` is who actually answered; ``failed_over`` is True when the
    primary client did not produce the answer (a fallback did).
    """

    text: str
    provider: str
    failed_over: bool = False
    hops: int = 1


@dataclass
class LLMRouter:
    """Holds one LLMClient per task. Built once at startup."""

    clients: dict[str, LLMClient]
    fallback: LLMClient
    # Per-task failover chain: [primary, *fallbacks, echo]. When LLM_FALLBACK_CHAIN
    # is empty this is just [primary] — i.e. today's single-client behavior.
    chains: dict[str, list[LLMClient]] = field(default_factory=dict)

    def for_task(self, task: str) -> LLMClient:
        """Return the client configured for the given task.

        Unknown task names return the fallback client rather than raising —
        this keeps callers safe even when env config is incomplete.
        """
        client = self.clients.get(task)
        if client is None:
            logger.debug("No client for task=%s, using fallback=%s",
                         task, self.fallback.provider_name)
            return self.fallback
        return client

    def chain_for(self, task: str) -> list[LLMClient]:
        """Return the failover chain for a task ([primary] if none configured)."""
        chain = self.chains.get(task)
        if chain:
            return chain
        return [self.for_task(task)]

    async def execute(
        self,
        task: str,
        messages: list[dict[str, str]],
        *,
        request_id: str = "",
    ) -> ExecuteResult:
        """Run ``task`` against its failover chain.

        Per provider: try once, and on a *retryable* error retry up to
        LLM_RETRY_MAX_ATTEMPTS, then fail over to the next provider; a *terminal*
        error (auth, context-length, content-filter) fails fast with no retry or
        failover. With an empty LLM_FALLBACK_CHAIN the chain is a single client
        and retries are disabled — byte-for-byte today's behavior. Every hop is
        logged with provider + error class + request_id.
        """
        chain = self.chain_for(task)
        multi = len(chain) > 1
        max_attempts = max(1, getattr(settings, "llm_retry_max_attempts", 2)) if multi else 1
        last_error: LLMError | None = None

        for hop, client in enumerate(chain):
            for attempt in range(1, max_attempts + 1):
                try:
                    text = await client.complete(messages)
                except Exception as exc:  # noqa: BLE001 - classified below
                    err = classify_provider_error(exc, client.provider_name)
                    last_error = err
                    logger.warning(
                        "LLM hop failed: task=%s provider=%s class=%s retryable=%s "
                        "hop=%d attempt=%d rid=%s",
                        task, client.provider_name, err.kind, err.retryable,
                        hop, attempt, request_id,
                    )
                    if not err.retryable:
                        raise err from exc  # terminal: fail fast
                    if attempt < max_attempts:
                        continue  # retry same provider
                    break  # exhausted retries → next provider in chain
                else:
                    if hop > 0 or attempt > 1:
                        logger.info(
                            "LLM execute recovered: task=%s provider=%s hop=%d "
                            "attempt=%d rid=%s",
                            task, client.provider_name, hop, attempt, request_id,
                        )
                    return ExecuteResult(
                        text=text, provider=client.provider_name,
                        failed_over=hop > 0, hops=hop + 1,
                    )

        # Chain exhausted on retryable errors.
        raise last_error or ProviderOutageError(detail="empty LLM chain")

    async def stream(
        self,
        task: str,
        messages: list[dict[str, str]],
        *,
        request_id: str = "",
    ) -> AsyncIterator[str]:
        """Stream ``task`` token deltas, failing over ONLY before the first token.

        Once a delta has been yielded we cannot un-send it, so a mid-stream error
        propagates; a failure before any output (or a terminal error) fails over
        to the next provider in the chain exactly like ``execute``.
        """
        chain = self.chain_for(task)
        last_error: LLMError | None = None

        for hop, client in enumerate(chain):
            started = False
            try:
                async for delta in client.stream(messages):
                    started = True
                    yield delta
                return  # provider completed the stream
            except Exception as exc:  # noqa: BLE001 - classified below
                err = classify_provider_error(exc, client.provider_name)
                last_error = err
                logger.warning(
                    "LLM stream hop failed: task=%s provider=%s class=%s started=%s hop=%d rid=%s",
                    task, client.provider_name, err.kind, started, hop, request_id,
                )
                if started or not err.retryable:
                    raise err from exc  # can't recover mid-stream, or terminal error
                # nothing yielded yet + retryable → try the next provider

        raise last_error or ProviderOutageError(detail="empty LLM chain")

    def describe(self) -> dict[str, str]:
        """Human-readable mapping of task → provider, for logs and debugging."""
        return {task: c.provider_name for task, c in self.clients.items()} | {
            "fallback": self.fallback.provider_name,
        }

    def describe_chains(self) -> dict[str, list[str]]:
        """Resolved failover chain per task, as provider names — for /ready + logs."""
        return {
            task: [c.provider_name for c in self.chain_for(task)]
            for task in (TASK_REASONING, TASK_SYNTHESIS)
        }


def _resolve_task_provider(task_provider: str, legacy_provider: str) -> str:
    """A task with no explicit provider inherits the legacy LLM_PROVIDER.

    This means existing single-provider deployments keep working with no
    config change at all.
    """
    return (task_provider or legacy_provider or "").strip()


def _task_overrides(task: str) -> dict[str, object]:
    """Per-task model / temperature / max-token overrides (B).

    Empty override = inherit the global LLM_* value (today's behavior). Advisor
    and synthesis temperature is clamped to <= 0.3 — ShipSmart is a grounded
    advisor, so a stray high-temperature override can't make it riff. Reads via
    getattr so partially-populated settings objects (e.g. tests) never raise.
    """
    if task == TASK_REASONING:
        model = getattr(settings, "llm_model_reasoning", "")
        temp_raw = getattr(settings, "llm_temperature_reasoning", "")
        mt_raw = getattr(settings, "llm_max_tokens_reasoning", "")
    elif task == TASK_SYNTHESIS:
        model = getattr(settings, "llm_model_synthesis", "")
        temp_raw = getattr(settings, "llm_temperature_synthesis", "")
        mt_raw = getattr(settings, "llm_max_tokens_synthesis", "")
    else:
        return {}

    out: dict[str, object] = {}
    if str(model or "").strip():
        out["model"] = str(model).strip()
    if str(temp_raw or "").strip():
        out["temperature"] = clamp_temperature(
            parse_float_or(temp_raw, settings.llm_temperature)
        )
    if str(mt_raw or "").strip():
        out["max_tokens"] = parse_int_or(mt_raw, settings.llm_max_tokens)
    return out


def _build_fallback_tail() -> list[LLMClient]:
    """Build the shared failover tail from LLM_FALLBACK_CHAIN (built once).

    Empty chain → no tail (today's single-client behavior). A non-empty chain is
    always terminated with EchoClient so a request can never dead-end with no
    answer. Unknown/unavailable providers are skipped with a warning.
    """
    raw = (getattr(settings, "llm_fallback_chain", "") or "").strip()
    if not raw:
        return []

    tail: list[LLMClient] = []
    seen: set[str] = set()
    for name in (n.strip() for n in raw.split(",")):
        key = name.lower()
        if not name or key in seen:
            continue
        seen.add(key)
        if key == "echo":
            tail.append(EchoClient())
        else:
            client = build_provider_client(name)
            if client is not None:
                tail.append(client)
            else:
                logger.warning(
                    "LLM_FALLBACK_CHAIN provider %r unavailable — skipping", name,
                )
    if not tail or not isinstance(tail[-1], EchoClient):
        tail.append(EchoClient())
    return tail


def create_llm_router() -> LLMRouter:
    """Build the LLMRouter from config.

    Resolution order for each task:
      1. LLM_PROVIDER_<TASK> if set
      2. LLM_PROVIDER (legacy single-provider) if set
      3. LLM_PROVIDER_FALLBACK
      4. EchoClient (always works)
    """
    legacy = settings.llm_provider

    fallback_name = (settings.llm_provider_fallback or "echo").strip()
    fallback_client = build_provider_client(fallback_name) or EchoClient()

    task_provider_names = {
        TASK_REASONING: _resolve_task_provider(
            settings.llm_provider_reasoning, legacy
        ),
        TASK_SYNTHESIS: _resolve_task_provider(
            settings.llm_provider_synthesis, legacy
        ),
    }

    clients: dict[str, LLMClient] = {}
    for task, provider_name in task_provider_names.items():
        overrides = _task_overrides(task)
        client = (
            build_provider_client(provider_name, **overrides) if provider_name else None
        )
        if client is None:
            logger.warning(
                "Task '%s' provider=%r unavailable — falling back to %s",
                task, provider_name or "<unset>", fallback_client.provider_name,
            )
            client = fallback_client
        clients[task] = client
        logger.info("LLM router: task=%s → provider=%s", task, client.provider_name)

    # Expose the fallback as a queryable task too, so callers can ask for it
    # explicitly (e.g. degraded paths).
    clients[TASK_FALLBACK] = fallback_client

    # Resolve the request-time failover chain per task. The tail is shared; each
    # task prepends its own primary. Empty LLM_FALLBACK_CHAIN → [primary] only.
    tail = _build_fallback_tail()
    chains: dict[str, list[LLMClient]] = {}
    if tail:
        for task in (TASK_REASONING, TASK_SYNTHESIS):
            chains[task] = [clients[task], *tail]
        logger.info(
            "LLM failover chains: %s",
            {t: [c.provider_name for c in ch] for t, ch in chains.items()},
        )

    return LLMRouter(clients=clients, fallback=fallback_client, chains=chains)
