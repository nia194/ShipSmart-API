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
from dataclasses import dataclass

from app.core.config import settings
from app.llm.client import EchoClient, LLMClient, build_provider_client

logger = logging.getLogger(__name__)


# Canonical task names. Services pass these strings to `for_task`.
TASK_REASONING = "reasoning"
TASK_SYNTHESIS = "synthesis"
TASK_FALLBACK = "fallback"

KNOWN_TASKS = (TASK_REASONING, TASK_SYNTHESIS, TASK_FALLBACK)


@dataclass
class LLMRouter:
    """Holds one LLMClient per task. Built once at startup."""

    clients: dict[str, LLMClient]
    fallback: LLMClient

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

    def describe(self) -> dict[str, str]:
        """Human-readable mapping of task → provider, for logs and debugging."""
        return {task: c.provider_name for task, c in self.clients.items()} | {
            "fallback": self.fallback.provider_name,
        }


def _resolve_task_provider(task_provider: str, legacy_provider: str) -> str:
    """A task with no explicit provider inherits the legacy LLM_PROVIDER.

    This means existing single-provider deployments keep working with no
    config change at all.
    """
    return (task_provider or legacy_provider or "").strip()


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
        client = build_provider_client(provider_name) if provider_name else None
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

    return LLMRouter(clients=clients, fallback=fallback_client)
