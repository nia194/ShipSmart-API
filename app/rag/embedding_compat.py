"""Embedding version compatibility (Governance & Guardrails §7.3).

Stored vectors carry no meaning across embedding models — a config flip or a
provider upgrade silently mixes incompatible vector spaces and degrades retrieval
with no error (the most invisible failure in the system). This fail-closed check
compares the configured embedding model/version against the store's recorded one
before serving retrieval. Emits ``guardrail:embedding_version``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

EMBEDDING_VERSION_TAG = "guardrail:embedding_version"


@dataclass
class CompatResult:
    ok: bool
    reason: str = ""
    tags: list[str] = field(default_factory=list)


def check_embedding_compat(
    *,
    configured_model: str,
    configured_version: str,
    store_model: str | None,
    store_version: str | None,
) -> CompatResult:
    """Fail-closed unless the configured embedding matches the store's recorded one.

    An empty store (no recorded model/version yet) is OK — it will be stamped on
    first ingest. A mismatch is not: it must fail closed, never serve silently.
    """
    if store_model is None and store_version is None:
        return CompatResult(ok=True, reason="empty store — stamped on first ingest")
    if configured_model != store_model or configured_version != store_version:
        return CompatResult(
            ok=False,
            reason=(
                f"embedding mismatch: configured {configured_model}@{configured_version} "
                f"!= store {store_model}@{store_version} — re-index before serving"
            ),
            tags=[EMBEDDING_VERSION_TAG],
        )
    return CompatResult(ok=True)
