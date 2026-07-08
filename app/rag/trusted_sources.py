"""Trusted-source registry (Governance & Guardrails §5.7).

The model is only as safe as the data it retrieves. Every corpus document is a
registered trusted source with a content hash (tamper-evident), a version, an
effective date, and a staleness window — so a stale-source answer can be
downgraded to "needs verification" and a citation to an unregistered source can
be rejected. Pure + keyless; derived from the corpus directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

DEFAULT_STALE_AFTER_DAYS = 365


@dataclass(frozen=True)
class TrustedSource:
    source: str          # rel path, e.g. "compliance/lithium-batteries-dangerous-goods.md"
    source_type: str     # top folder: carriers | compliance | guides | policies | scenarios
    content_hash: str
    version: str
    effective_date: str  # ISO date
    stale_after_days: int = DEFAULT_STALE_AFTER_DAYS
    allowed_use: str = "grounding"

    def is_stale(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(tz=UTC)
        eff = datetime.fromisoformat(self.effective_date).replace(tzinfo=UTC)
        return now - eff > timedelta(days=self.stale_after_days)


def build_registry(corpus_dir: str | Path) -> dict[str, TrustedSource]:
    """Scan a corpus directory into a {rel_path: TrustedSource} registry."""
    root = Path(corpus_dir)
    registry: dict[str, TrustedSource] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        h = sha256(path.read_bytes()).hexdigest()
        eff = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).date().isoformat()
        registry[rel] = TrustedSource(
            source=rel,
            source_type=rel.split("/")[0],
            content_hash=h,
            version=h[:12],
            effective_date=eff,
        )
    return registry


def is_trusted(registry: dict[str, TrustedSource], source: str) -> bool:
    """True if ``source`` (a rel path or bare filename) is a registered source."""
    if source in registry:
        return True
    return any(s.source == source or s.source.endswith("/" + source) for s in registry.values())
