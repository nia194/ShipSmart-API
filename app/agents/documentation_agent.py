"""
Documentation agent (UC3) — render the customs/shipping documents.

Deterministic pass-through to the injected ``DocRenderer`` over a plain context
dict (built by the workflow node from the finished state). No LLM. The workflow
node wraps this and emits the ``workflow:docs:*`` decision tags.
"""

from __future__ import annotations

from app.domain.models import GeneratedDoc
from app.domain.ports import DocRenderer


def generate(context: dict, *, renderer: DocRenderer) -> list[GeneratedDoc]:
    """Render the documents for a finished workflow from its context dict."""
    return renderer.render(context)
