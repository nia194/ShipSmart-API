"""PII redaction + write-time pseudonymization (Governance & Guardrails §6.1/§7.5).

The audit/observability plane must never persist raw PII, and identity is
pseudonymized **at write time** so a later DSAR erasure (deleting the raw-identity
store, §6.1) renders events unlinkable while preserving the audit trail.

- :func:`redact` scrubs emails, phone numbers, tracking numbers, and street
  addresses from free text before it lands in a log/trace/event.
- :func:`redact_mapping` scrubs the string values of a dict (structured logs).
- :func:`pseudonymize` deterministically maps an identity to a stable opaque
  token via HMAC-SHA256(secret, kind:identity) — raw identity is never stored.

All pure + keyless. The secret is a config value (dev default; override in prod).
"""

from __future__ import annotations

import hmac
import re
from hashlib import sha256
from typing import Any

_EMAIL = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
# UPS 1Z + 16 alnum; generic long digit runs cover FedEx/USPS/DHL tracking + ids.
_TRACKING = re.compile(r"\b1Z[0-9A-Z]{16}\b|\b\d{12,22}\b")
# +CC and 7+ digit sequences with common separators (kept conservative).
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
_STREET = re.compile(
    r"\b\d{1,6}\s+(?:[A-Z][a-zA-Z]+\.?\s+){1,4}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl)\b",
    re.IGNORECASE,
)
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# Order matters: email + tracking + street + ip before the broad phone matcher.
_RULES: list[tuple[re.Pattern[str], str]] = [
    (_EMAIL, "[REDACTED_EMAIL]"),
    (_STREET, "[REDACTED_ADDRESS]"),
    (_TRACKING, "[REDACTED_TRACKING]"),
    (_IP, "[REDACTED_IP]"),
    (_PHONE, "[REDACTED_PHONE]"),
]


def redact(text: str) -> str:
    """Return ``text`` with detectable PII replaced by typed placeholders."""
    if not text:
        return text
    for pattern, repl in _RULES:
        text = pattern.sub(repl, text)
    return text


def redact_value(value: Any) -> Any:
    """Recursively redact strings inside a value (str / list / dict / scalar)."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    return value


def redact_mapping(data: dict[str, Any]) -> dict[str, Any]:
    """Redact PII from the values of a structured-log/event dict."""
    return {k: redact_value(v) for k, v in data.items()}


def pseudonymize(identity: str | None, *, secret: str, kind: str = "user") -> str | None:
    """Stable opaque token for an identity (HMAC-SHA256). ``None`` stays ``None``.

    ``kind`` namespaces the token (``user`` -> ``usr_…``, ``session`` -> ``sess_…``)
    so a session hash and a user pseudonym never collide.
    """
    if not identity:
        return None
    digest = hmac.new(secret.encode(), f"{kind}:{identity}".encode(), sha256).hexdigest()
    prefix = {"user": "usr", "session": "sess"}.get(kind, kind[:4])
    return f"{prefix}_{digest[:16]}"
