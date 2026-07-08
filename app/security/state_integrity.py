"""Client-state integrity (Governance & Guardrails §7.2).

The concierge conversation state is client-owned and echoed back each turn. A
signature makes it tamper-evident: the server signs the state it issues, the
client echoes state + signature, and the server verifies before trusting it. An
unsigned or invalid state is treated as **empty** (re-gathered via clarification),
never as authority — so a forged "you already confirmed this is allowed" turn
changes nothing. Emits ``guardrail:state_unsigned``.
"""

from __future__ import annotations

import hmac
import json
from hashlib import sha256
from typing import Any

STATE_UNSIGNED_TAG = "guardrail:state_unsigned"


def _canonical(state: dict[str, Any]) -> bytes:
    return json.dumps(state, sort_keys=True, separators=(",", ":"), default=str).encode()


def sign_state(state: dict[str, Any], *, secret: str) -> str:
    """Return the HMAC-SHA256 signature the server issues alongside a state."""
    return hmac.new(secret.encode(), _canonical(state), sha256).hexdigest()


def verify_state(state: dict[str, Any], signature: str | None, *, secret: str) -> bool:
    """Constant-time check that ``signature`` matches ``state`` (both required)."""
    if not signature:
        return False
    expected = sign_state(state, secret=secret)
    return hmac.compare_digest(expected, signature)


def trust_state(
    state: dict[str, Any] | None,
    signature: str | None,
    *,
    secret: str,
) -> tuple[dict[str, Any], list[str]]:
    """Return the state to trust: the verified state, or ``{}`` if unsigned/invalid.

    An unsigned/forged state degrades to empty (tagged) rather than being trusted —
    the load-bearing §7.2 invariant.
    """
    if state and verify_state(state, signature, secret=secret):
        return state, []
    return {}, [STATE_UNSIGNED_TAG]
