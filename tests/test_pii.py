"""PII redaction + write-time pseudonymization (guardrails §6.1/§7.5)."""

from __future__ import annotations

from app.security.pii import (
    pseudonymize,
    redact,
    redact_mapping,
    redact_value,
)

SECRET = "test-secret"


def test_redacts_email_phone_tracking_address_ip():
    text = (
        "Email jane.doe@example.com or call +1 (415) 555-2671. "
        "Ship to 123 Main Street, tracking 1Z999AA10123456784, "
        "or the 20-digit 12345678901234567890, host 10.0.0.5."
    )
    out = redact(text)
    assert "jane.doe@example.com" not in out and "[REDACTED_EMAIL]" in out
    assert "555-2671" not in out and "[REDACTED_PHONE]" in out
    assert "123 Main Street" not in out and "[REDACTED_ADDRESS]" in out
    assert "1Z999AA10123456784" not in out and "[REDACTED_TRACKING]" in out
    assert "12345678901234567890" not in out
    assert "10.0.0.5" not in out and "[REDACTED_IP]" in out


def test_redact_is_noop_on_clean_text():
    clean = "Compare Ground vs Express for a 5 lb box."
    assert redact(clean) == clean


def test_redact_mapping_and_nested_value():
    data = {"note": "reach me at a@b.com", "nested": {"list": ["call 415-555-2671"]}}
    out = redact_mapping(data)
    assert "a@b.com" not in out["note"]
    assert "555-2671" not in out["nested"]["list"][0]
    assert redact_value(5) == 5 and redact_value(None) is None


def test_pseudonymize_is_deterministic_and_namespaced():
    a = pseudonymize("user-123", secret=SECRET, kind="user")
    assert a == pseudonymize("user-123", secret=SECRET, kind="user")   # stable
    assert a.startswith("usr_")
    assert a != pseudonymize("user-999", secret=SECRET, kind="user")   # distinct
    # same identity, different kind => different token (no session/user collision)
    assert a != pseudonymize("user-123", secret=SECRET, kind="session")
    assert pseudonymize("user-123", secret="other", kind="user") != a  # secret matters
    assert pseudonymize(None, secret=SECRET) is None
