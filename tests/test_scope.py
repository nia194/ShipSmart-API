"""Unit tests for the shipping-scope policy guard (app/core/scope.py)."""

from __future__ import annotations

import pytest

from app.core import scope
from app.core.errors import AppError


def _domestic(monkeypatch, country: str = "US") -> None:
    monkeypatch.setattr(scope.settings, "shipping_scope", "domestic", raising=False)
    monkeypatch.setattr(scope.settings, "domestic_country", country, raising=False)


def test_worldwide_is_a_noop(monkeypatch):
    monkeypatch.setattr(scope.settings, "shipping_scope", "worldwide", raising=False)
    assert scope.violates_domestic_scope("US", "DE") is None
    scope.enforce_scope("US", "DE")  # must not raise


def test_domestic_allows_home_to_home(monkeypatch):
    _domestic(monkeypatch)
    assert scope.violates_domestic_scope("US", "US") is None
    scope.enforce_scope("US", "US")  # ok


def test_domestic_rejects_cross_border_with_422(monkeypatch):
    _domestic(monkeypatch)
    assert scope.violates_domestic_scope("US", "DE") == "DE"
    with pytest.raises(AppError) as excinfo:
        scope.enforce_scope("US", "DE")
    assert excinfo.value.status_code == 422


def test_domestic_rejects_foreign_origin(monkeypatch):
    _domestic(monkeypatch)
    assert scope.violates_domestic_scope("DE", "US") == "DE"


def test_domestic_ignores_empty_country(monkeypatch):
    # Empty defaults to the home country downstream, so it never violates.
    _domestic(monkeypatch)
    assert scope.violates_domestic_scope("", "") is None


def test_domestic_is_case_insensitive(monkeypatch):
    _domestic(monkeypatch, country="us")
    assert scope.violates_domestic_scope("us", "Us") is None
