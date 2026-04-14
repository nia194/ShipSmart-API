"""
Base provider interface.
All external service providers (shipping, address validation, etc.)
implement this interface so they can be swapped without changing tool logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ProviderResult:
    """Standardized result from any provider call."""

    success: bool
    data: dict = field(default_factory=dict)
    provider: str = ""
    error: str | None = None


class Provider(ABC):
    """Abstract base for all external service providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique provider identifier (e.g. 'mock', 'ups', 'fedex')."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable and configured."""
