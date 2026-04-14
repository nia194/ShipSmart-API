"""
Address validation tool.
Validates and normalizes shipping addresses through the provider abstraction.
"""

from __future__ import annotations

import logging

from app.providers.shipping_provider import AddressInput, ShippingProvider
from app.tools.base import Tool, ToolInput, ToolOutput, ToolParameter

logger = logging.getLogger(__name__)


class ValidateAddressTool(Tool):
    """Validate and normalize a shipping address."""

    def __init__(self, provider: ShippingProvider) -> None:
        self._provider = provider

    @property
    def name(self) -> str:
        return "validate_address"

    @property
    def description(self) -> str:
        return (
            "Validate a shipping address and return a normalized version. "
            "Checks for required fields and format issues."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter("street", "string", "Street address line"),
            ToolParameter("city", "string", "City name"),
            ToolParameter("state", "string", "State code (e.g. CA, NY)"),
            ToolParameter("zip_code", "string", "ZIP code (e.g. 90210)"),
            ToolParameter(
                "country", "string", "Country code (default US)",
                required=False,
            ),
        ]

    async def execute(self, tool_input: ToolInput) -> ToolOutput:
        params = tool_input.params
        address = AddressInput(
            street=params.get("street", ""),
            city=params.get("city", ""),
            state=params.get("state", ""),
            zip_code=params.get("zip_code", ""),
            country=params.get("country", "US"),
        )

        logger.info("Validating address via provider=%s", self._provider.name)
        result = await self._provider.validate_address(address)

        return ToolOutput(
            success=result.success,
            data=result.data,
            error=result.error,
            metadata={"provider": result.provider, "tool": self.name},
        )
