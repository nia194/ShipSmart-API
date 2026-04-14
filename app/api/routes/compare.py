"""
Compare endpoint for decision-cockpit shipping comparisons.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from app.core.config import settings
from app.core.errors import AppError
from app.core.rate_limit import limiter
from app.llm.router import TASK_REASONING, LLMRouter
from app.schemas.compare import CompareOption, CompareRequest, CompareResponse
from app.services.compare_service import generate_compare_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/compare", tags=["compare"])


@router.post("", response_model=CompareResponse)
@limiter.limit(settings.rate_limit_compare)
async def compare_options(
    body: CompareRequest,
    request: Request,
) -> CompareResponse:
    """
    Compare 2-3 shipping options with LLM-driven priority reasoning.

    Request body:
    {
      "shipment": { origin_zip, destination_zip, item_description, ... },
      "option_ids": ["id1", "id2", "id3"]
    }

    Note: The frontend sends option_ids, but the backend must look up
    the full option data from somewhere (Java API quote cache or parameter).
    For MVP, option details are expected to be provided by the frontend
    in a separate parameter or we call Java API.

    Response: All 4 scenarios (ontime, damage, price, speed) precomputed.
    """
    # Get the LLM router from app state
    llm_router: LLMRouter = request.app.state.llm_router
    llm_client = llm_router.for_task(TASK_REASONING)

    logger.info(
        "Compare request: %d options, provider=%s",
        len(body.option_ids),
        llm_client.provider_name,
    )

    # Use the enriched options provided by the frontend
    options = body.options

    # Generate compare response
    response = await generate_compare_response(body, options, llm_client)

    logger.info(
        "Compare response generated: scenarios=%s",
        ", ".join(response.scenarios.keys()),
    )

    return response
