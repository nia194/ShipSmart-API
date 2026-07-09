"""Runtime AI-controls admin endpoint (Governance & Guardrails §12).

The operator-facing side of the kill-switch registry: inspect which AI features
are live and flip one during an incident, without a redeploy. Fail-closed twice:

* If ``admin_api_token`` is not configured, the endpoint does not exist (404 on
  every call) — no token, no admin surface.
* With a token configured, callers must present it in ``X-Admin-Token``;
  comparison is constant-time.

Every flip is audited as an append-only AIEvent by the registry itself.
"""

from __future__ import annotations

import hmac

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.errors import AppError
from app.core.kill_switch import KILLABLE_FEATURES, registry

router = APIRouter(prefix="/admin", tags=["admin"])


def _require_admin(x_admin_token: str) -> None:
    token = settings.admin_api_token
    if not token:
        # Unconfigured -> the admin surface does not exist.
        raise AppError(status_code=404, message="Not found")
    if not hmac.compare_digest(x_admin_token.encode(), token.encode()):
        raise AppError(status_code=403, message="Invalid admin token")


class AiControlsResponse(BaseModel):
    features: dict[str, bool]


class FlipRequest(BaseModel):
    feature: str
    enabled: bool
    reason: str = Field(default="", max_length=500)


@router.get("/ai-controls", response_model=AiControlsResponse)
async def get_ai_controls(x_admin_token: str = Header(default="")) -> AiControlsResponse:
    """Current runtime state of every killable AI feature."""
    _require_admin(x_admin_token)
    return AiControlsResponse(features=registry.snapshot())


@router.post("/ai-controls", response_model=AiControlsResponse)
async def flip_ai_control(
    body: FlipRequest, x_admin_token: str = Header(default="")
) -> AiControlsResponse:
    """Kill or restore one AI feature at runtime (audited)."""
    _require_admin(x_admin_token)
    if body.feature not in KILLABLE_FEATURES:
        raise AppError(
            status_code=422,
            message=f"unknown feature {body.feature!r}",
            detail=f"killable features: {', '.join(KILLABLE_FEATURES)}",
        )
    registry.set_enabled(body.feature, body.enabled, actor="admin_api", reason=body.reason)
    return AiControlsResponse(features=registry.snapshot())
