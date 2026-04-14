"""
Shared rate limiter for advisor / orchestration endpoints.

Uses slowapi (FastAPI-friendly wrapper around limits). Keys requests by the
client's remote address. Limits are tunable via env vars in app.core.config.

Routes opt-in by decorating with `@limiter.limit(settings.rate_limit_*)` and
including a `request: Request` parameter (slowapi inspects it).
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

# Single shared limiter instance, registered onto `app.state.limiter` in main.
limiter = Limiter(key_func=get_remote_address)
