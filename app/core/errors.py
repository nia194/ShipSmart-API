"""
Centralized error handling.
Registers global exception handlers on the FastAPI app so all errors
return a consistent JSON response format.
"""

import logging
from datetime import UTC, datetime

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Application-level error with an HTTP status code."""

    def __init__(
        self,
        status_code: int = 500,
        message: str = "Internal server error",
        detail: str | None = None,
    ):
        self.status_code = status_code
        self.message = message
        self.detail = detail
        super().__init__(message)


def _error_response(status: int, error: str, message: str, path: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "status": status,
            "error": error,
            "message": message,
            "path": path,
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the FastAPI application."""

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError,
    ) -> JSONResponse:
        errors = "; ".join(
            f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        )
        logger.warning("Validation error on %s: %s", request.url.path, errors)
        return _error_response(422, "Validation Error", errors, request.url.path)

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        logger.warning("AppError on %s: %s", request.url.path, exc.message)
        return _error_response(exc.status_code, "Error", exc.message, request.url.path)

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unexpected error on %s: %s", request.url.path, exc, exc_info=True)
        return _error_response(
            500, "Internal Server Error", "An unexpected error occurred", request.url.path,
        )
