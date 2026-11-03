"""Middleware for timing, privacy protection, and request size limiting."""

from __future__ import annotations

import logging
import time
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.status import HTTP_413_CONTENT_TOO_LARGE

logger = logging.getLogger("fedsecureai.api")

# 2 MB request body limit
MAX_CONTENT_LENGTH = 2 * 1024 * 1024


class RequestTimingAndPrivacyMiddleware(BaseHTTPMiddleware):
    """Logs endpoint execution duration while strictly preventing transaction leakage."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Enforce size limit
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_CONTENT_LENGTH:
            return JSONResponse(
                status_code=HTTP_413_CONTENT_TOO_LARGE,
                content={"detail": "Request payload exceeds 2 MB limit."},
            )

        start_time = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Privacy-safe log: path, method, status code, latency.
        # Zero body or account id logging.
        logger.info(
            "HTTP %s %s -> status=%d duration=%.2fms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response
