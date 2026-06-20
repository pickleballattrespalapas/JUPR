from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("jupr.api.request")


class StructuredRequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers.setdefault("x-request-id", request_id)

        logger.info(
            "request.complete",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "elapsed_ms": elapsed_ms,
                "request_id": request_id,
            },
        )
        return response
