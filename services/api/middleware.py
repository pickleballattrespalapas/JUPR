from __future__ import annotations

import logging
import os
import re
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from scripts.staging_write_waves import wave_allows_request

logger = logging.getLogger("jupr.api.request")


class StructuredRequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        supplied_id = request.headers.get("x-request-id", "")
        request_id = supplied_id if re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", supplied_id) else str(uuid.uuid4())
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.info(
                "request.failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": 500,
                    "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
                    "request_id": request_id,
                    # Exception messages can embed user input; retain only the type.
                    "exception_type": type(exc).__name__,
                },
            )
            raise
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


class StagingWriteWaveMiddleware(BaseHTTPMiddleware):
    """Fail closed before routing unless the runtime and write policy are explicit."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        environment = os.getenv("JUPR_ENV", "").strip().lower()
        unsafe = request.method.upper() in {"POST", "PUT", "PATCH", "DELETE"}
        if unsafe and environment not in {
            "local",
            "test",
            "development",
            "dev",
            "production",
            "staging",
        }:
            return JSONResponse(
                status_code=403,
                content={"detail": "Unsafe requests require an explicit runtime environment."},
            )
        if unsafe and environment == "production":
            policy = os.getenv("JUPR_PRODUCTION_WRITE_POLICY", "").strip().lower()
            if policy != "enabled":
                return JSONResponse(
                    status_code=403,
                    content={
                        "detail": (
                            "Production business-data writes are disabled by the "
                            "fail-closed deployment policy."
                        )
                    },
                )
        if environment == "staging" and unsafe:
            wave = os.getenv("JUPR_STAGING_WRITE_WAVE", "").strip()
            if not wave_allows_request(wave, request.method, request.url.path):
                return JSONResponse(
                    status_code=403,
                    content={
                        "detail": "This unsafe request is outside the selected staging write wave."
                    },
                )
        return await call_next(request)
