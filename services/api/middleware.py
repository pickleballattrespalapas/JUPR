from __future__ import annotations

import logging
import os
import re
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Match

from scripts.staging_write_waves import wave_allows_request
from jupr_app.observability.request_timing import RequestTiming, current_timing

logger = logging.getLogger("jupr.api.request")


class StructuredRequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        supplied_id = request.headers.get("x-request-id", "")
        request_id = supplied_id if re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", supplied_id) else str(uuid.uuid4())
        start = time.perf_counter()
        timing = RequestTiming(request_id)
        token = current_timing.set(timing)
        try:
            response = await call_next(request)
        except Exception as exc:
            _log_request("request.failed", request, timing, start, 500, type(exc).__name__)
            raise
        finally:
            current_timing.reset(token)
        response.headers.setdefault("x-request-id", request_id)
        _log_request("request.complete", request, timing, start, response.status_code)
        return response


def _request_route_template(request):
    matched = getattr(request.scope.get("route"), "path", None)
    if matched:
        return matched
    # Guards can reject a request before the router has populated scope.route.
    # Match only registered route metadata; never call the endpoint or echo a
    # raw URL (which could contain an unknown capability or private value).
    for route in getattr(request.app, "routes", ()):
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return getattr(route, "path", None) or "<unmatched>"
    return "<unmatched>"


def _log_request(event, request, timing, start, status_code, exception_type=None):
    try:
        # Matched route templates exclude player IDs, capability paths and names.
        # Unmatched/early-denied paths are not echoed back into the logs.
        route = _request_route_template(request)
        machine = os.getenv("FLY_MACHINE_ID", "")
        logger.info(event, extra={
            "method": request.method,
            "path": route,
            "status_code": status_code,
            "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
            "request_id": timing.request_id,
            "exception_type": exception_type,
            "machine_id": machine if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", machine) else None,
            **timing.snapshot(),
        })
    except Exception:
        # An unavailable log sink must never fail the application request.
        pass


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
