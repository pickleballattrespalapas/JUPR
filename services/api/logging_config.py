"""Production log formatters; request bodies, credentials and queries stay out."""
from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone

from uvicorn.logging import AccessFormatter


class RequestJSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        for field in ("method", "path", "status_code", "elapsed_ms", "request_id", "exception_type"):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


class RedactedAccessFormatter(AccessFormatter):
    def formatMessage(self, record: logging.LogRecord) -> str:
        # Uvicorn's separate access logger normally includes all query parameters.
        # Omit every query, including future capability names we do not yet know.
        sanitized = copy.copy(record)
        if isinstance(record.args, tuple) and len(record.args) == 5:
            client, method, full_path, version, status = record.args
            sanitized.args = (client, method, str(full_path).partition("?")[0], version, status)
        return super().formatMessage(sanitized)
