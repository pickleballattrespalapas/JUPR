from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Iterable

from postgrest.exceptions import APIError


BADGE_QUEUE_TABLE = "badge_eval_queue"
_MISSING_TABLE_CODES = {"PGRST205", "42P01"}

logger = logging.getLogger(__name__)


def enqueue_badge_eval(
    supabase: Any,
    *,
    club_id: str,
    event_type: str,
    player_ids: Iterable[int] | None = None,
    context_id: str | None = None,
    match_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if supabase is None or not club_id or not event_type:
        return {"queued": False, "reason": "invalid_input"}
    payload_json = dict(payload or {})
    row = {
        "club_id": str(club_id),
        "event_type": str(event_type),
        "player_ids": [int(pid) for pid in (player_ids or [])],
        "context_id": context_id,
        "match_id": match_id,
        "payload_json": payload_json,
        "status": "pending",
    }
    try:
        table = supabase.table(BADGE_QUEUE_TABLE)
        if match_id:
            table.upsert(row, on_conflict="event_type,match_id").execute()
        else:
            table.insert(row).execute()
    except APIError as exc:
        code = _get_api_error_code(exc)
        message = _get_api_error_message(exc)
        if code in _MISSING_TABLE_CODES:
            logger.warning(
                "Badge queue table %s missing in PostgREST schema cache (code=%s message=%s). "
                "Skipping badge enqueue.",
                BADGE_QUEUE_TABLE,
                code,
                message,
            )
            return {"queued": False, "reason": "missing_table"}
        logger.warning(
            "Failed to enqueue badge evaluation (code=%s message=%s). Skipping badge enqueue.",
            code,
            message,
        )
        return {"queued": False, "reason": "api_error"}
    except Exception as exc:  # noqa: BLE001 - badge enqueue should never crash uploads
        logger.warning("Unexpected error while enqueueing badge evaluation: %s", exc)
        return {"queued": False, "reason": "unexpected_error"}
    return {"queued": True, "reason": "ok"}


def dequeue_badge_eval(supabase: Any) -> dict[str, Any] | None:
    if supabase is None:
        return None
    try:
        resp = (
            supabase.table(BADGE_QUEUE_TABLE)
            .select("*")
            .eq("status", "pending")
            .order("created_at", desc=False)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            return None
        job = rows[0]
        attempts = int(job.get("attempts") or 0) + 1
        supabase.table(BADGE_QUEUE_TABLE).update(
            {"status": "processing", "attempts": attempts}
        ).eq("id", job.get("id")).execute()
        job["attempts"] = attempts
        job["status"] = "processing"
        return job
    except APIError as exc:
        code = _get_api_error_code(exc)
        message = _get_api_error_message(exc)
        if code in _MISSING_TABLE_CODES:
            logger.warning(
                "Badge queue table %s missing in PostgREST schema cache (code=%s message=%s). "
                "Skipping badge dequeue.",
                BADGE_QUEUE_TABLE,
                code,
                message,
            )
            return None
        raise


def ack_badge_eval(
    supabase: Any,
    *,
    job_id: str,
    status: str,
    error: str | None = None,
) -> None:
    if supabase is None or not job_id:
        return
    payload: dict[str, Any] = {
        "status": status,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    if error:
        payload["last_error"] = error
    supabase.table(BADGE_QUEUE_TABLE).update(payload).eq("id", job_id).execute()


def _get_api_error_code(exc: APIError) -> str | None:
    code = getattr(exc, "code", None)
    if code:
        return code
    if exc.args and isinstance(exc.args[0], dict):
        return exc.args[0].get("code")
    return None


def _get_api_error_message(exc: APIError) -> str:
    message = getattr(exc, "message", None)
    if message:
        return message
    if exc.args and isinstance(exc.args[0], dict):
        return exc.args[0].get("message", str(exc))
    return str(exc)
