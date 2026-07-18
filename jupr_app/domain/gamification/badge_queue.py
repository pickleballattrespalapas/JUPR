from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Iterable

from postgrest.exceptions import APIError


BADGE_QUEUE_TABLE = "badge_eval_queue"
BADGE_QUEUE_CLAIM_RPC = "claim_badge_eval_queue_job"
_MISSING_TABLE_CODES = {"PGRST205", "42P01"}
_MISSING_CLAIM_RPC_CODES = {"PGRST202", "42883"}

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
            table.upsert(row, on_conflict="club_id,event_type,match_id").execute()
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


def dequeue_badge_eval(supabase: Any, *, club_id: str) -> dict[str, Any] | None:
    clean_club_id = str(club_id or "").strip()
    if supabase is None:
        return None
    if not clean_club_id:
        raise ValueError("club_id is required to dequeue a badge evaluation")
    try:
        resp = supabase.rpc(
            BADGE_QUEUE_CLAIM_RPC,
            {"p_club_id": clean_club_id},
        ).execute()
        rows = resp.data or []
        if isinstance(rows, dict):
            rows = [rows]
        if not rows:
            return None
        job = dict(rows[0])
        if str(job.get("club_id") or "") != clean_club_id:
            raise RuntimeError("Atomic badge queue claim returned a job for another club.")
        return job
    except APIError as exc:
        code = _get_api_error_code(exc)
        message = _get_api_error_message(exc)
        if code in _MISSING_CLAIM_RPC_CODES:
            raise RuntimeError(
                "Atomic badge queue claims are unavailable. Apply "
                "supabase/migrations/20260718141016_badge_eval_queue_atomic_club_claim.sql "
                "before running badge workers."
            ) from exc
        if code in _MISSING_TABLE_CODES:
            logger.warning(
                "Badge queue table %s missing in PostgREST schema cache (code=%s message=%s). "
                "Skipping badge queue claim.",
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
