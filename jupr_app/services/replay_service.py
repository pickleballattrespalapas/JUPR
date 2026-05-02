from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from jupr_app.domain.replay_history import replay_history


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_replay_job(
    *,
    supabase,
    club_id: str,
    target_reset: str,
    actor_email: str | None = None,
    actor_role: str | None = None,
) -> dict[str, Any]:
    row = {
        "club_id": club_id,
        "target_reset": str(target_reset),
        "status": "pending",
        "actor_email": actor_email,
        "actor_role": actor_role,
    }
    resp = supabase.table("replay_jobs").insert(row).execute()
    data = (resp.data or [{}])[0]
    return data


def mark_replay_job_running(*, supabase, job_id: str) -> None:
    supabase.table("replay_jobs").update(
        {"status": "running", "started_at": _utc_now_iso(), "updated_at": _utc_now_iso()}
    ).eq("id", job_id).execute()


def mark_replay_job_succeeded(*, supabase, job_id: str, result_json: dict[str, Any]) -> None:
    supabase.table("replay_jobs").update(
        {
            "status": "succeeded",
            "finished_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "result_json": result_json or {},
            "error_text": None,
        }
    ).eq("id", job_id).execute()


def mark_replay_job_failed(*, supabase, job_id: str, error_text: str) -> None:
    supabase.table("replay_jobs").update(
        {
            "status": "failed",
            "finished_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "error_text": str(error_text),
        }
    ).eq("id", job_id).execute()



def is_replay_jobs_table_missing_error(exc: Exception) -> bool:
    code = str(getattr(exc, "code", "") or "").upper()
    args0 = exc.args[0] if getattr(exc, "args", None) else None
    payload_code = ""
    payload_text = ""
    if isinstance(args0, dict):
        payload_code = str(args0.get("code") or "").upper()
        payload_text = " ".join(
            str(args0.get(k) or "") for k in ("message", "details", "hint")
        )

    if code == "42P01" or payload_code == "42P01":
        return True
    if code == "PGRST205" or payload_code == "PGRST205":
        return True

    text = f"{exc} {payload_text}".lower()
    return (
        "replay_jobs" in text
        and (
            "does not exist" in text
            or "could not find" in text
            or "schema cache" in text
        )
    )

def run_replay_with_job_tracking(
    *,
    supabase,
    club_id: str,
    df_meta,
    target_reset: str,
    actor_email: str | None = None,
    actor_role: str | None = None,
    progress_cb: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    job = create_replay_job(
        supabase=supabase,
        club_id=club_id,
        target_reset=target_reset,
        actor_email=actor_email,
        actor_role=actor_role,
    )
    job_id = str(job.get("id") or "")
    try:
        mark_replay_job_running(supabase=supabase, job_id=job_id)
        result = replay_history(
            supabase=supabase,
            club_id=club_id,
            df_meta=df_meta,
            target_reset=target_reset,
            progress_cb=progress_cb,
        )
        mark_replay_job_succeeded(supabase=supabase, job_id=job_id, result_json=result)
        return {"job_id": job_id, "job_status": "succeeded", "result": result}
    except Exception as exc:
        mark_replay_job_failed(supabase=supabase, job_id=job_id, error_text=str(exc))
        raise
