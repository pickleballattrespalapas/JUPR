from __future__ import annotations

from typing import Any, Callable
from uuid import uuid4

from jupr_app.domain.replay_history import (
    FULL_RESET_LABEL,
    ReplayLeaseLostError,
    replay_history,
)


DEFAULT_REPLAY_LEASE_SECONDS = 1800
MIN_REPLAY_LEASE_SECONDS = 120
MAX_REPLAY_LEASE_SECONDS = 3600


def _rpc_payload(response: Any, *, label: str) -> dict[str, Any]:
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        return dict(data)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return dict(data[0])
    raise RuntimeError(
        f"{label} returned no durable result. Apply the replay lease migration "
        "before enabling writes."
    )


def _bounded_lease_seconds(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_REPLAY_LEASE_SECONDS
    return max(MIN_REPLAY_LEASE_SECONDS, min(parsed, MAX_REPLAY_LEASE_SECONDS))


def create_replay_job(
    *,
    supabase,
    club_id: str,
    target_reset: str,
    actor_email: str | None = None,
    actor_role: str | None = None,
    idempotency_key: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    clean_key = str(idempotency_key or "").strip()[:160] or None
    if clean_key:
        existing = (
            supabase.table("replay_jobs")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("idempotency_key", clean_key)
            .limit(1)
            .execute()
        )
        rows = existing.data or []
        if rows:
            existing_job = dict(rows[0])
            if str(existing_job.get("target_reset") or "") != str(target_reset):
                raise ValueError(
                    "Replay idempotency key is already bound to a different target."
                )
            return {**existing_job, "_created": False}
    row = {
        "club_id": str(club_id),
        "target_reset": str(target_reset),
        "status": "pending",
        "actor_email": actor_email,
        "actor_role": actor_role,
        "idempotency_key": clean_key,
        "source": str(source or "")[:120] or None,
    }
    try:
        resp = supabase.table("replay_jobs").insert(row).execute()
        data = (resp.data or [{}])[0]
    except Exception:
        if not clean_key:
            raise
        existing = (
            supabase.table("replay_jobs")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("idempotency_key", clean_key)
            .limit(1)
            .execute()
        )
        rows = existing.data or []
        if not rows:
            raise
        existing_job = dict(rows[0])
        if str(existing_job.get("target_reset") or "") != str(target_reset):
            raise ValueError(
                "Replay idempotency key is already bound to a different target."
            )
        return {**existing_job, "_created": False}
    return {**dict(data), "_created": True}


def _get_replay_job(
    *, supabase, club_id: str, job_id: str
) -> dict[str, Any] | None:
    response = (
        supabase.table("replay_jobs")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("id", str(job_id))
        .limit(1)
        .execute()
    )
    rows = response.data or []
    return dict(rows[0]) if rows else None


def claim_replay_job(
    *,
    supabase,
    club_id: str,
    job_id: str,
    worker_id: str,
    lease_seconds: int = DEFAULT_REPLAY_LEASE_SECONDS,
    retry_failed: bool = False,
) -> dict[str, Any]:
    response = supabase.rpc(
        "claim_replay_job_atomic",
        {
            "p_job_id": str(job_id),
            "p_club_id": str(club_id),
            "p_worker_id": str(worker_id),
            "p_lease_seconds": _bounded_lease_seconds(lease_seconds),
            "p_retry_failed": bool(retry_failed),
        },
    ).execute()
    payload = _rpc_payload(response, label="Replay claim RPC")
    if payload.get("ok") is not True and str(
        payload.get("status") or ""
    ) not in {"pending", "running", "failed", "succeeded"}:
        raise RuntimeError(
            str(
                payload.get("message")
                or payload.get("code")
                or "Replay job could not be claimed."
            )
        )
    return payload


def heartbeat_replay_job(
    *,
    supabase,
    club_id: str,
    job_id: str,
    lease_token: str,
    worker_id: str,
    lease_seconds: int = DEFAULT_REPLAY_LEASE_SECONDS,
) -> dict[str, Any]:
    response = supabase.rpc(
        "heartbeat_replay_job_atomic",
        {
            "p_job_id": str(job_id),
            "p_club_id": str(club_id),
            "p_lease_token": str(lease_token),
            "p_worker_id": str(worker_id),
            "p_lease_seconds": _bounded_lease_seconds(lease_seconds),
        },
    ).execute()
    payload = _rpc_payload(response, label="Replay heartbeat RPC")
    if payload.get("ok") is not True or payload.get("renewed") is not True:
        raise ReplayLeaseLostError(
            "Replay lease could not be renewed; the job must be recovered by a "
            "new worker after the active lease expires."
        )
    return payload


def finish_replay_job(
    *,
    supabase,
    club_id: str,
    job_id: str,
    lease_token: str,
    worker_id: str,
    status: str,
    result_json: dict[str, Any] | None = None,
    error_text: str | None = None,
) -> dict[str, Any]:
    clean_status = str(status or "").strip().lower()
    if clean_status not in {"succeeded", "failed"}:
        raise ValueError("Replay finish status must be succeeded or failed.")
    response = supabase.rpc(
        "finish_replay_job_atomic",
        {
            "p_job_id": str(job_id),
            "p_club_id": str(club_id),
            "p_lease_token": str(lease_token),
            "p_worker_id": str(worker_id),
            "p_status": clean_status,
            "p_result_json": dict(result_json or {}),
            "p_error_text": str(error_text or "")[:4000] or None,
        },
    ).execute()
    payload = _rpc_payload(response, label="Replay finish RPC")
    if payload.get("ok") is not True or payload.get("finished") is not True:
        raise ReplayLeaseLostError(
            "Replay result could not be committed because this worker no longer "
            "owns the active lease."
        )
    return payload


def _require_complete_replay_result(
    *, target_reset: str, result: dict[str, Any]
) -> None:
    if (
        str(target_reset).strip() == FULL_RESET_LABEL
        and result.get("singles_replay_supported") is not True
    ):
        raise RuntimeError(
            "Full replay did not attest replay-managed singles recovery."
        )


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
    idempotency_key: str | None = None,
    source: str | None = None,
    retry_failed: bool = False,
    worker_id: str | None = None,
    lease_seconds: int = DEFAULT_REPLAY_LEASE_SECONDS,
    replay_job_id: str | None = None,
) -> dict[str, Any]:
    if replay_job_id:
        job = _get_replay_job(
            supabase=supabase,
            club_id=str(club_id),
            job_id=str(replay_job_id),
        )
        if job is None:
            raise ValueError("Replay job was not found for this club.")
        if str(job.get("target_reset") or "") != str(target_reset):
            raise ValueError("Replay job target no longer matches this operation.")
    else:
        job = create_replay_job(
            supabase=supabase,
            club_id=str(club_id),
            target_reset=target_reset,
            actor_email=actor_email,
            actor_role=actor_role,
            idempotency_key=idempotency_key,
            source=source,
        )
    job_id = str(job.get("id") or "")
    if not job_id:
        raise RuntimeError("Replay job creation returned no durable job identity.")

    clean_worker_id = (
        str(worker_id or "").strip()[:160]
        or f"replay-worker:{uuid4()}"
    )
    safe_lease_seconds = _bounded_lease_seconds(lease_seconds)
    claim = claim_replay_job(
        supabase=supabase,
        club_id=str(club_id),
        job_id=job_id,
        worker_id=clean_worker_id,
        lease_seconds=safe_lease_seconds,
        retry_failed=retry_failed,
    )
    status = str(claim.get("status") or job.get("status") or "pending")
    result_json = dict(claim.get("result_json") or job.get("result_json") or {})

    if status == "succeeded":
        _require_complete_replay_result(
            target_reset=target_reset,
            result=result_json,
        )
        return {
            "job_id": job_id,
            "job_status": "succeeded",
            "result": result_json,
            "idempotent_replay": True,
            "lease_expires_at": claim.get("lease_expires_at"),
            "attempt_count": claim.get("attempt_count"),
        }
    if status == "failed" and claim.get("claimed") is not True:
        raise RuntimeError(
            str(
                claim.get("error_text")
                or job.get("error_text")
                or "The prior replay attempt failed. Use the guarded recovery action after review."
            )
        )
    if claim.get("claimed") is not True:
        current = (
            _get_replay_job(
                supabase=supabase,
                club_id=str(club_id),
                job_id=job_id,
            )
            or claim
            or job
        )
        current_status = str(current.get("status") or status or "pending")
        if current_status == "failed":
            raise RuntimeError(
                str(
                    current.get("error_text")
                    or "Replay failed before this request could claim it."
                )
            )
        current_result = dict(current.get("result_json") or {})
        if current_status == "succeeded":
            _require_complete_replay_result(
                target_reset=target_reset,
                result=current_result,
            )
        return {
            "job_id": job_id,
            "job_status": current_status,
            "result": current_result,
            "idempotent_replay": True,
            "lease_expires_at": current.get("lease_expires_at"),
            "attempt_count": current.get("attempt_count"),
        }

    lease_token = str(claim.get("lease_token") or "").strip()
    if not lease_token:
        raise RuntimeError("Replay claim returned no lease token.")

    def _tracked_progress(value: float) -> None:
        if progress_cb is not None:
            progress_cb(value)

    def _before_write_batch() -> None:
        heartbeat_replay_job(
            supabase=supabase,
            club_id=str(club_id),
            job_id=job_id,
            lease_token=lease_token,
            worker_id=clean_worker_id,
            lease_seconds=safe_lease_seconds,
        )

    try:
        result = replay_history(
            supabase=supabase,
            club_id=str(club_id),
            df_meta=df_meta,
            target_reset=target_reset,
            progress_cb=_tracked_progress,
            write_fence={
                "job_id": job_id,
                "lease_token": lease_token,
                "worker_id": clean_worker_id,
            },
            before_write_batch=_before_write_batch,
        )
        _require_complete_replay_result(
            target_reset=target_reset,
            result=result,
        )
        finished = finish_replay_job(
            supabase=supabase,
            club_id=str(club_id),
            job_id=job_id,
            lease_token=lease_token,
            worker_id=clean_worker_id,
            status="succeeded",
            result_json=result,
        )
        return {
            "job_id": job_id,
            "job_status": "succeeded",
            "result": dict(finished.get("result_json") or result),
            "idempotent_replay": bool(claim.get("idempotent_replay")),
            "lease_expires_at": claim.get("lease_expires_at"),
            "attempt_count": claim.get("attempt_count"),
        }
    except Exception as exc:
        if isinstance(exc, ReplayLeaseLostError):
            raise
        try:
            finish_replay_job(
                supabase=supabase,
                club_id=str(club_id),
                job_id=job_id,
                lease_token=lease_token,
                worker_id=clean_worker_id,
                status="failed",
                result_json={},
                error_text=str(exc),
            )
        except ReplayLeaseLostError as lease_exc:
            raise ReplayLeaseLostError(
                "Replay failed, and its failure could not be recorded because "
                "the worker lease was lost."
            ) from lease_exc
        raise
