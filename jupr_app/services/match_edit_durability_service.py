from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from jupr_app.domain.bulk_match_editor import compute_recompute_scope
from jupr_app.services.admin_replay_service import is_admin_replay_enabled, run_admin_replay_history


FULL_RESET_LABEL = "ALL (Full System Reset)"


class MatchEditRecoveryRequired(RuntimeError):
    def __init__(self, operation_id: str, message: str):
        self.operation_id = str(operation_id or "")
        super().__init__(message)


def _rpc_payload(response: Any) -> dict[str, Any]:
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        return dict(data)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return dict(data[0])
    raise RuntimeError("Atomic Match Log RPC returned no operation record. Apply the staging migration before enabling writes.")


def _clean_key(value: Any) -> str:
    key = str(value or "").strip()[:160]
    if not key:
        raise ValueError("A stable idempotency key is required for Match Log edits.")
    return key


def _update_operation(
    supabase: Any,
    *,
    operation_id: str,
    status: str,
    replay_job_id: str | None = None,
    result_json: dict[str, Any] | None = None,
    error_text: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "error_text": str(error_text or "")[:2000] or None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if replay_job_id:
        payload["replay_job_id"] = str(replay_job_id)
    if result_json is not None:
        payload["result_json"] = result_json
    if status == "succeeded":
        payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    supabase.table("match_edit_operations").update(payload).eq("id", str(operation_id)).execute()


def _replay_key(idempotency_key: str) -> str:
    return f"match-edit:{idempotency_key}"[:160]


def apply_atomic_match_edits(
    supabase: Any,
    *,
    club_id: str,
    patches: list[dict[str, Any]],
    actor_email: str,
    actor_role: str,
    correction_note: str | None,
    source: str,
    idempotency_key: str,
    replay_target: str = FULL_RESET_LABEL,
) -> dict[str, Any]:
    clean_patches = [dict(patch) for patch in patches if isinstance(patch, dict)]
    scope = compute_recompute_scope(clean_patches)
    if scope.get("ratings") and not is_admin_replay_enabled():
        raise PermissionError("Rating-affecting Match Log edits require JUPR_ENABLE_NEXT_ADMIN_REPLAY=1 before any write is attempted.")
    clean_key = _clean_key(idempotency_key)
    response = supabase.rpc(
        "apply_match_log_patches_atomic",
        {
            "p_club_id": str(club_id),
            "p_patches": clean_patches,
            "p_actor_email": str(actor_email or ""),
            "p_actor_role": str(actor_role or ""),
            "p_source": str(source or "next_match_log"),
            "p_correction_note": str(correction_note or "") or None,
            "p_idempotency_key": clean_key,
            "p_replay_target": str(replay_target or FULL_RESET_LABEL),
        },
    ).execute()
    operation = _rpc_payload(response)
    operation_id = str(operation.get("operation_id") or "")
    updated_ids = [int(value) for value in operation.get("updated_ids") or []]
    result_base = {
        "updated_ids": updated_ids,
        "updated_count": int(operation.get("updated_count") or len(updated_ids)),
        "recompute_scope": dict(operation.get("recompute_scope") or scope),
    }
    if not bool((operation.get("recompute_scope") or scope).get("ratings")):
        return {
            "ok": True,
            "mode": "applied",
            "atomic": True,
            "operation_id": operation_id,
            "operation_status": "succeeded",
            "idempotent": bool(operation.get("idempotent")),
            **result_base,
            "warnings": [],
        }

    operation_status = str(operation.get("status") or "pending_replay")
    if bool(operation.get("idempotent")) and operation_status == "succeeded":
        stored_result = dict(operation.get("result_json") or {})
        replay = dict(stored_result.get("replay") or {})
        return {
            "ok": True,
            "mode": "applied_and_replayed",
            "atomic": True,
            "operation_id": operation_id,
            "operation_status": "succeeded",
            "idempotent": True,
            **result_base,
            "replay_job_id": replay.get("job_id") or operation.get("replay_job_id"),
            "replay_status": replay.get("job_status") or "succeeded",
            "replay_result": replay.get("result") or {},
            "warnings": list(replay.get("warnings") or []),
        }
    if operation_status == "recovery_required":
        raise MatchEditRecoveryRequired(
            operation_id,
            f"Match edits were already committed, but mandatory replay recovery is required for operation {operation_id}.",
        )

    try:
        replay = run_admin_replay_history(
            supabase,
            club_id=str(club_id),
            target_reset=str(operation.get("replay_target") or replay_target or FULL_RESET_LABEL),
            actor_email=str(actor_email or ""),
            actor_role=str(actor_role or ""),
            source=f"{source}:atomic_replay",
            confirmation_text="REPLAY",
            idempotency_key=_replay_key(clean_key),
        )
    except Exception as exc:
        _update_operation(
            supabase,
            operation_id=operation_id,
            status="recovery_required",
            replay_job_id=str(operation.get("replay_job_id") or "") or None,
            result_json=result_base,
            error_text=str(exc),
        )
        raise MatchEditRecoveryRequired(
            operation_id,
            f"Match edits were committed atomically, but mandatory replay failed. Recovery operation {operation_id} must be completed before further edits.",
        ) from exc

    replay_status = str(replay.get("job_status") or "")
    succeeded = replay_status == "succeeded"
    result_json = {**result_base, "replay": replay}
    _update_operation(
        supabase,
        operation_id=operation_id,
        status="succeeded" if succeeded else "pending_replay",
        replay_job_id=str(replay.get("job_id") or operation.get("replay_job_id") or "") or None,
        result_json=result_json,
    )
    return {
        "ok": succeeded,
        "mode": "applied_and_replayed" if succeeded else "replay_in_progress",
        "atomic": True,
        "operation_id": operation_id,
        "operation_status": "succeeded" if succeeded else "pending_replay",
        "idempotent": bool(operation.get("idempotent")),
        **result_base,
        "replay_job_id": replay.get("job_id"),
        "replay_status": replay_status,
        "replay_result": replay.get("result") or {},
        "warnings": list(replay.get("warnings") or []),
    }


def recover_atomic_match_edit(
    supabase: Any,
    *,
    club_id: str,
    operation_id: str,
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any]:
    response = (
        supabase.table("match_edit_operations")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("id", str(operation_id))
        .limit(1)
        .execute()
    )
    rows = response.data or []
    if not rows:
        raise ValueError("Match edit recovery operation was not found for this club.")
    operation = dict(rows[0])
    if str(operation.get("status")) == "succeeded":
        return {
            "ok": True,
            "mode": "already_recovered",
            "operation_id": str(operation_id),
            "operation_status": "succeeded",
            "idempotent": True,
            "result": dict(operation.get("result_json") or {}),
        }
    operation_status = str(operation.get("status") or "")
    if operation_status not in {"pending_replay", "recovery_required"}:
        raise ValueError("This operation is not ready for recovery.")

    replay = run_admin_replay_history(
        supabase,
        club_id=str(club_id),
        target_reset=str(operation.get("replay_target") or FULL_RESET_LABEL),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        source=str(source or "next_match_log_recovery"),
        confirmation_text="REPLAY",
        idempotency_key=_replay_key(str(operation.get("idempotency_key") or "")),
        retry_failed=operation_status == "recovery_required",
    )
    if str(replay.get("job_status")) != "succeeded":
        raise MatchEditRecoveryRequired(str(operation_id), "Replay recovery is still in progress.")
    result_json = {**dict(operation.get("result_json") or {}), "replay": replay}
    _update_operation(
        supabase,
        operation_id=str(operation_id),
        status="succeeded",
        replay_job_id=str(replay.get("job_id") or "") or None,
        result_json=result_json,
    )
    return {
        "ok": True,
        "mode": "recovered",
        "operation_id": str(operation_id),
        "operation_status": "succeeded",
        "replay_job_id": replay.get("job_id"),
        "replay_result": replay.get("result") or {},
        "warnings": list(replay.get("warnings") or []),
    }
