from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from jupr_app.domain.replay_history import FULL_RESET_LABEL
from jupr_app.services.admin_replay_service import (
    is_admin_replay_enabled,
    run_admin_replay_history,
)


MAX_EXCLUSION_TARGETS = 100
VALID_EXCLUSION_MODES = {"exclude", "duplicate_cleanup"}
VALID_OPERATION_STATUSES = {
    "pending_replay",
    "pending_badge_reconcile",
    "recovery_required",
    "succeeded",
}


class MatchExclusionStaleError(RuntimeError):
    def __init__(self, message: str, *, operation_id: str | None = None):
        self.operation_id = str(operation_id or "") or None
        super().__init__(message)


class MatchExclusionIdempotencyConflict(RuntimeError):
    def __init__(self, message: str, *, operation_id: str | None = None):
        self.operation_id = str(operation_id or "") or None
        super().__init__(message)


class MatchExclusionRecoveryRequired(RuntimeError):
    def __init__(
        self,
        operation_id: str,
        message: str,
        *,
        replay_job_id: str | None = None,
        recovery_stage: str | None = None,
    ):
        self.operation_id = str(operation_id or "")
        self.replay_job_id = str(replay_job_id or "") or None
        self.recovery_stage = str(recovery_stage or "") or None
        super().__init__(message)


class MatchExclusionWorkActive(RuntimeError):
    """A healthy lease already owns the operation's current work stage."""

    def __init__(
        self,
        operation_id: str,
        message: str,
        *,
        replay_job_id: str | None = None,
        recovery_stage: str | None = None,
        operation_status: str | None = None,
    ):
        self.operation_id = str(operation_id or "")
        self.replay_job_id = str(replay_job_id or "") or None
        self.recovery_stage = str(recovery_stage or "") or None
        self.operation_status = (
            _operation_status(operation_status) or None
        )
        super().__init__(message)


class MatchExclusionReplayActive(MatchExclusionWorkActive):
    """A healthy leased replay is already executing for this operation."""


def _rpc_payload(response: Any, *, label: str) -> dict[str, Any]:
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        return dict(data)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return dict(data[0])
    raise RuntimeError(
        f"{label} returned no durable result. Apply the match exclusion "
        "recovery migration before enabling destructive Match Log writes."
    )


def _exception_payload(exc: Exception) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    args = getattr(exc, "args", ())
    if args and isinstance(args[0], dict):
        payload.update(dict(args[0]))
    json_error = getattr(exc, "json", None)
    if callable(json_error):
        try:
            raw_error = json_error()
        except Exception:
            raw_error = None
        if isinstance(raw_error, dict):
            payload.update(dict(raw_error))
    for key in ("code", "message", "details", "hint"):
        value = getattr(exc, key, None)
        if value not in (None, ""):
            payload.setdefault(key, value)
    payload.setdefault("message", str(exc))
    return payload


def _operation_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    if status == "pending_badges":
        return "pending_badge_reconcile"
    return status


def _clean_uuid(value: Any, *, label: str) -> str:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{label} must be a valid UUID.") from exc


def _clean_targets(targets: list[dict[str, Any]]) -> list[dict[str, int]]:
    if not isinstance(targets, list):
        raise ValueError("Match exclusion targets must be a list.")
    if not targets:
        raise ValueError("Select at least one match to exclude.")
    if len(targets) > MAX_EXCLUSION_TARGETS:
        raise ValueError(
            f"No more than {MAX_EXCLUSION_TARGETS} matches can be excluded at once."
        )

    clean: list[dict[str, int]] = []
    seen_ids: set[int] = set()
    for target in targets:
        if not isinstance(target, dict):
            raise ValueError(
                "Each match exclusion target must include match_id and "
                "expected_row_version."
            )
        try:
            match_id = int(target.get("match_id"))
            row_version = int(target.get("expected_row_version"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Each match exclusion target must include integer match_id and "
                "expected_row_version values."
            ) from exc
        if match_id <= 0:
            raise ValueError("Match IDs must be positive integers.")
        if row_version < 1:
            raise ValueError("Expected match row versions must be at least 1.")
        if match_id in seen_ids:
            raise ValueError("Each match may appear only once in an exclusion request.")
        seen_ids.add(match_id)
        clean.append(
            {
                "match_id": match_id,
                "expected_row_version": row_version,
            }
        )
    return sorted(clean, key=lambda item: item["match_id"])


def _raise_operation_failure(
    payload: dict[str, Any],
    *,
    fallback_message: str,
) -> None:
    code = str(payload.get("code") or "").strip().upper()
    message = str(payload.get("message") or payload.get("error_text") or fallback_message)
    operation_id = str(payload.get("operation_id") or "") or None
    replay_job_id = str(payload.get("replay_job_id") or "") or None
    recovery_stage = str(payload.get("recovery_stage") or "") or None
    operation_status = _operation_status(
        payload.get("operation_status") or payload.get("status")
    )
    searchable = " ".join(
        str(payload.get(key) or "")
        for key in ("code", "message", "details", "hint", "error_text")
    ).upper()

    if "STALE" in searchable or "ROW_VERSION" in searchable:
        raise MatchExclusionStaleError(message, operation_id=operation_id)
    if "IDEMPOTENCY" in searchable or "BODY_CONFLICT" in searchable:
        raise MatchExclusionIdempotencyConflict(
            message,
            operation_id=operation_id,
        )
    active_stage: str | None = None
    if (
        "ACTIVE" in searchable
        or "IN_PROGRESS" in searchable
        or "CONTENDED" in searchable
    ):
        if recovery_stage in {"replay", "badge_reconcile"}:
            active_stage = recovery_stage
        elif "REPLAY" in searchable:
            active_stage = "replay"
        elif "BADGE" in searchable:
            active_stage = "badge_reconcile"
    if (
        code == "MATCH_EXCLUSION_BADGE_RECONCILE_INCOMPLETE"
        and int(payload.get("failed_count") or 0) == 0
        and (
            int(payload.get("pending_count") or 0) > 0
            or int(payload.get("running_count") or 0) > 0
        )
    ):
        active_stage = "badge_reconcile"
    if active_stage:
        active_type = (
            MatchExclusionReplayActive
            if active_stage == "replay"
            else MatchExclusionWorkActive
        )
        raise active_type(
            operation_id or "",
            message,
            replay_job_id=replay_job_id,
            recovery_stage=recovery_stage or active_stage,
            operation_status=operation_status,
        )
    if (
        code == "MATCH_EXCLUSION_RECOVERY_REQUIRED"
        or _operation_status(payload.get("operation_status") or payload.get("status"))
        == "recovery_required"
    ):
        raise MatchExclusionRecoveryRequired(
            operation_id or "",
            message,
            replay_job_id=replay_job_id,
            recovery_stage=recovery_stage,
        )
    raise RuntimeError(message)


def _call_rpc(
    supabase: Any,
    name: str,
    params: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    try:
        payload = _rpc_payload(
            supabase.rpc(name, params).execute(),
            label=label,
        )
    except (
        MatchExclusionStaleError,
        MatchExclusionIdempotencyConflict,
        MatchExclusionWorkActive,
        MatchExclusionRecoveryRequired,
    ):
        raise
    except Exception as exc:
        error_payload = _exception_payload(exc)
        try:
            _raise_operation_failure(
                error_payload,
                fallback_message=f"{label} failed.",
            )
        except RuntimeError as classified:
            raise classified from exc
        raise
    if payload.get("ok") is False:
        _raise_operation_failure(payload, fallback_message=f"{label} failed.")
    return payload


def _fetch_operation_row(
    supabase: Any,
    *,
    club_id: str,
    operation_id: str,
) -> dict[str, Any]:
    response = (
        supabase.table("match_exclusion_operations")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("id", str(operation_id))
        .limit(1)
        .execute()
    )
    rows = response.data or []
    if not rows:
        raise ValueError("Match exclusion operation was not found for this club.")
    return dict(rows[0])


def _fetch_operation_by_idempotency_key(
    supabase: Any,
    *,
    club_id: str,
    idempotency_key: str,
) -> dict[str, Any] | None:
    response = (
        supabase.table("match_exclusion_operations")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("idempotency_key", str(idempotency_key))
        .limit(1)
        .execute()
    )
    rows = response.data or []
    return dict(rows[0]) if rows else None


def find_match_exclusion_operation_by_idempotency_key(
    supabase: Any,
    *,
    club_id: str,
    idempotency_key: str,
) -> dict[str, Any] | None:
    """Return an existing same-club operation for an exact UUID key."""
    clean_idempotency_key = _clean_uuid(
        idempotency_key,
        label="Idempotency key",
    )
    return _fetch_operation_by_idempotency_key(
        supabase,
        club_id=str(club_id),
        idempotency_key=clean_idempotency_key,
    )


def _operation_result(
    operation: dict[str, Any],
    *,
    mode: str | None = None,
) -> dict[str, Any]:
    stored_result = dict(operation.get("result_json") or {})
    operation_id = str(
        operation.get("operation_id") or operation.get("id") or ""
    )
    operation_mode = str(mode or operation.get("mode") or "exclude")
    status = _operation_status(
        operation.get("operation_status") or operation.get("status")
    )
    excluded_ids = [
        int(value)
        for value in (
            operation.get("excluded_ids")
            or operation.get("excluded_match_ids")
            or stored_result.get("excluded_ids")
            or []
        )
    ]
    affected_player_ids = [
        int(value)
        for value in (
            operation.get("affected_player_ids")
            or stored_result.get("affected_player_ids")
            or []
        )
    ]
    replay = dict(stored_result.get("replay") or {})
    replay_result_json = dict(operation.get("replay_result_json") or {})
    durable_badges: dict[str, Any] = {}
    if any(
        key in stored_result
        for key in (
            "inserted_count",
            "updated_count",
            "revoked_count",
            "badge_results",
        )
    ):
        durable_badges = {
            "ok": status == "succeeded",
            "contract_version": str(
                operation.get("badge_contract_version")
                or stored_result.get("badge_contract_version")
                or ""
            )
            or None,
            "badge_ids": list(
                operation.get("badge_ids")
                or stored_result.get("badge_ids")
                or []
            ),
            "player_ids": list(affected_player_ids),
            "processed_player_ids": (
                list(affected_player_ids) if status == "succeeded" else []
            ),
            "inserted_count": int(stored_result.get("inserted_count") or 0),
            "awarded_count": int(stored_result.get("inserted_count") or 0),
            "updated_count": int(stored_result.get("updated_count") or 0),
            "revoked_count": int(stored_result.get("revoked_count") or 0),
            "results": list(stored_result.get("badge_results") or []),
        }
    badges = (
        durable_badges
        or dict(stored_result.get("badge_reconcile") or {})
        or dict(stored_result.get("badges") or {})
        or dict(operation.get("badge_reconcile") or {})
    )
    replay_job_id = str(
        operation.get("replay_job_id")
        or replay.get("job_id")
        or ""
    ) or None
    succeeded = status == "succeeded"
    result = {
        "ok": succeeded,
        "mode": (
            "duplicates_cleaned"
            if succeeded and operation_mode == "duplicate_cleanup"
            else "matches_excluded"
            if succeeded
            else "match_exclusion_recovery_required"
            if status == "recovery_required"
            else "match_exclusion_in_progress"
        ),
        "atomic": True,
        "operation_id": operation_id,
        "operation_status": status,
        "recovery_stage": str(operation.get("recovery_stage") or "") or None,
        "idempotent": bool(operation.get("idempotent")),
        "excluded_count": int(
            operation.get("excluded_count")
            or stored_result.get("excluded_count")
            or len(excluded_ids)
        ),
        "excluded_ids": excluded_ids,
        "affected_player_ids": affected_player_ids,
        "replay_job_id": replay_job_id,
        "replay_status": str(
            replay.get("job_status")
            or operation.get("replay_status")
            or stored_result.get("replay_status")
            or ""
        )
        or None,
        "replay_result": dict(
            replay.get("result")
            or replay_result_json
            or stored_result.get("replay_result")
            or {}
        ),
        "badge_reconcile": badges,
        "warnings": list(stored_result.get("warnings") or []),
    }
    if operation_mode == "duplicate_cleanup":
        result["deleted_count"] = result["excluded_count"]
        result["deleted_ids"] = list(excluded_ids)
    return result


def get_match_exclusion_operation(
    supabase: Any,
    *,
    club_id: str,
    operation_id: str,
) -> dict[str, Any]:
    clean_operation_id = _clean_uuid(operation_id, label="Operation ID")
    operation = _fetch_operation_row(
        supabase,
        club_id=str(club_id),
        operation_id=clean_operation_id,
    )
    public_result = _operation_result(operation)
    return {
        **public_result,
        "targets": list(operation.get("targets_json") or []),
        "badge_ids": list(
            operation.get("badge_ids")
            or operation.get("badge_allowlist")
            or []
        ),
        "badge_contract_version": str(
            operation.get("badge_contract_version") or ""
        )
        or None,
        "error_text": str(operation.get("error_text") or "")[:2000] or None,
        "created_at": operation.get("created_at"),
        "updated_at": operation.get("updated_at"),
        "finished_at": operation.get("finished_at"),
        "result_json": dict(operation.get("result_json") or {}),
    }


def _mark_recovery_required(
    supabase: Any,
    *,
    club_id: str,
    operation_id: str,
    recovery_stage: str,
    error_text: str,
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any]:
    return _call_rpc(
        supabase,
        "mark_match_exclusion_recovery_required",
        {
            "p_operation_id": str(operation_id),
            "p_club_id": str(club_id),
            "p_recovery_stage": str(recovery_stage),
            "p_error_text": str(error_text)[:4000],
            "p_actor_email": str(actor_email or ""),
            "p_actor_role": str(actor_role or ""),
            "p_source": str(source or "next_match_log_recovery")[:120],
        },
        label="Match exclusion recovery-state RPC",
    )


def _transition_after_replay(
    supabase: Any,
    *,
    club_id: str,
    operation_id: str,
    replay_job_id: str,
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any]:
    return _call_rpc(
        supabase,
        "transition_match_exclusion_after_replay",
        {
            "p_operation_id": str(operation_id),
            "p_club_id": str(club_id),
            "p_replay_job_id": str(replay_job_id),
            "p_actor_email": str(actor_email or ""),
            "p_actor_role": str(actor_role or ""),
            "p_source": str(source or "next_match_log")[:120],
        },
        label="Match exclusion replay transition RPC",
    )


def _finalize_operation(
    supabase: Any,
    *,
    club_id: str,
    operation_id: str,
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any]:
    return _call_rpc(
        supabase,
        "finalize_match_exclusion_operation",
        {
            "p_operation_id": str(operation_id),
            "p_club_id": str(club_id),
            "p_actor_email": str(actor_email or ""),
            "p_actor_role": str(actor_role or ""),
            "p_source": str(source or "next_match_log")[:120],
        },
        label="Match exclusion finalize RPC",
    )


def _run_replay_stage(
    supabase: Any,
    *,
    operation: dict[str, Any],
    club_id: str,
    operation_id: str,
    actor_email: str,
    actor_role: str,
    source: str,
    retry_failed: bool,
) -> dict[str, Any]:
    replay_job_id = str(operation.get("replay_job_id") or "").strip()
    if not replay_job_id:
        raise MatchExclusionRecoveryRequired(
            operation_id,
            "Match exclusion has no durable replay job identity.",
            recovery_stage="replay",
        )
    replay_target = str(
        operation.get("replay_target") or FULL_RESET_LABEL
    )
    try:
        replay = run_admin_replay_history(
            supabase,
            club_id=str(club_id),
            target_reset=replay_target,
            actor_email=str(actor_email or ""),
            actor_role=str(actor_role or ""),
            source=f"{source}:match_exclusion_replay"[:120],
            confirmation_text="REPLAY",
            idempotency_key=f"match-exclusion:{operation_id}",
            retry_failed=retry_failed,
            worker_id=f"match-exclusion-replay:{operation_id}:{uuid4()}",
            replay_job_id=replay_job_id,
        )
    except MatchExclusionWorkActive:
        raise
    except Exception as exc:
        try:
            _mark_recovery_required(
                supabase,
                club_id=str(club_id),
                operation_id=operation_id,
                recovery_stage="replay",
                error_text=str(exc),
                actor_email=actor_email,
                actor_role=actor_role,
                source=source,
            )
        except Exception:
            pass
        raise MatchExclusionRecoveryRequired(
            operation_id,
            "Matches were excluded atomically, but mandatory rating replay "
            "did not complete. Use the guarded recovery action.",
            replay_job_id=replay_job_id,
            recovery_stage="replay",
        ) from exc

    replay_status = str(replay.get("job_status") or "").strip().lower()
    if replay_status in {"pending", "running"}:
        raise MatchExclusionReplayActive(
            operation_id,
            "The mandatory rating replay is already running. Wait for its "
            "active lease to finish before retrying recovery.",
            replay_job_id=replay_job_id,
            recovery_stage="replay",
            operation_status=_operation_status(
                operation.get("operation_status") or operation.get("status")
            ),
        )
    if replay_status != "succeeded":
        failure = RuntimeError(
            "Mandatory rating replay did not succeed "
            f"(job status: {replay_status or 'unknown'})."
        )
        try:
            _mark_recovery_required(
                supabase,
                club_id=str(club_id),
                operation_id=operation_id,
                recovery_stage="replay",
                error_text=str(failure),
                actor_email=actor_email,
                actor_role=actor_role,
                source=source,
            )
        except Exception:
            pass
        raise MatchExclusionRecoveryRequired(
            operation_id,
            "Matches were excluded atomically, but mandatory rating replay "
            "did not complete. Use the guarded recovery action.",
            replay_job_id=replay_job_id,
            recovery_stage="replay",
        ) from failure
    try:
        return _transition_after_replay(
            supabase,
            club_id=str(club_id),
            operation_id=operation_id,
            replay_job_id=replay_job_id,
            actor_email=actor_email,
            actor_role=actor_role,
            source=source,
        )
    except MatchExclusionWorkActive:
        raise
    except Exception as exc:
        try:
            _mark_recovery_required(
                supabase,
                club_id=str(club_id),
                operation_id=operation_id,
                recovery_stage="replay",
                error_text=str(exc),
                actor_email=actor_email,
                actor_role=actor_role,
                source=source,
            )
        except Exception:
            pass
        raise MatchExclusionRecoveryRequired(
            operation_id,
            "Rating replay succeeded, but the durable exclusion operation "
            "could not advance to badge reconciliation.",
            replay_job_id=replay_job_id,
            recovery_stage="replay",
        ) from exc


def _run_badge_stage(
    supabase: Any,
    *,
    operation: dict[str, Any],
    club_id: str,
    operation_id: str,
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any]:
    from jupr_app.domain.gamification.match_exclusion_reconcile import (
        reconcile_match_exclusion_badges,
    )

    affected_player_ids = [
        int(value) for value in operation.get("affected_player_ids") or []
    ]
    try:
        result = reconcile_match_exclusion_badges(
            supabase,
            club_id=str(club_id),
            operation_id=str(operation_id),
            player_ids=affected_player_ids,
            actor_email=str(actor_email or ""),
        )
        if result.get("ok") is not True:
            raise RuntimeError(
                "Badge reconciliation did not attest complete success."
            )
        finalized = _finalize_operation(
            supabase,
            club_id=str(club_id),
            operation_id=operation_id,
            actor_email=actor_email,
            actor_role=actor_role,
            source=source,
        )
        persisted = _fetch_operation_row(
            supabase,
            club_id=str(club_id),
            operation_id=operation_id,
        )
        return {
            **persisted,
            "operation_status": finalized.get("operation_status")
            or persisted.get("status"),
            "idempotent": bool(finalized.get("idempotent")),
            "badge_reconcile": result,
        }
    except MatchExclusionWorkActive:
        raise
    except Exception as exc:
        searchable = str(exc).upper()
        if (
            "BADGE_RECONCILE_IN_PROGRESS" in searchable
            or "BADGE_RECONCILE_CLAIM_CONTENDED" in searchable
        ):
            raise MatchExclusionWorkActive(
                operation_id,
                "Narrow badge reconciliation is already running. Wait for "
                "its active lease to finish before retrying.",
                replay_job_id=str(operation.get("replay_job_id") or "")
                or None,
                recovery_stage="badge_reconcile",
                operation_status=_operation_status(
                    operation.get("operation_status")
                    or operation.get("status")
                ),
            ) from exc
        try:
            _mark_recovery_required(
                supabase,
                club_id=str(club_id),
                operation_id=operation_id,
                recovery_stage="badge_reconcile",
                error_text=str(exc),
                actor_email=actor_email,
                actor_role=actor_role,
                source=source,
            )
        except Exception:
            pass
        raise MatchExclusionRecoveryRequired(
            operation_id,
            "Rating replay completed, but narrow match-trigger badge "
            "reconciliation requires recovery.",
            replay_job_id=str(operation.get("replay_job_id") or "") or None,
            recovery_stage="badge_reconcile",
        ) from exc


def _complete_operation(
    supabase: Any,
    *,
    operation: dict[str, Any],
    club_id: str,
    operation_id: str,
    actor_email: str,
    actor_role: str,
    source: str,
    retry_failed: bool,
) -> dict[str, Any]:
    status = _operation_status(
        operation.get("operation_status") or operation.get("status")
    )
    recovery_stage = str(operation.get("recovery_stage") or "")

    if status == "succeeded":
        return operation
    if status == "recovery_required" and recovery_stage not in {
        "replay",
        "badge_reconcile",
    }:
        raise MatchExclusionRecoveryRequired(
            operation_id,
            "Match exclusion recovery stage is unavailable; inspect the "
            "durable operation before retrying.",
            replay_job_id=str(operation.get("replay_job_id") or "") or None,
            recovery_stage=recovery_stage or None,
        )
    if status == "pending_replay" or (
        status == "recovery_required" and recovery_stage == "replay"
    ):
        operation = _run_replay_stage(
            supabase,
            operation=operation,
            club_id=str(club_id),
            operation_id=operation_id,
            actor_email=actor_email,
            actor_role=actor_role,
            source=source,
            retry_failed=retry_failed,
        )
        status = _operation_status(
            operation.get("operation_status") or operation.get("status")
        )
    if status == "pending_badge_reconcile" or (
        status == "recovery_required"
        and str(operation.get("recovery_stage") or "") == "badge_reconcile"
    ):
        operation = _run_badge_stage(
            supabase,
            operation=operation,
            club_id=str(club_id),
            operation_id=operation_id,
            actor_email=actor_email,
            actor_role=actor_role,
            source=source,
        )
        status = _operation_status(
            operation.get("operation_status") or operation.get("status")
        )
    if status != "succeeded":
        raise MatchExclusionRecoveryRequired(
            operation_id,
            "Match exclusion did not reach a terminal succeeded state.",
            replay_job_id=str(operation.get("replay_job_id") or "") or None,
            recovery_stage=str(operation.get("recovery_stage") or "") or None,
        )
    return operation


def apply_atomic_match_exclusions(
    supabase: Any,
    *,
    club_id: str,
    targets: list[dict[str, Any]],
    actor_email: str,
    actor_role: str,
    source: str,
    note: str | None,
    idempotency_key: str,
    mode: str = "exclude",
) -> dict[str, Any]:
    if not is_admin_replay_enabled():
        raise PermissionError(
            "Rated match exclusion requires JUPR_ENABLE_NEXT_ADMIN_REPLAY=1 "
            "before any write is attempted."
        )
    clean_mode = str(mode or "").strip().lower()
    if clean_mode not in VALID_EXCLUSION_MODES:
        raise ValueError("Match exclusion mode must be exclude or duplicate_cleanup.")
    clean_targets = _clean_targets(targets)
    clean_idempotency_key = _clean_uuid(
        idempotency_key,
        label="Idempotency key",
    )
    stored_operation = find_match_exclusion_operation_by_idempotency_key(
        supabase,
        club_id=str(club_id),
        idempotency_key=clean_idempotency_key,
    )
    if stored_operation is not None:
        operation_id = _clean_uuid(
            stored_operation.get("id"),
            label="Stored operation ID",
        )
        badge_ids = [
            str(value)
            for value in (
                stored_operation.get("badge_ids")
                or stored_operation.get("badge_allowlist")
                or []
            )
        ]
        badge_contract_version = str(
            stored_operation.get("badge_contract_version") or ""
        )
        if not badge_ids or not badge_contract_version:
            raise MatchExclusionRecoveryRequired(
                operation_id,
                "Stored match exclusion badge contract is incomplete.",
                replay_job_id=str(
                    stored_operation.get("replay_job_id") or ""
                )
                or None,
                recovery_stage=str(
                    stored_operation.get("recovery_stage") or "badge_reconcile"
                ),
            )
    else:
        operation_id = str(uuid4())
        from jupr_app.domain.gamification.match_exclusion_reconcile import (
            MATCH_EXCLUSION_BADGE_CONTRACT_VERSION,
            resolve_match_exclusion_badge_ids,
        )

        badge_ids = resolve_match_exclusion_badge_ids(
            supabase,
            club_id=str(club_id),
        )
        badge_contract_version = MATCH_EXCLUSION_BADGE_CONTRACT_VERSION
    try:
        operation = _call_rpc(
            supabase,
            "apply_match_exclusions_atomic",
            {
                "p_operation_id": operation_id,
                "p_club_id": str(club_id),
                "p_mode": clean_mode,
                "p_targets": clean_targets,
                "p_badge_ids": list(badge_ids),
                "p_badge_contract_version": badge_contract_version,
                "p_actor_email": str(actor_email or ""),
                "p_actor_role": str(actor_role or ""),
                "p_source": str(source or "next_match_log")[:120],
                "p_delete_note": str(
                    note
                    or (
                        "Duplicate cleanup from Next Match Log"
                        if clean_mode == "duplicate_cleanup"
                        else "Rated match exclusion from Next Match Log"
                    )
                )[:2000],
                "p_idempotency_key": clean_idempotency_key,
                "p_replay_target": FULL_RESET_LABEL,
            },
            label="Atomic match exclusion RPC",
        )
    except Exception as exc:
        if isinstance(
            exc,
            (
                MatchExclusionStaleError,
                MatchExclusionIdempotencyConflict,
                MatchExclusionWorkActive,
                MatchExclusionRecoveryRequired,
            ),
        ):
            raise
        _raise_operation_failure(
            _exception_payload(exc),
            fallback_message="Atomic match exclusion failed before completion.",
        )
        raise

    operation_id = str(operation.get("operation_id") or operation_id)
    status = _operation_status(
        operation.get("operation_status") or operation.get("status")
    )
    if status == "recovery_required":
        _raise_operation_failure(
            operation,
            fallback_message="Match exclusion recovery is required.",
        )
    if status == "succeeded":
        persisted = _fetch_operation_row(
            supabase,
            club_id=str(club_id),
            operation_id=operation_id,
        )
        return _operation_result(
            {
                **persisted,
                **operation,
                "idempotent": bool(operation.get("idempotent")),
            },
            mode=clean_mode,
        )

    completed = _complete_operation(
        supabase,
        operation=operation,
        club_id=str(club_id),
        operation_id=operation_id,
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        source=str(source or "next_match_log"),
        retry_failed=False,
    )
    return _operation_result(
        {
            **completed,
            "idempotent": bool(operation.get("idempotent")),
        },
        mode=clean_mode,
    )


def recover_atomic_match_exclusion(
    supabase: Any,
    *,
    club_id: str,
    operation_id: str,
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any]:
    clean_operation_id = _clean_uuid(operation_id, label="Operation ID")
    operation = _fetch_operation_row(
        supabase,
        club_id=str(club_id),
        operation_id=clean_operation_id,
    )
    status = _operation_status(operation.get("status"))
    if status == "succeeded":
        return {
            **_operation_result(
                {**operation, "idempotent": True},
            ),
            "mode": "already_recovered",
            "idempotent": True,
        }
    if status not in {
        "pending_replay",
        "pending_badge_reconcile",
        "recovery_required",
    }:
        raise ValueError("This match exclusion operation is not recoverable.")

    completed = _complete_operation(
        supabase,
        operation=operation,
        club_id=str(club_id),
        operation_id=clean_operation_id,
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        source=str(source or "next_match_log_recovery"),
        retry_failed=True,
    )
    return {
        **_operation_result(completed),
        "mode": "match_exclusion_recovered",
    }
