from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any
from uuid import UUID

import pandas as pd
from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import (
    PERMISSION_DELETE_MATCHES,
    PERMISSION_ENTER_SCORES,
    PERMISSION_MANAGE_MATCHES,
    PERMISSION_RUN_REPLAY,
    has_permission,
    resolve_admin_role,
)
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.live_social import (
    SocialTablesNotInstalledError,
    delete_social_matches_with_snapshot,
    list_social_match_log_rows,
    restore_social_matches,
    update_social_match_row,
)
from jupr_app.services.admin_match_log_service import (
    apply_admin_match_log_duplicate_cleanup,
    apply_admin_match_log_edits,
    apply_admin_match_log_exclusions,
    build_admin_match_log,
    is_admin_match_log_apply_enabled,
    is_admin_match_log_destructive_enabled,
    is_admin_match_log_enabled,
    resolve_admin_match_log_duplicate_false_positive,
)
from jupr_app.services.match_exclusion_durability_service import (
    MatchExclusionIdempotencyConflict,
    MatchExclusionRecoveryRequired,
    MatchExclusionStaleError,
    MatchExclusionWorkActive,
    get_match_exclusion_operation,
    recover_atomic_match_exclusion,
)
from jupr_app.services.match_edit_durability_service import MatchEditRecoveryRequired, recover_atomic_match_edit
from jupr_app.services.match_log_recovery_lock_service import (
    MATCH_EDIT_KIND,
    MATCH_EXCLUSION_KIND,
    MatchLogRecoveryLocked,
    MatchLogRecoveryLockUnavailable,
    enforce_match_log_recovery_lock,
)
from jupr_app.services.admin_guarded_write_service import (
    GuardedWriteRecoveryRequired,
    begin_guarded_operation,
    get_guarded_operation,
    operation_result,
    update_guarded_operation,
)
from services.api.auth import authenticate_bearer, auth_header


class AdminMatchLogEditRequest(BaseModel):
    patches: list[dict[str, Any]] = Field(default_factory=list)
    confirmation_text: str = ""
    correction_note: str | None = None
    source: str = "next_match_log"
    idempotency_key: str = Field(default="", max_length=160)
    replay_target: str = Field(default="ALL (Full System Reset)", max_length=160)


class AdminMatchLogEditRecoveryRequest(BaseModel):
    confirmation_text: str = ""
    source: str = "next_match_log_recovery"


class AdminMatchLogExclusionTarget(BaseModel):
    match_id: int = Field(gt=0)
    expected_row_version: int = Field(ge=1)


class AdminMatchLogDuplicateCleanupRequest(BaseModel):
    targets: list[AdminMatchLogExclusionTarget] = Field(
        min_length=1,
        max_length=100,
    )
    confirmation_text: str = ""
    idempotency_key: UUID
    note: str | None = Field(default=None, max_length=2000)
    source: str = "next_match_log_duplicate_cleanup"


class AdminMatchLogDuplicateResolutionRequest(BaseModel):
    match_ids: list[int] = Field(default_factory=list)
    dup_key: str | None = None
    reason: str = ""
    confirmation_text: str = ""
    source: str = "next_match_log_duplicate_no_issue"


class AdminMatchLogExcludeRequest(BaseModel):
    targets: list[AdminMatchLogExclusionTarget] = Field(
        min_length=1,
        max_length=100,
    )
    confirmation_text: str = ""
    idempotency_key: UUID
    note: str | None = Field(default=None, max_length=2000)
    source: str = "next_match_log_bulk_exclude"


class AdminMatchLogExclusionRecoveryRequest(BaseModel):
    confirmation_text: str = ""
    source: str = "next_match_log_exclusion_recovery"


class AdminMatchLogSocialUpdateRequest(BaseModel):
    event_name: str | None = None
    played_on: str | None = None
    round_number: int | None = None
    court_number: int | None = None
    mini_round_number: int | None = None
    score_t1: int | None = Field(default=None, ge=0)
    score_t2: int | None = Field(default=None, ge=0)
    expected_current: dict[str, Any]
    idempotency_key: str = Field(
        min_length=8,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]+$",
    )
    confirmation_text: str = Field(default="", max_length=80)
    source: str = "next_match_log_social_editor"


class AdminMatchLogSocialDeleteRequest(BaseModel):
    social_match_ids: list[str] = Field(default_factory=list)
    confirmation_text: str = ""
    source: str = "next_match_log_social_editor"


class AdminMatchLogSocialReconcileRequest(BaseModel):
    confirmation_text: str = Field(default="", max_length=80)
    source: str = "next_match_log_social_reconcile"


def _dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def _is_api_audit_log_required() -> bool:
    return os.getenv("JUPR_REQUIRE_API_AUDIT_LOG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _safe_rows(response: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in (getattr(response, "data", None) or [])]


def _social_review_value(field: str, value: Any) -> Any:
    if field == "event_name":
        return " ".join(str(value or "").split())
    if field in {"score_t1", "score_t2", "round_number", "court_number", "mini_round_number"}:
        return None if value is None else int(value)
    return None if value is None else str(value).strip()


def _clean_text(value: Any, *, limit: int = 200) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _dataframe_rows(df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        rows.append({str(key): _json_safe(value) for key, value in row.items()})
    return rows


def _list_match_log_player_options(supabase: Any, *, club_id: str) -> dict[str, Any]:
    try:
        rows = (
            supabase.table("players")
            .select("id,name")
            .eq("club_id", str(club_id))
            .order("name", desc=False)
            .execute()
            .data
            or []
        )
    except Exception as exc:  # noqa: BLE001 - surface schema/configuration issues to the operator
        raise RuntimeError(f"Could not load Match Log player options: {exc.__class__.__name__}") from exc

    players: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for row in rows:
        player_id = _safe_int(dict(row).get("id") if isinstance(row, dict) else None)
        if player_id is None or int(player_id) in seen_ids:
            continue
        seen_ids.add(int(player_id))
        name = _clean_text(dict(row).get("name") if isinstance(row, dict) else None, limit=160) or f"Player {int(player_id)}"
        players.append({"id": int(player_id), "name": name, "label": f"{name} (#{int(player_id)})"})
    players = sorted(players, key=lambda player: (str(player.get("name") or "").lower(), int(player.get("id") or 0)))
    return {"ok": True, "mode": "match_log_player_options", "players": players, "count": len(players)}


def _resolve_role_or_403(
    *,
    supabase: Any,
    club_id: str,
    authorization: str | None,
    permission: str | tuple[str, ...],
    source: str,
    require_all: bool = False,
) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    try:
        role_resolution = resolve_admin_role(
            supabase=supabase,
            club_id=str(club_id),
            email=user.email,
            user_id=user.user_id,
            allowlist=set(),
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - expose pilot auth configuration errors without opaque 500s
        raise HTTPException(status_code=503, detail=f"Admin role lookup failed: {exc.__class__.__name__}") from exc
    required_permissions = (permission,) if isinstance(permission, str) else permission
    permission_checks = [
        has_permission(role_resolution.role, required_permission)
        for required_permission in required_permissions
    ]
    has_required_permissions = (
        all(permission_checks) if require_all else any(permission_checks)
    )
    if not role_resolution.assigned or not has_required_permissions:
        reason = "missing_club_assignment" if not role_resolution.assigned else "insufficient_permission"
        denied_payload = build_activity_payload(
            club_id=str(club_id),
            actor_email=user.email,
            actor_role=role_resolution.role,
            action_type="match_log_write_denied",
            entity_type="match",
            entity_id="bulk",
            after_json={
                "source_client": "fastapi/nextjs",
                "reason": reason,
                "required_permission": list(required_permissions) if len(required_permissions) > 1 else required_permissions[0],
            },
            source_page=source,
            flagged_for_review=True,
        )
        write_admin_activity_log(supabase, denied_payload)
        raise HTTPException(status_code=403, detail="insufficient permission")
    return user.email, role_resolution.role


def _require_service_role() -> None:
    if not os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail=(
                "Match Log exclusion/recovery requires "
                "SUPABASE_SERVICE_ROLE_KEY on FastAPI; browser and anonymous "
                "keys are not accepted."
            ),
        )


def _enforce_match_log_recovery_guard(
    supabase: Any,
    *,
    club_id: str,
    recovery_kind: str | None = None,
    recovery_operation_id: str | None = None,
) -> None:
    try:
        enforce_match_log_recovery_lock(
            supabase,
            club_id=str(club_id),
            recovery_kind=recovery_kind,
            recovery_operation_id=recovery_operation_id,
        )
    except MatchLogRecoveryLocked as exc:
        raise HTTPException(
            status_code=409,
            detail=exc.lock.as_detail(code=exc.code),
        ) from exc
    except MatchLogRecoveryLockUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _exclusion_target_payloads(
    targets: list[AdminMatchLogExclusionTarget],
) -> list[dict[str, int]]:
    return [
        {
            "match_id": int(target.match_id),
            "expected_row_version": int(target.expected_row_version),
        }
        for target in targets
    ]


def _is_atomic_match_log_recovery_conflict(exc: Exception) -> bool:
    text = str(exc).upper()
    return (
        "JUPR_MATCH_LOG_RECOVERY_LOCKED" in text
        or "JUPR_MATCH_LOG_RECOVERY_LOCK_AMBIGUOUS" in text
    )


def _raise_atomic_match_log_recovery_conflict(exc: Exception) -> None:
    raise HTTPException(
        status_code=409,
        detail={
            "code": "MATCH_LOG_RECOVERY_LOCKED",
            "message": (
                "Another Match Log recovery operation claimed this club "
                "before the write could start. Refresh Match Log and complete "
                "that exact recovery."
            ),
        },
    ) from exc


def _raise_match_exclusion_http(exc: Exception) -> None:
    if _is_atomic_match_log_recovery_conflict(exc):
        _raise_atomic_match_log_recovery_conflict(exc)
    if isinstance(exc, MatchExclusionStaleError):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "MATCH_EXCLUSION_STALE",
                "operation_id": exc.operation_id,
                "message": str(exc),
            },
        ) from exc
    if isinstance(exc, MatchExclusionIdempotencyConflict):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "MATCH_EXCLUSION_IDEMPOTENCY_CONFLICT",
                "operation_id": exc.operation_id,
                "message": str(exc),
            },
        ) from exc
    if isinstance(exc, MatchExclusionWorkActive):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "MATCH_EXCLUSION_ACTIVE",
                "operation_id": exc.operation_id,
                "replay_job_id": exc.replay_job_id,
                "recovery_stage": exc.recovery_stage,
                "operation_status": exc.operation_status,
                "message": str(exc),
            },
        ) from exc
    if isinstance(exc, MatchExclusionRecoveryRequired):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "MATCH_EXCLUSION_RECOVERY_REQUIRED",
                "operation_id": exc.operation_id,
                "replay_job_id": exc.replay_job_id,
                "recovery_stage": exc.recovery_stage,
                "message": str(exc),
            },
        ) from exc
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise exc


def install_admin_match_log_routes(app, *, get_supabase_client) -> None:
    """Register Match Log planning and guarded write routes for the Next admin pilot."""

    @app.get("/admin/clubs/{club_id}/match-log")
    def get_admin_match_log(
        club_id: str,
        filter_type: str = Query(default="All", alias="filter"),
        match_id: int | None = Query(default=None),
        match_ids: str | None = Query(default=None, max_length=4000),
        league: str | None = Query(default=None),
        week_tag: str | None = Query(default=None),
        context_type: str | None = Query(default=None, max_length=80),
        context_id: str | None = Query(default=None, max_length=200),
        context_ids: str | None = Query(default=None, max_length=8000),
        start_date: str | None = Query(default=None),
        end_date: str | None = Query(default=None),
        limit: int = Query(default=500, ge=1, le=1000),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_match_log_enabled() else None
        if supabase is not None:
            _resolve_role_or_403(
                supabase=supabase,
                club_id=str(club_id),
                authorization=authorization,
                permission=(PERMISSION_MANAGE_MATCHES, PERMISSION_ENTER_SCORES),
                source="next_match_log_read",
            )
        try:
            return build_admin_match_log(
                supabase,
                club_id=str(club_id),
                filter_type=filter_type,
                match_id=match_id,
                match_ids=match_ids,
                league=league,
                week_tag=week_tag,
                context_type=context_type,
                context_id=context_id,
                context_ids=context_ids,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/admin/clubs/{club_id}/match-log/player-options")
    def get_admin_match_log_player_options(
        club_id: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_log_enabled():
            raise HTTPException(status_code=403, detail="Next Match Log is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            permission=PERMISSION_MANAGE_MATCHES,
            source="next_match_log_player_options",
        )
        try:
            return _list_match_log_player_options(supabase, club_id=str(club_id))
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/admin/clubs/{club_id}/match-log/social")
    def get_admin_match_log_social_rows(
        club_id: str,
        limit: int = Query(default=500, ge=1, le=1000),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_log_enabled():
            raise HTTPException(status_code=403, detail="Next Match Log is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            permission=PERMISSION_MANAGE_MATCHES,
            source="next_match_log_social_list",
        )
        try:
            rows = _dataframe_rows(list_social_match_log_rows(supabase, club_id=str(club_id), limit=int(limit)))
            return {"ok": True, "mode": "social_match_log_rows", "rows": rows, "count": len(rows), "warnings": []}
        except SocialTablesNotInstalledError as exc:
            return {"ok": True, "mode": "social_match_log_unavailable", "rows": [], "count": 0, "warnings": [str(exc)]}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Unable to load Club Social Match Log rows: {exc.__class__.__name__}") from exc

    @app.patch("/admin/clubs/{club_id}/match-log/social/{social_match_id}")
    def patch_admin_match_log_social_row(
        club_id: str,
        social_match_id: str,
        payload: AdminMatchLogSocialUpdateRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_log_apply_enabled():
            raise HTTPException(status_code=403, detail="Next Match Log apply is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            permission=PERMISSION_MANAGE_MATCHES,
            source=payload.source,
        )
        _enforce_match_log_recovery_guard(
            supabase,
            club_id=str(club_id),
        )
        patch = _dump_model(payload)
        source = str(patch.pop("source", payload.source))
        expected_current = dict(patch.pop("expected_current", payload.expected_current) or {})
        idempotency_key = str(patch.pop("idempotency_key", payload.idempotency_key))
        confirmation_text = str(patch.pop("confirmation_text", payload.confirmation_text))
        if not patch:
            raise HTTPException(status_code=400, detail="No Club Social changes provided.")
        if confirmation_text.strip().upper() != "SAVE SOCIAL MATCH":
            raise HTTPException(status_code=400, detail="Type SAVE SOCIAL MATCH to save this Club Social edit.")
        missing_expected_fields = sorted(set(patch) - set(expected_current))
        if missing_expected_fields:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Reload this Club Social row before saving. Expected values are missing for: "
                    + ", ".join(missing_expected_fields)
                ),
            )
        if "event_name" in patch and not " ".join(str(patch.get("event_name") or "").split()):
            raise HTTPException(status_code=400, detail="Club Social event name cannot be blank.")
        if all(_social_review_value(field, value) == _social_review_value(field, expected_current.get(field)) for field, value in patch.items()):
            raise HTTPException(status_code=400, detail="No Club Social changes detected.")
        if "event_name" in patch and len(patch) > 1:
            raise HTTPException(status_code=400, detail="Update the Club Social event name separately from match fields.")
        operation: dict[str, Any] | None = None
        idempotent = False
        try:
            operation, idempotent = begin_guarded_operation(
                supabase,
                club_id=str(club_id),
                workflow="social_match_edit",
                action="social_match_log_update",
                operation_key=idempotency_key,
                request_payload={
                    "social_match_id": str(social_match_id),
                    "patch": patch,
                    "expected_current": expected_current,
                    "confirmation_text": "SAVE SOCIAL MATCH",
                },
                actor_email=actor_email,
                actor_role=actor_role,
                source=source,
                before_json={"expected_current": expected_current},
            )
            if idempotent:
                return operation_result(operation)
            planned_evidence = {
                "social_match_id": str(social_match_id),
                "patch": patch,
                "expected_current": expected_current,
            }
            update_guarded_operation(
                supabase,
                operation_id=operation.get("id"),
                operation_key=idempotency_key,
                status="intent_recorded",
                result_json={"phase": "preflight", "planned": planned_evidence},
            )
            result = update_social_match_row(
                supabase,
                club_id=str(club_id),
                social_match_id=str(social_match_id),
                patch=patch,
                expected_current=expected_current,
            )
        except SocialTablesNotInstalledError as exc:
            if operation is not None and not idempotent:
                try:
                    update_guarded_operation(
                        supabase,
                        operation_id=operation.get("id"),
                        operation_key=idempotency_key,
                        status="failed",
                        error_text=str(exc),
                    )
                except GuardedWriteRecoveryRequired as ledger_exc:
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "code": "RECOVERY_REQUIRED",
                            "kind": "uncertain",
                            "message": str(ledger_exc),
                            "operation_key": ledger_exc.operation_key,
                            "recovery_required": True,
                        },
                    ) from ledger_exc
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except GuardedWriteRecoveryRequired as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "RECOVERY_REQUIRED",
                    "kind": "uncertain",
                    "message": str(exc),
                    "operation_key": exc.operation_key,
                    "recovery_required": True,
                },
            ) from exc
        except RuntimeError as exc:
            if operation is None:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "code": "DURABLE_INTENT_UNAVAILABLE",
                        "kind": "failed",
                        "message": "The Club Social edit was not started because its durable intent could not be recorded.",
                        "operation_key": idempotency_key,
                        "recovery_required": False,
                    },
                ) from exc
            if not idempotent:
                try:
                    update_guarded_operation(
                        supabase,
                        operation_id=operation.get("id"),
                        operation_key=idempotency_key,
                        status="failed",
                        error_text=str(exc),
                    )
                except GuardedWriteRecoveryRequired as ledger_exc:
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "code": "RECOVERY_REQUIRED",
                            "kind": "uncertain",
                            "message": str(ledger_exc),
                            "operation_key": ledger_exc.operation_key,
                            "recovery_required": True,
                        },
                    ) from ledger_exc
            error_text = str(exc).lower()
            if "not found" in error_text or "does not belong" in error_text:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if "missing event linkage" in error_text:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "INVALID_SOCIAL_MATCH",
                        "kind": "conflict",
                        "message": str(exc),
                        "operation_key": idempotency_key,
                        "recovery_required": False,
                    },
                ) from exc
            if not any(
                marker in error_text
                for marker in ("changed before", "newer data", "did not persist")
            ):
                raise HTTPException(
                    status_code=500,
                    detail={
                        "code": "SOCIAL_MATCH_EDIT_FAILED",
                        "kind": "failed",
                        "message": "The Club Social edit failed before a verified completion.",
                        "operation_key": idempotency_key,
                        "recovery_required": False,
                    },
                ) from exc
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "STALE_VERSION",
                    "kind": "conflict",
                    "message": "The Club Social row changed after it was loaded. Reload it and review your edit.",
                    "operation_key": idempotency_key,
                    "recovery_required": False,
                },
            ) from exc
        except ValueError as exc:
            if operation is not None and not idempotent:
                try:
                    update_guarded_operation(
                        supabase,
                        operation_id=operation.get("id"),
                        operation_key=idempotency_key,
                        status="failed",
                        error_text=str(exc),
                    )
                except GuardedWriteRecoveryRequired as ledger_exc:
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "code": "RECOVERY_REQUIRED",
                            "kind": "uncertain",
                            "message": str(ledger_exc),
                            "operation_key": ledger_exc.operation_key,
                            "recovery_required": True,
                        },
                    ) from ledger_exc
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - an interrupted write is intentionally uncertain
            if operation is None:
                raise HTTPException(status_code=500, detail="Unable to start the Club Social edit.") from exc
            try:
                update_guarded_operation(
                    supabase,
                    operation_id=operation.get("id"),
                    operation_key=idempotency_key,
                    status="recovery_required",
                    result_json={"planned": planned_evidence},
                    error_text=f"Club Social mutation outcome could not be verified: {exc}",
                )
            except Exception:
                pass
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "RECOVERY_REQUIRED",
                    "kind": "uncertain",
                    "message": "The Club Social edit outcome could not be verified. Inspect the operation before retrying.",
                    "operation_key": idempotency_key,
                    "recovery_required": True,
                },
            ) from exc
        audit_result = write_admin_activity_log(
            supabase,
            build_activity_payload(
                club_id=str(club_id),
                actor_email=actor_email,
                actor_role=actor_role,
                action_type="social_match_log_update",
                entity_type="live_event_match",
                entity_id=str(social_match_id),
                before_json=result["before"],
                after_json={
                    "source_client": "fastapi/nextjs",
                    "source_page": source,
                    "patch": result["patch"],
                    "result": result,
                },
                source_page=source,
                flagged_for_review=True,
            ),
        )
        if not audit_result.ok and _is_api_audit_log_required():
            try:
                update_social_match_row(
                    supabase,
                    club_id=str(club_id),
                    social_match_id=str(social_match_id),
                    patch=dict(result["before"]),
                    expected_current=dict(result["after"]),
                )
            except Exception as rollback_exc:  # noqa: BLE001 - surface a critical partial-write state
                try:
                    update_guarded_operation(
                        supabase,
                        operation_id=operation.get("id"),
                        operation_key=idempotency_key,
                        status="recovery_required",
                        result_json={"planned": planned_evidence, "social_match_id": str(social_match_id), "result": result},
                        error_text="Required audit failed and CAS rollback could not be verified.",
                    )
                except Exception:
                    # Intent is already durable. Even if the recovery marker
                    # cannot be updated, the client must receive an uncertain
                    # envelope and must not blind-retry with a new operation.
                    pass
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "RECOVERY_REQUIRED",
                        "kind": "uncertain",
                        "message": "The Club Social edit may be committed, but its audit and rollback could not be verified.",
                        "operation_key": idempotency_key,
                        "recovery_required": True,
                    },
                ) from rollback_exc
            try:
                update_guarded_operation(
                    supabase,
                    operation_id=operation.get("id"),
                    operation_key=idempotency_key,
                    status="failed",
                    result_json={"rolled_back": True},
                    error_text="Required completion audit failed; the Club Social edit was rolled back.",
                )
            except GuardedWriteRecoveryRequired as ledger_exc:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "RECOVERY_REQUIRED",
                        "kind": "uncertain",
                        "message": (
                            "The Club Social edit was rolled back, but its operation ledger could not be finalized. "
                            "Inspect the operation before retrying."
                        ),
                        "operation_key": ledger_exc.operation_key,
                        "recovery_required": True,
                    },
                ) from ledger_exc
            raise HTTPException(
                status_code=500,
                detail="Audit log write required but unavailable; the Club Social update was rolled back.",
            )
        warnings = [audit_result.warning] if audit_result.warning else []
        response = {
            "ok": True,
            "mode": "social_match_updated",
            "social_match_id": str(social_match_id),
            "operation_key": idempotency_key,
            "idempotent_replay": False,
            "result": result,
            "recovery": {
                "operation_status": f"/admin/clubs/{{club_id}}/match-log/social/operations/{idempotency_key}",
                "operator_rule": "Retry the exact unchanged request with the same idempotency key after an interrupted response.",
            },
            "warnings": warnings,
        }
        try:
            update_guarded_operation(
                supabase,
                operation_id=operation.get("id"),
                operation_key=idempotency_key,
                status="completed",
                after_json=result.get("after"),
                result_json=response,
            )
        except GuardedWriteRecoveryRequired as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "RECOVERY_REQUIRED",
                    "kind": "uncertain",
                    "message": str(exc),
                    "operation_key": exc.operation_key,
                    "recovery_required": True,
                },
            ) from exc
        return response

    @app.get("/admin/clubs/{club_id}/match-log/social/operations/{operation_key}")
    def get_admin_match_log_social_operation(
        club_id: str,
        operation_key: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_log_enabled():
            raise HTTPException(status_code=403, detail="Next Match Log is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            permission=PERMISSION_MANAGE_MATCHES,
            source="next_match_log_social_operation",
        )
        operation = get_guarded_operation(
            supabase,
            club_id=str(club_id),
            workflow="social_match_edit",
            operation_key=str(operation_key),
        )
        if operation is None:
            raise HTTPException(status_code=404, detail="Club Social edit operation was not found.")
        result_json = operation.get("result_json") or {}; error_text = operation.get("error_text")
        return {
            "ok": True,
            "operation_key": operation.get("operation_key"),
            "status": operation.get("status"),
            "result_json": result_json,
            "error_text": error_text,
            "result": result_json,
            "error": error_text,
            "recovery_required": operation.get("status") in {"intent_recorded", "recovery_required"},
            "updated_at": operation.get("updated_at"),
        }

    @app.post("/admin/clubs/{club_id}/match-log/social/operations/{operation_key}/reconcile")
    def post_admin_match_log_social_operation_reconcile(
        club_id: str,
        operation_key: str,
        payload: AdminMatchLogSocialReconcileRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_log_apply_enabled():
            raise HTTPException(status_code=403, detail="Next Match Log apply is disabled.")
        if str(payload.confirmation_text or "").strip().upper() != "RECONCILE SOCIAL MATCH":
            raise HTTPException(status_code=400, detail="Type RECONCILE SOCIAL MATCH to reconcile this exact edit.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            permission=PERMISSION_MANAGE_MATCHES,
            source=payload.source,
        )
        operation = get_guarded_operation(
            supabase,
            club_id=str(club_id),
            workflow="social_match_edit",
            operation_key=str(operation_key),
        )
        if operation is None:
            raise HTTPException(status_code=404, detail="Club Social edit operation was not found.")
        if str(operation.get("status") or "") == "completed":
            return operation_result(operation)
        if str(operation.get("status") or "") not in {"intent_recorded", "recovery_required"}:
            raise HTTPException(status_code=400, detail="This Club Social operation is not reconcilable.")
        evidence = operation.get("result_json") or {}
        planned = evidence.get("planned") if isinstance(evidence, dict) else None
        if not isinstance(planned, dict):
            raise HTTPException(
                status_code=409,
                detail={"code": "RECOVERY_REQUIRED", "kind": "uncertain", "message": "This operation lacks exact preflight evidence; inspect the row manually.", "operation_key": str(operation_key), "recovery_required": True},
            )
        social_match_id = str(planned.get("social_match_id") or "")
        patch = dict(planned.get("patch") or {})
        expected_current = dict(planned.get("expected_current") or {})
        try:
            match_rows = _safe_rows(supabase.table("live_event_matches").select("*").eq("id", social_match_id).limit(1).execute())
        except Exception as exc:
            raise HTTPException(status_code=409, detail={"code": "RECOVERY_REQUIRED", "kind": "uncertain", "message": "The Club Social row could not be read authoritatively.", "operation_key": str(operation_key), "recovery_required": True}) from exc
        if len(match_rows) != 1:
            raise HTTPException(status_code=409, detail={"code": "RECOVERY_REQUIRED", "kind": "uncertain", "message": "The Club Social row cannot be uniquely read.", "operation_key": str(operation_key), "recovery_required": True})
        match_row = match_rows[0]
        try:
            event_rows = _safe_rows(supabase.table("live_events").select("id,name,club_id,result_mode").eq("id", str(match_row.get("event_id") or "")).eq("club_id", str(club_id)).eq("result_mode", "social_unrated").limit(1).execute())
        except Exception as exc:
            raise HTTPException(status_code=409, detail={"code": "RECOVERY_REQUIRED", "kind": "uncertain", "message": "The Club Social event could not be read authoritatively.", "operation_key": str(operation_key), "recovery_required": True}) from exc
        if len(event_rows) != 1:
            raise HTTPException(status_code=409, detail={"code": "RECOVERY_REQUIRED", "kind": "uncertain", "message": "The Club Social event cannot be uniquely read.", "operation_key": str(operation_key), "recovery_required": True})

        def current_value(field: str) -> Any:
            value = event_rows[0].get("name") if field == "event_name" else match_row.get(field)
            return _social_review_value(field, value)

        proves_after = all(current_value(field) == _social_review_value(field, value) for field, value in patch.items())
        proves_before = all(current_value(field) == _social_review_value(field, value) for field, value in expected_current.items() if field in patch)
        if not proves_after:
            if proves_before:
                try:
                    update_guarded_operation(supabase, operation_id=operation.get("id"), operation_key=str(operation_key), status="failed", result_json={"reconciled": True, "proof": "authoritative_original_state"}, error_text="Authoritative readback proves the social edit did not commit.")
                except Exception as exc:
                    raise HTTPException(status_code=409, detail={"code": "RECOVERY_REQUIRED", "kind": "uncertain", "message": "The original Club Social state is proven, but the operation ledger could not be finalized.", "operation_key": str(operation_key), "recovery_required": True}) from exc
                return {"ok": False, "mode": "social_match_edit_reconciled_failed", "operation_key": str(operation_key), "status": "failed", "recovery_required": False}
            raise HTTPException(status_code=409, detail={"code": "RECOVERY_REQUIRED", "kind": "uncertain", "message": "Current Club Social state proves neither the reviewed before nor intended after values.", "operation_key": str(operation_key), "recovery_required": True})
        after = {field: current_value(field) for field in patch}
        audit = write_admin_activity_log(supabase, build_activity_payload(club_id=str(club_id), actor_email=actor_email, actor_role=actor_role, action_type="reconcile_social_match_log_update", entity_type="live_event_match", entity_id=social_match_id, before_json=expected_current, after_json={"source_client": "fastapi/nextjs", "source_page": payload.source, "operation_key": str(operation_key), "patch": patch, "proof": "authoritative_social_readback"}, source_page=payload.source, flagged_for_review=True))
        if not audit.ok:
            try:
                update_guarded_operation(supabase, operation_id=operation.get("id"), operation_key=str(operation_key), status="recovery_required", result_json={**evidence, "reconcile_audit_failed": True}, error_text="Authoritative social proof succeeded, but reconciliation audit did not persist.")
            except Exception:
                pass
            raise HTTPException(status_code=409, detail={"code": "RECOVERY_REQUIRED", "kind": "uncertain", "message": "Social edit proof succeeded, but reconciliation audit is unavailable.", "operation_key": str(operation_key), "recovery_required": True})
        result = {"ok": True, "mode": "social_match_updated", "social_match_id": social_match_id, "operation_key": str(operation_key), "idempotent_replay": True, "reconciled": True, "result": {"social_match_id": social_match_id, "patch": patch, "before": expected_current, "after": after}, "warnings": []}
        try:
            update_guarded_operation(supabase, operation_id=operation.get("id"), operation_key=str(operation_key), status="completed", after_json=after, result_json=result)
        except Exception as exc:
            raise HTTPException(status_code=409, detail={"code": "RECOVERY_REQUIRED", "kind": "uncertain", "message": "The reconciled Club Social edit is proven and audited, but its operation ledger could not be finalized.", "operation_key": str(operation_key), "recovery_required": True}) from exc
        return result

    @app.post("/admin/clubs/{club_id}/match-log/social/delete")
    def post_admin_match_log_social_delete(
        club_id: str,
        payload: AdminMatchLogSocialDeleteRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_log_apply_enabled():
            raise HTTPException(status_code=403, detail="Next Match Log apply is disabled.")
        if str(payload.confirmation_text or "").strip().upper() != "DELETE":
            raise HTTPException(status_code=400, detail="Type DELETE to confirm Club Social row deletion.")
        social_ids = [str(value).strip() for value in (payload.social_match_ids or []) if str(value).strip()]
        if not social_ids:
            raise HTTPException(status_code=400, detail="Select at least one Club Social row to delete.")
        if len(social_ids) > 100:
            raise HTTPException(status_code=400, detail="No more than 100 Club Social rows can be deleted at once.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            permission=PERMISSION_DELETE_MATCHES,
            source=payload.source,
        )
        _enforce_match_log_recovery_guard(
            supabase,
            club_id=str(club_id),
        )
        try:
            deleted, deleted_snapshots = delete_social_matches_with_snapshot(
                supabase,
                club_id=str(club_id),
                social_match_ids=social_ids,
            )
        except SocialTablesNotInstalledError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        audit_result = write_admin_activity_log(
            supabase,
            build_activity_payload(
                club_id=str(club_id),
                actor_email=actor_email,
                actor_role=actor_role,
                action_type="social_match_log_delete",
                entity_type="live_event_match",
                entity_id="bulk",
                before_json=deleted_snapshots,
                after_json={"source_client": "fastapi/nextjs", "source_page": payload.source, "requested_ids": social_ids, "deleted_count": deleted},
                source_page=payload.source,
                flagged_for_review=True,
            ),
        )
        if not audit_result.ok and _is_api_audit_log_required():
            try:
                restored = restore_social_matches(
                    supabase,
                    club_id=str(club_id),
                    snapshots=deleted_snapshots,
                )
                if restored != deleted:
                    raise RuntimeError("Club Social deleted rows were not fully restored.")
            except Exception as rollback_exc:  # noqa: BLE001 - surface a critical destructive state
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Critical: audit log write failed and the Club Social delete could not be rolled back. "
                        "Manual review is required."
                    ),
                ) from rollback_exc
            raise HTTPException(
                status_code=500,
                detail="Audit log write required but unavailable; the Club Social delete was rolled back.",
            )
        warnings = [audit_result.warning] if audit_result.warning else []
        return {"ok": True, "mode": "social_matches_deleted", "deleted_count": deleted, "requested_ids": social_ids, "warnings": warnings}

    @app.patch("/admin/clubs/{club_id}/match-log/edits")
    def patch_admin_match_log_edits(
        club_id: str,
        payload: AdminMatchLogEditRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_log_apply_enabled():
            raise HTTPException(status_code=403, detail="Next Match Log apply is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            permission=PERMISSION_MANAGE_MATCHES,
            source=payload.source,
        )
        _enforce_match_log_recovery_guard(
            supabase,
            club_id=str(club_id),
        )
        try:
            return apply_admin_match_log_edits(
                supabase,
                club_id=str(club_id),
                patches=payload.patches,
                actor_email=actor_email,
                actor_role=actor_role,
                correction_note=payload.correction_note,
                source=payload.source,
                confirmation_text=payload.confirmation_text,
                idempotency_key=payload.idempotency_key,
                replay_target=payload.replay_target,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except MatchEditRecoveryRequired as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "MATCH_EDIT_RECOVERY_REQUIRED",
                    "operation_id": exc.operation_id,
                    "message": str(exc),
                },
            ) from exc
        except Exception as exc:
            if _is_atomic_match_log_recovery_conflict(exc):
                _raise_atomic_match_log_recovery_conflict(exc)
            raise

    @app.post("/admin/clubs/{club_id}/match-log/edits/{operation_id}/recover")
    def post_admin_match_log_edit_recovery(
        club_id: str,
        operation_id: str,
        payload: AdminMatchLogEditRecoveryRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_log_apply_enabled():
            raise HTTPException(status_code=403, detail="Next Match Log apply is disabled.")
        if str(payload.confirmation_text or "").strip().upper() != "RECOVER":
            raise HTTPException(status_code=400, detail="Type RECOVER to confirm mandatory replay recovery.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            permission=PERMISSION_MANAGE_MATCHES,
            source=payload.source,
        )
        if not has_permission(actor_role, PERMISSION_RUN_REPLAY):
            raise HTTPException(status_code=403, detail="Replay recovery permission is required.")
        _enforce_match_log_recovery_guard(
            supabase,
            club_id=str(club_id),
            recovery_kind=MATCH_EDIT_KIND,
            recovery_operation_id=str(operation_id),
        )
        try:
            return recover_atomic_match_edit(
                supabase,
                club_id=str(club_id),
                operation_id=str(operation_id),
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except MatchEditRecoveryRequired as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "MATCH_EDIT_RECOVERY_REQUIRED",
                    "operation_id": exc.operation_id,
                    "message": str(exc),
                },
            ) from exc

    @app.post("/admin/clubs/{club_id}/match-log/duplicates/cleanup")
    def post_admin_match_log_duplicate_cleanup(
        club_id: str,
        payload: AdminMatchLogDuplicateCleanupRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_log_apply_enabled():
            raise HTTPException(status_code=403, detail="Next Match Log apply is disabled.")
        if not is_admin_match_log_destructive_enabled():
            raise HTTPException(
                status_code=403,
                detail="Next Match Log destructive actions are disabled.",
            )
        _require_service_role()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            permission=(PERMISSION_DELETE_MATCHES, PERMISSION_RUN_REPLAY),
            source=payload.source,
            require_all=True,
        )
        _enforce_match_log_recovery_guard(
            supabase,
            club_id=str(club_id),
        )
        try:
            return apply_admin_match_log_duplicate_cleanup(
                supabase,
                club_id=str(club_id),
                targets=_exclusion_target_payloads(payload.targets),
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                confirmation_text=payload.confirmation_text,
                idempotency_key=str(payload.idempotency_key),
                note=payload.note,
            )
        except Exception as exc:
            _raise_match_exclusion_http(exc)

    @app.post("/admin/clubs/{club_id}/match-log/exclude")
    def post_admin_match_log_exclude_matches(
        club_id: str,
        payload: AdminMatchLogExcludeRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_log_apply_enabled():
            raise HTTPException(status_code=403, detail="Next Match Log apply is disabled.")
        if not is_admin_match_log_destructive_enabled():
            raise HTTPException(
                status_code=403,
                detail="Next Match Log destructive actions are disabled.",
            )
        _require_service_role()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            permission=(PERMISSION_DELETE_MATCHES, PERMISSION_RUN_REPLAY),
            source=payload.source,
            require_all=True,
        )
        _enforce_match_log_recovery_guard(
            supabase,
            club_id=str(club_id),
        )
        try:
            return apply_admin_match_log_exclusions(
                supabase,
                club_id=str(club_id),
                targets=_exclusion_target_payloads(payload.targets),
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                confirmation_text=payload.confirmation_text,
                idempotency_key=str(payload.idempotency_key),
                note=payload.note,
            )
        except Exception as exc:
            _raise_match_exclusion_http(exc)

    @app.get(
        "/admin/clubs/{club_id}/match-log/exclusions/{operation_id}"
    )
    def get_admin_match_log_exclusion_operation(
        club_id: str,
        operation_id: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_log_enabled():
            raise HTTPException(status_code=403, detail="Next Match Log is disabled.")
        _require_service_role()
        supabase = get_supabase_client()
        _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            permission=(PERMISSION_DELETE_MATCHES, PERMISSION_RUN_REPLAY),
            source="next_match_log_exclusion_status",
            require_all=True,
        )
        try:
            return get_match_exclusion_operation(
                supabase,
                club_id=str(club_id),
                operation_id=str(operation_id),
            )
        except Exception as exc:
            _raise_match_exclusion_http(exc)

    @app.post(
        "/admin/clubs/{club_id}/match-log/exclusions/{operation_id}/recover"
    )
    def post_admin_match_log_exclusion_recovery(
        club_id: str,
        operation_id: str,
        payload: AdminMatchLogExclusionRecoveryRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_log_apply_enabled():
            raise HTTPException(status_code=403, detail="Next Match Log apply is disabled.")
        if not is_admin_match_log_destructive_enabled():
            raise HTTPException(
                status_code=403,
                detail="Next Match Log destructive actions are disabled.",
            )
        if str(payload.confirmation_text or "").strip().upper() != "RECOVER":
            raise HTTPException(
                status_code=400,
                detail="Type RECOVER to confirm mandatory exclusion recovery.",
            )
        _require_service_role()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            permission=(PERMISSION_DELETE_MATCHES, PERMISSION_RUN_REPLAY),
            source=payload.source,
            require_all=True,
        )
        _enforce_match_log_recovery_guard(
            supabase,
            club_id=str(club_id),
            recovery_kind=MATCH_EXCLUSION_KIND,
            recovery_operation_id=str(operation_id),
        )
        try:
            return recover_atomic_match_exclusion(
                supabase,
                club_id=str(club_id),
                operation_id=str(operation_id),
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
            )
        except Exception as exc:
            _raise_match_exclusion_http(exc)

    @app.post("/admin/clubs/{club_id}/match-log/duplicates/resolve")
    def post_admin_match_log_duplicate_resolution(
        club_id: str,
        payload: AdminMatchLogDuplicateResolutionRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_log_apply_enabled():
            raise HTTPException(status_code=403, detail="Next Match Log apply is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            permission=PERMISSION_MANAGE_MATCHES,
            source=payload.source,
        )
        _enforce_match_log_recovery_guard(
            supabase,
            club_id=str(club_id),
        )
        try:
            return resolve_admin_match_log_duplicate_false_positive(
                supabase,
                club_id=str(club_id),
                match_ids=payload.match_ids,
                dup_key=payload.dup_key,
                actor_email=actor_email,
                actor_role=actor_role,
                reason=payload.reason,
                source=payload.source,
                confirmation_text=payload.confirmation_text,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            if _is_atomic_match_log_recovery_conflict(exc):
                _raise_atomic_match_log_recovery_conflict(exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
