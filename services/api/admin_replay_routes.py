from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel

from jupr_app.domain.admin.roles import PERMISSION_RUN_REPLAY, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_replay_service import build_admin_replay_status, is_admin_replay_enabled, run_admin_replay_history
from jupr_app.services.match_log_recovery_lock_service import (
    MatchLogRecoveryLocked,
    MatchLogRecoveryLockUnavailable,
    enforce_match_log_recovery_lock,
)
from services.api.auth import authenticate_bearer, auth_header


class AdminReplayRequest(BaseModel):
    target_reset: str = "ALL (Full System Reset)"
    confirmation_text: str = ""
    source: str = "next_replay_history"
    idempotency_key: str | None = None


def _resolve_replay_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str) -> tuple[str, str]:
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
    if not has_permission(role_resolution.role, PERMISSION_RUN_REPLAY):
        denied_payload = build_activity_payload(
            club_id=str(club_id),
            actor_email=user.email,
            actor_role=role_resolution.role,
            action_type="replay_history_denied",
            entity_type="replay_history",
            entity_id="request",
            after_json={"source_client": "fastapi/nextjs", "reason": "insufficient_permission", "required_permission": PERMISSION_RUN_REPLAY},
            source_page=source,
            flagged_for_review=True,
        )
        write_admin_activity_log(supabase, denied_payload)
        raise HTTPException(status_code=403, detail="insufficient permission")
    return user.email, role_resolution.role


def _enforce_match_log_recovery_guard(
    supabase: Any,
    *,
    club_id: str,
) -> None:
    try:
        enforce_match_log_recovery_lock(
            supabase,
            club_id=str(club_id),
        )
    except MatchLogRecoveryLocked as exc:
        raise HTTPException(
            status_code=409,
            detail=exc.lock.as_detail(code=exc.code),
        ) from exc
    except MatchLogRecoveryLockUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


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
                "before replay could start. Refresh Match Log and complete "
                "that exact recovery."
            ),
        },
    ) from exc


def install_admin_replay_routes(app, *, get_supabase_client) -> None:
    """Register guarded replay routes for the Next admin pilot."""

    @app.get("/admin/clubs/{club_id}/replay-history")
    def get_admin_replay_history_status(
        club_id: str,
        include_jobs: bool = False,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_replay_enabled() else None
        if include_jobs and is_admin_replay_enabled():
            _resolve_replay_role_or_403(
                supabase=supabase,
                club_id=str(club_id),
                authorization=authorization,
                source="next_replay_history_status",
            )
        return build_admin_replay_status(
            supabase,
            club_id=str(club_id),
            include_recent_jobs=bool(include_jobs),
        )

    @app.post("/admin/clubs/{club_id}/replay-history")
    def post_admin_replay_history(
        club_id: str,
        payload: AdminReplayRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_replay_enabled():
            raise HTTPException(status_code=403, detail="Next replay is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_replay_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        _enforce_match_log_recovery_guard(
            supabase,
            club_id=str(club_id),
        )
        try:
            return run_admin_replay_history(
                supabase,
                club_id=str(club_id),
                target_reset=payload.target_reset,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                confirmation_text=payload.confirmation_text,
                idempotency_key=payload.idempotency_key,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            if _is_atomic_match_log_recovery_conflict(exc):
                _raise_atomic_match_log_recovery_conflict(exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            if _is_atomic_match_log_recovery_conflict(exc):
                _raise_atomic_match_log_recovery_conflict(exc)
            raise
