from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel

from jupr_app.domain.admin.roles import PERMISSION_MANAGE_MATCHES, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_challenge_ladder_service import (
    build_admin_challenge_ladder_status,
    get_admin_challenge_ladder_dashboard,
    is_admin_challenge_ladder_enabled,
    update_admin_challenge_ladder_challenge,
)
from services.api.auth import authenticate_bearer, auth_header


class ChallengeUpdateRequest(BaseModel):
    status: str
    admin_note: str | None = None
    confirmation_text: str = ""
    source: str = "next_challenge_ladder_admin"


def _resolve_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(supabase=supabase, club_id=str(club_id), email=user.email, user_id=user.user_id, allowlist=set())
    if not has_permission(role_resolution.role, PERMISSION_MANAGE_MATCHES):
        denied = build_activity_payload(
            club_id=str(club_id), actor_email=user.email, actor_role=role_resolution.role, action_type="admin_challenge_ladder_denied", entity_type="challenge_ladder", entity_id="challenge_ladder",
            after_json={"source_client": "fastapi/nextjs", "reason": "insufficient_permission"}, source_page=source, flagged_for_review=True,
        )
        write_admin_activity_log(supabase, denied)
        raise HTTPException(status_code=403, detail="insufficient permission")
    return user.email, role_resolution.role


def _handle(exc: Exception) -> None:
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise exc


def install_admin_challenge_ladder_routes(app, *, get_supabase_client) -> None:
    """Register guarded Challenge Ladder Admin routes."""

    @app.get("/admin/clubs/{club_id}/challenge-ladder/status")
    def get_admin_challenge_ladder_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_challenge_ladder_enabled() else None
        return build_admin_challenge_ladder_status(supabase, club_id=str(club_id))

    @app.get("/admin/clubs/{club_id}/challenge-ladder/dashboard")
    def get_admin_challenge_ladder_dashboard_route(club_id: str, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_challenge_ladder_admin_dashboard")
        try:
            return get_admin_challenge_ladder_dashboard(supabase, club_id=str(club_id))
        except Exception as exc:
            _handle(exc)

    @app.patch("/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}")
    def patch_admin_challenge_ladder_challenge(club_id: str, challenge_id: int, payload: ChallengeUpdateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return update_admin_challenge_ladder_challenge(
                supabase,
                club_id=str(club_id),
                challenge_id=int(challenge_id),
                status=payload.status,
                admin_note=payload.admin_note,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)
