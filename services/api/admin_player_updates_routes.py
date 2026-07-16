from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel

from jupr_app.domain.admin.roles import PERMISSION_MANAGE_SUBSCRIPTIONS, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_player_updates_service import (
    build_admin_player_updates_status,
    is_admin_player_updates_enabled,
    run_admin_player_update_range,
)
from services.api.auth import authenticate_bearer, auth_header


class AdminPlayerUpdateRangeRequest(BaseModel):
    start_date: str
    end_date: str
    only_players_with_matches: bool = True
    send_now: bool = True
    confirmation_text: str = ""
    source: str = "next_player_updates_admin_range"


def _resolve_player_updates_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(
        supabase=supabase,
        club_id=str(club_id),
        email=user.email,
        user_id=user.user_id,
        allowlist=set(),
    )
    if not has_permission(role_resolution.role, PERMISSION_MANAGE_SUBSCRIPTIONS):
        denied_payload = build_activity_payload(
            club_id=str(club_id),
            actor_email=user.email,
            actor_role=role_resolution.role,
            action_type="admin_player_updates_denied",
            entity_type="player_updates",
            entity_id="player_updates",
            after_json={"source_client": "fastapi/nextjs", "reason": "insufficient_permission"},
            source_page=source,
            flagged_for_review=True,
        )
        write_admin_activity_log(supabase, denied_payload)
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


def install_admin_player_updates_routes(app, *, get_supabase_client) -> None:
    """Register guarded Player Updates Admin routes for the Next admin pilot."""

    @app.get("/admin/clubs/{club_id}/player-updates/status")
    def get_admin_player_updates_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_player_updates_enabled() else None
        return build_admin_player_updates_status(supabase, club_id=str(club_id))

    @app.post("/admin/clubs/{club_id}/player-updates/send-range")
    def post_admin_player_updates_send_range(
        club_id: str,
        payload: AdminPlayerUpdateRangeRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_player_updates_enabled():
            raise HTTPException(status_code=403, detail="Next Player Updates Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_player_updates_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return run_admin_player_update_range(
                supabase,
                club_id=str(club_id),
                start_date=payload.start_date,
                end_date=payload.end_date,
                only_players_with_matches=payload.only_players_with_matches,
                send_now=payload.send_now,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)
