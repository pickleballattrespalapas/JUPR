from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from jupr_app.domain.admin.roles import PERMISSION_MANAGE_MATCHES, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_league_manager_service import (
    build_admin_league_manager_status,
    get_admin_league_manager_detail,
    is_admin_league_manager_enabled,
    list_admin_league_manager_leagues,
)
from services.api.auth import authenticate_bearer, auth_header


def _resolve_league_manager_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(
        supabase=supabase,
        club_id=str(club_id),
        email=user.email,
        user_id=user.user_id,
        allowlist=set(),
    )
    if not has_permission(role_resolution.role, PERMISSION_MANAGE_MATCHES):
        denied_payload = build_activity_payload(
            club_id=str(club_id),
            actor_email=user.email,
            actor_role=role_resolution.role,
            action_type="admin_league_manager_denied",
            entity_type="league_manager",
            entity_id="league_manager",
            after_json={"source_client": "fastapi/nextjs", "reason": "insufficient_permission"},
            source_page=source,
            flagged_for_review=True,
        )
        write_admin_activity_log(supabase, denied_payload)
        raise HTTPException(status_code=403, detail="insufficient permission")
    return user.email, role_resolution.role


def install_admin_league_manager_routes(app, *, get_supabase_client) -> None:
    """Register guarded League Manager read-foundation routes for the Next admin pilot."""

    @app.get("/admin/clubs/{club_id}/league-manager/status")
    def get_admin_league_manager_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_league_manager_enabled() else None
        return build_admin_league_manager_status(supabase, club_id=str(club_id))

    @app.get("/admin/clubs/{club_id}/league-manager/leagues")
    def get_admin_league_manager_leagues(
        club_id: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        supabase = get_supabase_client()
        _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_league_manager_list",
        )
        try:
            return list_admin_league_manager_leagues(supabase, club_id=str(club_id))
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/admin/clubs/{club_id}/league-manager/leagues/{league_name}")
    def get_admin_league_manager_league_detail(
        club_id: str,
        league_name: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        supabase = get_supabase_client()
        _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_league_manager_detail",
        )
        try:
            return get_admin_league_manager_detail(supabase, club_id=str(club_id), league_name=str(league_name))
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
