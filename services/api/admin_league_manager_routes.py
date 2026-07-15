from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel

from jupr_app.domain.admin.roles import PERMISSION_MANAGE_MATCHES, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_league_manager_roster_service import update_admin_league_manager_roster_membership
from jupr_app.services.admin_league_manager_service import (
    build_admin_league_manager_status,
    get_admin_league_manager_detail,
    is_admin_league_manager_enabled,
    list_admin_league_manager_leagues,
)
from jupr_app.services.admin_league_manager_update_service import update_admin_league_manager_settings
from services.api.auth import authenticate_bearer, auth_header


class AdminLeagueManagerSettingsUpdateRequest(BaseModel):
    status: str | None = None
    k_factor: int | None = None
    min_games: int | None = None
    schedule_config: dict[str, Any] | None = None
    court_board_defaults: dict[str, Any] | None = None
    rules_config: dict[str, Any] | None = None
    awards_config: dict[str, Any] | None = None
    event_tags: dict[str, Any] | None = None
    confirmation_text: str = ""
    source: str = "next_league_manager_settings_update"


class AdminLeagueManagerRosterMembershipRequest(BaseModel):
    action: str
    starting_rating: float | None = None
    confirmation_text: str = ""
    source: str = "next_league_manager_roster_update"


def _dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


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


def _handle_common(exc: Exception) -> None:
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise exc


def install_admin_league_manager_routes(app, *, get_supabase_client) -> None:
    """Register guarded League Manager routes for the Next admin pilot."""

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
        except Exception as exc:
            _handle_common(exc)

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
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            _handle_common(exc)

    @app.patch("/admin/clubs/{club_id}/league-manager/leagues/{league_name}")
    def patch_admin_league_manager_settings(
        club_id: str,
        league_name: str,
        payload: AdminLeagueManagerSettingsUpdateRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        patch = _dump_model(payload)
        source = str(patch.pop("source", payload.source))
        confirmation_text = str(patch.pop("confirmation_text", payload.confirmation_text))
        try:
            return update_admin_league_manager_settings(
                supabase,
                club_id=str(club_id),
                league_name=str(league_name),
                patch=patch,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=confirmation_text,
                source=source,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.patch("/admin/clubs/{club_id}/league-manager/leagues/{league_name}/roster/{player_id}")
    def patch_admin_league_manager_roster_membership(
        club_id: str,
        league_name: str,
        player_id: int,
        payload: AdminLeagueManagerRosterMembershipRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return update_admin_league_manager_roster_membership(
                supabase,
                club_id=str(club_id),
                league_name=str(league_name),
                player_id=player_id,
                action=payload.action,
                starting_rating=payload.starting_rating,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle_common(exc)
