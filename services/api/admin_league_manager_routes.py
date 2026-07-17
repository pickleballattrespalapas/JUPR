from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import PERMISSION_MANAGE_MATCHES, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_league_awards_service import close_admin_league_and_award, preview_admin_league_awards
from jupr_app.services.admin_league_live_service import (
    build_admin_league_live_status,
    create_admin_league_live_session,
    get_admin_league_live_session,
    list_admin_league_live_sessions,
    save_admin_league_live_round,
    update_admin_league_live_session_snapshot,
)
from jupr_app.services.admin_league_manager_create_service import create_admin_league_manager_draft
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
    description: str | None = Field(default=None, max_length=2000)
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


class AdminLeagueManagerCreateRequest(BaseModel):
    league_name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    min_games: int = Field(default=6, ge=0, le=1000)
    k_factor: int = Field(default=32, ge=1, le=128)
    confirmation_text: str = ""
    source: str = "next_league_manager_create"


class AdminLeagueManagerRosterMembershipRequest(BaseModel):
    action: str
    starting_rating: float | None = None
    confirmation_text: str = ""
    source: str = "next_league_manager_roster_update"


class AdminLeagueAwardsCloseRequest(BaseModel):
    award_badges: bool = True
    confirmation_text: str = ""
    source: str = "next_league_manager_awards_close"


class AdminLeagueLiveSessionCreateRequest(BaseModel):
    league_name: str
    week_tag: str = "Week 1"
    total_rounds: int = Field(default=1, ge=1, le=50)
    current_round: int = Field(default=1, ge=1, le=50)
    roster: list[dict[str, Any]] = Field(default_factory=list)
    courts: list[dict[str, Any]] = Field(default_factory=list)
    notes: str | None = None
    confirmation_text: str = ""
    source: str = "next_league_live_session_create"


class AdminLeagueLiveSessionSnapshotRequest(BaseModel):
    status: str | None = None
    week_tag: str | None = None
    total_rounds: int | None = Field(default=None, ge=1, le=50)
    current_round: int | None = Field(default=None, ge=1, le=50)
    roster: list[dict[str, Any]] | None = None
    courts: list[dict[str, Any]] | None = None
    notes: str | None = None
    confirmation_text: str = ""
    source: str = "next_league_live_session_snapshot"


class AdminLeagueLiveRoundSaveRequest(BaseModel):
    round_label: str | None = None
    match_date: str | None = None
    preview: dict[str, Any] | None = None
    matches: list[dict[str, Any]] = Field(default_factory=list)
    movement: dict[str, Any] | None = None
    submitted_match_count: int | None = None
    submitted_match_ids: list[Any] = Field(default_factory=list)
    courts: list[dict[str, Any]] = Field(default_factory=list)
    advance_after_save: bool = True
    confirmation_text: str = ""
    source: str = "next_league_live_round_save"


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

    @app.get("/admin/clubs/{club_id}/league-manager/live/status")
    def get_admin_league_live_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_league_manager_enabled() else None
        return build_admin_league_live_status(supabase, club_id=str(club_id))

    @app.get("/admin/clubs/{club_id}/league-manager/live-sessions")
    def get_admin_league_live_sessions(
        club_id: str,
        status: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        supabase = get_supabase_client()
        _resolve_league_manager_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_league_live_sessions_list")
        try:
            return list_admin_league_live_sessions(supabase, club_id=str(club_id), status=status, limit=limit)
        except Exception as exc:
            _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/league-manager/live-sessions")
    def post_admin_league_live_session(
        club_id: str,
        payload: AdminLeagueLiveSessionCreateRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return create_admin_league_live_session(
                supabase,
                club_id=str(club_id),
                league_name=payload.league_name,
                week_tag=payload.week_tag,
                total_rounds=payload.total_rounds,
                current_round=payload.current_round,
                roster=payload.roster,
                courts=payload.courts,
                notes=payload.notes,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.get("/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}")
    def get_admin_league_live_session_detail(
        club_id: str,
        session_id: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        supabase = get_supabase_client()
        _resolve_league_manager_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_league_live_session_detail")
        try:
            return get_admin_league_live_session(supabase, club_id=str(club_id), session_id=str(session_id))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            _handle_common(exc)

    @app.patch("/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/snapshot")
    def patch_admin_league_live_session_snapshot(
        club_id: str,
        session_id: str,
        payload: AdminLeagueLiveSessionSnapshotRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        patch = _dump_model(payload)
        confirmation_text = str(patch.pop("confirmation_text", payload.confirmation_text))
        source = str(patch.pop("source", payload.source))
        try:
            return update_admin_league_live_session_snapshot(
                supabase,
                club_id=str(club_id),
                session_id=str(session_id),
                patch=patch,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=confirmation_text,
                source=source,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}")
    def post_admin_league_live_round(
        club_id: str,
        session_id: str,
        round_number: int,
        payload: AdminLeagueLiveRoundSaveRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return save_admin_league_live_round(
                supabase,
                club_id=str(club_id),
                session_id=str(session_id),
                round_number=int(round_number),
                round_label=payload.round_label,
                match_date=payload.match_date,
                preview=payload.preview,
                matches=payload.matches,
                movement=payload.movement,
                submitted_match_count=payload.submitted_match_count,
                submitted_match_ids=payload.submitted_match_ids,
                courts=payload.courts,
                advance_after_save=payload.advance_after_save,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle_common(exc)

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

    @app.post("/admin/clubs/{club_id}/league-manager/leagues")
    def post_admin_league_manager_league(
        club_id: str,
        payload: AdminLeagueManagerCreateRequest,
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
            return create_admin_league_manager_draft(
                supabase,
                club_id=str(club_id),
                league_name=payload.league_name,
                description=payload.description,
                min_games=payload.min_games,
                k_factor=payload.k_factor,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
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

    @app.get("/admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards/preview")
    def get_admin_league_awards_preview(
        club_id: str,
        league_name: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        supabase = get_supabase_client()
        _resolve_league_manager_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_league_manager_awards_preview")
        try:
            return preview_admin_league_awards(supabase, club_id=str(club_id), league_name=str(league_name))
        except Exception as exc:
            _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards/close")
    def post_admin_league_awards_close(
        club_id: str,
        league_name: str,
        payload: AdminLeagueAwardsCloseRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return close_admin_league_and_award(
                supabase,
                club_id=str(club_id),
                league_name=str(league_name),
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                award_badges=payload.award_badges,
                source=payload.source,
            )
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
