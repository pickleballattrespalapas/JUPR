from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import PERMISSION_MANAGE_TOURNAMENTS, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_tournament_setup_service import (
    build_admin_tournament_setup_status,
    get_admin_tournament_setup_detail,
    is_admin_tournament_setup_enabled,
    list_admin_tournament_setup_tournaments,
    publish_admin_tournament_setup,
    save_admin_tournament_setup_draft,
    update_admin_tournament_setup_settings,
)
from services.api.auth import authenticate_bearer, auth_header


class TournamentSetupSettingsRequest(BaseModel):
    registration_slug: str | None = None
    locale: str | None = "en"
    registration_status: str | None = None
    registration_open_at: str | None = None
    registration_close_at: str | None = None
    waitlist_enabled: bool | None = None
    partner_board_enabled: bool | None = None
    rules_markdown: str | None = None
    refund_policy_markdown: str | None = None
    sponsor_markdown: str | None = None
    confirmation_text: str = ""
    source: str = "next_tournament_setup_settings"


class TournamentSetupDraftRequest(BaseModel):
    days: list[dict[str, Any]] = Field(default_factory=list)
    event_families: list[dict[str, Any]] = Field(default_factory=list)
    event_options: list[dict[str, Any]] = Field(default_factory=list)
    saved_step: str | None = "next_setup"
    confirmation_text: str = ""
    source: str = "next_tournament_setup_draft"


class TournamentSetupPublishRequest(BaseModel):
    days: list[dict[str, Any]] = Field(default_factory=list)
    event_options: list[dict[str, Any]] = Field(default_factory=list)
    confirmation_text: str = ""
    source: str = "next_tournament_setup_publish"


def _dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def _resolve_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(supabase=supabase, club_id=str(club_id), email=user.email, user_id=user.user_id, allowlist=set())
    if not has_permission(role_resolution.role, PERMISSION_MANAGE_TOURNAMENTS):
        write_admin_activity_log(
            supabase,
            build_activity_payload(
                club_id=str(club_id),
                actor_email=user.email,
                actor_role=role_resolution.role,
                action_type="admin_tournament_setup_denied",
                entity_type="tournament_setup",
                entity_id="tournament_setup",
                after_json={"source_client": "fastapi/nextjs", "reason": "insufficient_permission"},
                source_page=source,
                flagged_for_review=True,
            ),
        )
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


def install_admin_tournament_setup_routes(app, *, get_supabase_client) -> None:
    """Register guarded Tournament Setup Manager routes for the Next staging surface."""

    @app.get("/admin/clubs/{club_id}/tournaments/setup/status")
    def get_setup_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_tournament_setup_enabled() else None
        return build_admin_tournament_setup_status(supabase, club_id=str(club_id))

    @app.get("/admin/clubs/{club_id}/tournaments/setup/tournaments")
    def get_setup_tournaments(club_id: str, include_archived: bool = Query(default=True), authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_setup_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Setup is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_tournament_setup_list")
        try:
            return list_admin_tournament_setup_tournaments(supabase, club_id=str(club_id), include_archived=bool(include_archived))
        except Exception as exc:
            _handle(exc)

    @app.get("/admin/clubs/{club_id}/tournaments/setup/tournaments/{tournament_id}")
    def get_setup_detail(club_id: str, tournament_id: str, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_setup_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Setup is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_tournament_setup_detail")
        try:
            return get_admin_tournament_setup_detail(supabase, club_id=str(club_id), tournament_id=str(tournament_id))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            _handle(exc)

    @app.patch("/admin/clubs/{club_id}/tournaments/setup/tournaments/{tournament_id}/settings")
    def patch_setup_settings(club_id: str, tournament_id: str, payload: TournamentSetupSettingsRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_setup_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Setup is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        patch = _dump_model(payload)
        source = str(patch.pop("source", payload.source))
        confirmation_text = str(patch.pop("confirmation_text", payload.confirmation_text))
        try:
            return update_admin_tournament_setup_settings(supabase, club_id=str(club_id), tournament_id=str(tournament_id), patch=patch, actor_email=actor_email, actor_role=actor_role, confirmation_text=confirmation_text, source=source)
        except Exception as exc:
            _handle(exc)

    @app.put("/admin/clubs/{club_id}/tournaments/setup/tournaments/{tournament_id}/draft")
    def put_setup_draft(club_id: str, tournament_id: str, payload: TournamentSetupDraftRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_setup_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Setup is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return save_admin_tournament_setup_draft(supabase, club_id=str(club_id), tournament_id=str(tournament_id), days=payload.days, event_families=payload.event_families, event_options=payload.event_options, saved_step=payload.saved_step, actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source)
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/setup/tournaments/{tournament_id}/publish")
    def post_setup_publish(club_id: str, tournament_id: str, payload: TournamentSetupPublishRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_setup_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Setup is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return publish_admin_tournament_setup(supabase, club_id=str(club_id), tournament_id=str(tournament_id), days=payload.days, event_options=payload.event_options, actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source)
        except Exception as exc:
            _handle(exc)
