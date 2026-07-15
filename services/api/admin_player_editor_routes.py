from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import PERMISSION_MANAGE_PLAYERS, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_player_editor_service import (
    build_admin_player_editor_status,
    create_admin_player_editor_player,
    get_admin_player_editor_detail,
    is_admin_player_editor_enabled,
    list_admin_player_editor_players,
    update_admin_player_editor_player,
)
from jupr_app.services.admin_player_league_rating_service import update_admin_player_editor_league_rating
from jupr_app.services.admin_player_merge_service import build_admin_player_merge_preview, execute_admin_player_merge
from jupr_app.services.admin_player_social_identity_service import (
    auto_link_admin_player_social_identities,
    list_admin_player_social_identities,
    update_admin_player_social_identity,
)
from services.api.auth import authenticate_bearer, auth_header


class AdminPlayerEditorCreateRequest(BaseModel):
    name: str
    starting_jupr: float = Field(default=3.5, ge=1.0, le=7.0)
    source: str = "next_player_editor"


class AdminPlayerEditorUpdateRequest(BaseModel):
    name: str | None = None
    rating_jupr: float | None = Field(default=None, ge=1.0, le=7.0)
    starting_jupr: float | None = Field(default=None, ge=1.0, le=7.0)
    active: bool | None = None
    source: str = "next_player_editor"


class AdminPlayerEditorLeagueRatingUpdateRequest(BaseModel):
    rating_jupr: float | None = Field(default=None, ge=1.0, le=7.0)
    starting_jupr: float | None = Field(default=None, ge=1.0, le=7.0)
    is_active: bool | None = None
    confirmation_text: str = ""
    source: str = "next_player_editor_league_rating"


class AdminPlayerSocialIdentityUpdateRequest(BaseModel):
    linked_player_id: int | None = None
    display_name: str | None = None
    confirmation_text: str = ""
    source: str = "next_player_editor_social_identity"


class AdminPlayerSocialIdentityAutoLinkRequest(BaseModel):
    confirmation_text: str = ""
    source: str = "next_player_editor_social_auto_link"


class AdminPlayerMergePreviewRequest(BaseModel):
    source_player_id: int
    target_player_id: int
    source: str = "next_player_editor_merge_preview"


class AdminPlayerMergeExecuteRequest(BaseModel):
    source_player_id: int
    target_player_id: int
    confirmation_text: str = ""
    source: str = "next_player_editor_merge"


def _dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def _resolve_player_editor_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(supabase=supabase, club_id=str(club_id), email=user.email, user_id=user.user_id, allowlist=set())
    if not has_permission(role_resolution.role, PERMISSION_MANAGE_PLAYERS):
        denied_payload = build_activity_payload(club_id=str(club_id), actor_email=user.email, actor_role=role_resolution.role, action_type="admin_player_editor_denied", entity_type="players", entity_id="player_editor", after_json={"source_client": "fastapi/nextjs", "reason": "insufficient_permission"}, source_page=source, flagged_for_review=True)
        write_admin_activity_log(supabase, denied_payload)
        raise HTTPException(status_code=403, detail="insufficient permission")
    return user.email, role_resolution.role


def _handle_common(exc: Exception) -> None:
    if isinstance(exc, PermissionError): raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, ValueError): raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, RuntimeError): raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise exc


def install_admin_player_editor_routes(app, *, get_supabase_client) -> None:
    """Register guarded Player Editor routes for the Next admin pilot."""

    @app.get("/admin/clubs/{club_id}/players/editor/status")
    def get_admin_player_editor_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_player_editor_enabled() else None
        return build_admin_player_editor_status(supabase, club_id=str(club_id))

    @app.post("/admin/clubs/{club_id}/players/editor/merge/preview")
    def post_admin_player_merge_preview(club_id: str, payload: AdminPlayerMergePreviewRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try: return build_admin_player_merge_preview(supabase, club_id=str(club_id), source_player_id=payload.source_player_id, target_player_id=payload.target_player_id)
        except Exception as exc: _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/players/editor/merge")
    def post_admin_player_merge_execute(club_id: str, payload: AdminPlayerMergeExecuteRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); actor_email, actor_role = _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try: return execute_admin_player_merge(supabase, club_id=str(club_id), source_player_id=payload.source_player_id, target_player_id=payload.target_player_id, actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source)
        except Exception as exc: _handle_common(exc)

    @app.get("/admin/clubs/{club_id}/players/editor/social-identities")
    def get_admin_player_social_identities(club_id: str, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_player_editor_social_identity_list")
        try: return list_admin_player_social_identities(supabase, club_id=str(club_id))
        except Exception as exc: _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/players/editor/social-identities/auto-link")
    def post_admin_player_social_auto_link(club_id: str, payload: AdminPlayerSocialIdentityAutoLinkRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); actor_email, actor_role = _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try: return auto_link_admin_player_social_identities(supabase, club_id=str(club_id), actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source)
        except Exception as exc: _handle_common(exc)

    @app.patch("/admin/clubs/{club_id}/players/editor/social-identities/{club_person_id}")
    def patch_admin_player_social_identity(club_id: str, club_person_id: str, payload: AdminPlayerSocialIdentityUpdateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); actor_email, actor_role = _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try: return update_admin_player_social_identity(supabase, club_id=str(club_id), club_person_id=str(club_person_id), linked_player_id=payload.linked_player_id, display_name=payload.display_name, confirmation_text=payload.confirmation_text, actor_email=actor_email, actor_role=actor_role, source=payload.source)
        except Exception as exc: _handle_common(exc)

    @app.get("/admin/clubs/{club_id}/players/editor/players")
    def get_admin_player_editor_players(club_id: str, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_player_editor_list")
        try: return list_admin_player_editor_players(supabase, club_id=str(club_id))
        except Exception as exc: _handle_common(exc)

    @app.get("/admin/clubs/{club_id}/players/editor/players/{player_id}")
    def get_admin_player_editor_player_detail(club_id: str, player_id: int, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_player_editor_detail")
        try: return get_admin_player_editor_detail(supabase, club_id=str(club_id), player_id=int(player_id))
        except ValueError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc: _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/players/editor/players")
    def post_admin_player_editor_player(club_id: str, payload: AdminPlayerEditorCreateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); actor_email, actor_role = _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try: return create_admin_player_editor_player(supabase, club_id=str(club_id), name=payload.name, starting_jupr=payload.starting_jupr, actor_email=actor_email, actor_role=actor_role, source=payload.source)
        except Exception as exc: _handle_common(exc)

    @app.patch("/admin/clubs/{club_id}/players/editor/players/{player_id}")
    def patch_admin_player_editor_player(club_id: str, player_id: int, payload: AdminPlayerEditorUpdateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); actor_email, actor_role = _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        patch = _dump_model(payload); source = str(patch.pop("source", payload.source))
        try: return update_admin_player_editor_player(supabase, club_id=str(club_id), player_id=int(player_id), patch=patch, actor_email=actor_email, actor_role=actor_role, source=source)
        except Exception as exc: _handle_common(exc)

    @app.patch("/admin/clubs/{club_id}/players/editor/players/{player_id}/league-ratings/{league_rating_id}")
    def patch_admin_player_editor_league_rating(club_id: str, player_id: int, league_rating_id: int, payload: AdminPlayerEditorLeagueRatingUpdateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); actor_email, actor_role = _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        patch = _dump_model(payload); source = str(patch.pop("source", payload.source)); confirmation_text = str(patch.pop("confirmation_text", payload.confirmation_text))
        try: return update_admin_player_editor_league_rating(supabase, club_id=str(club_id), player_id=int(player_id), league_rating_id=int(league_rating_id), patch=patch, actor_email=actor_email, actor_role=actor_role, confirmation_text=confirmation_text, source=source)
        except Exception as exc: _handle_common(exc)
