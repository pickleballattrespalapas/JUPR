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


def _dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def _resolve_player_editor_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(
        supabase=supabase,
        club_id=str(club_id),
        email=user.email,
        user_id=user.user_id,
        allowlist=set(),
    )
    if not has_permission(role_resolution.role, PERMISSION_MANAGE_PLAYERS):
        denied_payload = build_activity_payload(
            club_id=str(club_id),
            actor_email=user.email,
            actor_role=role_resolution.role,
            action_type="admin_player_editor_denied",
            entity_type="players",
            entity_id="player_editor",
            after_json={"source_client": "fastapi/nextjs", "reason": "insufficient_permission"},
            source_page=source,
            flagged_for_review=True,
        )
        write_admin_activity_log(supabase, denied_payload)
        raise HTTPException(status_code=403, detail="insufficient permission")
    return user.email, role_resolution.role


def install_admin_player_editor_routes(app, *, get_supabase_client) -> None:
    """Register guarded Player Editor routes for the Next admin pilot."""

    @app.get("/admin/clubs/{club_id}/players/editor/status")
    def get_admin_player_editor_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_player_editor_enabled() else None
        return build_admin_player_editor_status(supabase, club_id=str(club_id))

    @app.get("/admin/clubs/{club_id}/players/editor/players")
    def get_admin_player_editor_players(
        club_id: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_player_editor_enabled():
            raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client()
        _resolve_player_editor_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_player_editor_list",
        )
        try:
            return list_admin_player_editor_players(supabase, club_id=str(club_id))
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/admin/clubs/{club_id}/players/editor/players/{player_id}")
    def get_admin_player_editor_player_detail(
        club_id: str,
        player_id: int,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_player_editor_enabled():
            raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client()
        _resolve_player_editor_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_player_editor_detail",
        )
        try:
            return get_admin_player_editor_detail(supabase, club_id=str(club_id), player_id=int(player_id))
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/admin/clubs/{club_id}/players/editor/players")
    def post_admin_player_editor_player(
        club_id: str,
        payload: AdminPlayerEditorCreateRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_player_editor_enabled():
            raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_player_editor_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return create_admin_player_editor_player(
                supabase,
                club_id=str(club_id),
                name=payload.name,
                starting_jupr=payload.starting_jupr,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.patch("/admin/clubs/{club_id}/players/editor/players/{player_id}")
    def patch_admin_player_editor_player(
        club_id: str,
        player_id: int,
        payload: AdminPlayerEditorUpdateRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_player_editor_enabled():
            raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_player_editor_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        patch = _dump_model(payload)
        source = str(patch.pop("source", payload.source))
        try:
            return update_admin_player_editor_player(
                supabase,
                club_id=str(club_id),
                player_id=int(player_id),
                patch=patch,
                actor_email=actor_email,
                actor_role=actor_role,
                source=source,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
