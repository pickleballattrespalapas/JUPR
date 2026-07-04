from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import PERMISSION_ENTER_SCORES, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_match_uploader_service import (
    build_admin_match_uploader_round_robin_preview,
    build_admin_match_uploader_status,
    create_admin_match_uploader_players,
    is_admin_match_uploader_enabled,
    submit_admin_match_uploader_batch,
)
from services.api.auth import authenticate_bearer, auth_header


class AdminMatchUploaderBatchRequest(BaseModel):
    matches: list[dict[str, Any]] = Field(default_factory=list)
    source: str = "next_match_uploader"


class AdminMatchUploaderRoundRobinCourtRequest(BaseModel):
    court: int | None = None
    format_type: str
    player_names: list[str] = Field(default_factory=list)
    names: str | None = None
    players_text: str | None = None


class AdminMatchUploaderRoundRobinPreviewRequest(BaseModel):
    courts: list[AdminMatchUploaderRoundRobinCourtRequest] = Field(default_factory=list)
    custom_schedule: str = ""
    schedule_mode: str = "full"
    source: str = "next_match_uploader_round_robin_preview"


class AdminMatchUploaderNewPlayerRequest(BaseModel):
    name: str
    starting_jupr: float = Field(default=3.5, ge=1.0, le=7.0)


class AdminMatchUploaderCreatePlayersRequest(BaseModel):
    players: list[AdminMatchUploaderNewPlayerRequest] = Field(default_factory=list)
    source: str = "next_match_uploader_new_players"


def _dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _resolve_score_entry_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(
        supabase=supabase,
        club_id=str(club_id),
        email=user.email,
        user_id=user.user_id,
        allowlist=set(),
    )
    if not has_permission(role_resolution.role, PERMISSION_ENTER_SCORES):
        denied_payload = build_activity_payload(
            club_id=str(club_id),
            actor_email=user.email,
            actor_role=role_resolution.role,
            action_type="submit_match_uploader_batch_denied",
            entity_type="matches",
            entity_id="batch",
            after_json={"source_client": "fastapi/nextjs", "reason": "insufficient_permission"},
            source_page=source,
            flagged_for_review=True,
        )
        write_admin_activity_log(supabase, denied_payload)
        raise HTTPException(status_code=403, detail="insufficient permission")
    return user.email, role_resolution.role


def install_admin_match_uploader_routes(app, *, get_supabase_client) -> None:
    """Register guarded Match Uploader routes for the Next admin pilot."""

    @app.get("/admin/clubs/{club_id}/match-uploader/status")
    def get_admin_match_uploader_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_match_uploader_enabled() else None
        return build_admin_match_uploader_status(supabase, club_id=str(club_id))

    @app.post("/admin/clubs/{club_id}/match-uploader/round-robin/preview")
    def post_admin_match_uploader_round_robin_preview(
        club_id: str,
        payload: AdminMatchUploaderRoundRobinPreviewRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_uploader_enabled():
            raise HTTPException(status_code=403, detail="Next Match Uploader is disabled.")
        supabase = get_supabase_client()
        _resolve_score_entry_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return build_admin_match_uploader_round_robin_preview(
                supabase,
                club_id=str(club_id),
                courts=[_dump_model(court) for court in payload.courts],
                custom_schedule=payload.custom_schedule,
                schedule_mode=payload.schedule_mode,
                source=payload.source,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/admin/clubs/{club_id}/match-uploader/players")
    def post_admin_match_uploader_players(
        club_id: str,
        payload: AdminMatchUploaderCreatePlayersRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_uploader_enabled():
            raise HTTPException(status_code=403, detail="Next Match Uploader is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_score_entry_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return create_admin_match_uploader_players(
                supabase,
                club_id=str(club_id),
                players=[_dump_model(player) for player in payload.players],
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

    @app.post("/admin/clubs/{club_id}/match-uploader/batch")
    def post_admin_match_uploader_batch(
        club_id: str,
        payload: AdminMatchUploaderBatchRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_uploader_enabled():
            raise HTTPException(status_code=403, detail="Next Match Uploader is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_score_entry_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return submit_admin_match_uploader_batch(
                supabase,
                club_id=str(club_id),
                matches=payload.matches,
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
