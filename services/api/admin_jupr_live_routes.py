from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import PERMISSION_ENTER_SCORES, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_jupr_live_service import (
    build_admin_jupr_live_status,
    create_admin_jupr_live_session,
    get_admin_jupr_live_session,
    is_admin_jupr_live_enabled,
    list_admin_jupr_live_sessions,
    publish_admin_jupr_live_matches,
    update_admin_jupr_live_scores,
    update_admin_jupr_live_session_status,
)
from services.api.auth import authenticate_bearer, auth_header


class JuprLiveSessionCreateRequest(BaseModel):
    title: str = "JUPR Live Session"
    event_type: str = "round_robin"
    participant_names: list[str] = Field(default_factory=list)
    player_ids: list[int] = Field(default_factory=list)
    confirmation_text: str = ""
    source: str = "next_jupr_live_admin_create"


class JuprLiveSessionStatusRequest(BaseModel):
    status: str
    title: str | None = None
    confirmation_text: str = ""
    source: str = "next_jupr_live_admin_status"


class JuprLiveScorePayload(BaseModel):
    match_id: str
    score_a: int | None = None
    score_b: int | None = None


class JuprLiveScoresRequest(BaseModel):
    scores: list[JuprLiveScorePayload] = Field(default_factory=list)
    confirmation_text: str = ""
    source: str = "next_jupr_live_admin_scores"


class JuprLivePublishRequest(BaseModel):
    match_date: str | None = None
    confirmation_text: str = ""
    source: str = "next_jupr_live_admin_publish"


def _resolve_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(supabase=supabase, club_id=str(club_id), email=user.email, user_id=user.user_id, allowlist=set())
    if not has_permission(role_resolution.role, PERMISSION_ENTER_SCORES):
        denied = build_activity_payload(
            club_id=str(club_id), actor_email=user.email, actor_role=role_resolution.role, action_type="admin_jupr_live_denied", entity_type="live_session", entity_id="live_session",
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


def install_admin_jupr_live_routes(app, *, get_supabase_client) -> None:
    """Register guarded JUPR Live Admin routes."""

    @app.get("/admin/clubs/{club_id}/jupr-live/status")
    def get_admin_jupr_live_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_jupr_live_enabled() else None
        return build_admin_jupr_live_status(supabase, club_id=str(club_id))

    @app.get("/admin/clubs/{club_id}/jupr-live/sessions")
    def get_admin_jupr_live_sessions(
        club_id: str,
        status: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=300),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_jupr_live_enabled():
            raise HTTPException(status_code=403, detail="Next JUPR Live Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_jupr_live_admin_list")
        try:
            return list_admin_jupr_live_sessions(supabase, club_id=str(club_id), status=status, limit=limit)
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/jupr-live/sessions")
    def post_admin_jupr_live_session(
        club_id: str,
        payload: JuprLiveSessionCreateRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_jupr_live_enabled():
            raise HTTPException(status_code=403, detail="Next JUPR Live Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return create_admin_jupr_live_session(
                supabase,
                club_id=str(club_id),
                title=payload.title,
                event_type=payload.event_type,
                participant_names=payload.participant_names,
                player_ids=payload.player_ids,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)

    @app.get("/admin/clubs/{club_id}/jupr-live/sessions/{session_key}")
    def get_admin_jupr_live_session_detail(
        club_id: str,
        session_key: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_jupr_live_enabled():
            raise HTTPException(status_code=403, detail="Next JUPR Live Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_jupr_live_admin_detail")
        try:
            return get_admin_jupr_live_session(supabase, club_id=str(club_id), session_key=str(session_key))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            _handle(exc)

    @app.patch("/admin/clubs/{club_id}/jupr-live/sessions/{session_key}")
    def patch_admin_jupr_live_session(
        club_id: str,
        session_key: str,
        payload: JuprLiveSessionStatusRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_jupr_live_enabled():
            raise HTTPException(status_code=403, detail="Next JUPR Live Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return update_admin_jupr_live_session_status(
                supabase,
                club_id=str(club_id),
                session_key=str(session_key),
                status=payload.status,
                title=payload.title,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)

    @app.patch("/admin/clubs/{club_id}/jupr-live/sessions/{session_key}/scores")
    def patch_admin_jupr_live_scores(
        club_id: str,
        session_key: str,
        payload: JuprLiveScoresRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_jupr_live_enabled():
            raise HTTPException(status_code=403, detail="Next JUPR Live Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return update_admin_jupr_live_scores(
                supabase,
                club_id=str(club_id),
                session_key=str(session_key),
                scores=[score.dict() for score in payload.scores],
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/jupr-live/sessions/{session_key}/publish")
    def post_admin_jupr_live_publish(
        club_id: str,
        session_key: str,
        payload: JuprLivePublishRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_jupr_live_enabled():
            raise HTTPException(status_code=403, detail="Next JUPR Live Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return publish_admin_jupr_live_matches(
                supabase,
                club_id=str(club_id),
                session_key=str(session_key),
                match_date=payload.match_date,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)
