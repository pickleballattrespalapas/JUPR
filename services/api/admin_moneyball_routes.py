from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import PERMISSION_ENTER_SCORES, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_moneyball_service import (
    build_admin_moneyball_status,
    build_moneyball_preview,
    is_admin_moneyball_enabled,
    submit_admin_moneyball,
)
from services.api.auth import authenticate_bearer, auth_header


class MoneyballPreviewRequest(BaseModel):
    player_ids: list[int] = Field(default_factory=list)
    rating_context: str = "OVERALL"
    win_rate: float = 5.0
    point_rate: float = 2.0


class MoneyballSubmitRequest(MoneyballPreviewRequest):
    scores: list[dict[str, Any]] = Field(default_factory=list)
    league_name: str = "Moneyball"
    week_tag: str = "Moneyball"
    match_type: str = "Moneyball RR"
    confirmation_text: str = ""
    source: str = "next_moneyball_admin"


def _resolve_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(supabase=supabase, club_id=str(club_id), email=user.email, user_id=user.user_id, allowlist=set())
    if not has_permission(role_resolution.role, PERMISSION_ENTER_SCORES):
        denied = build_activity_payload(
            club_id=str(club_id), actor_email=user.email, actor_role=role_resolution.role, action_type="admin_moneyball_denied", entity_type="moneyball", entity_id="moneyball",
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


def install_admin_moneyball_routes(app, *, get_supabase_client) -> None:
    """Register guarded Moneyball admin routes."""

    @app.get("/admin/clubs/{club_id}/moneyball/status")
    def get_admin_moneyball_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_moneyball_enabled() else None
        return build_admin_moneyball_status(supabase, club_id=str(club_id))

    @app.post("/admin/clubs/{club_id}/moneyball/preview")
    def post_admin_moneyball_preview(club_id: str, payload: MoneyballPreviewRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_moneyball_enabled():
            raise HTTPException(status_code=403, detail="Next Moneyball is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_moneyball_preview")
        try:
            return build_moneyball_preview(supabase, club_id=str(club_id), player_ids=payload.player_ids, rating_context=payload.rating_context, win_rate=payload.win_rate, point_rate=payload.point_rate)
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/moneyball/submit")
    def post_admin_moneyball_submit(club_id: str, payload: MoneyballSubmitRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_moneyball_enabled():
            raise HTTPException(status_code=403, detail="Next Moneyball is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return submit_admin_moneyball(
                supabase,
                club_id=str(club_id),
                player_ids=payload.player_ids,
                scores=payload.scores,
                rating_context=payload.rating_context,
                league_name=payload.league_name,
                week_tag=payload.week_tag,
                match_type=payload.match_type,
                win_rate=payload.win_rate,
                point_rate=payload.point_rate,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)
