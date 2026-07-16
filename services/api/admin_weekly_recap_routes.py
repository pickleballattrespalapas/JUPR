from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import PERMISSION_MANAGE_MATCHES, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_weekly_recap_service import (
    build_admin_weekly_recap_status,
    generate_admin_weekly_recap,
    get_admin_weekly_recap,
    is_admin_weekly_recap_enabled,
    list_admin_weekly_recaps,
    publish_admin_weekly_recap,
    save_admin_weekly_recap,
)
from services.api.auth import authenticate_bearer, auth_header


class AdminWeeklyRecapGenerateRequest(BaseModel):
    week_start: str
    week_end: str
    tz_name: str = "America/Mazatlan"
    confirmation_text: str = ""
    source: str = "next_weekly_recap_generate"


class AdminWeeklyRecapSaveRequest(BaseModel):
    edits_json: dict[str, Any] = Field(default_factory=dict)
    tz_name: str = "America/Mazatlan"
    confirmation_text: str = ""
    source: str = "next_weekly_recap_save"


class AdminWeeklyRecapPublishRequest(BaseModel):
    action: str = "publish"
    edits_json: dict[str, Any] | None = None
    tz_name: str = "America/Mazatlan"
    confirmation_text: str = ""
    source: str = "next_weekly_recap_publish"


def _resolve_weekly_recap_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str) -> tuple[str, str]:
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
            action_type="admin_weekly_recap_denied",
            entity_type="weekly_recap",
            entity_id="weekly_recap",
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


def install_admin_weekly_recap_routes(app, *, get_supabase_client) -> None:
    """Register guarded Weekly Recap Admin routes for the Next admin pilot."""

    @app.get("/admin/clubs/{club_id}/weekly-recap/status")
    def get_weekly_recap_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_weekly_recap_enabled() else None
        return build_admin_weekly_recap_status(supabase, club_id=str(club_id))

    @app.get("/admin/clubs/{club_id}/weekly-recap/recaps")
    def get_weekly_recap_list(
        club_id: str,
        limit: int = Query(default=50, ge=1, le=200),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_weekly_recap_enabled():
            raise HTTPException(status_code=403, detail="Next Weekly Recap Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_weekly_recap_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_weekly_recap_list")
        try:
            return list_admin_weekly_recaps(supabase, club_id=str(club_id), limit=limit)
        except Exception as exc:
            _handle_common(exc)

    @app.get("/admin/clubs/{club_id}/weekly-recap/recaps/{week_start}")
    def get_weekly_recap_detail(
        club_id: str,
        week_start: str,
        include_candidates: bool = Query(default=True),
        tz_name: str = Query(default="America/Mazatlan"),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_weekly_recap_enabled():
            raise HTTPException(status_code=403, detail="Next Weekly Recap Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_weekly_recap_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_weekly_recap_detail")
        try:
            return get_admin_weekly_recap(supabase, club_id=str(club_id), week_start=str(week_start), include_candidates=include_candidates, tz_name=tz_name)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/weekly-recap/generate")
    def post_weekly_recap_generate(
        club_id: str,
        payload: AdminWeeklyRecapGenerateRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_weekly_recap_enabled():
            raise HTTPException(status_code=403, detail="Next Weekly Recap Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_weekly_recap_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return generate_admin_weekly_recap(
                supabase,
                club_id=str(club_id),
                week_start=payload.week_start,
                week_end=payload.week_end,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                tz_name=payload.tz_name,
                source=payload.source,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.patch("/admin/clubs/{club_id}/weekly-recap/recaps/{week_start}")
    def patch_weekly_recap_save(
        club_id: str,
        week_start: str,
        payload: AdminWeeklyRecapSaveRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_weekly_recap_enabled():
            raise HTTPException(status_code=403, detail="Next Weekly Recap Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_weekly_recap_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return save_admin_weekly_recap(
                supabase,
                club_id=str(club_id),
                week_start=str(week_start),
                edits_json=payload.edits_json,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                tz_name=payload.tz_name,
                source=payload.source,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/weekly-recap/recaps/{week_start}/publish")
    def post_weekly_recap_publish(
        club_id: str,
        week_start: str,
        payload: AdminWeeklyRecapPublishRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_weekly_recap_enabled():
            raise HTTPException(status_code=403, detail="Next Weekly Recap Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_weekly_recap_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return publish_admin_weekly_recap(
                supabase,
                club_id=str(club_id),
                week_start=str(week_start),
                action=payload.action,
                edits_json=payload.edits_json,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                tz_name=payload.tz_name,
                source=payload.source,
            )
        except Exception as exc:
            _handle_common(exc)
