from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel

from jupr_app.domain.admin.roles import PERMISSION_MANAGE_SUBSCRIPTIONS, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_verified_updates_service import (
    build_admin_verified_updates_status,
    is_admin_verified_updates_enabled,
    list_admin_verified_update_requests,
    update_admin_verified_update_request,
)
from services.api.auth import authenticate_bearer, auth_header


class AdminVerifiedUpdateRequestAction(BaseModel):
    action: str
    admin_note: str | None = None
    confirmation_text: str = ""
    source: str = "next_verified_updates_request_review"


def _resolve_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(supabase=supabase, club_id=str(club_id), email=user.email, user_id=user.user_id, allowlist=set())
    if not has_permission(role_resolution.role, PERMISSION_MANAGE_SUBSCRIPTIONS):
        denied = build_activity_payload(
            club_id=str(club_id),
            actor_email=user.email,
            actor_role=role_resolution.role,
            action_type="admin_verified_updates_denied",
            entity_type="player_profile_update_subscription",
            entity_id="verified_updates",
            after_json={"source_client": "fastapi/nextjs", "reason": "insufficient_permission"},
            source_page=source,
            flagged_for_review=True,
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


def install_admin_verified_updates_routes(app, *, get_supabase_client) -> None:
    """Register guarded verified player-update request review routes."""

    @app.get("/admin/clubs/{club_id}/verified-updates/status")
    def get_admin_verified_updates_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_verified_updates_enabled() else None
        return build_admin_verified_updates_status(supabase, club_id=str(club_id))

    @app.get("/admin/clubs/{club_id}/verified-updates/requests")
    def get_admin_verified_update_requests(
        club_id: str,
        status: str = Query(default="pending"),
        limit: int = Query(default=100, ge=1, le=500),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_verified_updates_enabled():
            raise HTTPException(status_code=403, detail="Next verified update requests are disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_verified_updates_request_list")
        try:
            return list_admin_verified_update_requests(supabase, club_id=str(club_id), status=status, limit=limit)
        except Exception as exc:
            _handle(exc)

    @app.patch("/admin/clubs/{club_id}/verified-updates/requests/{subscription_id}")
    def patch_admin_verified_update_request(
        club_id: str,
        subscription_id: str,
        payload: AdminVerifiedUpdateRequestAction,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_verified_updates_enabled():
            raise HTTPException(status_code=403, detail="Next verified update requests are disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return update_admin_verified_update_request(
                supabase,
                club_id=str(club_id),
                subscription_id=str(subscription_id),
                action=payload.action,
                admin_note=payload.admin_note,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)
