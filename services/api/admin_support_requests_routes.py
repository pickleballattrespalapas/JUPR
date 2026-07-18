from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel

from jupr_app.domain.admin.roles import (
    PERMISSION_MANAGE_MATCHES,
    PERMISSION_MANAGE_PLAYERS,
    PERMISSION_MANAGE_SUBSCRIPTIONS,
    PERMISSION_MANAGE_TOURNAMENTS,
    PERMISSION_VIEW_AUDIT_LOG,
    has_permission,
    resolve_admin_role,
)
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_support_requests_service import (
    build_admin_support_requests_status,
    is_admin_support_requests_enabled,
    list_admin_support_requests,
    update_admin_support_request,
)
from services.api.auth import authenticate_bearer, auth_header


class AdminSupportRequestUpdatePayload(BaseModel):
    status: str
    admin_note: str | None = None
    confirmation_text: str = ""
    source: str = "next_admin_support_requests"


def _resolve_support_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str, write: bool = False) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(
        supabase=supabase,
        club_id=str(club_id),
        email=user.email,
        user_id=user.user_id,
        allowlist=set(),
    )
    read_allowed = has_permission(role_resolution.role, PERMISSION_VIEW_AUDIT_LOG) or any(
        has_permission(role_resolution.role, permission)
        for permission in (PERMISSION_MANAGE_PLAYERS, PERMISSION_MANAGE_MATCHES, PERMISSION_MANAGE_TOURNAMENTS, PERMISSION_MANAGE_SUBSCRIPTIONS)
    )
    write_allowed = any(
        has_permission(role_resolution.role, permission)
        for permission in (PERMISSION_MANAGE_PLAYERS, PERMISSION_MANAGE_MATCHES, PERMISSION_MANAGE_TOURNAMENTS, PERMISSION_MANAGE_SUBSCRIPTIONS)
    )
    if not role_resolution.assigned or not (write_allowed if write else read_allowed):
        reason = "missing_club_assignment" if not role_resolution.assigned else "insufficient_permission"
        denied_payload = build_activity_payload(
            club_id=str(club_id),
            actor_email=user.email,
            actor_role=role_resolution.role,
            action_type="admin_support_request_denied",
            entity_type="public_support_request",
            entity_id="support_requests",
            after_json={"source_client": "fastapi/nextjs", "reason": reason, "write": bool(write)},
            source_page=source,
            flagged_for_review=True,
        )
        write_admin_activity_log(supabase, denied_payload)
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


def install_admin_support_requests_routes(app, *, get_supabase_client) -> None:
    """Register guarded support/data-correction/privacy request review routes."""

    @app.get("/admin/clubs/{club_id}/support-requests/status")
    def get_admin_support_requests_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_support_requests_enabled() else None
        return build_admin_support_requests_status(supabase, club_id=str(club_id))

    @app.get("/admin/clubs/{club_id}/support-requests")
    def get_admin_support_requests(
        club_id: str,
        status: str | None = Query(default=None),
        request_type: str | None = Query(default=None),
        limit: int = Query(default=200, ge=1, le=500),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_support_requests_enabled():
            raise HTTPException(status_code=403, detail="Next Support Requests is disabled.")
        supabase = get_supabase_client()
        _resolve_support_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_admin_support_requests_list")
        try:
            return list_admin_support_requests(supabase, club_id=str(club_id), status=status, request_type=request_type, limit=limit)
        except Exception as exc:
            _handle(exc)

    @app.patch("/admin/clubs/{club_id}/support-requests/{request_id}")
    def patch_admin_support_request(
        club_id: str,
        request_id: str,
        payload: AdminSupportRequestUpdatePayload,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_support_requests_enabled():
            raise HTTPException(status_code=403, detail="Next Support Requests is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_support_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
            write=True,
        )
        try:
            return update_admin_support_request(
                supabase,
                club_id=str(club_id),
                request_id=str(request_id),
                status=payload.status,
                admin_note=payload.admin_note,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)
