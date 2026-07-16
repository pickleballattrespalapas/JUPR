from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel

from jupr_app.domain.admin.roles import (
    PERMISSION_MANAGE_ROLES,
    PERMISSION_VIEW_AUDIT_LOG,
    has_permission,
    resolve_admin_role,
)
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_tools_service import (
    build_admin_tools_overview,
    build_admin_tools_status,
    is_admin_tools_enabled,
    update_admin_role_assignment,
)
from services.api.auth import authenticate_bearer, auth_header


class AdminRoleAssignmentRequest(BaseModel):
    email: str
    role: str = "read_only"
    user_id: str | None = None
    action: str = "upsert"
    confirmation_text: str = ""
    source: str = "next_admin_tools_roles"


def _resolve_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str, permission: str) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(
        supabase=supabase,
        club_id=str(club_id),
        email=user.email,
        user_id=user.user_id,
        allowlist=set(),
    )
    if not has_permission(role_resolution.role, permission):
        denied_payload = build_activity_payload(
            club_id=str(club_id),
            actor_email=user.email,
            actor_role=role_resolution.role,
            action_type="admin_tools_denied",
            entity_type="admin_tools",
            entity_id="admin_tools",
            after_json={"source_client": "fastapi/nextjs", "reason": "insufficient_permission", "required_permission": permission},
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


def install_admin_tools_routes(app, *, get_supabase_client) -> None:
    """Register guarded Admin Tools routes for the Next admin pilot."""

    @app.get("/admin/clubs/{club_id}/tools/status")
    def get_admin_tools_status(club_id: str) -> dict[str, Any]:
        return build_admin_tools_status(club_id=str(club_id))

    @app.get("/admin/clubs/{club_id}/tools/overview")
    def get_admin_tools_overview(
        club_id: str,
        flagged_only: bool = Query(default=False),
        limit: int = Query(default=200, ge=1, le=500),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tools_enabled():
            raise HTTPException(status_code=403, detail="Next Admin Tools are disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_admin_tools_overview", permission=PERMISSION_VIEW_AUDIT_LOG)
        try:
            return build_admin_tools_overview(supabase, club_id=str(club_id), include_flagged_only=bool(flagged_only), activity_limit=int(limit))
        except Exception as exc:
            _handle(exc)

    @app.patch("/admin/clubs/{club_id}/tools/roles")
    def patch_admin_tools_role_assignment(
        club_id: str,
        payload: AdminRoleAssignmentRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tools_enabled():
            raise HTTPException(status_code=403, detail="Next Admin Tools are disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source, permission=PERMISSION_MANAGE_ROLES)
        try:
            return update_admin_role_assignment(
                supabase,
                club_id=str(club_id),
                target_email=payload.email,
                target_role=payload.role,
                user_id=payload.user_id,
                action=payload.action,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)
