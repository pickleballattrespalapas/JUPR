from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query

from jupr_app.domain.admin.roles import PERMISSION_VIEW_AUDIT_LOG, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_badge_diagnostics_service import (
    build_admin_badge_audit,
    build_admin_badge_debug,
    build_admin_badge_diagnostics_status,
    is_admin_badge_diagnostics_enabled,
    list_admin_badge_diagnostic_options,
)
from services.api.auth import authenticate_bearer, auth_header


def _resolve_badge_diagnostics_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(
        supabase=supabase,
        club_id=str(club_id),
        email=user.email,
        user_id=user.user_id,
        allowlist=set(),
    )
    if not has_permission(role_resolution.role, PERMISSION_VIEW_AUDIT_LOG):
        denied_payload = build_activity_payload(
            club_id=str(club_id),
            actor_email=user.email,
            actor_role=role_resolution.role,
            action_type="admin_badge_diagnostics_denied",
            entity_type="badge_diagnostics",
            entity_id="badge_diagnostics",
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


def install_admin_badge_diagnostics_routes(app, *, get_supabase_client) -> None:
    """Register read-only Badge Debug and Badge Audit routes for Next admin."""

    @app.get("/admin/clubs/{club_id}/badges/status")
    def get_admin_badge_diagnostics_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_badge_diagnostics_enabled() else None
        return build_admin_badge_diagnostics_status(supabase, club_id=str(club_id))

    @app.get("/admin/clubs/{club_id}/badges/options")
    def get_admin_badge_diagnostic_options(
        club_id: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_badge_diagnostics_enabled():
            raise HTTPException(status_code=403, detail="Next Badge Diagnostics is disabled.")
        supabase = get_supabase_client()
        _resolve_badge_diagnostics_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_badge_diagnostics_options")
        try:
            return list_admin_badge_diagnostic_options(supabase, club_id=str(club_id))
        except Exception as exc:
            _handle_common(exc)

    @app.get("/admin/clubs/{club_id}/badges/debug")
    def get_admin_badge_debug(
        club_id: str,
        player_id: int = Query(..., ge=1),
        badge_id: str = Query(...),
        league_id: str | None = Query(default=None),
        match_limit: int = Query(default=5000, ge=1, le=20000),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_badge_diagnostics_enabled():
            raise HTTPException(status_code=403, detail="Next Badge Diagnostics is disabled.")
        supabase = get_supabase_client()
        _resolve_badge_diagnostics_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_badge_debug")
        try:
            return build_admin_badge_debug(
                supabase,
                club_id=str(club_id),
                player_id=int(player_id),
                badge_id=str(badge_id),
                league_id=league_id,
                match_limit=int(match_limit),
            )
        except Exception as exc:
            _handle_common(exc)

    @app.get("/admin/clubs/{club_id}/badges/audit")
    def get_admin_badge_audit(
        club_id: str,
        league_id: str | None = Query(default=None),
        player_id: int | None = Query(default=None, ge=1),
        badge_id: str | None = Query(default=None),
        context_id: str | None = Query(default=None),
        since: str | None = Query(default=None),
        until: str | None = Query(default=None),
        include_non_live: bool = Query(default=False),
        include_revoked: bool = Query(default=False),
        match_limit: int = Query(default=5000, ge=1, le=20000),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_badge_diagnostics_enabled():
            raise HTTPException(status_code=403, detail="Next Badge Diagnostics is disabled.")
        supabase = get_supabase_client()
        _resolve_badge_diagnostics_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_badge_audit")
        try:
            return build_admin_badge_audit(
                supabase,
                club_id=str(club_id),
                league_id=league_id,
                player_id=player_id,
                badge_id=badge_id,
                context_id=context_id,
                since=since,
                until=until,
                include_non_live=bool(include_non_live),
                include_revoked=bool(include_revoked),
                match_limit=int(match_limit),
            )
        except Exception as exc:
            _handle_common(exc)
