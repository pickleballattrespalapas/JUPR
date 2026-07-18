from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel

from jupr_app.domain.admin.roles import PERMISSION_MANAGE_ROLES, PERMISSION_RUN_REPLAY, PERMISSION_VIEW_AUDIT_LOG, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_badge_diagnostics_service import (
    build_admin_badge_audit,
    build_admin_badge_debug,
    build_admin_badge_diagnostics_status,
    is_admin_badge_diagnostics_enabled,
    list_admin_badge_diagnostic_options,
    revoke_admin_player_badge,
    run_admin_badge_recompute,
    update_admin_badge_definition_state,
)
from services.api.auth import authenticate_bearer, auth_header


class BadgeRecomputeRequest(BaseModel):
    mode: str = "dry-run"
    league_id: str | None = None
    player_id: int | None = None
    badge_id: str | None = None
    context_id: str | None = None
    since: str | None = None
    until: str | None = None
    include_non_live: bool = False
    match_limit: int = 5000
    revoke_reason: str | None = None
    confirmation_text: str = ""
    source: str = "next_badge_recompute"


class BadgeRevokeRequest(BaseModel):
    player_badge_id: str | None = None
    player_id: int | None = None
    badge_id: str | None = None
    context_id: str | None = None
    revoke_reason: str | None = None
    confirmation_text: str = ""
    source: str = "next_badge_revoke"


class BadgeDefinitionStateRequest(BaseModel):
    expected_state: str
    target_state: str
    reason: str
    force: bool = False
    confirmation_text: str = ""
    source: str = "next_badge_definition_state"


def _resolve_badge_diagnostics_role_or_403(
    *,
    supabase: Any,
    club_id: str,
    authorization: str | None,
    source: str,
    permission: str = PERMISSION_VIEW_AUDIT_LOG,
) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(
        supabase=supabase,
        club_id=str(club_id),
        email=user.email,
        user_id=user.user_id,
        allowlist=set(),
    )
    if not role_resolution.assigned or not has_permission(role_resolution.role, permission):
        reason = "missing_club_assignment" if not role_resolution.assigned else "insufficient_permission"
        denied_payload = build_activity_payload(
            club_id=str(club_id),
            actor_email=user.email,
            actor_role=role_resolution.role,
            action_type="admin_badge_diagnostics_denied",
            entity_type="badge_diagnostics",
            entity_id="badge_diagnostics",
            after_json={"source_client": "fastapi/nextjs", "reason": reason, "required_permission": permission},
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
    """Register Badge Debug, Badge Audit, recompute, and revoke routes for Next admin."""

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

    @app.patch("/admin/clubs/{club_id}/badges/{badge_id}/state")
    def patch_admin_badge_definition_state(
        club_id: str,
        badge_id: str,
        payload: BadgeDefinitionStateRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_badge_diagnostics_enabled():
            raise HTTPException(status_code=403, detail="Next Badge Diagnostics is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_badge_diagnostics_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
            permission=PERMISSION_MANAGE_ROLES,
        )
        try:
            return update_admin_badge_definition_state(
                supabase,
                club_id=str(club_id),
                badge_id=str(badge_id),
                expected_state=payload.expected_state,
                target_state=payload.target_state,
                reason=payload.reason,
                force=payload.force,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
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

    @app.post("/admin/clubs/{club_id}/badges/recompute")
    def post_admin_badge_recompute(
        club_id: str,
        payload: BadgeRecomputeRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_badge_diagnostics_enabled():
            raise HTTPException(status_code=403, detail="Next Badge Diagnostics is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_badge_diagnostics_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
            permission=PERMISSION_RUN_REPLAY,
        )
        try:
            return run_admin_badge_recompute(
                supabase,
                club_id=str(club_id),
                mode=payload.mode,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                league_id=payload.league_id,
                player_id=payload.player_id,
                badge_id=payload.badge_id,
                context_id=payload.context_id,
                since=payload.since,
                until=payload.until,
                include_non_live=payload.include_non_live,
                match_limit=int(payload.match_limit or 5000),
                revoke_reason=payload.revoke_reason,
                source=payload.source,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.patch("/admin/clubs/{club_id}/badges/revoke")
    def patch_admin_badge_revoke(
        club_id: str,
        payload: BadgeRevokeRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_badge_diagnostics_enabled():
            raise HTTPException(status_code=403, detail="Next Badge Diagnostics is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_badge_diagnostics_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
            permission=PERMISSION_RUN_REPLAY,
        )
        try:
            return revoke_admin_player_badge(
                supabase,
                club_id=str(club_id),
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                player_badge_id=payload.player_badge_id,
                player_id=payload.player_id,
                badge_id=payload.badge_id,
                context_id=payload.context_id,
                revoke_reason=payload.revoke_reason,
                source=payload.source,
            )
        except Exception as exc:
            _handle_common(exc)
