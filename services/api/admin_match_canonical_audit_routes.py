from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import (
    PERMISSION_MANAGE_MATCHES,
    PERMISSION_VIEW_AUDIT_LOG,
    has_permission,
    resolve_admin_role,
)
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_match_canonical_audit_service import (
    build_admin_match_canonical_audit_status,
    build_admin_match_canonical_options,
    is_admin_match_canonical_audit_enabled,
    run_admin_match_canonical_audit,
    run_admin_match_canonical_normalize,
)
from services.api.auth import authenticate_bearer, auth_header


class MatchCanonicalAuditRequest(BaseModel):
    player_id: int
    league_id: str | None = None
    limit: int = Field(default=1200, ge=100, le=5000)
    source: str = "next_match_canonical_audit"


class MatchCanonicalNormalizeRequest(BaseModel):
    player_id: int
    match_ids: list[int] = Field(default_factory=list)
    dry_run: bool = True
    confirmation_text: str = ""
    source: str = "next_match_canonical_audit"


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
            action_type="admin_match_canonical_audit_denied",
            entity_type="match_canonical_audit",
            entity_id="match_canonical_audit",
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


def install_admin_match_canonical_audit_routes(app, *, get_supabase_client) -> None:
    """Register guarded Match Canonical Audit routes for Next admin."""

    @app.get("/admin/clubs/{club_id}/match-canonical-audit/status")
    def get_admin_match_canonical_audit_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_match_canonical_audit_enabled() else None
        return build_admin_match_canonical_audit_status(supabase, club_id=str(club_id))

    @app.get("/admin/clubs/{club_id}/match-canonical-audit/options")
    def get_admin_match_canonical_options(
        club_id: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_canonical_audit_enabled():
            raise HTTPException(status_code=403, detail="Next Match Canonical Audit is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_match_canonical_audit_options", permission=PERMISSION_VIEW_AUDIT_LOG)
        try:
            return build_admin_match_canonical_options(supabase, club_id=str(club_id))
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/match-canonical-audit/run")
    def post_admin_match_canonical_audit(
        club_id: str,
        payload: MatchCanonicalAuditRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_canonical_audit_enabled():
            raise HTTPException(status_code=403, detail="Next Match Canonical Audit is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source, permission=PERMISSION_VIEW_AUDIT_LOG)
        try:
            return run_admin_match_canonical_audit(
                supabase,
                club_id=str(club_id),
                player_id=int(payload.player_id),
                league_id=payload.league_id,
                limit=int(payload.limit),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/match-canonical-audit/normalize")
    def post_admin_match_canonical_normalize(
        club_id: str,
        payload: MatchCanonicalNormalizeRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_canonical_audit_enabled():
            raise HTTPException(status_code=403, detail="Next Match Canonical Audit is disabled.")
        supabase = get_supabase_client()
        required_permission = PERMISSION_VIEW_AUDIT_LOG if payload.dry_run else PERMISSION_MANAGE_MATCHES
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source, permission=required_permission)
        try:
            return run_admin_match_canonical_normalize(
                supabase,
                club_id=str(club_id),
                player_id=int(payload.player_id),
                match_ids=[int(mid) for mid in payload.match_ids],
                dry_run=bool(payload.dry_run),
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)
