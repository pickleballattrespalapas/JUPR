from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import (
    PERMISSION_MANAGE_ROLES,
    PERMISSION_RUN_REPLAY,
    PERMISSION_VIEW_AUDIT_LOG,
    has_permission,
    resolve_admin_role,
)
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_tools_service import (
    apply_admin_tournament_match_backfill,
    build_admin_rating_report,
    build_admin_tools_overview,
    build_admin_tools_status,
    build_admin_tournament_match_backfill_preview,
    build_admin_worker_status,
    is_admin_tools_enabled,
    run_admin_badge_queue_worker,
    run_admin_badge_recompute_job,
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


class AdminBadgeQueueRequest(BaseModel):
    mode: str = "batch"
    max_jobs: int = 10
    time_budget_seconds: float = 5.0
    confirmation_text: str = ""
    source: str = "next_admin_tools_workers"


class AdminBadgeRecomputeRequest(BaseModel):
    mode: str = "dry-run"
    player_id: int | None = None
    badge_id: str | None = None
    league_id: str | None = None
    context_id: str | None = None
    since: str | None = None
    until: str | None = None
    include_non_live: bool = False
    allow_strict_global: bool = False
    match_limit: int = 5000
    confirmation_text: str = ""
    source: str = "next_admin_tools_badge_recompute"


class AdminTournamentMatchBackfillApplyRequest(BaseModel):
    game_ids: list[str] = Field(default_factory=list)
    preview_fingerprint: str = ""
    preview_limit: int = Field(default=500, ge=1, le=1000)
    confirmation_text: str = ""
    source: str = "next_admin_tools_tournament_match_backfill"


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

    @app.get("/admin/clubs/{club_id}/tools/reports/ratings")
    def get_admin_tools_rating_report(
        club_id: str,
        league: str = Query(default="OVERALL", min_length=1, max_length=160),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tools_enabled():
            raise HTTPException(status_code=403, detail="Next Admin Tools are disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_admin_tools_rating_report",
            permission=PERMISSION_VIEW_AUDIT_LOG,
        )
        try:
            return build_admin_rating_report(
                supabase,
                club_id=str(club_id),
                league_name=str(league),
            )
        except Exception as exc:
            _handle(exc)

    @app.get("/admin/clubs/{club_id}/tools/workers/status")
    def get_admin_tools_worker_status(
        club_id: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tools_enabled():
            raise HTTPException(status_code=403, detail="Next Admin Tools are disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_admin_tools_workers_status", permission=PERMISSION_VIEW_AUDIT_LOG)
        try:
            return build_admin_worker_status(supabase, club_id=str(club_id))
        except Exception as exc:
            _handle(exc)

    @app.get("/admin/clubs/{club_id}/tools/backfills/tournament-matches/preview")
    def get_admin_tools_tournament_match_backfill_preview(
        club_id: str,
        limit: int = Query(default=500, ge=1, le=1000),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tools_enabled():
            raise HTTPException(status_code=403, detail="Next Admin Tools are disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_admin_tools_tournament_match_backfill_preview",
            permission=PERMISSION_VIEW_AUDIT_LOG,
        )
        try:
            return build_admin_tournament_match_backfill_preview(
                supabase,
                club_id=str(club_id),
                candidate_limit=int(limit),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tools/backfills/tournament-matches/apply")
    def post_admin_tools_tournament_match_backfill_apply(
        club_id: str,
        payload: AdminTournamentMatchBackfillApplyRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tools_enabled():
            raise HTTPException(status_code=403, detail="Next Admin Tools are disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
            permission=PERMISSION_RUN_REPLAY,
        )
        try:
            return apply_admin_tournament_match_backfill(
                supabase,
                club_id=str(club_id),
                game_ids=payload.game_ids,
                preview_fingerprint=payload.preview_fingerprint,
                preview_limit=payload.preview_limit,
                confirmation_text=payload.confirmation_text,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
            )
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

    @app.post("/admin/clubs/{club_id}/tools/workers/badge-queue")
    def post_admin_tools_badge_queue_worker(
        club_id: str,
        payload: AdminBadgeQueueRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tools_enabled():
            raise HTTPException(status_code=403, detail="Next Admin Tools are disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source, permission=PERMISSION_RUN_REPLAY)
        try:
            return run_admin_badge_queue_worker(
                supabase,
                club_id=str(club_id),
                mode=payload.mode,
                max_jobs=payload.max_jobs,
                time_budget_seconds=payload.time_budget_seconds,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tools/workers/badge-recompute")
    def post_admin_tools_badge_recompute(
        club_id: str,
        payload: AdminBadgeRecomputeRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tools_enabled():
            raise HTTPException(status_code=403, detail="Next Admin Tools are disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source, permission=PERMISSION_RUN_REPLAY)
        try:
            return run_admin_badge_recompute_job(
                supabase,
                club_id=str(club_id),
                mode=payload.mode,
                player_id=payload.player_id,
                badge_id=payload.badge_id,
                league_id=payload.league_id,
                context_id=payload.context_id,
                since=payload.since,
                until=payload.until,
                include_non_live=payload.include_non_live,
                allow_strict_global=payload.allow_strict_global,
                match_limit=payload.match_limit,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)
