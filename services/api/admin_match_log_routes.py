from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import (
    PERMISSION_DELETE_MATCHES,
    PERMISSION_MANAGE_MATCHES,
    has_permission,
    resolve_admin_role,
)
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_match_log_service import (
    apply_admin_match_log_duplicate_cleanup,
    apply_admin_match_log_edits,
    build_admin_match_log,
    is_admin_match_log_apply_enabled,
    is_admin_match_log_enabled,
    resolve_admin_match_log_duplicate_false_positive,
)
from services.api.auth import authenticate_bearer, auth_header


class AdminMatchLogEditRequest(BaseModel):
    patches: list[dict[str, Any]] = Field(default_factory=list)
    confirmation_text: str = ""
    correction_note: str | None = None
    source: str = "next_match_log"


class AdminMatchLogDuplicateCleanupRequest(BaseModel):
    delete_ids: list[int] = Field(default_factory=list)
    confirmation_text: str = ""
    source: str = "next_match_log_duplicate_cleanup"


class AdminMatchLogDuplicateResolutionRequest(BaseModel):
    match_ids: list[int] = Field(default_factory=list)
    dup_key: str | None = None
    reason: str = ""
    confirmation_text: str = ""
    source: str = "next_match_log_duplicate_no_issue"


def _resolve_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, permission: str, source: str) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    try:
        role_resolution = resolve_admin_role(
            supabase=supabase,
            club_id=str(club_id),
            email=user.email,
            user_id=user.user_id,
            allowlist=set(),
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - expose pilot auth configuration errors without opaque 500s
        raise HTTPException(status_code=503, detail=f"Admin role lookup failed: {exc.__class__.__name__}") from exc
    if not has_permission(role_resolution.role, permission):
        denied_payload = build_activity_payload(
            club_id=str(club_id),
            actor_email=user.email,
            actor_role=role_resolution.role,
            action_type="match_log_write_denied",
            entity_type="match",
            entity_id="bulk",
            after_json={"source_client": "fastapi/nextjs", "reason": "insufficient_permission", "required_permission": permission},
            source_page=source,
            flagged_for_review=True,
        )
        write_admin_activity_log(supabase, denied_payload)
        raise HTTPException(status_code=403, detail="insufficient permission")
    return user.email, role_resolution.role


def install_admin_match_log_routes(app, *, get_supabase_client) -> None:
    """Register Match Log planning and guarded write routes for the Next admin pilot."""

    @app.get("/admin/clubs/{club_id}/match-log")
    def get_admin_match_log(
        club_id: str,
        filter_type: str = Query(default="All", alias="filter"),
        match_id: int | None = Query(default=None),
        league: str | None = Query(default=None),
        week_tag: str | None = Query(default=None),
        start_date: str | None = Query(default=None),
        end_date: str | None = Query(default=None),
        limit: int = Query(default=500, ge=1, le=1000),
    ) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_match_log_enabled() else None
        return build_admin_match_log(
            supabase,
            club_id=str(club_id),
            filter_type=filter_type,
            match_id=match_id,
            league=league,
            week_tag=week_tag,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )

    @app.patch("/admin/clubs/{club_id}/match-log/edits")
    def patch_admin_match_log_edits(
        club_id: str,
        payload: AdminMatchLogEditRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_log_apply_enabled():
            raise HTTPException(status_code=403, detail="Next Match Log apply is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            permission=PERMISSION_MANAGE_MATCHES,
            source=payload.source,
        )
        try:
            return apply_admin_match_log_edits(
                supabase,
                club_id=str(club_id),
                patches=payload.patches,
                actor_email=actor_email,
                actor_role=actor_role,
                correction_note=payload.correction_note,
                source=payload.source,
                confirmation_text=payload.confirmation_text,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/admin/clubs/{club_id}/match-log/duplicates/cleanup")
    def post_admin_match_log_duplicate_cleanup(
        club_id: str,
        payload: AdminMatchLogDuplicateCleanupRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_log_apply_enabled():
            raise HTTPException(status_code=403, detail="Next Match Log apply is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            permission=PERMISSION_DELETE_MATCHES,
            source=payload.source,
        )
        try:
            return apply_admin_match_log_duplicate_cleanup(
                supabase,
                club_id=str(club_id),
                delete_ids=payload.delete_ids,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                confirmation_text=payload.confirmation_text,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/admin/clubs/{club_id}/match-log/duplicates/resolve")
    def post_admin_match_log_duplicate_resolution(
        club_id: str,
        payload: AdminMatchLogDuplicateResolutionRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_log_apply_enabled():
            raise HTTPException(status_code=403, detail="Next Match Log apply is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            permission=PERMISSION_MANAGE_MATCHES,
            source=payload.source,
        )
        try:
            return resolve_admin_match_log_duplicate_false_positive(
                supabase,
                club_id=str(club_id),
                match_ids=payload.match_ids,
                dup_key=payload.dup_key,
                actor_email=actor_email,
                actor_role=actor_role,
                reason=payload.reason,
                source=payload.source,
                confirmation_text=payload.confirmation_text,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
