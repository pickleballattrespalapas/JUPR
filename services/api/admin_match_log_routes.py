from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd
from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import (
    PERMISSION_DELETE_MATCHES,
    PERMISSION_MANAGE_MATCHES,
    has_permission,
    resolve_admin_role,
)
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.live_social import (
    SocialTablesNotInstalledError,
    delete_social_matches,
    list_social_match_log_rows,
    update_social_match_row,
)
from jupr_app.domain.match_delete import delete_rated_matches_with_replay
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


class AdminMatchLogExcludeRequest(BaseModel):
    match_ids: list[int] = Field(default_factory=list)
    confirmation_text: str = ""
    note: str | None = None
    source: str = "next_match_log_bulk_exclude"


class AdminMatchLogSocialUpdateRequest(BaseModel):
    event_name: str | None = None
    played_on: str | None = None
    round_number: int | None = None
    court_number: int | None = None
    mini_round_number: int | None = None
    score_t1: int | None = Field(default=None, ge=0)
    score_t2: int | None = Field(default=None, ge=0)
    source: str = "next_match_log_social_editor"


class AdminMatchLogSocialDeleteRequest(BaseModel):
    social_match_ids: list[str] = Field(default_factory=list)
    confirmation_text: str = ""
    source: str = "next_match_log_social_editor"


def _dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _clean_text(value: Any, *, limit: int = 200) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _dataframe_rows(df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        rows.append({str(key): _json_safe(value) for key, value in row.items()})
    return rows


def _list_match_log_player_options(supabase: Any, *, club_id: str) -> dict[str, Any]:
    try:
        rows = (
            supabase.table("players")
            .select("id,name")
            .eq("club_id", str(club_id))
            .order("name", desc=False)
            .execute()
            .data
            or []
        )
    except Exception as exc:  # noqa: BLE001 - surface schema/configuration issues to the operator
        raise RuntimeError(f"Could not load Match Log player options: {exc.__class__.__name__}") from exc

    players: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for row in rows:
        player_id = _safe_int(dict(row).get("id") if isinstance(row, dict) else None)
        if player_id is None or int(player_id) in seen_ids:
            continue
        seen_ids.add(int(player_id))
        name = _clean_text(dict(row).get("name") if isinstance(row, dict) else None, limit=160) or f"Player {int(player_id)}"
        players.append({"id": int(player_id), "name": name, "label": f"{name} (#{int(player_id)})"})
    players = sorted(players, key=lambda player: (str(player.get("name") or "").lower(), int(player.get("id") or 0)))
    return {"ok": True, "mode": "match_log_player_options", "players": players, "count": len(players)}


def _fetch_league_metadata_df(supabase: Any, *, club_id: str) -> pd.DataFrame:
    try:
        rows = (
            supabase.table("leagues_metadata")
            .select("league_name,k_factor,is_active,status")
            .eq("club_id", str(club_id))
            .execute()
            .data
            or []
        )
    except Exception:
        return pd.DataFrame(columns=["league_name", "k_factor", "is_active", "status"])
    return pd.DataFrame([dict(row) for row in rows])


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

    @app.get("/admin/clubs/{club_id}/match-log/player-options")
    def get_admin_match_log_player_options(
        club_id: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_log_enabled():
            raise HTTPException(status_code=403, detail="Next Match Log is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            permission=PERMISSION_MANAGE_MATCHES,
            source="next_match_log_player_options",
        )
        try:
            return _list_match_log_player_options(supabase, club_id=str(club_id))
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/admin/clubs/{club_id}/match-log/social")
    def get_admin_match_log_social_rows(
        club_id: str,
        limit: int = Query(default=500, ge=1, le=1000),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_log_enabled():
            raise HTTPException(status_code=403, detail="Next Match Log is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            permission=PERMISSION_MANAGE_MATCHES,
            source="next_match_log_social_list",
        )
        try:
            rows = _dataframe_rows(list_social_match_log_rows(supabase, club_id=str(club_id), limit=int(limit)))
            return {"ok": True, "mode": "social_match_log_rows", "rows": rows, "count": len(rows), "warnings": []}
        except SocialTablesNotInstalledError as exc:
            return {"ok": True, "mode": "social_match_log_unavailable", "rows": [], "count": 0, "warnings": [str(exc)]}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Unable to load Club Social Match Log rows: {exc.__class__.__name__}") from exc

    @app.patch("/admin/clubs/{club_id}/match-log/social/{social_match_id}")
    def patch_admin_match_log_social_row(
        club_id: str,
        social_match_id: str,
        payload: AdminMatchLogSocialUpdateRequest,
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
        patch = _dump_model(payload)
        source = str(patch.pop("source", payload.source))
        if not patch:
            raise HTTPException(status_code=400, detail="No Club Social changes provided.")
        try:
            result = update_social_match_row(
                supabase,
                club_id=str(club_id),
                social_match_id=str(social_match_id),
                patch=patch,
            )
        except SocialTablesNotInstalledError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        audit_result = write_admin_activity_log(
            supabase,
            build_activity_payload(
                club_id=str(club_id),
                actor_email=actor_email,
                actor_role=actor_role,
                action_type="social_match_log_update",
                entity_type="live_event_match",
                entity_id=str(social_match_id),
                after_json={"source_client": "fastapi/nextjs", "source_page": source, "patch": patch, "result": result},
                source_page=source,
                flagged_for_review=True,
            ),
        )
        warnings = [audit_result.warning] if audit_result.warning else []
        return {"ok": True, "mode": "social_match_updated", "social_match_id": str(social_match_id), "result": result, "warnings": warnings}

    @app.post("/admin/clubs/{club_id}/match-log/social/delete")
    def post_admin_match_log_social_delete(
        club_id: str,
        payload: AdminMatchLogSocialDeleteRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_log_apply_enabled():
            raise HTTPException(status_code=403, detail="Next Match Log apply is disabled.")
        if str(payload.confirmation_text or "").strip().upper() != "DELETE":
            raise HTTPException(status_code=400, detail="Type DELETE to confirm Club Social row deletion.")
        social_ids = [str(value).strip() for value in (payload.social_match_ids or []) if str(value).strip()]
        if not social_ids:
            raise HTTPException(status_code=400, detail="Select at least one Club Social row to delete.")
        if len(social_ids) > 100:
            raise HTTPException(status_code=400, detail="No more than 100 Club Social rows can be deleted at once.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            permission=PERMISSION_DELETE_MATCHES,
            source=payload.source,
        )
        try:
            deleted = delete_social_matches(supabase, club_id=str(club_id), social_match_ids=social_ids)
        except SocialTablesNotInstalledError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        audit_result = write_admin_activity_log(
            supabase,
            build_activity_payload(
                club_id=str(club_id),
                actor_email=actor_email,
                actor_role=actor_role,
                action_type="social_match_log_delete",
                entity_type="live_event_match",
                entity_id="bulk",
                after_json={"source_client": "fastapi/nextjs", "source_page": payload.source, "requested_ids": social_ids, "deleted_count": deleted},
                source_page=payload.source,
                flagged_for_review=True,
            ),
        )
        warnings = [audit_result.warning] if audit_result.warning else []
        return {"ok": True, "mode": "social_matches_deleted", "deleted_count": deleted, "requested_ids": social_ids, "warnings": warnings}

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

    @app.post("/admin/clubs/{club_id}/match-log/exclude")
    def post_admin_match_log_exclude_matches(
        club_id: str,
        payload: AdminMatchLogExcludeRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_log_apply_enabled():
            raise HTTPException(status_code=403, detail="Next Match Log apply is disabled.")
        normalized_confirmation = str(payload.confirmation_text or "").strip().upper()
        if normalized_confirmation != "DELETE":
            raise HTTPException(status_code=400, detail="Type DELETE to confirm rated match exclusion.")
        match_ids = sorted({int(match_id) for match_id in (payload.match_ids or []) if _safe_int(match_id) is not None})
        if not match_ids:
            raise HTTPException(status_code=400, detail="Select at least one match to exclude.")
        if len(match_ids) > 100:
            raise HTTPException(status_code=400, detail="No more than 100 matches can be excluded at once.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            permission=PERMISSION_DELETE_MATCHES,
            source=payload.source,
        )
        try:
            result = delete_rated_matches_with_replay(
                supabase=supabase,
                club_id=str(club_id),
                match_ids=match_ids,
                df_meta=_fetch_league_metadata_df(supabase, club_id=str(club_id)),
                actor=actor_email,
                actor_role=actor_role,
                source=payload.source,
                note=payload.note,
                flagged_for_review=True,
            )
            return {"ok": True, "mode": "matches_excluded", **result}
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
