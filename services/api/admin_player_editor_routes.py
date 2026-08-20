from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import PERMISSION_MANAGE_PLAYERS, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_player_editor_service import (
    PlayerEditorConflictError,
    build_admin_player_editor_status,
    create_admin_player_editor_player,
    get_admin_player_editor_detail,
    is_admin_player_editor_enabled,
    list_admin_player_editor_players,
    reconcile_admin_player_editor_operation,
    update_admin_player_editor_player,
)
from jupr_app.services.admin_player_league_rating_service import update_admin_player_editor_league_rating
from jupr_app.services.admin_player_merge_service import (
    PlayerMergeConflictError,
    PlayerMergeSetupError,
    build_admin_player_merge_preview,
    compensate_admin_player_merge,
    execute_admin_player_merge,
    get_admin_player_merge_operation,
    verify_admin_player_merge_replay,
)
from jupr_app.services.admin_player_social_identity_service import (
    auto_link_admin_player_social_identities,
    list_admin_player_social_identities,
    update_admin_player_social_identity,
)
from jupr_app.services.admin_guarded_write_service import (
    GuardedWriteRecoveryRequired,
    get_guarded_operation,
)
from services.api.auth import authenticate_bearer, auth_header


class AdminPlayerEditorCreateRequest(BaseModel):
    name: str
    starting_jupr: float = Field(ge=1.0, le=7.0)
    idempotency_key: str = Field(
        min_length=8,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]+$",
    )
    source: str = "next_player_editor"


class AdminPlayerEditorUpdateRequest(BaseModel):
    name: str | None = None
    rating_jupr: float | None = Field(default=None, ge=1.0, le=7.0)
    starting_jupr: float | None = Field(default=None, ge=1.0, le=7.0)
    active: bool | None = None
    expected_state_fingerprint: str = Field(min_length=64, max_length=64)
    idempotency_key: str = Field(
        min_length=8,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]+$",
    )
    source: str = "next_player_editor"


class AdminPlayerEditorLeagueRatingUpdateRequest(BaseModel):
    rating_jupr: float | None = Field(default=None, ge=1.0, le=7.0)
    starting_jupr: float | None = Field(default=None, ge=1.0, le=7.0)
    is_active: bool | None = None
    expected_state_fingerprint: str = Field(min_length=64, max_length=64)
    idempotency_key: str = Field(
        min_length=8,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]+$",
    )
    confirmation_text: str = ""
    source: str = "next_player_editor_league_rating"


class AdminPlayerEditorOperationReconcileRequest(BaseModel):
    confirmation_text: str = Field(default="", max_length=80)
    source: str = "next_player_editor_operation_reconcile"


class AdminPlayerSocialIdentityUpdateRequest(BaseModel):
    linked_player_id: int | None = None
    display_name: str | None = None
    confirmation_text: str = ""
    source: str = "next_player_editor_social_identity"


class AdminPlayerSocialIdentityAutoLinkRequest(BaseModel):
    confirmation_text: str = ""
    source: str = "next_player_editor_social_auto_link"


class AdminPlayerMergePreviewRequest(BaseModel):
    source_player_id: int
    target_player_id: int
    source: str = "next_player_editor_merge_preview"


class AdminPlayerMergeExecuteRequest(BaseModel):
    source_player_id: int
    target_player_id: int
    preview_fingerprint: str
    operation_id: str | None = None
    confirmation_text: str = ""
    source: str = "next_player_editor_merge"


class AdminPlayerMergeCompensateRequest(BaseModel):
    confirmation_text: str = ""
    source: str = "next_player_editor_merge_compensation"


class AdminPlayerMergeReplayEvidenceRequest(BaseModel):
    replay_job_id: str
    confirmation_text: str = ""
    source: str = "next_player_editor_merge_replay_evidence"


def _dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def _resolve_player_editor_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(supabase=supabase, club_id=str(club_id), email=user.email, user_id=user.user_id, allowlist=set())
    if not has_permission(role_resolution.role, PERMISSION_MANAGE_PLAYERS):
        denied_payload = build_activity_payload(club_id=str(club_id), actor_email=user.email, actor_role=role_resolution.role, action_type="admin_player_editor_denied", entity_type="players", entity_id="player_editor", after_json={"source_client": "fastapi/nextjs", "reason": "insufficient_permission"}, source_page=source, flagged_for_review=True)
        write_admin_activity_log(supabase, denied_payload)
        raise HTTPException(status_code=403, detail="insufficient permission")
    return user.email, role_resolution.role


def _handle_common(exc: Exception) -> None:
    if isinstance(exc, GuardedWriteRecoveryRequired):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "RECOVERY_REQUIRED",
                "kind": "uncertain",
                "message": str(exc),
                "operation_key": exc.operation_key,
                "recovery_required": True,
            },
        ) from exc
    if isinstance(exc, PlayerEditorConflictError):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "STALE_VERSION",
                "kind": "conflict",
                "message": str(exc),
                "operation_key": exc.operation_key,
                "recovery_required": False,
            },
        ) from exc
    if isinstance(exc, PermissionError): raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, PlayerMergeConflictError): raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, ValueError): raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, PlayerMergeSetupError): raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, RuntimeError): raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise exc


def install_admin_player_editor_routes(app, *, get_supabase_client) -> None:
    """Register guarded Player Editor routes for the Next admin pilot."""

    @app.get("/admin/clubs/{club_id}/players/editor/status")
    def get_admin_player_editor_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_player_editor_enabled() else None
        return build_admin_player_editor_status(supabase, club_id=str(club_id))

    @app.post("/admin/clubs/{club_id}/players/editor/merge/preview")
    def post_admin_player_merge_preview(club_id: str, payload: AdminPlayerMergePreviewRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try: return build_admin_player_merge_preview(supabase, club_id=str(club_id), source_player_id=payload.source_player_id, target_player_id=payload.target_player_id)
        except Exception as exc: _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/players/editor/merge")
    def post_admin_player_merge_execute(club_id: str, payload: AdminPlayerMergeExecuteRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); actor_email, actor_role = _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try: return execute_admin_player_merge(supabase, club_id=str(club_id), source_player_id=payload.source_player_id, target_player_id=payload.target_player_id, actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, preview_fingerprint=payload.preview_fingerprint, operation_id=payload.operation_id, source=payload.source)
        except Exception as exc: _handle_common(exc)

    @app.get("/admin/clubs/{club_id}/players/editor/merge/{operation_id}")
    def get_admin_player_merge_recovery(club_id: str, operation_id: str, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_player_editor_merge_recovery")
        try: return get_admin_player_merge_operation(supabase, club_id=str(club_id), operation_id=operation_id)
        except Exception as exc: _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/players/editor/merge/{operation_id}/compensate")
    def post_admin_player_merge_compensation(club_id: str, operation_id: str, payload: AdminPlayerMergeCompensateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); actor_email, actor_role = _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try: return compensate_admin_player_merge(supabase, club_id=str(club_id), operation_id=operation_id, actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source)
        except Exception as exc: _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/players/editor/merge/{operation_id}/replay-evidence")
    def post_admin_player_merge_replay_evidence(club_id: str, operation_id: str, payload: AdminPlayerMergeReplayEvidenceRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); actor_email, actor_role = _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try: return verify_admin_player_merge_replay(supabase, club_id=str(club_id), operation_id=operation_id, replay_job_id=payload.replay_job_id, actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source)
        except Exception as exc: _handle_common(exc)

    @app.get("/admin/clubs/{club_id}/players/editor/social-identities")
    def get_admin_player_social_identities(club_id: str, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_player_editor_social_identity_list")
        try: return list_admin_player_social_identities(supabase, club_id=str(club_id))
        except Exception as exc: _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/players/editor/social-identities/auto-link")
    def post_admin_player_social_auto_link(club_id: str, payload: AdminPlayerSocialIdentityAutoLinkRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); actor_email, actor_role = _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try: return auto_link_admin_player_social_identities(supabase, club_id=str(club_id), actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source)
        except Exception as exc: _handle_common(exc)

    @app.patch("/admin/clubs/{club_id}/players/editor/social-identities/{club_person_id}")
    def patch_admin_player_social_identity(club_id: str, club_person_id: str, payload: AdminPlayerSocialIdentityUpdateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); actor_email, actor_role = _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try: return update_admin_player_social_identity(supabase, club_id=str(club_id), club_person_id=str(club_person_id), linked_player_id=payload.linked_player_id, display_name=payload.display_name, confirmation_text=payload.confirmation_text, actor_email=actor_email, actor_role=actor_role, source=payload.source)
        except Exception as exc: _handle_common(exc)

    @app.get("/admin/clubs/{club_id}/players/editor/players")
    def get_admin_player_editor_players(club_id: str, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_player_editor_list")
        try: return list_admin_player_editor_players(supabase, club_id=str(club_id))
        except Exception as exc: _handle_common(exc)

    @app.get("/admin/clubs/{club_id}/players/editor/players/{player_id}")
    def get_admin_player_editor_player_detail(club_id: str, player_id: int, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_player_editor_detail")
        try: return get_admin_player_editor_detail(supabase, club_id=str(club_id), player_id=int(player_id))
        except ValueError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc: _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/players/editor/players")
    def post_admin_player_editor_player(club_id: str, payload: AdminPlayerEditorCreateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); actor_email, actor_role = _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try: return create_admin_player_editor_player(supabase, club_id=str(club_id), name=payload.name, starting_jupr=payload.starting_jupr, actor_email=actor_email, actor_role=actor_role, idempotency_key=payload.idempotency_key, source=payload.source)
        except Exception as exc: _handle_common(exc)

    @app.patch("/admin/clubs/{club_id}/players/editor/players/{player_id}")
    def patch_admin_player_editor_player(club_id: str, player_id: int, payload: AdminPlayerEditorUpdateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); actor_email, actor_role = _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        patch = _dump_model(payload); source = str(patch.pop("source", payload.source)); expected_state_fingerprint = str(patch.pop("expected_state_fingerprint", payload.expected_state_fingerprint)); idempotency_key = str(patch.pop("idempotency_key", payload.idempotency_key))
        try: return update_admin_player_editor_player(supabase, club_id=str(club_id), player_id=int(player_id), patch=patch, actor_email=actor_email, actor_role=actor_role, expected_state_fingerprint=expected_state_fingerprint, idempotency_key=idempotency_key, source=source)
        except Exception as exc: _handle_common(exc)

    @app.patch("/admin/clubs/{club_id}/players/editor/players/{player_id}/league-ratings/{league_rating_id}")
    def patch_admin_player_editor_league_rating(club_id: str, player_id: int, league_rating_id: int, payload: AdminPlayerEditorLeagueRatingUpdateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); actor_email, actor_role = _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        patch = _dump_model(payload); source = str(patch.pop("source", payload.source)); confirmation_text = str(patch.pop("confirmation_text", payload.confirmation_text)); expected_state_fingerprint = str(patch.pop("expected_state_fingerprint", payload.expected_state_fingerprint)); idempotency_key = str(patch.pop("idempotency_key", payload.idempotency_key))
        try: return update_admin_player_editor_league_rating(supabase, club_id=str(club_id), player_id=int(player_id), league_rating_id=int(league_rating_id), patch=patch, actor_email=actor_email, actor_role=actor_role, expected_state_fingerprint=expected_state_fingerprint, idempotency_key=idempotency_key, confirmation_text=confirmation_text, source=source)
        except Exception as exc: _handle_common(exc)

    @app.get("/admin/clubs/{club_id}/players/editor/operations/{operation_key}")
    def get_admin_player_editor_operation(club_id: str, operation_key: str, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_player_editor_operation")
        workflows = ("player_editor_create", "player_editor_update", "player_editor_league_rating_update")
        operation = next((row for workflow in workflows if (row := get_guarded_operation(supabase, club_id=str(club_id), workflow=workflow, operation_key=str(operation_key))) is not None), None)
        if operation is None: raise HTTPException(status_code=404, detail="Player Editor operation was not found.")
        result_json = operation.get("result_json") or {}; error_text = operation.get("error_text")
        return {"ok": True, "operation_key": operation.get("operation_key"), "workflow": operation.get("workflow"), "status": operation.get("status"), "result_json": result_json, "error_text": error_text, "result": result_json, "error": error_text, "recovery_required": operation.get("status") in {"intent_recorded", "recovery_required"}, "updated_at": operation.get("updated_at")}

    @app.post("/admin/clubs/{club_id}/players/editor/operations/{operation_key}/reconcile")
    def post_admin_player_editor_operation_reconcile(club_id: str, operation_key: str, payload: AdminPlayerEditorOperationReconcileRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_player_editor_enabled(): raise HTTPException(status_code=403, detail="Next Player Editor is disabled.")
        supabase = get_supabase_client(); actor_email, actor_role = _resolve_player_editor_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try: return reconcile_admin_player_editor_operation(supabase, club_id=str(club_id), operation_key=str(operation_key), confirmation_text=payload.confirmation_text, actor_email=actor_email, actor_role=actor_role, source=payload.source)
        except Exception as exc: _handle_common(exc)
