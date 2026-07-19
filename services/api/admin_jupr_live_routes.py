from __future__ import annotations

import os
from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import PERMISSION_ENTER_SCORES, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_jupr_live_service import (
    CONFIRM_ADVANCE,
    CONFIRM_CREATE,
    CONFIRM_PUBLISH,
    CONFIRM_SCORES,
    CONFIRM_STATUS,
    JUPR_LIVE_WRITE_FLAG,
    admin_jupr_live_pending_operation_key,
    admin_jupr_live_publish_contexts,
    advance_admin_jupr_live_league_round,
    build_admin_jupr_live_status,
    create_admin_jupr_live_session,
    get_admin_jupr_live_session,
    is_admin_jupr_live_enabled,
    list_admin_jupr_live_sessions,
    publish_admin_jupr_live_matches,
    update_admin_jupr_live_scores,
    update_admin_jupr_live_session_status,
)
from jupr_app.services.admin_live_ladder_operation_service import (
    LiveLadderConflictError,
    LiveLadderPersistenceError,
    LiveLadderUncertainError,
    deterministic_operation_key,
    get_durable_admin_operation,
    operation_recovery_handoff,
    reconcile_durable_admin_operation,
    require_staging_write_gate,
    run_durable_admin_operation,
)
from services.api.auth import authenticate_bearer, auth_header


class JuprLiveDurableMutationRequest(BaseModel):
    expected_version: str = ""
    idempotency_key: str = ""


class JuprLiveSessionCreateRequest(JuprLiveDurableMutationRequest):
    title: str = "JUPR Live Session"
    event_type: str = "round_robin"
    participant_names: list[str] = Field(default_factory=list)
    player_ids: list[int] = Field(default_factory=list)
    total_rounds: int = 3
    court_sizes: list[int] = Field(default_factory=list)
    confirmation_text: str = ""
    source: str = "next_jupr_live_admin_create"


class JuprLiveSessionStatusRequest(JuprLiveDurableMutationRequest):
    status: str
    title: str | None = None
    confirmation_text: str = ""
    source: str = "next_jupr_live_admin_status"


class JuprLiveScorePayload(BaseModel):
    match_id: str
    score_a: int | None = None
    score_b: int | None = None


class JuprLiveScoresRequest(JuprLiveDurableMutationRequest):
    scores: list[JuprLiveScorePayload] = Field(default_factory=list)
    confirmation_text: str = ""
    source: str = "next_jupr_live_admin_scores"


class JuprLivePublishRequest(JuprLiveDurableMutationRequest):
    match_date: str | None = None
    confirmation_text: str = ""
    source: str = "next_jupr_live_admin_publish"


class JuprLiveAdvanceRequest(JuprLiveDurableMutationRequest):
    confirmation_text: str = ""
    source: str = "next_jupr_live_admin_advance"


class JuprLiveReconcileRequest(BaseModel):
    confirmation_text: str = ""
    source: str = "next_jupr_live_admin_reconcile"


CONFIRM_RECONCILE_LIVE = "RECONCILE LIVE OPERATION"


def _resolve_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(supabase=supabase, club_id=str(club_id), email=user.email, user_id=user.user_id, allowlist=set())
    if not has_permission(role_resolution.role, PERMISSION_ENTER_SCORES):
        denied = build_activity_payload(
            club_id=str(club_id), actor_email=user.email, actor_role=role_resolution.role, action_type="admin_jupr_live_denied", entity_type="live_session", entity_id="live_session",
            after_json={"source_client": "fastapi/nextjs", "reason": "insufficient_permission"}, source_page=source, flagged_for_review=True,
        )
        write_admin_activity_log(supabase, denied)
        raise HTTPException(status_code=403, detail="insufficient permission")
    return user.email, role_resolution.role


def _handle(exc: Exception) -> None:
    if isinstance(exc, LiveLadderConflictError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, LiveLadderUncertainError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, LiveLadderPersistenceError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise exc


def _require_service_role() -> None:
    if not os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail="JUPR Live Admin writes/recovery require SUPABASE_SERVICE_ROLE_KEY on FastAPI; browser and anonymous keys are not accepted.",
        )


def _require_confirmation(actual: str, expected: str) -> None:
    if str(actual or "").strip().upper() != str(expected).upper():
        raise HTTPException(status_code=400, detail=f"Type {expected} to continue.")


def _require_staging_recovery() -> None:
    if os.getenv("JUPR_ENV", "").strip().lower() != "staging":
        raise HTTPException(status_code=403, detail="JUPR Live operation recovery is staging-only.")


def _model_payload(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[attr-defined]
    return model.dict()


def _score_payloads(payload: JuprLiveScoresRequest) -> list[dict[str, Any]]:
    return [_model_payload(score) for score in payload.scores]


def install_admin_jupr_live_routes(app, *, get_supabase_client) -> None:
    """Register guarded JUPR Live Admin routes."""

    @app.get("/admin/clubs/{club_id}/jupr-live/status")
    def get_admin_jupr_live_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_jupr_live_enabled() else None
        return build_admin_jupr_live_status(supabase, club_id=str(club_id))

    @app.get("/admin/clubs/{club_id}/jupr-live/sessions")
    def get_admin_jupr_live_sessions(club_id: str, status: str | None = Query(default=None), limit: int = Query(default=100, ge=1, le=300), authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_jupr_live_enabled():
            raise HTTPException(status_code=403, detail="Next JUPR Live Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_jupr_live_admin_list")
        try:
            return list_admin_jupr_live_sessions(supabase, club_id=str(club_id), status=status, limit=limit)
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/jupr-live/sessions")
    def post_admin_jupr_live_session(club_id: str, payload: JuprLiveSessionCreateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_jupr_live_enabled():
            raise HTTPException(status_code=403, detail="Next JUPR Live Admin is disabled.")
        authenticate_bearer(authorization)
        _require_service_role()
        try:
            require_staging_write_gate(surface_label="JUPR Live Admin", flag_name=JUPR_LIVE_WRITE_FLAG)
        except Exception as exc:
            _handle(exc)
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        _require_confirmation(payload.confirmation_text, CONFIRM_CREATE)
        try:
            recovery = operation_recovery_handoff(surface="jupr_live_admin", entity_id="new", match_context_ids=[])
            return run_durable_admin_operation(
                supabase,
                club_id=str(club_id),
                surface="jupr_live_admin",
                operation_type="create_session",
                entity_id="new",
                idempotency_key=payload.idempotency_key,
                expected_version=payload.expected_version,
                current_version="new",
                request_payload=_model_payload(payload),
                recovery=recovery,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: create_admin_jupr_live_session(
                    supabase,
                    club_id=str(club_id),
                    title=payload.title,
                    event_type=payload.event_type,
                    participant_names=payload.participant_names,
                    player_ids=payload.player_ids,
                    total_rounds=payload.total_rounds,
                    court_sizes=payload.court_sizes or None,
                    actor_email=actor_email,
                    actor_role=actor_role,
                    confirmation_text=payload.confirmation_text,
                    source=payload.source,
                ),
                current_version_resolver=lambda: "new",
            )
        except Exception as exc:
            _handle(exc)

    @app.get("/admin/clubs/{club_id}/jupr-live/sessions/{session_key}")
    def get_admin_jupr_live_session_detail(club_id: str, session_key: str, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_jupr_live_enabled():
            raise HTTPException(status_code=403, detail="Next JUPR Live Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_jupr_live_admin_detail")
        try:
            return get_admin_jupr_live_session(supabase, club_id=str(club_id), session_key=str(session_key))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            _handle(exc)

    @app.patch("/admin/clubs/{club_id}/jupr-live/sessions/{session_key}")
    def patch_admin_jupr_live_session(club_id: str, session_key: str, payload: JuprLiveSessionStatusRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_jupr_live_enabled():
            raise HTTPException(status_code=403, detail="Next JUPR Live Admin is disabled.")
        authenticate_bearer(authorization)
        _require_service_role()
        try:
            require_staging_write_gate(surface_label="JUPR Live Admin", flag_name=JUPR_LIVE_WRITE_FLAG)
        except Exception as exc:
            _handle(exc)
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        _require_confirmation(payload.confirmation_text, CONFIRM_STATUS)
        try:
            current = get_admin_jupr_live_session(supabase, club_id=str(club_id), session_key=str(session_key))["session"]
            return run_durable_admin_operation(
                supabase,
                club_id=str(club_id),
                surface="jupr_live_admin",
                operation_type="update_session",
                entity_id=str(session_key),
                idempotency_key=payload.idempotency_key,
                expected_version=payload.expected_version,
                current_version=str(current.get("version") or ""),
                request_payload=_model_payload(payload),
                recovery=operation_recovery_handoff(surface="jupr_live_admin", entity_id=str(session_key)),
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: update_admin_jupr_live_session_status(supabase, club_id=str(club_id), session_key=str(session_key), status=payload.status, title=payload.title, actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, expected_version=payload.expected_version, source=payload.source),
                current_version_resolver=lambda: str(
                    get_admin_jupr_live_session(supabase, club_id=str(club_id), session_key=str(session_key))["session"].get("version") or ""
                ),
            )
        except Exception as exc:
            _handle(exc)

    @app.patch("/admin/clubs/{club_id}/jupr-live/sessions/{session_key}/scores")
    def patch_admin_jupr_live_scores(club_id: str, session_key: str, payload: JuprLiveScoresRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_jupr_live_enabled():
            raise HTTPException(status_code=403, detail="Next JUPR Live Admin is disabled.")
        authenticate_bearer(authorization)
        _require_service_role()
        try:
            require_staging_write_gate(surface_label="JUPR Live Admin", flag_name=JUPR_LIVE_WRITE_FLAG)
        except Exception as exc:
            _handle(exc)
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        _require_confirmation(payload.confirmation_text, CONFIRM_SCORES)
        try:
            current = get_admin_jupr_live_session(supabase, club_id=str(club_id), session_key=str(session_key))["session"]
            return run_durable_admin_operation(
                supabase,
                club_id=str(club_id),
                surface="jupr_live_admin",
                operation_type="save_scores",
                entity_id=str(session_key),
                idempotency_key=payload.idempotency_key,
                expected_version=payload.expected_version,
                current_version=str(current.get("version") or ""),
                request_payload=_model_payload(payload),
                recovery=operation_recovery_handoff(surface="jupr_live_admin", entity_id=str(session_key)),
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: update_admin_jupr_live_scores(supabase, club_id=str(club_id), session_key=str(session_key), scores=_score_payloads(payload), actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, expected_version=payload.expected_version, source=payload.source),
                current_version_resolver=lambda: str(
                    get_admin_jupr_live_session(supabase, club_id=str(club_id), session_key=str(session_key))["session"].get("version") or ""
                ),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/jupr-live/sessions/{session_key}/advance")
    def post_admin_jupr_live_advance(club_id: str, session_key: str, payload: JuprLiveAdvanceRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_jupr_live_enabled():
            raise HTTPException(status_code=403, detail="Next JUPR Live Admin is disabled.")
        authenticate_bearer(authorization)
        _require_service_role()
        try:
            require_staging_write_gate(surface_label="JUPR Live Admin", flag_name=JUPR_LIVE_WRITE_FLAG)
        except Exception as exc:
            _handle(exc)
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        _require_confirmation(payload.confirmation_text, CONFIRM_ADVANCE)
        try:
            current = get_admin_jupr_live_session(supabase, club_id=str(club_id), session_key=str(session_key))["session"]
            return run_durable_admin_operation(
                supabase,
                club_id=str(club_id),
                surface="jupr_live_admin",
                operation_type="advance_round",
                entity_id=str(session_key),
                idempotency_key=payload.idempotency_key,
                expected_version=payload.expected_version,
                current_version=str(current.get("version") or ""),
                request_payload=_model_payload(payload),
                recovery=operation_recovery_handoff(surface="jupr_live_admin", entity_id=str(session_key)),
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: advance_admin_jupr_live_league_round(supabase, club_id=str(club_id), session_key=str(session_key), actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, expected_version=payload.expected_version, source=payload.source),
                current_version_resolver=lambda: str(
                    get_admin_jupr_live_session(supabase, club_id=str(club_id), session_key=str(session_key))["session"].get("version") or ""
                ),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/jupr-live/sessions/{session_key}/publish")
    def post_admin_jupr_live_publish(club_id: str, session_key: str, payload: JuprLivePublishRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_jupr_live_enabled():
            raise HTTPException(status_code=403, detail="Next JUPR Live Admin is disabled.")
        authenticate_bearer(authorization)
        _require_service_role()
        try:
            require_staging_write_gate(surface_label="JUPR Live Admin", flag_name=JUPR_LIVE_WRITE_FLAG)
        except Exception as exc:
            _handle(exc)
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        _require_confirmation(payload.confirmation_text, CONFIRM_PUBLISH)
        try:
            current = get_admin_jupr_live_session(supabase, club_id=str(club_id), session_key=str(session_key))["session"]
            operation_key = deterministic_operation_key(
                club_id=str(club_id),
                surface="jupr_live_admin",
                operation_type="official_publish",
                entity_id=str(session_key),
                idempotency_key=payload.idempotency_key,
            )
            pending_operation_key = admin_jupr_live_pending_operation_key(current)
            if pending_operation_key and pending_operation_key != operation_key:
                raise LiveLadderConflictError(
                    "This session already has an interrupted official publish. Reconcile operation "
                    f"{pending_operation_key} and inspect Match Log/Replay History before creating a new publish."
                )
            match_context_ids = admin_jupr_live_publish_contexts(current, operation_key=operation_key)
            if not match_context_ids:
                raise ValueError("No unpublished scored JUPR Live matches are ready to publish.")
            return run_durable_admin_operation(
                supabase,
                club_id=str(club_id),
                surface="jupr_live_admin",
                operation_type="official_publish",
                entity_id=str(session_key),
                idempotency_key=payload.idempotency_key,
                expected_version=payload.expected_version,
                current_version=str(current.get("version") or ""),
                request_payload=_model_payload(payload),
                recovery=operation_recovery_handoff(
                    surface="jupr_live_admin",
                    entity_id=str(session_key),
                    match_context_ids=match_context_ids,
                ),
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: publish_admin_jupr_live_matches(supabase, club_id=str(club_id), session_key=str(session_key), match_date=payload.match_date, actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, expected_version=payload.expected_version, publish_context_prefix=operation_key, source=payload.source),
                current_version_resolver=lambda: str(
                    get_admin_jupr_live_session(supabase, club_id=str(club_id), session_key=str(session_key))["session"].get("version") or ""
                ),
            )
        except Exception as exc:
            _handle(exc)

    @app.get("/admin/clubs/{club_id}/jupr-live/operations/{operation_key}")
    def get_admin_jupr_live_operation(club_id: str, operation_key: str, authorization: str | None = auth_header()) -> dict[str, Any]:
        authenticate_bearer(authorization)
        _require_service_role()
        _require_staging_recovery()
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_jupr_live_admin_operation_status")
        try:
            return get_durable_admin_operation(supabase, club_id=str(club_id), operation_key=str(operation_key), surface="jupr_live_admin")
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/jupr-live/operations/{operation_key}/reconcile")
    def post_admin_jupr_live_reconcile(club_id: str, operation_key: str, payload: JuprLiveReconcileRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        authenticate_bearer(authorization)
        _require_service_role()
        _require_staging_recovery()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return reconcile_durable_admin_operation(
                supabase,
                club_id=str(club_id),
                operation_key=str(operation_key),
                surface="jupr_live_admin",
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                expected_confirmation=CONFIRM_RECONCILE_LIVE,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)
