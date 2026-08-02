from __future__ import annotations

import os
from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import (
    PERMISSION_ENTER_SCORES,
    has_permission,
    resolve_admin_role,
)
from jupr_app.domain.admin_activity_log import (
    build_activity_payload,
    write_admin_activity_log,
)
from jupr_app.services.admin_jupr_live_service import JUPR_LIVE_WRITE_FLAG
from jupr_app.services.admin_live_ladder_operation_service import (
    LiveLadderConflictError,
    LiveLadderPersistenceError,
    LiveLadderUncertainError,
    deterministic_operation_key,
    operation_recovery_handoff,
    require_staging_write_gate,
    run_durable_admin_operation,
)
from jupr_app.services.admin_play_generator_service import (
    advance_play_generator_session,
    build_play_generator_status,
    complete_play_generator_session,
    create_play_generator_session,
    get_play_generator_session,
    list_play_generator_sessions,
    mutate_play_generator_roster,
    preview_play_generator,
    publish_play_generator_matches,
    save_play_generator_round,
    skip_play_generator_round,
)
from services.api.auth import authenticate_bearer, auth_header


class GeneratorPreviewRequest(BaseModel):
    generator_kind: str = Field(pattern=r"^(round_robin|ladder)$")
    play_format: str = Field(pattern=r"^(singles|doubles)$")
    title: str = Field(default="Play session", max_length=160)
    participant_names: list[str] = Field(min_length=2, max_length=40)
    player_ids: list[int] = Field(default_factory=list, max_length=40)
    total_rounds: int = Field(default=3, ge=1, le=50)
    court_count: int = Field(default=0, ge=0, le=20)
    standings_sort: str = Field(default="wins", pattern=r"^(wins|points|differential)$")


class GeneratorDurableRequest(BaseModel):
    expected_version: str = ""
    idempotency_key: str = Field(min_length=8, max_length=160)


class GeneratorStartRequest(GeneratorPreviewRequest, GeneratorDurableRequest):
    preview_fingerprint: str | None = Field(default=None, max_length=128)
    source: str = "next_play_generator_start"


class GeneratorScorePayload(BaseModel):
    match_id: str = Field(min_length=1, max_length=160)
    score_a: int | None = Field(default=None, ge=0, le=99)
    score_b: int | None = Field(default=None, ge=0, le=99)


class GeneratorScoresRequest(GeneratorDurableRequest):
    scores: list[GeneratorScorePayload] = Field(min_length=1, max_length=1000)
    source: str = "next_play_generator_scores"


class GeneratorSkipRequest(GeneratorDurableRequest):
    reason: str = Field(default="", max_length=300)
    source: str = "next_play_generator_skip"


class GeneratorAdvanceRequest(GeneratorDurableRequest):
    source: str = "next_play_generator_advance"


class GeneratorRosterRequest(GeneratorDurableRequest):
    action: str = Field(pattern=r"^(add|remove|substitute|reorder)$")
    participant_id: str | None = Field(default=None, max_length=160)
    name: str | None = Field(default=None, max_length=160)
    player_id: int | None = None
    substitute_scope: str = Field(default="rest", pattern=r"^(round|rest)$")
    roster_order: list[str] = Field(default_factory=list, max_length=40)
    source: str = "next_play_generator_roster"


class GeneratorCompleteRequest(GeneratorDurableRequest):
    source: str = "next_play_generator_complete"


class GeneratorPublishRequest(GeneratorDurableRequest):
    match_date: str | None = Field(default=None, max_length=80)
    source: str = "next_play_generator_publish"


def _model_payload(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[attr-defined]
    return model.dict()


def _score_payloads(payload: GeneratorScoresRequest) -> list[dict[str, Any]]:
    return [_model_payload(row) for row in payload.scores]


def _require_service_role() -> None:
    if not os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail=(
                "Round-Robin and Ladder Generator writes require "
                "SUPABASE_SERVICE_ROLE_KEY on FastAPI."
            ),
        )


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


def _resolve_role_or_403(
    *,
    supabase: Any,
    club_id: str,
    authorization: str | None,
    source: str,
) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(
        supabase=supabase,
        club_id=str(club_id),
        email=user.email,
        user_id=user.user_id,
        allowlist=set(),
    )
    if not has_permission(role_resolution.role, PERMISSION_ENTER_SCORES):
        denied = build_activity_payload(
            club_id=str(club_id),
            actor_email=user.email,
            actor_role=role_resolution.role,
            action_type="play_generator_denied",
            entity_type="play_generator_session",
            entity_id="play_generator",
            after_json={
                "source_client": "fastapi/nextjs",
                "reason": "insufficient_permission",
            },
            source_page=source,
            flagged_for_review=True,
        )
        write_admin_activity_log(supabase, denied)
        raise HTTPException(status_code=403, detail="insufficient permission")
    return user.email, role_resolution.role


def _require_write_gate() -> None:
    _require_service_role()
    try:
        require_staging_write_gate(
            surface_label="Round-Robin and Ladder Generators",
            flag_name=JUPR_LIVE_WRITE_FLAG,
        )
    except Exception as exc:
        _handle(exc)


def install_admin_play_generator_routes(app, *, get_supabase_client) -> None:
    """Install durable Round-Robin and Ladder Generator administration routes."""

    @app.get("/admin/clubs/{club_id}/play-generators/status")
    def get_generator_status(club_id: str) -> dict[str, Any]:
        try:
            supabase = get_supabase_client()
        except Exception:
            supabase = None
        return build_play_generator_status(supabase, club_id=str(club_id))

    @app.post("/admin/clubs/{club_id}/play-generators/preview")
    def post_generator_preview(
        club_id: str,
        payload: GeneratorPreviewRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        supabase = get_supabase_client()
        _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_play_generator_preview",
        )
        try:
            return preview_play_generator(
                supabase,
                club_id=str(club_id),
                generator_kind=payload.generator_kind,
                play_format=payload.play_format,
                title=payload.title,
                participant_names=payload.participant_names,
                player_ids=payload.player_ids,
                total_rounds=payload.total_rounds,
                court_count=payload.court_count,
                standings_sort=payload.standings_sort,
            )
        except Exception as exc:
            _handle(exc)

    @app.get("/admin/clubs/{club_id}/play-generators/sessions")
    def get_generator_sessions(
        club_id: str,
        generator_kind: str | None = Query(default=None),
        status: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=300),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        supabase = get_supabase_client()
        _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_play_generator_list",
        )
        try:
            return list_play_generator_sessions(
                supabase,
                club_id=str(club_id),
                generator_kind=generator_kind,
                status=status,
                limit=limit,
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/play-generators/sessions")
    def post_generator_session(
        club_id: str,
        payload: GeneratorStartRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_write_gate()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return run_durable_admin_operation(
                supabase,
                club_id=str(club_id),
                surface="play_generator",
                operation_type="start_session",
                entity_id="new",
                idempotency_key=payload.idempotency_key,
                expected_version=payload.expected_version,
                current_version="new",
                request_payload=_model_payload(payload),
                recovery=operation_recovery_handoff(
                    surface="play_generator",
                    entity_id="new",
                ),
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: create_play_generator_session(
                    supabase,
                    club_id=str(club_id),
                    generator_kind=payload.generator_kind,
                    play_format=payload.play_format,
                    title=payload.title,
                    participant_names=payload.participant_names,
                    player_ids=payload.player_ids,
                    total_rounds=payload.total_rounds,
                    court_count=payload.court_count,
                    preview_fingerprint=payload.preview_fingerprint,
                    standings_sort=payload.standings_sort,
                    actor_email=actor_email,
                    actor_role=actor_role,
                    source=payload.source,
                ),
                current_version_resolver=lambda: "new",
            )
        except Exception as exc:
            _handle(exc)

    @app.get("/admin/clubs/{club_id}/play-generators/sessions/{session_key}")
    def get_generator_session(
        club_id: str,
        session_key: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        supabase = get_supabase_client()
        _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_play_generator_detail",
        )
        try:
            return get_play_generator_session(
                supabase,
                club_id=str(club_id),
                session_key=str(session_key),
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            _handle(exc)

    @app.patch(
        "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/rounds/{round_number}/scores"
    )
    def patch_generator_round_scores(
        club_id: str,
        session_key: str,
        round_number: int,
        payload: GeneratorScoresRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_write_gate()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            current = get_play_generator_session(
                supabase,
                club_id=str(club_id),
                session_key=str(session_key),
            )["session"]
            return run_durable_admin_operation(
                supabase,
                club_id=str(club_id),
                surface="play_generator",
                operation_type="save_round_scores",
                entity_id=str(session_key),
                idempotency_key=payload.idempotency_key,
                expected_version=payload.expected_version,
                current_version=str(current.get("version") or ""),
                request_payload=_model_payload(payload),
                recovery=operation_recovery_handoff(
                    surface="play_generator",
                    entity_id=str(session_key),
                ),
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: save_play_generator_round(
                    supabase,
                    club_id=str(club_id),
                    session_key=str(session_key),
                    round_number=int(round_number),
                    scores=_score_payloads(payload),
                    expected_version=payload.expected_version,
                    actor_email=actor_email,
                    actor_role=actor_role,
                    source=payload.source,
                ),
                current_version_resolver=lambda: str(
                    get_play_generator_session(
                        supabase,
                        club_id=str(club_id),
                        session_key=str(session_key),
                    )["session"].get("version")
                    or ""
                ),
            )
        except Exception as exc:
            _handle(exc)

    @app.post(
        "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/rounds/{round_number}/skip"
    )
    def post_generator_round_skip(
        club_id: str,
        session_key: str,
        round_number: int,
        payload: GeneratorSkipRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_write_gate()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            current = get_play_generator_session(
                supabase,
                club_id=str(club_id),
                session_key=str(session_key),
            )["session"]
            return run_durable_admin_operation(
                supabase,
                club_id=str(club_id),
                surface="play_generator",
                operation_type="skip_round",
                entity_id=str(session_key),
                idempotency_key=payload.idempotency_key,
                expected_version=payload.expected_version,
                current_version=str(current.get("version") or ""),
                request_payload=_model_payload(payload),
                recovery=operation_recovery_handoff(
                    surface="play_generator",
                    entity_id=str(session_key),
                ),
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: skip_play_generator_round(
                    supabase,
                    club_id=str(club_id),
                    session_key=str(session_key),
                    round_number=int(round_number),
                    reason=payload.reason,
                    expected_version=payload.expected_version,
                    actor_email=actor_email,
                    actor_role=actor_role,
                    source=payload.source,
                ),
                current_version_resolver=lambda: str(
                    get_play_generator_session(
                        supabase,
                        club_id=str(club_id),
                        session_key=str(session_key),
                    )["session"].get("version")
                    or ""
                ),
            )
        except Exception as exc:
            _handle(exc)

    @app.post(
        "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/advance"
    )
    def post_generator_advance(
        club_id: str,
        session_key: str,
        payload: GeneratorAdvanceRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_write_gate()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            current = get_play_generator_session(
                supabase,
                club_id=str(club_id),
                session_key=str(session_key),
            )["session"]
            return run_durable_admin_operation(
                supabase,
                club_id=str(club_id),
                surface="play_generator",
                operation_type="advance_round",
                entity_id=str(session_key),
                idempotency_key=payload.idempotency_key,
                expected_version=payload.expected_version,
                current_version=str(current.get("version") or ""),
                request_payload=_model_payload(payload),
                recovery=operation_recovery_handoff(
                    surface="play_generator",
                    entity_id=str(session_key),
                ),
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: advance_play_generator_session(
                    supabase,
                    club_id=str(club_id),
                    session_key=str(session_key),
                    expected_version=payload.expected_version,
                    actor_email=actor_email,
                    actor_role=actor_role,
                    source=payload.source,
                ),
                current_version_resolver=lambda: str(
                    get_play_generator_session(
                        supabase,
                        club_id=str(club_id),
                        session_key=str(session_key),
                    )["session"].get("version")
                    or ""
                ),
            )
        except Exception as exc:
            _handle(exc)

    @app.post(
        "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/roster"
    )
    def post_generator_roster(
        club_id: str,
        session_key: str,
        payload: GeneratorRosterRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_write_gate()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            current = get_play_generator_session(
                supabase,
                club_id=str(club_id),
                session_key=str(session_key),
            )["session"]
            return run_durable_admin_operation(
                supabase,
                club_id=str(club_id),
                surface="play_generator",
                operation_type=f"roster_{payload.action}",
                entity_id=str(session_key),
                idempotency_key=payload.idempotency_key,
                expected_version=payload.expected_version,
                current_version=str(current.get("version") or ""),
                request_payload=_model_payload(payload),
                recovery=operation_recovery_handoff(
                    surface="play_generator",
                    entity_id=str(session_key),
                ),
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: mutate_play_generator_roster(
                    supabase,
                    club_id=str(club_id),
                    session_key=str(session_key),
                    action=payload.action,
                    participant_id=payload.participant_id,
                    name=payload.name,
                    player_id=payload.player_id,
                    substitute_scope=payload.substitute_scope,
                    roster_order=payload.roster_order,
                    expected_version=payload.expected_version,
                    actor_email=actor_email,
                    actor_role=actor_role,
                    source=payload.source,
                ),
                current_version_resolver=lambda: str(
                    get_play_generator_session(
                        supabase,
                        club_id=str(club_id),
                        session_key=str(session_key),
                    )["session"].get("version")
                    or ""
                ),
            )
        except Exception as exc:
            _handle(exc)

    @app.post(
        "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/complete"
    )
    def post_generator_complete(
        club_id: str,
        session_key: str,
        payload: GeneratorCompleteRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_write_gate()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            current = get_play_generator_session(
                supabase,
                club_id=str(club_id),
                session_key=str(session_key),
            )["session"]
            return run_durable_admin_operation(
                supabase,
                club_id=str(club_id),
                surface="play_generator",
                operation_type="complete_session",
                entity_id=str(session_key),
                idempotency_key=payload.idempotency_key,
                expected_version=payload.expected_version,
                current_version=str(current.get("version") or ""),
                request_payload=_model_payload(payload),
                recovery=operation_recovery_handoff(
                    surface="play_generator",
                    entity_id=str(session_key),
                ),
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: complete_play_generator_session(
                    supabase,
                    club_id=str(club_id),
                    session_key=str(session_key),
                    expected_version=payload.expected_version,
                    actor_email=actor_email,
                    actor_role=actor_role,
                    source=payload.source,
                ),
                current_version_resolver=lambda: str(
                    get_play_generator_session(
                        supabase,
                        club_id=str(club_id),
                        session_key=str(session_key),
                    )["session"].get("version")
                    or ""
                ),
            )
        except Exception as exc:
            _handle(exc)

    @app.post(
        "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/publish"
    )
    def post_generator_publish(
        club_id: str,
        session_key: str,
        payload: GeneratorPublishRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_write_gate()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        operation_key = deterministic_operation_key(
            club_id=str(club_id),
            surface="play_generator",
            operation_type="official_publish",
            entity_id=str(session_key),
            idempotency_key=payload.idempotency_key,
        )
        try:
            current = get_play_generator_session(
                supabase,
                club_id=str(club_id),
                session_key=str(session_key),
            )["session"]
            return run_durable_admin_operation(
                supabase,
                club_id=str(club_id),
                surface="play_generator",
                operation_type="official_publish",
                entity_id=str(session_key),
                idempotency_key=payload.idempotency_key,
                expected_version=payload.expected_version,
                current_version=str(current.get("version") or ""),
                request_payload=_model_payload(payload),
                recovery=operation_recovery_handoff(
                    surface="play_generator",
                    entity_id=str(session_key),
                ),
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: publish_play_generator_matches(
                    supabase,
                    club_id=str(club_id),
                    session_key=str(session_key),
                    match_date=payload.match_date,
                    expected_version=payload.expected_version,
                    idempotency_key=payload.idempotency_key,
                    operation_key=operation_key,
                    actor_email=actor_email,
                    actor_role=actor_role,
                    source=payload.source,
                ),
                current_version_resolver=lambda: str(
                    get_play_generator_session(
                        supabase,
                        club_id=str(club_id),
                        session_key=str(session_key),
                    )["session"].get("version")
                    or ""
                ),
            )
        except Exception as exc:
            _handle(exc)
