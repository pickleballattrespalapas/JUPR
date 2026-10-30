from __future__ import annotations

from services.api.generator_request_models import GeneratorScoreInput, require_unique_match_ids

import os
from uuid import UUID
from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator

from jupr_app.services.generator_session_details_service import update_admin_generator_details
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
    mark_play_generator_round_played,
    mutate_play_generator_roster,
    preview_play_generator,
    publish_play_generator_matches,
    save_play_generator_round,
    skip_play_generator_round,
    reopen_play_generator_round,
    select_play_generator_round,
    start_play_generator_playoff,
)
from services.api.auth import authenticate_bearer, auth_header
from jupr_app.services.generator_submission_service import (
    list_generator_submissions, require_reviewer, review_generator_submission, submit_generator_session,
    set_generator_starting_rating, update_generator_submission_title, correct_generator_submission_scores,
)
from jupr_app.services.direct_match_entry_service import (
    DirectMatchConflictError, DirectMatchRecoveryRequiredError,
)
from jupr_app.services.public_live_operation_service import PublicLiveConflictError
from jupr_app.services.generator_sharing_service import manage_generator_sharing
from services.api.shared_play_generator_routes import GeneratorSharingRequest


class GeneratorPreviewRequest(BaseModel):
    event_format: str = Field(default="standard", pattern=r"^(standard|moneyball_expectations)$")
    generator_kind: str = Field(pattern=r"^(round_robin|ladder)$")
    play_format: str = Field(pattern=r"^(singles|doubles|doubles_singles)$")
    title: str = Field(default="Play session", max_length=160)
    participant_names: list[str] = Field(min_length=2, max_length=40)
    player_ids: list[int] = Field(default_factory=list, max_length=40)
    participant_player_ids: dict[str, int] = Field(default_factory=dict, max_length=40)
    total_rounds: int = Field(default=3, ge=1, le=50)
    court_count: int = Field(default=0, ge=0, le=20)
    doubles_court_count: int = Field(default=0, ge=0, le=20)
    singles_court_count: int = Field(default=0, ge=0, le=20)
    standings_sort: str = Field(default="wins", pattern=r"^(wins|points|differential)$")
    scoring_mode: str = Field(default="scored", pattern=r"^(scored|unscored)$")
    rating_mode: str = Field(default="unrated", pattern=r"^(rated|unrated)$")


class GeneratorDurableRequest(BaseModel):
    expected_version: str = ""
    idempotency_key: str = Field(min_length=8, max_length=160)


class GeneratorStartRequest(GeneratorPreviewRequest, GeneratorDurableRequest):
    participant_names: list[str] = Field(default_factory=list, max_length=40)
    registration_hub_id: UUID | None = None
    preview_fingerprint: str | None = Field(default=None, max_length=128)
    source: str = "next_play_generator_start"


class GeneratorScorePayload(GeneratorScoreInput):
    pass

class GeneratorScoresRequest(GeneratorDurableRequest):
    scores: list[GeneratorScorePayload] = Field(min_length=1, max_length=1000)

    @field_validator("scores")
    @classmethod
    def unique_games(cls, value: list[GeneratorScorePayload]) -> list[GeneratorScorePayload]:
        return require_unique_match_ids(value)
    source: str = "next_play_generator_scores"


class GeneratorPlayedRequest(GeneratorDurableRequest):
    source: str = "next_play_generator_played"


class GeneratorSkipRequest(GeneratorDurableRequest):
    reason: str = Field(default="", max_length=300)
    source: str = "next_play_generator_skip"


class GeneratorReopenRequest(GeneratorDurableRequest):
    source: str = "next_play_generator_reopen"


class GeneratorSelectRequest(GeneratorDurableRequest):
    source: str = "next_play_generator_select"


class GeneratorAdvanceRequest(GeneratorDurableRequest):
    source: str = "next_play_generator_advance"


class GeneratorPlayoffRequest(GeneratorDurableRequest):
    playoff_format: str = Field(pattern=r"^(groups_of_four|top_eight|top_four)$")
    source: str = "next_play_generator_playoff"


class GeneratorRosterRequest(GeneratorDurableRequest):
    action: str = Field(pattern=r"^(add|remove|substitute|reorder|seat_arrivals)$")
    participant_id: str | None = Field(default=None, max_length=160)
    name: str | None = Field(default=None, max_length=160)
    player_id: int | None = None
    substitute_scope: str = Field(default="rest", pattern=r"^(round|rest)$")
    roster_order: list[str] = Field(default_factory=list, max_length=40)
    participant_ids: list[str] = Field(default_factory=list, max_length=40)
    court_number: int | None = Field(default=None, ge=1, le=20)
    source: str = "next_play_generator_roster"


class GeneratorCompleteRequest(GeneratorDurableRequest):
    confirmation_text: str = Field(default="", max_length=80)
    source: str = "next_play_generator_complete"


class GeneratorPublishRequest(GeneratorDurableRequest):
    match_date: str | None = Field(default=None, max_length=80)
    confirmation_text: str = Field(default="", max_length=80)
    source: str = "next_play_generator_publish"


class GeneratorSubmitRequest(BaseModel):
    expected_version: str = Field(min_length=1)
    organizer_name: str = Field(min_length=1, max_length=160)
    match_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


class GeneratorReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_version: int = Field(ge=1)
    action: str = Field(pattern=r"^(approve|reject)$")
    player_ids: dict[str, int] = Field(default_factory=dict, max_length=2500)
    match_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    reason: str = Field(default="", max_length=500)


class GeneratorStartingRatingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    participant_id: str = Field(min_length=1, max_length=160)
    match_format: str = Field(pattern=r"^doubles$")
    rating_jupr: float = Field(ge=1.0, le=7.0, allow_inf_nan=False)
    expected_state_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")


class GeneratorDetailsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_version: str | int
    title: str = Field(min_length=1, max_length=160)
    rating_mode: str = Field(pattern=r"^(rated|unrated)$")


class GeneratorSubmissionTitleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_version: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=160)

    @field_validator("title", mode="before")
    @classmethod
    def normalize_title(cls, value: Any) -> Any:
        return " ".join(value.replace("<", "").replace(">", "").split()) if isinstance(value, str) else value


class GeneratorSubmissionScoreCorrection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round_number: int = Field(strict=True, ge=1)
    match_id: str = Field(strict=True, min_length=1, max_length=160)
    score_a: int = Field(strict=True, ge=0, le=99)
    score_b: int = Field(strict=True, ge=0, le=99)


class GeneratorSubmissionScoresRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_version: int = Field(strict=True, ge=1)
    scores: list[GeneratorSubmissionScoreCorrection] = Field(min_length=1, max_length=2500)


def _model_payload(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        payload = model.model_dump()  # type: ignore[attr-defined]
    else:
        payload = model.dict()
    # Preserve durable request fingerprints for starts saved before sparse
    # profile links were supported. Nonempty maps remain part of the intent.
    if payload.get("participant_player_ids") == {}:
        payload.pop("participant_player_ids")
    if payload.get("registration_hub_id") is None:
        payload.pop("registration_hub_id", None)
    return payload


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
    from jupr_app.services.generator_compute import compute_unavailable_cause
    if unavailable := compute_unavailable_cause(exc):
        raise HTTPException(status_code=503, detail=str(unavailable), headers={"Retry-After": "2"}) from exc
    if isinstance(exc, DirectMatchConflictError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, DirectMatchRecoveryRequiredError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, PublicLiveConflictError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, LiveLadderConflictError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, LiveLadderUncertainError):
        raise HTTPException(status_code=409, detail={"message": str(exc), "code": "RECOVERY_REQUIRED", "recovery_required": True}) from exc
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

    @app.post("/admin/clubs/{club_id}/play-generators/sessions/{session_key}/sharing")
    def manage_sharing(club_id: str, session_key: str, payload: GeneratorSharingRequest,
                       authorization: str | None = auth_header()) -> dict[str, Any]:
        _require_write_gate()
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=club_id, authorization=authorization,
                             source="generator_sharing")
        try:
            return manage_generator_sharing(supabase, club_id=club_id, session_key=session_key,
                action=payload.action, expected_version=payload.expected_version, admin_authorized=True)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            _handle(exc)

    @app.get("/admin/clubs/{club_id}/play-generators/submissions")
    def get_generator_submissions(club_id: str,
        status: str = Query(default="pending", pattern=r"^(pending|approved|rejected)$"),
        authorization: str | None = auth_header()) -> dict[str, Any]:
        supabase = get_supabase_client()
        _, role = _resolve_role_or_403(supabase=supabase, club_id=club_id,
            authorization=authorization, source="generator_submissions")
        try:
            return list_generator_submissions(supabase, club_id=club_id, actor_role=role, status=status)
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/play-generators/sessions/{session_key}/submit")
    def post_generator_submission(club_id: str, session_key: str, payload: GeneratorSubmitRequest,
        authorization: str | None = auth_header()) -> dict[str, Any]:
        _require_write_gate()
        supabase = get_supabase_client()
        email, role = _resolve_role_or_403(supabase=supabase, club_id=club_id,
            authorization=authorization, source="generator_submission")
        try:
            return submit_generator_session(supabase, club_id=club_id, session_key=session_key,
                actor_email=email, actor_role=role, **_model_payload(payload))
        except Exception as exc:
            _handle(exc)

    @app.patch("/admin/clubs/{club_id}/play-generators/sessions/{session_key}/details")
    def patch_generator_details(club_id: str, session_key: str,
        payload: GeneratorDetailsRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        _require_write_gate()
        supabase = get_supabase_client()
        email, role = _resolve_role_or_403(supabase=supabase, club_id=club_id,
            authorization=authorization, source="generator_session_details")
        try:
            return update_admin_generator_details(supabase, club_id=club_id, session_key=session_key,
                actor_email=email, actor_role=role, **_model_payload(payload))
        except Exception as exc:
            _handle(exc)

    @app.patch("/admin/clubs/{club_id}/play-generators/sessions/{session_key}/submission/title")
    def patch_generator_submission_title(club_id: str, session_key: str,
        payload: GeneratorSubmissionTitleRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        _require_write_gate()
        supabase = get_supabase_client()
        email, role = _resolve_role_or_403(supabase=supabase, club_id=club_id,
            authorization=authorization, source="generator_submission_title")
        try:
            return update_generator_submission_title(supabase, club_id=club_id, session_key=session_key,
                actor_email=email, actor_role=role, **_model_payload(payload))
        except Exception as exc:
            _handle(exc)

    @app.patch("/admin/clubs/{club_id}/play-generators/sessions/{session_key}/submission/scores")
    def patch_generator_submission_scores(club_id: str, session_key: str,
        payload: GeneratorSubmissionScoresRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        _require_write_gate()
        supabase = get_supabase_client()
        email, role = _resolve_role_or_403(supabase=supabase, club_id=club_id,
            authorization=authorization, source="generator_submission_scores")
        try:
            return correct_generator_submission_scores(supabase, club_id=club_id, session_key=session_key,
                actor_email=email, actor_role=role, **_model_payload(payload))
        except Exception as exc:
            _handle(exc)

    @app.patch("/admin/clubs/{club_id}/play-generators/sessions/{session_key}/starting-ratings/{player_id}")
    def patch_generator_starting_rating(club_id: str, session_key: str, player_id: int,
        payload: GeneratorStartingRatingRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        _require_write_gate()
        supabase = get_supabase_client()
        email, role = _resolve_role_or_403(supabase=supabase, club_id=club_id,
            authorization=authorization, source="generator_starting_rating")
        try:
            return set_generator_starting_rating(supabase, club_id=club_id, session_key=session_key,
                player_id=player_id, actor_email=email, actor_role=role, **_model_payload(payload))
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/play-generators/sessions/{session_key}/review")
    def post_generator_review(club_id: str, session_key: str, payload: GeneratorReviewRequest,
        authorization: str | None = auth_header()) -> dict[str, Any]:
        _require_write_gate()
        supabase = get_supabase_client()
        email, role = _resolve_role_or_403(supabase=supabase, club_id=club_id,
            authorization=authorization, source="generator_review")
        try:
            return review_generator_submission(supabase, club_id=club_id, session_key=session_key,
                actor_email=email, actor_role=role, **_model_payload(payload))
        except Exception as exc:
            _handle(exc)

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
                participant_player_ids=payload.participant_player_ids,
                total_rounds=payload.total_rounds,
                court_count=payload.court_count,
                doubles_court_count=payload.doubles_court_count,
                singles_court_count=payload.singles_court_count,
                standings_sort=payload.standings_sort,
                scoring_mode=payload.scoring_mode,
                rating_mode=payload.rating_mode,
                event_format=payload.event_format,
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
                    participant_player_ids=payload.participant_player_ids,
                    total_rounds=payload.total_rounds,
                    court_count=payload.court_count,
                    doubles_court_count=payload.doubles_court_count,
                    singles_court_count=payload.singles_court_count,
                    preview_fingerprint=payload.preview_fingerprint,
                    registration_hub_id=str(payload.registration_hub_id) if payload.registration_hub_id else None,
                    standings_sort=payload.standings_sort,
                    scoring_mode=payload.scoring_mode,
                    rating_mode=payload.rating_mode,
                    event_format=payload.event_format,
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
                request_payload={**_model_payload(payload), "round_number": int(round_number)},
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
        "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/rounds/{round_number}/played"
    )
    def post_generator_round_played(
        club_id: str,
        session_key: str,
        round_number: int,
        payload: GeneratorPlayedRequest,
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
                operation_type="mark_round_played",
                entity_id=str(session_key),
                idempotency_key=payload.idempotency_key,
                expected_version=payload.expected_version,
                current_version=str(current.get("version") or ""),
                request_payload={**_model_payload(payload), "round_number": int(round_number)},
                recovery=operation_recovery_handoff(
                    surface="play_generator",
                    entity_id=str(session_key),
                ),
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: mark_play_generator_round_played(
                    supabase,
                    club_id=str(club_id),
                    session_key=str(session_key),
                    round_number=int(round_number),
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
                request_payload={**_model_payload(payload), "round_number": int(round_number)},
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
        "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/rounds/{round_number}/reopen"
    )
    def post_generator_round_reopen(
        club_id: str,
        session_key: str,
        round_number: int,
        payload: GeneratorReopenRequest,
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
                operation_type="reopen_round",
                entity_id=str(session_key),
                idempotency_key=payload.idempotency_key,
                expected_version=payload.expected_version,
                current_version=str(current.get("version") or ""),
                request_payload={**_model_payload(payload), "round_number": int(round_number)},
                recovery=operation_recovery_handoff(
                    surface="play_generator",
                    entity_id=str(session_key),
                ),
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: reopen_play_generator_round(
                    supabase,
                    club_id=str(club_id),
                    session_key=str(session_key),
                    round_number=int(round_number),
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
        "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/rounds/{round_number}/select"
    )
    def post_generator_round_select(
        club_id: str,
        session_key: str,
        round_number: int,
        payload: GeneratorSelectRequest,
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
                operation_type="select_round",
                entity_id=str(session_key),
                idempotency_key=payload.idempotency_key,
                expected_version=payload.expected_version,
                current_version=str(current.get("version") or ""),
                request_payload={**_model_payload(payload), "round_number": int(round_number)},
                recovery=operation_recovery_handoff(
                    surface="play_generator",
                    entity_id=str(session_key),
                ),
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: select_play_generator_round(
                    supabase,
                    club_id=str(club_id),
                    session_key=str(session_key),
                    round_number=int(round_number),
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

    @app.post("/admin/clubs/{club_id}/play-generators/sessions/{session_key}/playoff")
    def post_generator_playoff(
        club_id: str, session_key: str, payload: GeneratorPlayoffRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_write_gate()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source,
        )
        try:
            current = get_play_generator_session(supabase, club_id=club_id, session_key=session_key)["session"]
            return run_durable_admin_operation(
                supabase, club_id=club_id, surface="play_generator", operation_type="start_playoff",
                entity_id=session_key, idempotency_key=payload.idempotency_key,
                expected_version=payload.expected_version, current_version=str(current.get("version") or ""),
                request_payload=_model_payload(payload),
                recovery=operation_recovery_handoff(surface="play_generator", entity_id=session_key),
                actor_email=actor_email, actor_role=actor_role, source=payload.source,
                mutate=lambda: start_play_generator_playoff(
                    supabase, club_id=club_id, session_key=session_key, playoff_format=payload.playoff_format,
                    expected_version=payload.expected_version, actor_email=actor_email, actor_role=actor_role, source=payload.source,
                ),
                current_version_resolver=lambda: str(get_play_generator_session(
                    supabase, club_id=club_id, session_key=session_key)["session"].get("version") or ""),
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
                    participant_ids=payload.participant_ids,
                    court_number=payload.court_number,
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
        if str(payload.confirmation_text or "").strip().upper() != "COMPLETE SESSION":
            raise HTTPException(
                status_code=400,
                detail="Type COMPLETE SESSION to complete this play session.",
            )
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
        if str(payload.confirmation_text or "").strip().upper() != "PUBLISH MATCHES":
            raise HTTPException(
                status_code=400,
                detail="Type PUBLISH MATCHES to publish official matches.",
            )
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            require_reviewer(actor_role)
        except PermissionError as exc:
            _handle(exc)
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
