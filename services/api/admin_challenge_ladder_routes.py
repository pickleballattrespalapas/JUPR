from __future__ import annotations

import os
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import PERMISSION_MANAGE_MATCHES, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_challenge_ladder_service import (
    CHALLENGE_LADDER_WRITE_FLAG,
    CONFIRM,
    CONFIRM_ACCEPT,
    CONFIRM_CLOCK,
    CONFIRM_CREATE,
    CONFIRM_FORFEIT,
    CONFIRM_OVERRIDES,
    CONFIRM_PASS,
    CONFIRM_RESULT,
    CONFIRM_ROSTER_ADD,
    CONFIRM_ROSTER_MOVE,
    CONFIRM_ROSTER_REPLACE,
    accept_admin_challenge_ladder_challenge,
    add_admin_challenge_ladder_roster_player,
    build_admin_challenge_ladder_status,
    create_admin_challenge_ladder_challenge,
    get_admin_challenge_ladder_dashboard,
    get_admin_challenge_ladder_tier_movement_review,
    is_admin_challenge_ladder_enabled,
    move_admin_challenge_ladder_roster_player,
    preview_admin_challenge_ladder_result_for_challenge,
    preview_admin_challenge_ladder_tier_roster_replacement,
    record_admin_challenge_ladder_forfeit,
    record_admin_challenge_ladder_pass,
    record_admin_challenge_ladder_result,
    replace_admin_challenge_ladder_tier_roster,
    save_admin_challenge_ladder_player_overrides,
    start_admin_challenge_ladder_clock,
    update_admin_challenge_ladder_challenge,
)
from jupr_app.services.admin_live_ladder_operation_service import (
    LiveLadderConflictError,
    LiveLadderPersistenceError,
    LiveLadderUncertainError,
    deterministic_match_context_id,
    deterministic_operation_key,
    get_durable_admin_operation,
    operation_recovery_handoff,
    reconcile_durable_admin_operation,
    replay_durable_admin_operation_if_present,
    require_staging_write_gate,
    run_durable_admin_operation,
)
from services.api.auth import authenticate_bearer, auth_header


class ChallengeDurableMutationRequest(BaseModel):
    expected_version: str = ""
    idempotency_key: str = ""


class ChallengeUpdateRequest(ChallengeDurableMutationRequest):
    status: str
    admin_note: str | None = None
    confirmation_text: str = ""
    source: str = "next_challenge_ladder_admin"


class ChallengeCreateRequest(ChallengeDurableMutationRequest):
    challenger_id: int
    defender_id: int
    tier_id: str
    challenger_contact: str | None = None
    ledger_ref: str | None = None
    override: bool = False
    start_clock: bool = False
    confirmation_text: str = ""
    source: str = "next_challenge_ladder_admin_create"


class ChallengeSimpleConfirmationRequest(ChallengeDurableMutationRequest):
    confirmation_text: str = ""
    source: str = "next_challenge_ladder_admin"


class ChallengeForfeitRequest(ChallengeDurableMutationRequest):
    forfeited_by_id: int
    admin_note: str | None = None
    confirmation_text: str = ""
    source: str = "next_challenge_ladder_forfeit"


class ChallengePassRequest(ChallengeDurableMutationRequest):
    player_id: int
    confirmation_text: str = ""
    source: str = "next_challenge_ladder_pass"


class ChallengeRosterAddRequest(ChallengeDurableMutationRequest):
    player_id: int
    tier_id: str
    admin_note: str | None = None
    confirmation_text: str = ""
    source: str = "next_challenge_ladder_roster_add"


class ChallengeRosterMoveRequest(ChallengeDurableMutationRequest):
    destination_tier: str
    recompress_old: bool = True
    admin_note: str | None = None
    confirmation_text: str = ""
    source: str = "next_challenge_ladder_roster_move"


class ChallengeRosterReplacePreviewRequest(BaseModel):
    tier_id: str
    ranked_names: list[str] = Field(default_factory=list)
    source: str = "next_challenge_ladder_roster_replace_preview"


class ChallengeRosterReplaceRequest(ChallengeDurableMutationRequest):
    tier_id: str
    ranked_player_ids: list[int] = Field(default_factory=list)
    preview_fingerprint: str = ""
    admin_note: str | None = None
    confirmation_text: str = ""
    source: str = "next_challenge_ladder_roster_replace"


class ChallengePlayerOverridesRequest(ChallengeDurableMutationRequest):
    vacation_until: str | None = None
    reinstate_required: bool = False
    reinstate_notes: str | None = None
    confirmation_text: str = ""
    source: str = "next_challenge_ladder_player_overrides"


class ChallengeResultRequest(ChallengeDurableMutationRequest):
    partner_a_challenger_id: int
    partner_a_defender_id: int
    partner_b_challenger_id: int
    partner_b_defender_id: int
    match_a_games: list[list[int]] = Field(default_factory=list)
    match_b_games: list[list[int]] = Field(default_factory=list)
    match_date: str = ""
    winner_override: str = "computed"
    publish_official_matches: bool = True
    preview_fingerprint: str = ""
    confirmation_text: str = ""
    source: str = "next_challenge_ladder_result"


class ChallengeResultPreviewRequest(BaseModel):
    partner_a_challenger_id: int
    partner_a_defender_id: int
    partner_b_challenger_id: int
    partner_b_defender_id: int
    match_a_games: list[list[int]] = Field(default_factory=list)
    match_b_games: list[list[int]] = Field(default_factory=list)
    match_date: str = ""
    winner_override: str = "computed"
    publish_official_matches: bool = True
    source: str = "next_challenge_ladder_result_preview"


class ChallengeReconcileRequest(BaseModel):
    confirmation_text: str = ""
    source: str = "next_challenge_ladder_reconcile"


CONFIRM_RECONCILE_LADDER = "RECONCILE LADDER OPERATION"


def _resolve_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(supabase=supabase, club_id=str(club_id), email=user.email, user_id=user.user_id, allowlist=set())
    if not has_permission(role_resolution.role, PERMISSION_MANAGE_MATCHES):
        denied = build_activity_payload(
            club_id=str(club_id), actor_email=user.email, actor_role=role_resolution.role, action_type="admin_challenge_ladder_denied", entity_type="challenge_ladder", entity_id="challenge_ladder",
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
            detail="Challenge Ladder writes/recovery require SUPABASE_SERVICE_ROLE_KEY on FastAPI; browser and anonymous keys are not accepted.",
        )


def _model_payload(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[attr-defined]
    return model.dict()


def _run_ladder_operation(
    supabase: Any,
    *,
    club_id: str,
    operation_type: str,
    entity_id: str,
    payload: ChallengeDurableMutationRequest,
    actor_email: str,
    actor_role: str,
    source: str,
    mutate,
    match_context_ids: list[str] | None = None,
) -> dict[str, Any]:
    current_version = str(get_admin_challenge_ladder_dashboard(supabase, club_id=str(club_id)).get("state_version") or "")
    return run_durable_admin_operation(
        supabase,
        club_id=str(club_id),
        surface="challenge_ladder",
        operation_type=str(operation_type),
        entity_id=str(entity_id),
        idempotency_key=payload.idempotency_key,
        expected_version=payload.expected_version,
        current_version=current_version,
        request_payload=_model_payload(payload),
        recovery=operation_recovery_handoff(surface="challenge_ladder", entity_id=str(entity_id), match_context_ids=match_context_ids),
        actor_email=actor_email,
        actor_role=actor_role,
        source=source,
        mutate=mutate,
        current_version_resolver=lambda: str(
            get_admin_challenge_ladder_dashboard(supabase, club_id=str(club_id)).get("state_version") or ""
        ),
    )


def _require_confirmation(actual: str, expected: str) -> None:
    if str(actual or "").strip().upper() != str(expected).upper():
        raise HTTPException(status_code=400, detail=f"Type {expected} to continue.")


def _require_staging_recovery() -> None:
    if os.getenv("JUPR_ENV", "").strip().lower() != "staging":
        raise HTTPException(status_code=403, detail="Challenge Ladder operation recovery is staging-only.")


def _prepare_write(
    get_supabase_client,
    *,
    club_id: str,
    authorization: str | None,
    source: str,
    confirmation_text: str,
    expected_confirmation: str,
) -> tuple[Any, str, str]:
    authenticate_bearer(authorization)
    _require_service_role()
    try:
        require_staging_write_gate(surface_label="Challenge Ladder", flag_name=CHALLENGE_LADDER_WRITE_FLAG)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    supabase = get_supabase_client()
    actor_email, actor_role = _resolve_role_or_403(
        supabase=supabase,
        club_id=str(club_id),
        authorization=authorization,
        source=source,
    )
    _require_confirmation(confirmation_text, expected_confirmation)
    return supabase, actor_email, actor_role


def install_admin_challenge_ladder_routes(app, *, get_supabase_client) -> None:
    """Register guarded Challenge Ladder Admin routes."""

    @app.get("/admin/clubs/{club_id}/challenge-ladder/status")
    def get_admin_challenge_ladder_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_challenge_ladder_enabled() else None
        return build_admin_challenge_ladder_status(supabase, club_id=str(club_id))

    @app.get("/admin/clubs/{club_id}/challenge-ladder/dashboard")
    def get_admin_challenge_ladder_dashboard_route(club_id: str, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_challenge_ladder_admin_dashboard")
        try:
            return get_admin_challenge_ladder_dashboard(supabase, club_id=str(club_id))
        except Exception as exc:
            _handle(exc)

    @app.get("/admin/clubs/{club_id}/challenge-ladder/tier-movement-review")
    def get_admin_challenge_ladder_tier_movement_review_route(club_id: str, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_challenge_ladder_tier_movement_review")
        try:
            return get_admin_challenge_ladder_tier_movement_review(supabase, club_id=str(club_id))
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/challenge-ladder/challenges")
    def post_admin_challenge_ladder_challenge(club_id: str, payload: ChallengeCreateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase, actor_email, actor_role = _prepare_write(get_supabase_client, club_id=str(club_id), authorization=authorization, source=payload.source, confirmation_text=payload.confirmation_text, expected_confirmation=CONFIRM_CREATE)
        try:
            return _run_ladder_operation(
                supabase,
                club_id=str(club_id),
                operation_type="create_challenge",
                entity_id=f"{payload.challenger_id}:{payload.defender_id}:{payload.tier_id}",
                payload=payload,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: create_admin_challenge_ladder_challenge(
                    supabase,
                    club_id=str(club_id),
                    challenger_id=payload.challenger_id,
                    defender_id=payload.defender_id,
                    tier_id=payload.tier_id,
                    ledger_ref=payload.ledger_ref,
                    override=payload.override,
                    start_clock=payload.start_clock,
                    actor_email=actor_email,
                    actor_role=actor_role,
                    confirmation_text=payload.confirmation_text,
                    challenger_contact=payload.challenger_contact,
                    source=payload.source,
                ),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/start-clock")
    def post_admin_challenge_ladder_start_clock(club_id: str, challenge_id: int, payload: ChallengeSimpleConfirmationRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase, actor_email, actor_role = _prepare_write(get_supabase_client, club_id=str(club_id), authorization=authorization, source=payload.source, confirmation_text=payload.confirmation_text, expected_confirmation=CONFIRM_CLOCK)
        try:
            return _run_ladder_operation(
                supabase,
                club_id=str(club_id),
                operation_type="start_clock",
                entity_id=str(challenge_id),
                payload=payload,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: start_admin_challenge_ladder_clock(supabase, club_id=str(club_id), challenge_id=int(challenge_id), actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/accept")
    def post_admin_challenge_ladder_accept(club_id: str, challenge_id: int, payload: ChallengeSimpleConfirmationRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase, actor_email, actor_role = _prepare_write(get_supabase_client, club_id=str(club_id), authorization=authorization, source=payload.source, confirmation_text=payload.confirmation_text, expected_confirmation=CONFIRM_ACCEPT)
        try:
            return _run_ladder_operation(
                supabase,
                club_id=str(club_id),
                operation_type="accept_challenge",
                entity_id=str(challenge_id),
                payload=payload,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: accept_admin_challenge_ladder_challenge(supabase, club_id=str(club_id), challenge_id=int(challenge_id), actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/forfeit")
    def post_admin_challenge_ladder_forfeit(club_id: str, challenge_id: int, payload: ChallengeForfeitRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase, actor_email, actor_role = _prepare_write(get_supabase_client, club_id=str(club_id), authorization=authorization, source=payload.source, confirmation_text=payload.confirmation_text, expected_confirmation=CONFIRM_FORFEIT)
        try:
            return _run_ladder_operation(
                supabase,
                club_id=str(club_id),
                operation_type="record_forfeit",
                entity_id=str(challenge_id),
                payload=payload,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: record_admin_challenge_ladder_forfeit(supabase, club_id=str(club_id), challenge_id=int(challenge_id), forfeited_by_id=payload.forfeited_by_id, admin_note=payload.admin_note, actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/pass")
    def post_admin_challenge_ladder_pass(club_id: str, challenge_id: int, payload: ChallengePassRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase, actor_email, actor_role = _prepare_write(get_supabase_client, club_id=str(club_id), authorization=authorization, source=payload.source, confirmation_text=payload.confirmation_text, expected_confirmation=CONFIRM_PASS)
        try:
            return _run_ladder_operation(
                supabase,
                club_id=str(club_id),
                operation_type="record_pass",
                entity_id=str(challenge_id),
                payload=payload,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: record_admin_challenge_ladder_pass(
                    supabase,
                    club_id=str(club_id),
                    challenge_id=int(challenge_id),
                    player_id=payload.player_id,
                    actor_email=actor_email,
                    actor_role=actor_role,
                    confirmation_text=payload.confirmation_text,
                    source=payload.source,
                ),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/challenge-ladder/roster")
    def post_admin_challenge_ladder_roster(club_id: str, payload: ChallengeRosterAddRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase, actor_email, actor_role = _prepare_write(get_supabase_client, club_id=str(club_id), authorization=authorization, source=payload.source, confirmation_text=payload.confirmation_text, expected_confirmation=CONFIRM_ROSTER_ADD)
        try:
            return _run_ladder_operation(
                supabase,
                club_id=str(club_id),
                operation_type="add_roster_player",
                entity_id=str(payload.player_id),
                payload=payload,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: add_admin_challenge_ladder_roster_player(
                    supabase,
                    club_id=str(club_id),
                    player_id=payload.player_id,
                    tier_id=payload.tier_id,
                    admin_note=payload.admin_note,
                    actor_email=actor_email,
                    actor_role=actor_role,
                    confirmation_text=payload.confirmation_text,
                    source=payload.source,
                ),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/challenge-ladder/roster/{player_id}/move")
    def post_admin_challenge_ladder_roster_move(club_id: str, player_id: int, payload: ChallengeRosterMoveRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase, actor_email, actor_role = _prepare_write(get_supabase_client, club_id=str(club_id), authorization=authorization, source=payload.source, confirmation_text=payload.confirmation_text, expected_confirmation=CONFIRM_ROSTER_MOVE)
        try:
            return _run_ladder_operation(
                supabase,
                club_id=str(club_id),
                operation_type="move_roster_player",
                entity_id=str(player_id),
                payload=payload,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: move_admin_challenge_ladder_roster_player(
                    supabase,
                    club_id=str(club_id),
                    player_id=int(player_id),
                    destination_tier=payload.destination_tier,
                    recompress_old=payload.recompress_old,
                    admin_note=payload.admin_note,
                    actor_email=actor_email,
                    actor_role=actor_role,
                    confirmation_text=payload.confirmation_text,
                    source=payload.source,
                ),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/challenge-ladder/roster/replace-tier/preview")
    def post_admin_challenge_ladder_roster_replace_preview(club_id: str, payload: ChallengeRosterReplacePreviewRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return preview_admin_challenge_ladder_tier_roster_replacement(
                supabase,
                club_id=str(club_id),
                tier_id=payload.tier_id,
                ranked_names=payload.ranked_names,
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/challenge-ladder/roster/replace-tier")
    def post_admin_challenge_ladder_roster_replace(club_id: str, payload: ChallengeRosterReplaceRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase, actor_email, actor_role = _prepare_write(get_supabase_client, club_id=str(club_id), authorization=authorization, source=payload.source, confirmation_text=payload.confirmation_text, expected_confirmation=CONFIRM_ROSTER_REPLACE)
        try:
            return _run_ladder_operation(
                supabase,
                club_id=str(club_id),
                operation_type="replace_tier_roster",
                entity_id=str(payload.tier_id),
                payload=payload,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: replace_admin_challenge_ladder_tier_roster(
                    supabase,
                    club_id=str(club_id),
                    tier_id=payload.tier_id,
                    ranked_player_ids=payload.ranked_player_ids,
                    preview_fingerprint=payload.preview_fingerprint,
                    admin_note=payload.admin_note,
                    actor_email=actor_email,
                    actor_role=actor_role,
                    confirmation_text=payload.confirmation_text,
                    source=payload.source,
                ),
            )
        except Exception as exc:
            _handle(exc)

    @app.put("/admin/clubs/{club_id}/challenge-ladder/roster/{player_id}/overrides")
    @app.patch("/admin/clubs/{club_id}/challenge-ladder/roster/{player_id}/overrides")
    def put_admin_challenge_ladder_player_overrides(club_id: str, player_id: int, payload: ChallengePlayerOverridesRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase, actor_email, actor_role = _prepare_write(get_supabase_client, club_id=str(club_id), authorization=authorization, source=payload.source, confirmation_text=payload.confirmation_text, expected_confirmation=CONFIRM_OVERRIDES)
        try:
            return _run_ladder_operation(
                supabase,
                club_id=str(club_id),
                operation_type="save_player_overrides",
                entity_id=str(player_id),
                payload=payload,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: save_admin_challenge_ladder_player_overrides(
                    supabase,
                    club_id=str(club_id),
                    player_id=int(player_id),
                    vacation_until=payload.vacation_until,
                    reinstate_required=payload.reinstate_required,
                    reinstate_notes=payload.reinstate_notes,
                    actor_email=actor_email,
                    actor_role=actor_role,
                    confirmation_text=payload.confirmation_text,
                    source=payload.source,
                ),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/result")
    def post_admin_challenge_ladder_result(club_id: str, challenge_id: int, payload: ChallengeResultRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase, actor_email, actor_role = _prepare_write(get_supabase_client, club_id=str(club_id), authorization=authorization, source=payload.source, confirmation_text=payload.confirmation_text, expected_confirmation=CONFIRM_RESULT)
        try:
            replay = replay_durable_admin_operation_if_present(
                supabase,
                club_id=str(club_id),
                surface="challenge_ladder",
                operation_type="publish_result",
                entity_id=str(challenge_id),
                idempotency_key=payload.idempotency_key,
                request_payload=_model_payload(payload),
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
            )
            if replay is not None:
                return replay
            current_preview = preview_admin_challenge_ladder_result_for_challenge(
                supabase,
                club_id=str(club_id),
                challenge_id=int(challenge_id),
                partner_a_challenger_id=payload.partner_a_challenger_id,
                partner_a_defender_id=payload.partner_a_defender_id,
                partner_b_challenger_id=payload.partner_b_challenger_id,
                partner_b_defender_id=payload.partner_b_defender_id,
                match_a_games=payload.match_a_games,
                match_b_games=payload.match_b_games,
                match_date=payload.match_date,
                winner_override=payload.winner_override,
                publish_official_matches=payload.publish_official_matches,
            )
            if not payload.preview_fingerprint or payload.preview_fingerprint != current_preview.get("preview_fingerprint"):
                raise LiveLadderConflictError("Ladder result changed. Review the Python preview again before official publish.")
            operation_key = deterministic_operation_key(
                club_id=str(club_id),
                surface="challenge_ladder",
                operation_type="publish_result",
                entity_id=str(challenge_id),
                idempotency_key=payload.idempotency_key,
            )
            contexts = (
                [
                    deterministic_match_context_id(operation_key=operation_key, slot="a"),
                    deterministic_match_context_id(operation_key=operation_key, slot="b"),
                ]
                if payload.publish_official_matches
                else []
            )
            return _run_ladder_operation(
                supabase,
                club_id=str(club_id),
                operation_type="publish_result",
                entity_id=str(challenge_id),
                payload=payload,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                match_context_ids=contexts,
                mutate=lambda: record_admin_challenge_ladder_result(
                    supabase,
                    club_id=str(club_id),
                    challenge_id=int(challenge_id),
                    partner_a_challenger_id=payload.partner_a_challenger_id,
                    partner_a_defender_id=payload.partner_a_defender_id,
                    partner_b_challenger_id=payload.partner_b_challenger_id,
                    partner_b_defender_id=payload.partner_b_defender_id,
                    match_a_games=payload.match_a_games,
                    match_b_games=payload.match_b_games,
                    match_date=payload.match_date,
                    winner_override=payload.winner_override,
                    publish_official_matches=payload.publish_official_matches,
                    actor_email=actor_email,
                    actor_role=actor_role,
                    confirmation_text=payload.confirmation_text,
                    expected_preview_fingerprint=payload.preview_fingerprint,
                    publish_context_prefix=operation_key,
                    source=payload.source,
                ),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/result/preview")
    def post_admin_challenge_ladder_result_preview(club_id: str, challenge_id: int, payload: ChallengeResultPreviewRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return preview_admin_challenge_ladder_result_for_challenge(
                supabase,
                club_id=str(club_id),
                challenge_id=int(challenge_id),
                partner_a_challenger_id=payload.partner_a_challenger_id,
                partner_a_defender_id=payload.partner_a_defender_id,
                partner_b_challenger_id=payload.partner_b_challenger_id,
                partner_b_defender_id=payload.partner_b_defender_id,
                match_a_games=payload.match_a_games,
                match_b_games=payload.match_b_games,
                match_date=payload.match_date,
                winner_override=payload.winner_override,
                publish_official_matches=payload.publish_official_matches,
            )
        except Exception as exc:
            _handle(exc)

    @app.patch("/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}")
    def patch_admin_challenge_ladder_challenge(club_id: str, challenge_id: int, payload: ChallengeUpdateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase, actor_email, actor_role = _prepare_write(get_supabase_client, club_id=str(club_id), authorization=authorization, source=payload.source, confirmation_text=payload.confirmation_text, expected_confirmation=CONFIRM)
        try:
            return _run_ladder_operation(
                supabase,
                club_id=str(club_id),
                operation_type="update_challenge",
                entity_id=str(challenge_id),
                payload=payload,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: update_admin_challenge_ladder_challenge(
                    supabase,
                    club_id=str(club_id),
                    challenge_id=int(challenge_id),
                    status=payload.status,
                    admin_note=payload.admin_note,
                    actor_email=actor_email,
                    actor_role=actor_role,
                    confirmation_text=payload.confirmation_text,
                    source=payload.source,
                ),
            )
        except Exception as exc:
            _handle(exc)

    @app.get("/admin/clubs/{club_id}/challenge-ladder/operations/{operation_key}")
    def get_admin_challenge_ladder_operation(club_id: str, operation_key: str, authorization: str | None = auth_header()) -> dict[str, Any]:
        authenticate_bearer(authorization)
        _require_service_role()
        _require_staging_recovery()
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_challenge_ladder_operation_status")
        try:
            return get_durable_admin_operation(supabase, club_id=str(club_id), operation_key=str(operation_key), surface="challenge_ladder")
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/challenge-ladder/operations/{operation_key}/reconcile")
    def post_admin_challenge_ladder_reconcile(club_id: str, operation_key: str, payload: ChallengeReconcileRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
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
                surface="challenge_ladder",
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                expected_confirmation=CONFIRM_RECONCILE_LADDER,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)
