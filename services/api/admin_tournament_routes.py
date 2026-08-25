from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import HTTPException, Query, Response
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import (
    PERMISSION_ENTER_SCORES,
    PERMISSION_MANAGE_MATCHES,
    PERMISSION_MANAGE_PLAYERS,
    PERMISSION_MANAGE_TOURNAMENTS,
    has_permission,
    resolve_admin_role,
)
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.tournament_registration_repo import (
    StaleTournamentRegistrationAdminError,
    StaleTournamentRegistrationSelectionError,
)
from jupr_app.services.admin_tournament_award_service import award_admin_tournament_draw_podium
from jupr_app.services.admin_tournament_bulk_service import bulk_update_admin_tournament_registrations
from jupr_app.services.admin_tournament_bulk_team_import_service import import_admin_tournament_bulk_teams
from jupr_app.services.admin_tournament_delete_service import delete_admin_tournament_draft
from jupr_app.services.admin_tournament_draw_service import (
    cancel_admin_tournament_empty_event,
    cancel_admin_tournament_empty_draw,
    create_admin_tournament_draw,
)
from jupr_app.services.admin_tournament_game_service import (
    generate_admin_tournament_round_robin_games,
    rebuild_admin_tournament_round_robin_games,
    reconcile_admin_tournament_round_robin_games,
)
from jupr_app.services.admin_tournament_match_publish_service import (
    build_admin_tournament_official_publish_plan,
    publish_admin_tournament_draw_matches,
    reconcile_admin_tournament_official_publish,
)
from jupr_app.domain.tournament_admin_operations import (
    build_tournament_admin_operation_request,
    stable_tournament_admin_fingerprint,
)
from jupr_app.services.admin_tournament_ops_service import (
    get_admin_tournament_ops_snapshot,
    get_admin_tournament_ops_state_fingerprint,
    require_admin_tournament_official_publish_runtime,
)
from jupr_app.services.admin_tournament_playoff_service import generate_admin_tournament_playoff_games
from jupr_app.services.admin_tournament_podium_service import generate_admin_tournament_draw_podium
from jupr_app.services.admin_tournament_podium_review_service import (
    review_admin_tournament_draw_podium,
)
from jupr_app.services.admin_tournament_registration_import_service import import_admin_tournament_registrations_to_draw
from jupr_app.services.admin_tournament_registration_import_recovery_service import (
    REGISTRATION_IMPORT_RECONCILE_CONFIRMATION,
    reconcile_admin_tournament_registration_import_operation,
)
from jupr_app.services.admin_tournament_recovery_reconciliation_service import (
    reconcile_admin_tournament_ops_recovery,
)
from jupr_app.services.admin_tournament_results_import_service import (
    apply_admin_tournament_results_import,
    build_admin_tournament_results_import_preview,
)
from jupr_app.services.admin_tournament_registration_reporting_service import (
    build_admin_tournament_broadcast_preview,
    build_admin_tournament_registration_export,
)
from jupr_app.services.admin_tournament_score_service import update_admin_tournament_game_score
from jupr_app.services.admin_tournament_service import (
    build_admin_tournament_registration_import_handoff,
    build_admin_tournament_status,
    create_admin_tournament_selection,
    delete_admin_tournament_selection,
    get_admin_tournament_detail,
    replace_admin_tournament_selection_partner,
    is_admin_tournament_admin_enabled,
    list_admin_tournaments,
    update_admin_tournament_registration,
    update_admin_tournament_selection,
    update_admin_tournament,
)
from jupr_app.services.admin_tournament_guarded_operation import (
    StaleTournamentAdminStateError,
    TournamentAdminRecoveryRequiredError,
    require_tournament_admin_mutation_runtime,
    run_tournament_admin_guarded_operation,
    tournament_admin_guarded_runtime_enabled,
)
from jupr_app.services.admin_tournament_lifecycle_service import (
    require_admin_tournament_official_publish_readiness,
)
from jupr_app.services.admin_tournament_status_service import (
    apply_admin_tournament_status_action,
    reconcile_admin_tournament_status_action,
)
from jupr_app.services.admin_tournament_team_service import replace_admin_tournament_draw_teams
from jupr_app.services.admin_tournament_team_competition_service import (
    is_admin_team_tournament_enabled,
)
from services.api.auth import authenticate_bearer, auth_header


class AdminTournamentRegistrationUpdateRequest(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    display_name: str | None = None
    email: str | None = None
    phone: str | None = None
    player_id: int | None = None
    gender: str | None = None
    age: int | None = None
    age_bracket: str | None = None
    dupr_id: str | None = None
    doubles_skill: float | None = None
    singles_skill: float | None = None
    wants_partner_board_contact: bool | None = None
    registration_status: str | None = None
    payment_status: str | None = None
    notes: str | None = None
    expected_updated_at: str
    confirmation_text: str = ""
    source: str = "next_tournament_admin_registration_update"


class AdminTournamentBroadcastPreviewRequest(BaseModel):
    subject: str = ""
    message: str = ""
    include_cancelled: bool = False
    registration_status: str | None = None
    payment_status: str | None = None
    partner_mode: str | None = None
    registration_day_id: str | None = None
    event_option_id: str | None = None
    search: str | None = None


class AdminTournamentRegistrationBulkUpdateRequest(BaseModel):
    registration_ids: list[str] = Field(default_factory=list)
    registration_status: str | None = None
    payment_status: str | None = None
    append_note: str | None = None
    expected_state_fingerprint: str | None = None
    expected_versions: dict[str, str] = Field(default_factory=dict)
    confirmation_text: str = ""
    source: str = "next_tournament_admin_registration_bulk_update"


class AdminTournamentStatusActionRequest(BaseModel):
    action: str
    expected_updated_at: str | None = None
    confirmation_text: str = ""
    source: str = "next_tournament_admin_status_action"


class AdminTournamentDeleteDraftRequest(BaseModel):
    expected_updated_at: str | None = None
    confirmation_text: str = ""
    source: str = "next_tournament_admin_delete_draft"


class AdminTournamentDrawCreateRequest(BaseModel):
    registration_day_id: str | None = None
    event_option_id: str | None = None
    name: str | None = None
    expected_state_fingerprint: str | None = None
    confirmation_text: str = ""
    source: str = "next_tournament_admin_create_draw"


class AdminTournamentDrawTeamRow(BaseModel):
    team_number: int
    player1_id: int
    player2_id: int | None = None
    seed: int | None = None
    source: str | None = "MANUAL"
    notes: str | None = None


class AdminTournamentDrawTeamsReplaceRequest(BaseModel):
    teams: list[AdminTournamentDrawTeamRow] = Field(default_factory=list)
    expected_state_fingerprint: str | None = None
    expected_draw_updated_at: str | None = None
    confirmation_text: str = ""
    source: str = "next_tournament_admin_replace_teams"


class AdminTournamentRegistrationImportRequest(BaseModel):
    import_mode: str = "REPLACE"
    idempotency_key: UUID
    expected_state_fingerprint: str | None = None
    expected_draw_updated_at: str | None = None
    confirmation_text: str = ""
    source: str = "next_tournament_admin_import_registrations"


class AdminTournamentRegistrationImportReconcileRequest(BaseModel):
    retained_request: AdminTournamentRegistrationImportRequest
    confirmation_text: str = Field(default="", max_length=120)
    source: str = Field(
        default="next_tournament_ops_registration_import_reconcile",
        max_length=180,
    )


class AdminTournamentBulkTeamImportRequest(BaseModel):
    raw_text: str = ""
    import_mode: str = "REPLACE"
    expected_state_fingerprint: str | None = None
    expected_draw_updated_at: str | None = None
    confirmation_text: str = ""
    source: str = "next_tournament_admin_import_bulk_teams"


class AdminTournamentResultsImportPreviewRequest(BaseModel):
    raw_text: str = ""
    import_mode: str = "REPLACE"
    mapping_decisions: dict[str, dict[str, Any]] | None = None
    match_reviews: dict[str, dict[str, Any]] | None = None
    podium_refs: dict[str, str | None] | None = None
    allow_duplicate_mapping: bool = False


class AdminTournamentResultsImportCommitRequest(BaseModel):
    raw_text: str = ""
    import_mode: str = "REPLACE"
    mapping_decisions: dict[str, dict[str, Any]] = Field(default_factory=dict)
    match_reviews: dict[str, dict[str, Any]] = Field(default_factory=dict)
    podium_refs: dict[str, str | None] = Field(default_factory=dict)
    allow_duplicate_mapping: bool = False
    unusual_scores_acknowledged: bool = False
    expected_review_fingerprint: str = ""
    expected_state_fingerprint: str | None = None
    expected_draw_updated_at: str | None = None
    confirmation_text: str = ""
    source: str = "next_tournament_ops_results_import"


class AdminTournamentRowVersion(BaseModel):
    id: str
    updated_at: str


class AdminTournamentRoundRobinGenerateRequest(BaseModel):
    expected_state_fingerprint: str | None = None
    expected_draw_updated_at: str | None = None
    expected_team_versions: list[AdminTournamentRowVersion] = Field(default_factory=list)
    confirmation_text: str = ""
    source: str = "next_tournament_admin_generate_round_robin"


class AdminTournamentEmptyDrawCancelRequest(BaseModel):
    expected_state_fingerprint: str | None = None
    expected_draw_updated_at: str
    confirmation_text: str = ""
    source: str = "next_tournament_admin_cancel_empty_draw"


class AdminTournamentEmptyEventCancelRequest(BaseModel):
    expected_state_fingerprint: str | None = None
    confirmation_text: str = ""
    source: str = "next_tournament_admin_cancel_empty_event"


class AdminTournamentPlayoffGenerateRequest(BaseModel):
    advance_count: int | None = None
    expected_state_fingerprint: str | None = None
    expected_draw_updated_at: str | None = None
    expected_team_versions: list[AdminTournamentRowVersion] = Field(default_factory=list)
    expected_source_game_versions: list[AdminTournamentRowVersion] = Field(default_factory=list)
    confirmation_text: str = ""
    source: str = "next_tournament_admin_generate_playoffs"


class AdminTournamentPodiumGenerateRequest(BaseModel):
    expected_state_fingerprint: str | None = None
    expected_draw_updated_at: str | None = None
    expected_team_versions: list[AdminTournamentRowVersion] = Field(default_factory=list)
    expected_source_game_versions: list[AdminTournamentRowVersion] = Field(default_factory=list)
    confirmation_text: str = ""
    source: str = "next_tournament_admin_generate_podium"


class AdminTournamentPodiumAwardRequest(BaseModel):
    expected_state_fingerprint: str | None = None
    expected_draw_updated_at: str | None = None
    expected_team_versions: list[AdminTournamentRowVersion] = Field(default_factory=list)
    expected_source_game_versions: list[AdminTournamentRowVersion] = Field(default_factory=list)
    expected_podium_versions: list[AdminTournamentRowVersion] = Field(default_factory=list)
    expected_podium: list[dict[str, Any]] = Field(default_factory=list)
    expected_awards: list[dict[str, Any]] = Field(default_factory=list)
    confirmation_text: str = ""
    source: str = "next_tournament_admin_award_podium"


class AdminTournamentPodiumReviewRequest(BaseModel):
    expected_state_fingerprint: str = Field(min_length=64, max_length=64)
    expected_draw_updated_at: str = Field(min_length=1, max_length=120)
    expected_team_versions: list[AdminTournamentRowVersion]
    expected_source_game_versions: list[AdminTournamentRowVersion]
    confirmation_text: str = Field(default="", max_length=120)
    source: str = Field(default="next_tournament_admin_review_podium", max_length=180)


class AdminTournamentOfficialMatchPublishRequest(BaseModel):
    playoff_winner_bonus_elo: float | None = 0.0
    expected_state_fingerprint: str | None = None
    confirmation_text: str = ""
    source: str = "next_tournament_admin_publish_matches"


class AdminTournamentGameScoreRequest(BaseModel):
    score_a: int
    score_b: int
    unusual_score_acknowledged: bool = False
    expected_state_fingerprint: str | None = None
    expected_game_updated_at: str | None = None
    expected_draw_updated_at: str | None = None
    confirmation_text: str = ""
    source: str = "next_tournament_admin_score_game"


class AdminTournamentSelectionUpdateRequest(BaseModel):
    expected_updated_at: str
    event_option_id: str | None = None
    partner_mode: str | None = None
    partner_name: str | None = None
    partner_email: str | None = None
    partner_phone: str | None = None
    partner_dupr_id: str | None = None
    partner_skill: float | None = None
    partner_age: int | None = None
    partner_gender: str | None = None
    partner_note: str | None = None
    show_on_partner_board: bool | None = None
    confirmation_text: str = ""
    source: str = "next_tournament_admin_selection_update"


class AdminTournamentSelectionCreateRequest(BaseModel):
    event_option_id: str
    partner_mode: str = Field(
        default="NONE", pattern=r"^(NONE|HAS_PARTNER|NEEDS_PARTNER)$"
    )
    partner_name: str | None = None
    partner_email: str | None = None
    partner_phone: str | None = None
    partner_dupr_id: str | None = None
    partner_skill: float | None = None
    partner_age: int | None = None
    partner_gender: str | None = None
    partner_note: str | None = None
    show_on_partner_board: bool | None = None
    expected_state_fingerprint: str
    confirmation_text: str = ""
    source: str = "next_tournament_registration_detail"


class AdminTournamentSelectionDeleteRequest(BaseModel):
    expected_updated_at: str
    confirmation_text: str = ""
    source: str = "next_tournament_registration_detail"


class AdminTournamentSelectionPartnerRequest(BaseModel):
    partner_selection_id: str | None = None
    unpaired_mode: str = Field(default="NEEDS_PARTNER", pattern=r"^(NONE|NEEDS_PARTNER)$")
    expected_updated_at: str
    confirmation_text: str = ""
    source: str = "next_tournament_registration_detail"


class AdminTournamentUpdateRequest(BaseModel):
    name: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    expected_updated_at: str | None = None
    confirmation_text: str = ""
    source: str = "next_tournament_admin_tournament_update"


def _dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def _dump_patch_model(model: BaseModel) -> dict[str, Any]:
    """Keep explicitly supplied nulls so editable fields can be cleared."""

    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_unset=True)
    return model.dict(exclude_unset=True)


def _resolve_tournament_role_or_403(
    *,
    supabase: Any,
    club_id: str,
    authorization: str | None,
    source: str,
    required_permissions: tuple[str, ...] = (PERMISSION_MANAGE_TOURNAMENTS,),
    require_all: bool = True,
) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(supabase=supabase, club_id=str(club_id), email=user.email, user_id=user.user_id, allowlist=set())
    permission_checks = [has_permission(role_resolution.role, permission) for permission in required_permissions]
    permitted = all(permission_checks) if require_all else any(permission_checks)
    if not permitted:
        denied_payload = build_activity_payload(
            club_id=str(club_id),
            actor_email=user.email,
            actor_role=role_resolution.role,
            action_type="admin_tournament_denied",
            entity_type="tournament_admin",
            entity_id="tournament_admin",
            after_json={
                "source_client": "fastapi/nextjs",
                "reason": "insufficient_permission",
                "required_permissions": list(required_permissions),
                "require_all": bool(require_all),
            },
            source_page=source,
            flagged_for_review=True,
        )
        write_admin_activity_log(supabase, denied_payload)
        raise HTTPException(status_code=403, detail="insufficient permission")
    return user.email, role_resolution.role


def _handle(exc: Exception) -> None:
    if isinstance(
        exc,
        (
            StaleTournamentRegistrationAdminError,
            StaleTournamentRegistrationSelectionError,
            StaleTournamentAdminStateError,
            TournamentAdminRecoveryRequiredError,
        ),
    ):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise exc


def _require_confirmation(actual: str, expected: str) -> None:
    if str(actual or "").strip().upper() != str(expected):
        raise ValueError(f"Type {expected} to confirm this Tournament Admin mutation.")


def _guarded_admin_mutation(
    supabase: Any,
    *,
    club_id: str,
    surface: str,
    action: str,
    entity_type: str,
    entity_id: str,
    lock_scope: str | None = None,
    expected_state: str,
    current_state,
    payload: dict[str, Any],
    actor_email: str,
    actor_role: str,
    source: str,
    preflight=None,
    reconcile=None,
    mutate,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    require_tournament_admin_mutation_runtime(surface)
    if not tournament_admin_guarded_runtime_enabled(surface):
        return mutate()
    return run_tournament_admin_guarded_operation(
        supabase,
        club_id=str(club_id),
        surface=surface,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        lock_scope=lock_scope,
        expected_state=expected_state,
        current_state=current_state,
        payload=payload,
        actor_email=actor_email,
        actor_role=actor_role,
        source=source,
        preflight=preflight,
        reconcile=reconcile,
        mutate=mutate,
        idempotency_key=idempotency_key,
    )


def _guarded_ops_mutation(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    action: str,
    entity_type: str,
    entity_id: str,
    expected_state: str,
    payload: dict[str, Any],
    actor_email: str,
    actor_role: str,
    source: str,
    preflight,
    reconcile=None,
    mutate,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    return _guarded_admin_mutation(
        supabase,
        club_id=str(club_id),
        surface="operations",
        action=action,
        entity_type=entity_type,
        entity_id=str(entity_id),
        lock_scope=str(tournament_id),
        expected_state=str(expected_state or ""),
        current_state=lambda: get_admin_tournament_ops_state_fingerprint(
            supabase,
            club_id=str(club_id),
            tournament_id=str(tournament_id),
        ),
        payload=payload,
        actor_email=actor_email,
        actor_role=actor_role,
        source=source,
        preflight=preflight,
        reconcile=reconcile,
        mutate=mutate,
        idempotency_key=idempotency_key,
    )


def install_admin_tournament_routes(app, *, get_supabase_client) -> None:
    """Register guarded Tournament Admin routes for the Next admin pilot."""

    @app.get("/admin/clubs/{club_id}/tournaments/admin/status")
    def get_admin_tournament_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_tournament_admin_enabled() else None
        return build_admin_tournament_status(supabase, club_id=str(club_id))

    @app.get("/admin/clubs/{club_id}/tournaments/admin/tournaments")
    def get_admin_tournaments(club_id: str, include_archived: bool = Query(default=False), authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_tournament_admin_list")
        try:
            return list_admin_tournaments(supabase, club_id=str(club_id), include_archived=bool(include_archived))
        except Exception as exc:
            _handle(exc)

    @app.get("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/ops")
    def get_admin_tournament_ops(club_id: str, tournament_id: str, draw_id: str | None = Query(default=None), authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_tournament_admin_ops",
            required_permissions=(PERMISSION_MANAGE_TOURNAMENTS, PERMISSION_ENTER_SCORES),
            require_all=False,
        )
        try:
            return get_admin_tournament_ops_snapshot(supabase, club_id=str(club_id), tournament_id=str(tournament_id), draw_id=draw_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            _handle(exc)

    @app.get("/admin/clubs/{club_id}/tournaments/admin/ops/tournaments")
    def get_admin_tournament_ops_tournaments(
        club_id: str,
        include_archived: bool = Query(default=False),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_tournament_ops_list",
            required_permissions=(PERMISSION_MANAGE_TOURNAMENTS, PERMISSION_ENTER_SCORES),
            require_all=False,
        )
        try:
            return list_admin_tournaments(
                supabase,
                club_id=str(club_id),
                include_archived=bool(include_archived),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws")
    def post_admin_tournament_draw(club_id: str, tournament_id: str, payload: AdminTournamentDrawCreateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            _require_confirmation(payload.confirmation_text, "CREATE DRAW")
            kwargs = {
                "club_id": str(club_id),
                "tournament_id": str(tournament_id),
                "registration_day_id": payload.registration_day_id,
                "event_option_id": payload.event_option_id,
                "name": payload.name,
                "actor_email": actor_email,
                "actor_role": actor_role,
                "confirmation_text": payload.confirmation_text,
                "source": payload.source,
            }
            return _guarded_ops_mutation(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                action="ops_draw_create",
                entity_type="tournament",
                entity_id=str(tournament_id),
                expected_state=str(payload.expected_state_fingerprint or ""),
                payload={
                    "registration_day_id": payload.registration_day_id,
                    "event_option_id": payload.event_option_id,
                    "name": payload.name,
                },
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                preflight=lambda: create_admin_tournament_draw(supabase, **kwargs, dry_run=True),
                mutate=lambda: create_admin_tournament_draw(supabase, **kwargs),
            )
        except Exception as exc:
            _handle(exc)

    @app.put("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/teams")
    def put_admin_tournament_draw_teams(club_id: str, tournament_id: str, draw_id: str, payload: AdminTournamentDrawTeamsReplaceRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            _require_confirmation(payload.confirmation_text, "SAVE TEAMS")
            team_rows = [_dump_model(team) for team in payload.teams]
            kwargs = {
                "club_id": str(club_id),
                "tournament_id": str(tournament_id),
                "draw_id": str(draw_id),
                "teams": team_rows,
                "actor_email": actor_email,
                "actor_role": actor_role,
                "confirmation_text": payload.confirmation_text,
                "expected_draw_updated_at": payload.expected_draw_updated_at,
                "source": payload.source,
                "atomic": tournament_admin_guarded_runtime_enabled("operations"),
            }
            return _guarded_ops_mutation(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                action="ops_teams_replace",
                entity_type="tournament_event_draw",
                entity_id=str(draw_id),
                expected_state=str(payload.expected_state_fingerprint or ""),
                payload={
                    "draw_id": str(draw_id),
                    "teams": team_rows,
                    "expected_draw_updated_at": payload.expected_draw_updated_at,
                },
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                preflight=lambda: replace_admin_tournament_draw_teams(supabase, **kwargs, dry_run=True),
                mutate=lambda: replace_admin_tournament_draw_teams(supabase, **kwargs),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/teams/import-registrations")
    def post_admin_tournament_registration_team_import(club_id: str, tournament_id: str, draw_id: str, payload: AdminTournamentRegistrationImportRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            _require_confirmation(payload.confirmation_text, "IMPORT REGISTRATIONS")
            kwargs = {
                "club_id": str(club_id),
                "tournament_id": str(tournament_id),
                "draw_id": str(draw_id),
                "import_mode": payload.import_mode,
                "actor_email": actor_email,
                "actor_role": actor_role,
                "confirmation_text": payload.confirmation_text,
                "expected_draw_updated_at": payload.expected_draw_updated_at,
                "source": payload.source,
                "atomic": tournament_admin_guarded_runtime_enabled("operations"),
            }
            return _guarded_ops_mutation(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                action="ops_registration_import",
                entity_type="tournament_event_draw",
                entity_id=str(draw_id),
                expected_state=str(payload.expected_state_fingerprint or ""),
                payload={
                    "draw_id": str(draw_id),
                    "import_mode": payload.import_mode,
                    "expected_draw_updated_at": payload.expected_draw_updated_at,
                },
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                preflight=lambda: import_admin_tournament_registrations_to_draw(supabase, **kwargs, dry_run=True),
                mutate=lambda: import_admin_tournament_registrations_to_draw(supabase, **kwargs),
                idempotency_key=str(payload.idempotency_key),
            )
        except TournamentAdminRecoveryRequiredError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "kind": "uncertain",
                    "code": "TOURNAMENT_REGISTRATION_IMPORT_RECOVERY_REQUIRED",
                    "message": str(exc),
                    "recovery_required": True,
                    "operation_reference": str(payload.idempotency_key),
                },
            ) from exc
        except Exception as exc:
            _handle(exc)

    @app.post(
        "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/teams/import-registrations/operations/{operation_reference}/reconcile"
    )
    def post_admin_tournament_registration_team_import_reconcile(
        club_id: str,
        tournament_id: str,
        draw_id: str,
        operation_reference: str,
        payload: AdminTournamentRegistrationImportReconcileRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
            required_permissions=(PERMISSION_MANAGE_TOURNAMENTS,),
        )
        try:
            if str(payload.confirmation_text or "").strip() != REGISTRATION_IMPORT_RECONCILE_CONFIRMATION:
                raise ValueError(
                    f"Type {REGISTRATION_IMPORT_RECONCILE_CONFIRMATION} exactly to reconcile this registration import."
                )
            return reconcile_admin_tournament_registration_import_operation(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                draw_id=str(draw_id),
                operation_reference=str(operation_reference),
                retained_request=payload.retained_request.model_dump(mode="json"),
                confirmation_text=payload.confirmation_text,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/teams/import-bulk")
    def post_admin_tournament_bulk_team_import(club_id: str, tournament_id: str, draw_id: str, payload: AdminTournamentBulkTeamImportRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            _require_confirmation(payload.confirmation_text, "IMPORT TEAMS")
            kwargs = {
                "club_id": str(club_id),
                "tournament_id": str(tournament_id),
                "draw_id": str(draw_id),
                "raw_text": payload.raw_text,
                "import_mode": payload.import_mode,
                "actor_email": actor_email,
                "actor_role": actor_role,
                "confirmation_text": payload.confirmation_text,
                "expected_draw_updated_at": payload.expected_draw_updated_at,
                "source": payload.source,
                "atomic": tournament_admin_guarded_runtime_enabled("operations"),
            }
            return _guarded_ops_mutation(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                action="ops_bulk_team_import",
                entity_type="tournament_event_draw",
                entity_id=str(draw_id),
                expected_state=str(payload.expected_state_fingerprint or ""),
                payload={
                    "draw_id": str(draw_id),
                    "import_mode": payload.import_mode,
                    "raw_text_fingerprint": stable_tournament_admin_fingerprint(payload.raw_text),
                    "expected_draw_updated_at": payload.expected_draw_updated_at,
                },
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                preflight=lambda: import_admin_tournament_bulk_teams(supabase, **kwargs, dry_run=True),
                mutate=lambda: import_admin_tournament_bulk_teams(supabase, **kwargs),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/games/round-robin")
    def post_admin_tournament_round_robin_games(club_id: str, tournament_id: str, draw_id: str, payload: AdminTournamentRoundRobinGenerateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            _require_confirmation(payload.confirmation_text, "GENERATE GAMES")
            kwargs = {
                "club_id": str(club_id),
                "tournament_id": str(tournament_id),
                "draw_id": str(draw_id),
                "actor_email": actor_email,
                "actor_role": actor_role,
                "confirmation_text": payload.confirmation_text,
                "expected_draw_updated_at": payload.expected_draw_updated_at,
                "expected_team_versions": [_dump_model(row) for row in payload.expected_team_versions],
                "source": payload.source,
                "atomic": tournament_admin_guarded_runtime_enabled("operations"),
            }
            return _guarded_ops_mutation(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                action="ops_round_robin_generate",
                entity_type="tournament_event_draw",
                entity_id=str(draw_id),
                expected_state=str(payload.expected_state_fingerprint or ""),
                payload={
                    "draw_id": str(draw_id),
                    "expected_draw_updated_at": payload.expected_draw_updated_at,
                    "expected_team_versions": [_dump_model(row) for row in payload.expected_team_versions],
                },
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                preflight=lambda: generate_admin_tournament_round_robin_games(supabase, **kwargs, dry_run=True),
                mutate=lambda: generate_admin_tournament_round_robin_games(supabase, **kwargs),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/games/round-robin/reconcile")
    def post_admin_tournament_round_robin_games_reconcile(
        club_id: str,
        tournament_id: str,
        draw_id: str,
        payload: AdminTournamentRoundRobinGenerateRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            _require_confirmation(payload.confirmation_text, "RECONCILE GAMES")
            kwargs = {
                "club_id": str(club_id),
                "tournament_id": str(tournament_id),
                "draw_id": str(draw_id),
                "actor_email": actor_email,
                "actor_role": actor_role,
                "confirmation_text": payload.confirmation_text,
                "expected_draw_updated_at": payload.expected_draw_updated_at,
                "expected_team_versions": [
                    _dump_model(row) for row in payload.expected_team_versions
                ],
                "source": payload.source,
                "atomic": True,
            }
            return _guarded_ops_mutation(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                action="ops_round_robin_reconcile",
                entity_type="tournament_event_draw",
                entity_id=str(draw_id),
                expected_state=str(payload.expected_state_fingerprint or ""),
                payload={
                    "draw_id": str(draw_id),
                    "expected_draw_updated_at": payload.expected_draw_updated_at,
                    "expected_team_versions": [
                        _dump_model(row) for row in payload.expected_team_versions
                    ],
                    "preserve_existing_games": True,
                },
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                preflight=lambda: reconcile_admin_tournament_round_robin_games(
                    supabase, **kwargs, dry_run=True
                ),
                reconcile=lambda operation: reconcile_admin_tournament_ops_recovery(
                    supabase,
                    club_id=str(club_id),
                    tournament_id=str(tournament_id),
                    action="ops_round_robin_reconcile",
                    entity_id=str(draw_id),
                    operation=operation,
                ),
                mutate=lambda: reconcile_admin_tournament_round_robin_games(
                    supabase, **kwargs
                ),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/games/round-robin/rebuild")
    def post_admin_tournament_round_robin_games_rebuild(
        club_id: str,
        tournament_id: str,
        draw_id: str,
        payload: AdminTournamentRoundRobinGenerateRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            _require_confirmation(payload.confirmation_text, "REBUILD GAMES")
            kwargs = {
                "club_id": str(club_id),
                "tournament_id": str(tournament_id),
                "draw_id": str(draw_id),
                "actor_email": actor_email,
                "actor_role": actor_role,
                "confirmation_text": payload.confirmation_text,
                "expected_draw_updated_at": payload.expected_draw_updated_at,
                "expected_team_versions": [
                    _dump_model(row) for row in payload.expected_team_versions
                ],
                "source": payload.source,
                "atomic": True,
            }
            return _guarded_ops_mutation(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                action="ops_round_robin_rebuild",
                entity_type="tournament_event_draw",
                entity_id=str(draw_id),
                expected_state=str(payload.expected_state_fingerprint or ""),
                payload={
                    "draw_id": str(draw_id),
                    "expected_draw_updated_at": payload.expected_draw_updated_at,
                    "expected_team_versions": [
                        _dump_model(row) for row in payload.expected_team_versions
                    ],
                    "replace_unstarted_games": True,
                },
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                preflight=lambda: rebuild_admin_tournament_round_robin_games(
                    supabase, **kwargs, dry_run=True
                ),
                reconcile=lambda operation: reconcile_admin_tournament_ops_recovery(
                    supabase,
                    club_id=str(club_id),
                    tournament_id=str(tournament_id),
                    action="ops_round_robin_rebuild",
                    entity_id=str(draw_id),
                    operation=operation,
                ),
                mutate=lambda: rebuild_admin_tournament_round_robin_games(
                    supabase, **kwargs
                ),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/cancel-empty")
    def post_admin_tournament_empty_draw_cancel(
        club_id: str,
        tournament_id: str,
        draw_id: str,
        payload: AdminTournamentEmptyDrawCancelRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            _require_confirmation(payload.confirmation_text, "CANCEL EMPTY DRAW")
            kwargs = {
                "club_id": str(club_id),
                "tournament_id": str(tournament_id),
                "draw_id": str(draw_id),
                "expected_draw_updated_at": payload.expected_draw_updated_at,
                "actor_email": actor_email,
                "actor_role": actor_role,
                "confirmation_text": payload.confirmation_text,
                "source": payload.source,
                "atomic": True,
            }
            return _guarded_ops_mutation(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                action="ops_empty_draw_cancel",
                entity_type="tournament_event_draw",
                entity_id=str(draw_id),
                expected_state=str(payload.expected_state_fingerprint or ""),
                payload={
                    "draw_id": str(draw_id),
                    "expected_draw_updated_at": payload.expected_draw_updated_at,
                    "status": "cancelled",
                },
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                preflight=lambda: cancel_admin_tournament_empty_draw(
                    supabase, **kwargs, dry_run=True
                ),
                reconcile=lambda operation: reconcile_admin_tournament_ops_recovery(
                    supabase,
                    club_id=str(club_id),
                    tournament_id=str(tournament_id),
                    action="ops_empty_draw_cancel",
                    entity_id=str(draw_id),
                    operation=operation,
                ),
                mutate=lambda: cancel_admin_tournament_empty_draw(
                    supabase, **kwargs
                ),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/events/{event_option_id}/cancel-empty")
    def post_admin_tournament_empty_event_cancel(
        club_id: str,
        tournament_id: str,
        event_option_id: str,
        payload: AdminTournamentEmptyEventCancelRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            _require_confirmation(payload.confirmation_text, "CANCEL EMPTY EVENT")
            kwargs = {
                "club_id": str(club_id),
                "tournament_id": str(tournament_id),
                "event_option_id": str(event_option_id),
                "actor_email": actor_email,
                "actor_role": actor_role,
                "confirmation_text": payload.confirmation_text,
                "source": payload.source,
                "atomic": True,
            }
            return _guarded_ops_mutation(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                action="ops_empty_event_cancel",
                entity_type="tournament_event_option",
                entity_id=str(event_option_id),
                expected_state=str(payload.expected_state_fingerprint or ""),
                payload={
                    "event_option_id": str(event_option_id),
                    "enabled": False,
                    "status": "cancelled",
                },
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                preflight=lambda: cancel_admin_tournament_empty_event(
                    supabase, **kwargs, dry_run=True
                ),
                reconcile=lambda operation: reconcile_admin_tournament_ops_recovery(
                    supabase,
                    club_id=str(club_id),
                    tournament_id=str(tournament_id),
                    action="ops_empty_event_cancel",
                    entity_id=str(event_option_id),
                    operation=operation,
                ),
                mutate=lambda: cancel_admin_tournament_empty_event(
                    supabase, **kwargs
                ),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/results-import/preview")
    def post_admin_tournament_results_import_preview(
        club_id: str,
        tournament_id: str,
        draw_id: str,
        payload: AdminTournamentResultsImportPreviewRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_tournament_ops_results_import_preview",
        )
        try:
            return build_admin_tournament_results_import_preview(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                draw_id=str(draw_id),
                raw_text=payload.raw_text,
                import_mode=payload.import_mode,
                mapping_decisions=payload.mapping_decisions,
                match_reviews=payload.match_reviews,
                podium_refs=payload.podium_refs,
                allow_duplicate_mapping=payload.allow_duplicate_mapping,
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/results-import/commit")
    def post_admin_tournament_results_import_commit(
        club_id: str,
        tournament_id: str,
        draw_id: str,
        payload: AdminTournamentResultsImportCommitRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        creates_players = any(
            str((decision or {}).get("action") or "") == "create_new"
            for decision in payload.mapping_decisions.values()
        )
        required_permissions = (
            (PERMISSION_MANAGE_TOURNAMENTS, PERMISSION_MANAGE_PLAYERS)
            if creates_players
            else (PERMISSION_MANAGE_TOURNAMENTS,)
        )
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
            required_permissions=required_permissions,
            require_all=True,
        )
        try:
            mode = str(payload.import_mode or "").strip().upper()
            expected_confirmation = "REPLACE RESULTS" if mode == "REPLACE" else "IMPORT RESULTS"
            _require_confirmation(payload.confirmation_text, expected_confirmation)
            kwargs = {
                "club_id": str(club_id),
                "tournament_id": str(tournament_id),
                "draw_id": str(draw_id),
                "raw_text": payload.raw_text,
                "import_mode": payload.import_mode,
                "mapping_decisions": payload.mapping_decisions,
                "match_reviews": payload.match_reviews,
                "podium_refs": payload.podium_refs,
                "allow_duplicate_mapping": payload.allow_duplicate_mapping,
                "unusual_scores_acknowledged": payload.unusual_scores_acknowledged,
                "expected_review_fingerprint": payload.expected_review_fingerprint,
                "actor_email": actor_email,
                "actor_role": actor_role,
                "confirmation_text": payload.confirmation_text,
                "expected_draw_updated_at": payload.expected_draw_updated_at,
                "source": payload.source,
                "atomic": tournament_admin_guarded_runtime_enabled("operations"),
            }
            return _guarded_ops_mutation(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                action="ops_results_import",
                entity_type="tournament_event_draw",
                entity_id=str(draw_id),
                expected_state=str(payload.expected_state_fingerprint or ""),
                payload={
                    "draw_id": str(draw_id),
                    "import_mode": mode,
                    "raw_text_fingerprint": stable_tournament_admin_fingerprint(payload.raw_text),
                    "review_fingerprint": payload.expected_review_fingerprint,
                    "allow_duplicate_mapping": payload.allow_duplicate_mapping,
                    "unusual_scores_acknowledged": payload.unusual_scores_acknowledged,
                    "expected_draw_updated_at": payload.expected_draw_updated_at,
                },
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                preflight=lambda: apply_admin_tournament_results_import(supabase, **kwargs, dry_run=True),
                mutate=lambda: apply_admin_tournament_results_import(supabase, **kwargs),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/games/playoffs")
    def post_admin_tournament_playoff_games(club_id: str, tournament_id: str, draw_id: str, payload: AdminTournamentPlayoffGenerateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            _require_confirmation(payload.confirmation_text, "GENERATE PLAYOFFS")
            kwargs = {
                "club_id": str(club_id),
                "tournament_id": str(tournament_id),
                "draw_id": str(draw_id),
                "advance_count": payload.advance_count,
                "actor_email": actor_email,
                "actor_role": actor_role,
                "confirmation_text": payload.confirmation_text,
                "expected_draw_updated_at": payload.expected_draw_updated_at,
                "expected_team_versions": [_dump_model(row) for row in payload.expected_team_versions],
                "expected_source_game_versions": [_dump_model(row) for row in payload.expected_source_game_versions],
                "source": payload.source,
                "atomic": tournament_admin_guarded_runtime_enabled("operations"),
            }
            return _guarded_ops_mutation(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                action="ops_playoffs_generate",
                entity_type="tournament_event_draw",
                entity_id=str(draw_id),
                expected_state=str(payload.expected_state_fingerprint or ""),
                payload={
                    "draw_id": str(draw_id),
                    "advance_count": payload.advance_count,
                    "expected_draw_updated_at": payload.expected_draw_updated_at,
                    "expected_team_versions": [_dump_model(row) for row in payload.expected_team_versions],
                    "expected_source_game_versions": [_dump_model(row) for row in payload.expected_source_game_versions],
                },
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                preflight=lambda: generate_admin_tournament_playoff_games(supabase, **kwargs, dry_run=True),
                mutate=lambda: generate_admin_tournament_playoff_games(supabase, **kwargs),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/podium")
    def post_admin_tournament_draw_podium(club_id: str, tournament_id: str, draw_id: str, payload: AdminTournamentPodiumGenerateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            _require_confirmation(payload.confirmation_text, "GENERATE PODIUM")
            kwargs = {
                "club_id": str(club_id),
                "tournament_id": str(tournament_id),
                "draw_id": str(draw_id),
                "actor_email": actor_email,
                "actor_role": actor_role,
                "confirmation_text": payload.confirmation_text,
                "expected_draw_updated_at": payload.expected_draw_updated_at,
                "expected_team_versions": [_dump_model(row) for row in payload.expected_team_versions],
                "expected_source_game_versions": [_dump_model(row) for row in payload.expected_source_game_versions],
                "source": payload.source,
                "atomic": tournament_admin_guarded_runtime_enabled("operations"),
            }
            return _guarded_ops_mutation(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                action="ops_podium_generate",
                entity_type="tournament_event_draw",
                entity_id=str(draw_id),
                expected_state=str(payload.expected_state_fingerprint or ""),
                payload={
                    "draw_id": str(draw_id),
                    "expected_draw_updated_at": payload.expected_draw_updated_at,
                    "expected_team_versions": [_dump_model(row) for row in payload.expected_team_versions],
                    "expected_source_game_versions": [_dump_model(row) for row in payload.expected_source_game_versions],
                },
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                preflight=lambda: generate_admin_tournament_draw_podium(supabase, **kwargs, dry_run=True),
                mutate=lambda: generate_admin_tournament_draw_podium(supabase, **kwargs),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/podium/awards")
    def post_admin_tournament_draw_podium_awards(club_id: str, tournament_id: str, draw_id: str, payload: AdminTournamentPodiumAwardRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            _require_confirmation(payload.confirmation_text, "AWARD PODIUM")
            reviewed_team_versions = [
                _dump_model(row) for row in payload.expected_team_versions
            ]
            reviewed_game_versions = [
                _dump_model(row) for row in payload.expected_source_game_versions
            ]
            reviewed_podium_versions = [
                _dump_model(row) for row in payload.expected_podium_versions
            ]
            reviewed_podium = [dict(row) for row in payload.expected_podium]
            reviewed_awards = [dict(row) for row in payload.expected_awards]
            kwargs = {
                "club_id": str(club_id),
                "tournament_id": str(tournament_id),
                "draw_id": str(draw_id),
                "actor_email": actor_email,
                "actor_role": actor_role,
                "confirmation_text": payload.confirmation_text,
                "expected_draw_updated_at": payload.expected_draw_updated_at,
                "expected_team_versions": reviewed_team_versions,
                "expected_source_game_versions": reviewed_game_versions,
                "expected_podium_versions": reviewed_podium_versions,
                "expected_podium": reviewed_podium,
                "expected_awards": reviewed_awards,
                "source": payload.source,
                # Podium awards must never fall back to the legacy per-row badge
                # writes. The exact reviewed draw/team/game/podium/award plan is
                # required even for local API callers; staging additionally wraps
                # this atomic domain write in the durable guarded operation.
                "atomic": True,
            }
            return _guarded_ops_mutation(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                action="ops_podium_award",
                entity_type="tournament_event_draw",
                entity_id=str(draw_id),
                expected_state=str(payload.expected_state_fingerprint or ""),
                payload={
                    "draw_id": str(draw_id),
                    "expected_draw_updated_at": payload.expected_draw_updated_at,
                    "expected_team_versions": reviewed_team_versions,
                    "expected_source_game_versions": reviewed_game_versions,
                    "expected_podium_versions": reviewed_podium_versions,
                    "expected_podium": reviewed_podium,
                    "expected_awards": reviewed_awards,
                },
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                preflight=lambda: award_admin_tournament_draw_podium(supabase, **kwargs, dry_run=True),
                mutate=lambda: award_admin_tournament_draw_podium(supabase, **kwargs),
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/podium/review")
    def post_admin_tournament_draw_podium_review(
        club_id: str,
        tournament_id: str,
        draw_id: str,
        payload: AdminTournamentPodiumReviewRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
            required_permissions=(PERMISSION_MANAGE_TOURNAMENTS,),
            require_all=True,
        )
        try:
            return review_admin_tournament_draw_podium(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                draw_id=str(draw_id),
                expected_state_fingerprint=payload.expected_state_fingerprint,
                expected_draw_updated_at=payload.expected_draw_updated_at,
                expected_team_versions=[_dump_model(row) for row in payload.expected_team_versions],
                expected_source_game_versions=[
                    _dump_model(row) for row in payload.expected_source_game_versions
                ],
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/matches/publish")
    def post_admin_tournament_draw_matches_publish(club_id: str, tournament_id: str, draw_id: str, payload: AdminTournamentOfficialMatchPublishRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        try:
            require_tournament_admin_mutation_runtime("operations")
            require_admin_tournament_official_publish_runtime()
            if (
                str(payload.source or "").strip()
                == "next_team_tournament_child_publish"
                and not is_admin_team_tournament_enabled()
            ):
                raise PermissionError("Team tournament management is disabled.")
        except Exception as exc:
            _handle(exc)
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
            required_permissions=(PERMISSION_MANAGE_TOURNAMENTS, PERMISSION_MANAGE_MATCHES),
            require_all=True,
        )
        try:
            _require_confirmation(payload.confirmation_text, "PUBLISH MATCHES")
            require_admin_tournament_official_publish_readiness(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                draw_id=str(draw_id),
            )
            kwargs = {
                "club_id": str(club_id),
                "tournament_id": str(tournament_id),
                "draw_id": str(draw_id),
                "actor_email": actor_email,
                "actor_role": actor_role,
                "confirmation_text": payload.confirmation_text,
                "playoff_winner_bonus_elo": payload.playoff_winner_bonus_elo,
                "source": payload.source,
            }
            publish_plan = build_admin_tournament_official_publish_plan(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                draw_id=str(draw_id),
                playoff_winner_bonus_elo=payload.playoff_winner_bonus_elo,
            )
            kwargs["expected_plan"] = publish_plan
            operation_payload = {
                "draw_id": str(draw_id),
                "playoff_winner_bonus_elo": payload.playoff_winner_bonus_elo,
                "publish_plan": publish_plan,
            }
            operation_identity = build_tournament_admin_operation_request(
                club_id=str(club_id),
                surface="operations",
                action="ops_official_publish",
                entity_type="tournament_event_draw",
                entity_id=str(draw_id),
                lock_scope=str(tournament_id),
                expected_state=str(payload.expected_state_fingerprint or ""),
                payload=operation_payload,
            )
            if tournament_admin_guarded_runtime_enabled("operations"):
                kwargs.update(
                    {
                        "guarded_operation_key": str(operation_identity["operation_key"]),
                        "guarded_request_fingerprint": str(operation_identity["request_fingerprint"]),
                    }
                )
            return _guarded_ops_mutation(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                action="ops_official_publish",
                entity_type="tournament_event_draw",
                entity_id=str(draw_id),
                expected_state=str(payload.expected_state_fingerprint or ""),
                payload=operation_payload,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                preflight=lambda: publish_admin_tournament_draw_matches(supabase, **kwargs, dry_run=True),
                reconcile=lambda operation: reconcile_admin_tournament_official_publish(
                    supabase,
                    club_id=str(club_id),
                    tournament_id=str(tournament_id),
                    draw_id=str(draw_id),
                    expected_plan=publish_plan,
                    guarded_operation_key=str(operation.get("operation_key") or ""),
                    guarded_request_fingerprint=str(operation.get("request_fingerprint") or ""),
                    client_idempotency_key=str(operation.get("client_idempotency_key") or ""),
                ),
                mutate=lambda: publish_admin_tournament_draw_matches(supabase, **kwargs),
            )
        except Exception as exc:
            _handle(exc)

    @app.patch("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/games/{game_id}/score")
    def patch_admin_tournament_game_score(club_id: str, tournament_id: str, game_id: str, payload: AdminTournamentGameScoreRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
            required_permissions=(PERMISSION_ENTER_SCORES,),
        )
        try:
            _require_confirmation(payload.confirmation_text, "SAVE SCORE")
            kwargs = {
                "club_id": str(club_id),
                "tournament_id": str(tournament_id),
                "game_id": str(game_id),
                "score_a": payload.score_a,
                "score_b": payload.score_b,
                "unusual_score_acknowledged": payload.unusual_score_acknowledged,
                "actor_email": actor_email,
                "actor_role": actor_role,
                "confirmation_text": payload.confirmation_text,
                "expected_updated_at": payload.expected_game_updated_at,
                "expected_draw_updated_at": payload.expected_draw_updated_at,
                "source": payload.source,
                "atomic": tournament_admin_guarded_runtime_enabled("operations"),
            }
            return _guarded_ops_mutation(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                action="ops_game_score",
                entity_type="tournament_game",
                entity_id=str(game_id),
                expected_state=str(payload.expected_state_fingerprint or ""),
                payload={
                    "game_id": str(game_id),
                    "score_a": payload.score_a,
                    "score_b": payload.score_b,
                    "unusual_score_acknowledged": payload.unusual_score_acknowledged,
                    "expected_game_updated_at": payload.expected_game_updated_at,
                    "expected_draw_updated_at": payload.expected_draw_updated_at,
                },
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                preflight=lambda: update_admin_tournament_game_score(supabase, **kwargs, dry_run=True),
                mutate=lambda: update_admin_tournament_game_score(supabase, **kwargs),
            )
        except Exception as exc:
            _handle(exc)

    @app.patch("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/status-action")
    def patch_admin_tournament_status_action(club_id: str, tournament_id: str, payload: AdminTournamentStatusActionRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            normalized_action = str(payload.action).strip().lower()
            if normalized_action not in {"complete", "archive", "unarchive"}:
                raise ValueError("action must be complete, archive, or unarchive")
            expected_confirmation = {
                "complete": "COMPLETE",
                "archive": "ARCHIVE",
                "unarchive": "UNARCHIVE",
            }[normalized_action]
            _require_confirmation(payload.confirmation_text, expected_confirmation)
            operation_payload = {"action": payload.action}
            operation_identity = build_tournament_admin_operation_request(
                club_id=str(club_id),
                surface="tournament",
                action=f"tournament_{normalized_action}",
                entity_type="tournament",
                entity_id=str(tournament_id),
                lock_scope=str(tournament_id),
                expected_state=str(payload.expected_updated_at or ""),
                payload=operation_payload,
            )
            guarded_terminal = tournament_admin_guarded_runtime_enabled("tournament")
            preflight = lambda: apply_admin_tournament_status_action(supabase, club_id=str(club_id), tournament_id=str(tournament_id), action=payload.action, expected_updated_at=str(payload.expected_updated_at or ""), actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source, dry_run=True, atomic=guarded_terminal)
            mutate = lambda: apply_admin_tournament_status_action(supabase, club_id=str(club_id), tournament_id=str(tournament_id), action=payload.action, expected_updated_at=str(payload.expected_updated_at or "") if guarded_terminal else None, actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, guarded_operation_key=str(operation_identity["operation_key"]) if guarded_terminal else None, request_fingerprint=str(operation_identity["request_fingerprint"]) if guarded_terminal else None, source=payload.source, atomic=guarded_terminal)
            return _guarded_admin_mutation(
                supabase,
                club_id=str(club_id),
                surface="tournament",
                action=f"tournament_{str(payload.action).strip().lower()}",
                entity_type="tournament",
                entity_id=str(tournament_id),
                lock_scope=str(tournament_id),
                expected_state=str(payload.expected_updated_at or ""),
                current_state=lambda: str(get_admin_tournament_detail(supabase, club_id=str(club_id), tournament_id=str(tournament_id))["tournament"].get("updated_at") or ""),
                payload=operation_payload,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                preflight=preflight,
                reconcile=lambda operation: reconcile_admin_tournament_status_action(
                    supabase,
                    club_id=str(club_id),
                    tournament_id=str(tournament_id),
                    action=normalized_action,
                    operation_key=str(operation.get("operation_key") or ""),
                ),
                mutate=mutate,
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/delete-draft")
    def post_admin_tournament_delete_draft(club_id: str, tournament_id: str, payload: AdminTournamentDeleteDraftRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            _require_confirmation(payload.confirmation_text, "DELETE DRAFT")
            preflight = lambda: delete_admin_tournament_draft(supabase, club_id=str(club_id), tournament_id=str(tournament_id), expected_updated_at=str(payload.expected_updated_at or ""), actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source, dry_run=True)
            mutate = lambda: delete_admin_tournament_draft(supabase, club_id=str(club_id), tournament_id=str(tournament_id), expected_updated_at=str(payload.expected_updated_at or "") if tournament_admin_guarded_runtime_enabled("tournament") else None, actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source)
            return _guarded_admin_mutation(
                supabase,
                club_id=str(club_id),
                surface="tournament",
                action="tournament_delete_draft",
                entity_type="tournament",
                entity_id=str(tournament_id),
                lock_scope=str(tournament_id),
                expected_state=str(payload.expected_updated_at or ""),
                current_state=lambda: str(get_admin_tournament_detail(supabase, club_id=str(club_id), tournament_id=str(tournament_id))["tournament"].get("updated_at") or ""),
                payload={"delete": True},
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                preflight=preflight,
                mutate=mutate,
            )
        except Exception as exc:
            _handle(exc)

    @app.get("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}")
    def get_admin_tournament(club_id: str, tournament_id: str, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_tournament_admin_detail")
        try:
            return get_admin_tournament_detail(supabase, club_id=str(club_id), tournament_id=str(tournament_id))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            _handle(exc)

    @app.patch("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}")
    def patch_admin_tournament(club_id: str, tournament_id: str, payload: AdminTournamentUpdateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        patch = _dump_patch_model(payload)
        source = str(patch.pop("source", payload.source))
        confirmation_text = str(patch.pop("confirmation_text", payload.confirmation_text))
        expected_updated_at = str(patch.pop("expected_updated_at", payload.expected_updated_at) or "")
        try:
            _require_confirmation(confirmation_text, "SAVE TOURNAMENT")
            preflight = lambda: update_admin_tournament(supabase, club_id=str(club_id), tournament_id=str(tournament_id), patch=patch, expected_updated_at=expected_updated_at or None, actor_email=actor_email, actor_role=actor_role, confirmation_text=confirmation_text, source=source, dry_run=True)
            mutate = lambda: update_admin_tournament(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                patch=patch,
                expected_updated_at=expected_updated_at or None,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=confirmation_text,
                source=source,
            )
            return _guarded_admin_mutation(
                supabase,
                club_id=str(club_id),
                surface="tournament",
                action="tournament_update",
                entity_type="tournament",
                entity_id=str(tournament_id),
                lock_scope=str(tournament_id),
                expected_state=expected_updated_at,
                current_state=lambda: str(get_admin_tournament_detail(supabase, club_id=str(club_id), tournament_id=str(tournament_id))["tournament"].get("updated_at") or ""),
                payload=patch,
                actor_email=actor_email,
                actor_role=actor_role,
                source=source,
                preflight=preflight,
                mutate=mutate,
            )
        except Exception as exc:
            _handle(exc)

    @app.get("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/registrations/import-handoff")
    def get_admin_tournament_registration_import_handoff(club_id: str, tournament_id: str, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_tournament_admin_import_handoff")
        try:
            return build_admin_tournament_registration_import_handoff(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
            )
        except Exception as exc:
            _handle(exc)

    @app.get("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/registrations/export.csv")
    def get_admin_tournament_registration_export(
        club_id: str,
        tournament_id: str,
        registration_status: str | None = Query(default=None),
        payment_status: str | None = Query(default=None),
        partner_mode: str | None = Query(default=None),
        registration_day_id: str | None = Query(default=None),
        event_option_id: str | None = Query(default=None),
        search: str | None = Query(default=None),
        authorization: str | None = auth_header(),
    ) -> Response:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_tournament_admin_registration_export",
        )
        try:
            export = build_admin_tournament_registration_export(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                registration_status=registration_status,
                payment_status=payment_status,
                partner_mode=partner_mode,
                registration_day_id=registration_day_id,
                event_option_id=event_option_id,
                search=search,
            )
            safe_id = "".join(
                character
                for character in str(tournament_id)
                if character.isalnum() or character in {"-", "_"}
            ) or "tournament"
            return Response(
                content=str(export["csv"]),
                media_type="text/csv; charset=utf-8",
                headers={
                    "Cache-Control": "private, no-store, max-age=0",
                    "Pragma": "no-cache",
                    "X-Content-Type-Options": "nosniff",
                    "Content-Disposition": (
                        f'attachment; filename="{safe_id}-registrations.csv"'
                    ),
                    "X-JUPR-Export-Row-Count": str(export["row_count"]),
                },
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/registrations/broadcast-preview")
    def post_admin_tournament_broadcast_preview(
        club_id: str,
        tournament_id: str,
        payload: AdminTournamentBroadcastPreviewRequest,
        response: Response,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_tournament_admin_broadcast_preview",
        )
        response.headers["Cache-Control"] = "private, no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        try:
            return build_admin_tournament_broadcast_preview(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                **_dump_model(payload),
            )
        except Exception as exc:
            _handle(exc)

    @app.patch("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/registrations/bulk")
    def patch_admin_tournament_registrations_bulk(club_id: str, tournament_id: str, payload: AdminTournamentRegistrationBulkUpdateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        patch = _dump_model(payload)
        source = str(patch.pop("source", payload.source))
        confirmation_text = str(patch.pop("confirmation_text", payload.confirmation_text))
        registration_ids = list(patch.pop("registration_ids", payload.registration_ids) or [])
        expected_state = str(patch.pop("expected_state_fingerprint", payload.expected_state_fingerprint) or "")
        expected_versions = dict(patch.pop("expected_versions", payload.expected_versions) or {})
        try:
            _require_confirmation(confirmation_text, "BULK UPDATE REGISTRATIONS")
            preflight = lambda: bulk_update_admin_tournament_registrations(supabase, club_id=str(club_id), tournament_id=str(tournament_id), registration_ids=registration_ids, patch=patch, expected_versions=expected_versions, actor_email=actor_email, actor_role=actor_role, confirmation_text=confirmation_text, source=source, dry_run=True)
            mutate = lambda: bulk_update_admin_tournament_registrations(supabase, club_id=str(club_id), tournament_id=str(tournament_id), registration_ids=registration_ids, patch=patch, expected_versions=expected_versions if tournament_admin_guarded_runtime_enabled("registration") else None, actor_email=actor_email, actor_role=actor_role, confirmation_text=confirmation_text, source=source)
            return _guarded_admin_mutation(
                supabase,
                club_id=str(club_id),
                surface="registration",
                action="tournament_registration_bulk_update",
                entity_type="tournament_registration",
                entity_id=f"{tournament_id}:bulk",
                lock_scope=str(tournament_id),
                expected_state=expected_state,
                current_state=lambda: str(get_admin_tournament_detail(supabase, club_id=str(club_id), tournament_id=str(tournament_id)).get("state_fingerprint") or ""),
                payload={"registration_ids": registration_ids, "patch": patch, "expected_versions": expected_versions},
                actor_email=actor_email,
                actor_role=actor_role,
                source=source,
                preflight=preflight,
                mutate=mutate,
            )
        except Exception as exc:
            _handle(exc)

    @app.patch("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/registrations/{registration_id}")
    def patch_admin_tournament_registration(club_id: str, tournament_id: str, registration_id: str, payload: AdminTournamentRegistrationUpdateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        patch = _dump_patch_model(payload)
        source = str(patch.pop("source", payload.source))
        confirmation_text = str(patch.pop("confirmation_text", payload.confirmation_text))
        expected_updated_at = str(patch.pop("expected_updated_at", payload.expected_updated_at))
        try:
            _require_confirmation(confirmation_text, "SAVE REGISTRATION")
            preflight = lambda: update_admin_tournament_registration(supabase, club_id=str(club_id), tournament_id=str(tournament_id), registration_id=str(registration_id), patch=patch, expected_updated_at=expected_updated_at, actor_email=actor_email, actor_role=actor_role, confirmation_text=confirmation_text, source=source, dry_run=True)
            mutate = lambda: update_admin_tournament_registration(supabase, club_id=str(club_id), tournament_id=str(tournament_id), registration_id=str(registration_id), patch=patch, expected_updated_at=expected_updated_at, actor_email=actor_email, actor_role=actor_role, confirmation_text=confirmation_text, source=source)
            return _guarded_admin_mutation(
                supabase,
                club_id=str(club_id),
                surface="registration",
                action="tournament_registration_update",
                entity_type="tournament_registration",
                entity_id=str(registration_id),
                lock_scope=str(tournament_id),
                expected_state=expected_updated_at,
                current_state=lambda: next((str(row.get("updated_at") or "") for row in get_admin_tournament_detail(supabase, club_id=str(club_id), tournament_id=str(tournament_id)).get("registrations") or [] if str(row.get("id") or "") == str(registration_id)), ""),
                payload=patch,
                actor_email=actor_email,
                actor_role=actor_role,
                source=source,
                preflight=preflight,
                mutate=mutate,
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/registrations/{registration_id}/selections")
    def post_admin_tournament_selection(
        club_id: str,
        tournament_id: str,
        registration_id: str,
        payload: AdminTournamentSelectionCreateRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        patch = _dump_patch_model(payload)
        source = str(patch.pop("source", payload.source))
        confirmation_text = str(
            patch.pop("confirmation_text", payload.confirmation_text)
        )
        expected_state_fingerprint = str(
            patch.pop("expected_state_fingerprint", payload.expected_state_fingerprint)
        ).strip()
        try:
            _require_confirmation(confirmation_text, "SAVE SELECTION")
            if not expected_state_fingerprint:
                raise ValueError("expected_state_fingerprint is required for event-entry creation.")
            preflight = lambda: create_admin_tournament_selection(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                registration_id=str(registration_id),
                patch=patch,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=confirmation_text,
                source=source,
                dry_run=True,
            )
            mutate = lambda: create_admin_tournament_selection(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                registration_id=str(registration_id),
                patch=patch,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=confirmation_text,
                source=source,
            )
            return _guarded_admin_mutation(
                supabase,
                club_id=str(club_id),
                surface="registration",
                action="tournament_registration_selection_create",
                entity_type="tournament_registration_selection",
                entity_id=f"{registration_id}:{patch['event_option_id']}",
                lock_scope=str(tournament_id),
                expected_state=expected_state_fingerprint,
                current_state=lambda: str(
                    get_admin_tournament_detail(
                        supabase,
                        club_id=str(club_id),
                        tournament_id=str(tournament_id),
                    ).get("state_fingerprint")
                    or ""
                ),
                payload=patch,
                actor_email=actor_email,
                actor_role=actor_role,
                source=source,
                preflight=preflight,
                mutate=mutate,
            )
        except Exception as exc:
            _handle(exc)

    @app.patch("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/selections/{selection_id}")
    def patch_admin_tournament_selection(club_id: str, tournament_id: str, selection_id: str, payload: AdminTournamentSelectionUpdateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        patch = _dump_patch_model(payload)
        source = str(patch.pop("source", payload.source))
        confirmation_text = str(patch.pop("confirmation_text", payload.confirmation_text))
        expected_updated_at = str(patch.pop("expected_updated_at", payload.expected_updated_at))
        try:
            _require_confirmation(confirmation_text, "SAVE SELECTION")
            preflight = lambda: update_admin_tournament_selection(supabase, club_id=str(club_id), tournament_id=str(tournament_id), selection_id=str(selection_id), patch=patch, expected_updated_at=expected_updated_at, actor_email=actor_email, actor_role=actor_role, confirmation_text=confirmation_text, source=source, dry_run=True)
            mutate = lambda: update_admin_tournament_selection(
                    supabase,
                    club_id=str(club_id),
                    tournament_id=str(tournament_id),
                    selection_id=str(selection_id),
                    patch=patch,
                    expected_updated_at=expected_updated_at,
                    actor_email=actor_email,
                    actor_role=actor_role,
                    confirmation_text=confirmation_text,
                    source=source,
                )
            return _guarded_admin_mutation(
                supabase,
                club_id=str(club_id),
                surface="registration",
                action="tournament_registration_selection_update",
                entity_type="tournament_registration_selection",
                entity_id=str(selection_id),
                lock_scope=str(tournament_id),
                expected_state=expected_updated_at,
                current_state=lambda: next((str(row.get("updated_at") or "") for row in get_admin_tournament_detail(supabase, club_id=str(club_id), tournament_id=str(tournament_id)).get("selections") or [] if str(row.get("id") or "") == str(selection_id)), ""),
                payload=patch,
                actor_email=actor_email,
                actor_role=actor_role,
                source=source,
                preflight=preflight,
                mutate=mutate,
            )
        except Exception as exc:
            _handle(exc)

    @app.delete("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/selections/{selection_id}")
    def delete_admin_tournament_selection_route(
        club_id: str,
        tournament_id: str,
        selection_id: str,
        payload: AdminTournamentSelectionDeleteRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        body = _dump_patch_model(payload)
        source = str(body.pop("source", payload.source))
        confirmation_text = str(
            body.pop("confirmation_text", payload.confirmation_text)
        )
        expected_updated_at = str(
            body.pop("expected_updated_at", payload.expected_updated_at)
        ).strip()
        try:
            _require_confirmation(confirmation_text, "REMOVE SELECTION")
            preflight = lambda: delete_admin_tournament_selection(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                selection_id=str(selection_id),
                expected_updated_at=expected_updated_at,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=confirmation_text,
                source=source,
                dry_run=True,
            )
            mutate = lambda: delete_admin_tournament_selection(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                selection_id=str(selection_id),
                expected_updated_at=expected_updated_at,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=confirmation_text,
                source=source,
            )
            return _guarded_admin_mutation(
                supabase,
                club_id=str(club_id),
                surface="registration",
                action="tournament_registration_selection_delete",
                entity_type="tournament_registration_selection",
                entity_id=str(selection_id),
                lock_scope=str(tournament_id),
                expected_state=expected_updated_at,
                current_state=lambda: next(
                    (
                        str(row.get("updated_at") or "")
                        for row in get_admin_tournament_detail(
                            supabase,
                            club_id=str(club_id),
                            tournament_id=str(tournament_id),
                        ).get("selections")
                        or []
                        if str(row.get("id") or "") == str(selection_id)
                    ),
                    "",
                ),
                payload={"selection_id": str(selection_id), **body},
                actor_email=actor_email,
                actor_role=actor_role,
                source=source,
                preflight=preflight,
                mutate=mutate,
            )
        except Exception as exc:
            _handle(exc)


    @app.put("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/selections/{selection_id}/partner")
    def put_admin_tournament_selection_partner(
        club_id: str,
        tournament_id: str,
        selection_id: str,
        payload: AdminTournamentSelectionPartnerRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        body = _dump_model(payload)
        source = str(body.pop("source", payload.source))
        confirmation_text = str(body.pop("confirmation_text", payload.confirmation_text))
        expected_updated_at = str(body.pop("expected_updated_at", payload.expected_updated_at))
        partner_selection_id = body.pop("partner_selection_id", payload.partner_selection_id)
        unpaired_mode = str(body.pop("unpaired_mode", payload.unpaired_mode) or "NEEDS_PARTNER").upper()
        try:
            _require_confirmation(confirmation_text, "SAVE PARTNER")
            preflight = lambda: replace_admin_tournament_selection_partner(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                selection_id=str(selection_id),
                partner_selection_id=str(partner_selection_id) if partner_selection_id else None,
                unpaired_mode=unpaired_mode,
                expected_updated_at=expected_updated_at,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=confirmation_text,
                source=source,
                dry_run=True,
            )
            mutate = lambda: replace_admin_tournament_selection_partner(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                selection_id=str(selection_id),
                partner_selection_id=str(partner_selection_id) if partner_selection_id else None,
                unpaired_mode=unpaired_mode,
                expected_updated_at=expected_updated_at,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=confirmation_text,
                source=source,
            )
            return _guarded_admin_mutation(
                supabase,
                club_id=str(club_id),
                surface="registration",
                action="tournament_registration_partner_update",
                entity_type="tournament_registration_selection",
                entity_id=str(selection_id),
                lock_scope=str(tournament_id),
                expected_state=expected_updated_at,
                current_state=lambda: next(
                    (
                        str(row.get("updated_at") or "")
                        for row in get_admin_tournament_detail(
                            supabase,
                            club_id=str(club_id),
                            tournament_id=str(tournament_id),
                        ).get("selections") or []
                        if str(row.get("id") or "") == str(selection_id)
                    ),
                    "",
                ),
                payload={
                    "partner_selection_id": partner_selection_id,
                    "unpaired_mode": unpaired_mode,
                },
                actor_email=actor_email,
                actor_role=actor_role,
                source=source,
                preflight=preflight,
                mutate=mutate,
            )
        except Exception as exc:
            _handle(exc)
