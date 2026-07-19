from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query, Response
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import PERMISSION_MANAGE_TOURNAMENTS, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.tournament_registration_repo import (
    StaleTournamentRegistrationAdminError,
    StaleTournamentRegistrationSelectionError,
)
from jupr_app.services.admin_tournament_award_service import award_admin_tournament_draw_podium
from jupr_app.services.admin_tournament_bulk_service import bulk_update_admin_tournament_registrations
from jupr_app.services.admin_tournament_bulk_team_import_service import import_admin_tournament_bulk_teams
from jupr_app.services.admin_tournament_delete_service import delete_admin_tournament_draft
from jupr_app.services.admin_tournament_draw_service import create_admin_tournament_draw
from jupr_app.services.admin_tournament_game_service import generate_admin_tournament_round_robin_games
from jupr_app.services.admin_tournament_match_publish_service import publish_admin_tournament_draw_matches
from jupr_app.services.admin_tournament_ops_service import get_admin_tournament_ops_snapshot
from jupr_app.services.admin_tournament_playoff_service import generate_admin_tournament_playoff_games
from jupr_app.services.admin_tournament_podium_service import generate_admin_tournament_draw_podium
from jupr_app.services.admin_tournament_registration_import_service import import_admin_tournament_registrations_to_draw
from jupr_app.services.admin_tournament_registration_reporting_service import (
    build_admin_tournament_broadcast_preview,
    build_admin_tournament_registration_export,
)
from jupr_app.services.admin_tournament_score_service import update_admin_tournament_game_score
from jupr_app.services.admin_tournament_service import (
    build_admin_tournament_registration_import_handoff,
    build_admin_tournament_status,
    get_admin_tournament_detail,
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
from jupr_app.services.admin_tournament_status_service import apply_admin_tournament_status_action
from jupr_app.services.admin_tournament_team_service import replace_admin_tournament_draw_teams
from services.api.auth import authenticate_bearer, auth_header


class AdminTournamentRegistrationUpdateRequest(BaseModel):
    registration_status: str | None = None
    payment_status: str | None = None
    notes: str | None = None
    expected_updated_at: str | None = None
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
    confirmation_text: str = ""
    source: str = "next_tournament_admin_replace_teams"


class AdminTournamentRegistrationImportRequest(BaseModel):
    import_mode: str = "REPLACE"
    confirmation_text: str = ""
    source: str = "next_tournament_admin_import_registrations"


class AdminTournamentBulkTeamImportRequest(BaseModel):
    raw_text: str = ""
    import_mode: str = "REPLACE"
    confirmation_text: str = ""
    source: str = "next_tournament_admin_import_bulk_teams"


class AdminTournamentRoundRobinGenerateRequest(BaseModel):
    confirmation_text: str = ""
    source: str = "next_tournament_admin_generate_round_robin"


class AdminTournamentPlayoffGenerateRequest(BaseModel):
    advance_count: int | None = None
    confirmation_text: str = ""
    source: str = "next_tournament_admin_generate_playoffs"


class AdminTournamentPodiumGenerateRequest(BaseModel):
    confirmation_text: str = ""
    source: str = "next_tournament_admin_generate_podium"


class AdminTournamentPodiumAwardRequest(BaseModel):
    confirmation_text: str = ""
    source: str = "next_tournament_admin_award_podium"


class AdminTournamentOfficialMatchPublishRequest(BaseModel):
    playoff_winner_bonus_elo: float | None = 0.0
    confirmation_text: str = ""
    source: str = "next_tournament_admin_publish_matches"


class AdminTournamentGameScoreRequest(BaseModel):
    score_a: int
    score_b: int
    confirmation_text: str = ""
    source: str = "next_tournament_admin_score_game"


class AdminTournamentSelectionUpdateRequest(BaseModel):
    expected_updated_at: str
    event_option_id: str | None = None
    partner_mode: str | None = None
    partner_name: str | None = None
    partner_email: str | None = None
    partner_phone: str | None = None
    partner_note: str | None = None
    confirmation_text: str = ""
    source: str = "next_tournament_admin_selection_update"


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


def _resolve_tournament_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(supabase=supabase, club_id=str(club_id), email=user.email, user_id=user.user_id, allowlist=set())
    if not has_permission(role_resolution.role, PERMISSION_MANAGE_TOURNAMENTS):
        denied_payload = build_activity_payload(
            club_id=str(club_id),
            actor_email=user.email,
            actor_role=role_resolution.role,
            action_type="admin_tournament_denied",
            entity_type="tournament_admin",
            entity_id="tournament_admin",
            after_json={"source_client": "fastapi/nextjs", "reason": "insufficient_permission"},
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
    mutate,
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
        mutate=mutate,
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
        _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_tournament_admin_ops")
        try:
            return get_admin_tournament_ops_snapshot(supabase, club_id=str(club_id), tournament_id=str(tournament_id), draw_id=draw_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws")
    def post_admin_tournament_draw(club_id: str, tournament_id: str, payload: AdminTournamentDrawCreateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return create_admin_tournament_draw(supabase, club_id=str(club_id), tournament_id=str(tournament_id), registration_day_id=payload.registration_day_id, event_option_id=payload.event_option_id, name=payload.name, actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source)
        except Exception as exc:
            _handle(exc)

    @app.put("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/teams")
    def put_admin_tournament_draw_teams(club_id: str, tournament_id: str, draw_id: str, payload: AdminTournamentDrawTeamsReplaceRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return replace_admin_tournament_draw_teams(supabase, club_id=str(club_id), tournament_id=str(tournament_id), draw_id=str(draw_id), teams=[_dump_model(team) for team in payload.teams], actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source)
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/teams/import-registrations")
    def post_admin_tournament_registration_team_import(club_id: str, tournament_id: str, draw_id: str, payload: AdminTournamentRegistrationImportRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return import_admin_tournament_registrations_to_draw(supabase, club_id=str(club_id), tournament_id=str(tournament_id), draw_id=str(draw_id), import_mode=payload.import_mode, actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source)
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/teams/import-bulk")
    def post_admin_tournament_bulk_team_import(club_id: str, tournament_id: str, draw_id: str, payload: AdminTournamentBulkTeamImportRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return import_admin_tournament_bulk_teams(supabase, club_id=str(club_id), tournament_id=str(tournament_id), draw_id=str(draw_id), raw_text=payload.raw_text, import_mode=payload.import_mode, actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source)
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/games/round-robin")
    def post_admin_tournament_round_robin_games(club_id: str, tournament_id: str, draw_id: str, payload: AdminTournamentRoundRobinGenerateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return generate_admin_tournament_round_robin_games(supabase, club_id=str(club_id), tournament_id=str(tournament_id), draw_id=str(draw_id), actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source)
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/games/playoffs")
    def post_admin_tournament_playoff_games(club_id: str, tournament_id: str, draw_id: str, payload: AdminTournamentPlayoffGenerateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return generate_admin_tournament_playoff_games(supabase, club_id=str(club_id), tournament_id=str(tournament_id), draw_id=str(draw_id), advance_count=payload.advance_count, actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source)
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/podium")
    def post_admin_tournament_draw_podium(club_id: str, tournament_id: str, draw_id: str, payload: AdminTournamentPodiumGenerateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return generate_admin_tournament_draw_podium(supabase, club_id=str(club_id), tournament_id=str(tournament_id), draw_id=str(draw_id), actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source)
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/podium/awards")
    def post_admin_tournament_draw_podium_awards(club_id: str, tournament_id: str, draw_id: str, payload: AdminTournamentPodiumAwardRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return award_admin_tournament_draw_podium(supabase, club_id=str(club_id), tournament_id=str(tournament_id), draw_id=str(draw_id), actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source)
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/matches/publish")
    def post_admin_tournament_draw_matches_publish(club_id: str, tournament_id: str, draw_id: str, payload: AdminTournamentOfficialMatchPublishRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return publish_admin_tournament_draw_matches(supabase, club_id=str(club_id), tournament_id=str(tournament_id), draw_id=str(draw_id), actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, playoff_winner_bonus_elo=payload.playoff_winner_bonus_elo, source=payload.source)
        except Exception as exc:
            _handle(exc)

    @app.patch("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/games/{game_id}/score")
    def patch_admin_tournament_game_score(club_id: str, tournament_id: str, game_id: str, payload: AdminTournamentGameScoreRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return update_admin_tournament_game_score(supabase, club_id=str(club_id), tournament_id=str(tournament_id), game_id=str(game_id), score_a=payload.score_a, score_b=payload.score_b, actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source)
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
            if normalized_action not in {"archive", "unarchive"}:
                raise ValueError("action must be archive or unarchive")
            expected_confirmation = "ARCHIVE" if normalized_action == "archive" else "UNARCHIVE"
            _require_confirmation(payload.confirmation_text, expected_confirmation)
            preflight = lambda: apply_admin_tournament_status_action(supabase, club_id=str(club_id), tournament_id=str(tournament_id), action=payload.action, expected_updated_at=str(payload.expected_updated_at or ""), actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source, dry_run=True)
            mutate = lambda: apply_admin_tournament_status_action(supabase, club_id=str(club_id), tournament_id=str(tournament_id), action=payload.action, expected_updated_at=str(payload.expected_updated_at or "") if tournament_admin_guarded_runtime_enabled("tournament") else None, actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source)
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
                payload={"action": payload.action},
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                preflight=preflight,
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
        patch = _dump_model(payload)
        source = str(patch.pop("source", payload.source))
        confirmation_text = str(patch.pop("confirmation_text", payload.confirmation_text))
        expected_updated_at = str(patch.pop("expected_updated_at", payload.expected_updated_at) or "")
        try:
            _require_confirmation(confirmation_text, "SAVE REGISTRATION")
            preflight = lambda: update_admin_tournament_registration(supabase, club_id=str(club_id), tournament_id=str(tournament_id), registration_id=str(registration_id), patch=patch, expected_updated_at=expected_updated_at or None, actor_email=actor_email, actor_role=actor_role, confirmation_text=confirmation_text, source=source, dry_run=True)
            mutate = lambda: update_admin_tournament_registration(supabase, club_id=str(club_id), tournament_id=str(tournament_id), registration_id=str(registration_id), patch=patch, expected_updated_at=expected_updated_at if tournament_admin_guarded_runtime_enabled("registration") else None, actor_email=actor_email, actor_role=actor_role, confirmation_text=confirmation_text, source=source)
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

    @app.patch("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/selections/{selection_id}")
    def patch_admin_tournament_selection(club_id: str, tournament_id: str, selection_id: str, payload: AdminTournamentSelectionUpdateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        patch = _dump_model(payload)
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
