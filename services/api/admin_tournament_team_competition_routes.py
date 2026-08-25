from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import (
    PERMISSION_MANAGE_TOURNAMENTS,
    has_permission,
    resolve_admin_role,
)
from jupr_app.domain.admin_activity_log import (
    build_activity_payload,
    write_admin_activity_log,
)
from jupr_app.services.admin_tournament_team_competition_service import (
    amend_four_player_team_roster,
    build_admin_team_tournament_status,
    close_combined_rating_reviews,
    create_four_player_team,
    create_team_playoffs,
    get_admin_team_tournament_snapshot,
    is_admin_team_tournament_enabled,
    lock_team_lineup,
    reconcile_team_match_game,
    record_combined_rating_review,
    reissue_four_player_team_invitation,
    replace_team_podium,
    replace_team_round_robin,
    score_team_match_game,
    update_tournament_competition_config,
    upsert_rating_verification,
)
from jupr_app.services.admin_tournament_guarded_operation import (
    require_tournament_admin_mutation_runtime,
)
from services.api.auth import authenticate_bearer, auth_header


class MutationBase(BaseModel):
    idempotency_key: str = Field(min_length=8, max_length=160)
    confirmation_text: str = Field(min_length=1, max_length=80)
    source: str = Field(default="next_team_tournament", max_length=120)


class CompetitionConfigRequest(MutationBase):
    expected_updated_at: str
    patch: dict[str, Any]


class RatingVerificationRequest(MutationBase):
    event_option_id: str
    registration_id: str
    rating: float = Field(ge=0, le=7)
    note: str = Field(default="", max_length=1000)
    expected_version: int | None = Field(default=None, ge=1)


class RatingReviewRequest(MutationBase):
    event_option_id: str
    selection_id: str
    review_phase: str
    override_state: str | None = None
    override_reason: str | None = Field(default=None, max_length=1000)
    expected_selection_updated_at: str


class RatingCloseRequest(MutationBase):
    event_option_id: str
    entries: list[dict[str, Any]] = Field(default_factory=list)


class FourPlayerTeamRequest(MutationBase):
    event_option_id: str
    team_name: str = Field(min_length=1, max_length=180)
    captain_registration_id: str
    members: list[dict[str, Any]] = Field(min_length=4, max_length=4)


class InviteReissueRequest(MutationBase):
    member_id: str
    expected_invitation_version: int = Field(ge=1)
    invited_email: str


class RosterRequest(MutationBase):
    expected_team_version: int = Field(ge=1)
    action: str
    members: list[dict[str, Any]] = Field(default_factory=list, max_length=4)
    reason: str = Field(min_length=1, max_length=1000)


class RoundRobinRequest(MutationBase):
    event_option_id: str
    team_ids: list[str] = Field(min_length=2)
    expected_draw_updated_at: str


class PlayoffRequest(MutationBase):
    playoff_format: str
    expected_draw_updated_at: str


class LineupRequest(MutationBase):
    team_id: str
    mixed_pairing: str
    singles_tiebreak_player_id: int | None = None
    expected_matchup_version: int = Field(ge=1)
    expected_lineup_version: int | None = Field(default=None, ge=1)


class ScoreRequest(MutationBase):
    score_a: int = Field(ge=0)
    score_b: int = Field(ge=0)
    unusual_score_acknowledged: bool = False
    expected_game_version: int = Field(ge=1)
    expected_matchup_version: int = Field(ge=1)


class ReconcileRequest(MutationBase):
    official_match_id: str
    expected_official_row_version: int = Field(ge=1)
    expected_game_version: int = Field(ge=1)
    expected_matchup_version: int = Field(ge=1)
    reason: str = Field(min_length=1, max_length=1000)


class PodiumRequest(MutationBase):
    expected_draw_updated_at: str
    publish: bool = False
    reason: str = Field(default="", max_length=1000)
    podium: list[dict[str, Any]] = Field(default_factory=list, max_length=3)


def _resolve_manage_role_or_403(
    *,
    supabase: Any,
    club_id: str,
    authorization: str | None,
    source: str,
) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    resolution = resolve_admin_role(
        supabase=supabase,
        club_id=str(club_id),
        email=user.email,
        user_id=user.user_id,
        allowlist=set(),
    )
    if not has_permission(resolution.role, PERMISSION_MANAGE_TOURNAMENTS):
        write_admin_activity_log(
            supabase,
            build_activity_payload(
                club_id=str(club_id),
                actor_email=user.email,
                actor_role=resolution.role,
                action_type="admin_team_tournament_denied",
                entity_type="tournament_team_competition",
                entity_id="tournament_team_competition",
                after_json={
                    "source_client": "fastapi/nextjs",
                    "reason": "manage_tournaments_required",
                },
                source_page=source,
                flagged_for_review=True,
            ),
        )
        raise HTTPException(status_code=403, detail="insufficient permission")
    return user.email, resolution.role


def _confirm(actual: str, expected: str) -> None:
    if str(actual or "").strip().upper() != expected:
        raise ValueError(f"Type {expected} to confirm this tournament change.")


def _handle(exc: Exception) -> None:
    text = str(exc)
    if "STALE" in text or "RECOVERY" in text or "LOCKED" in text:
        raise HTTPException(status_code=409, detail=text) from exc
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=text) from exc
    if isinstance(exc, ValueError):
        status = 404 if "not found" in text.lower() else 400
        raise HTTPException(status_code=status, detail=text) from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=503, detail=text) from exc
    raise exc


def install_admin_tournament_team_competition_routes(
    app,
    *,
    get_supabase_client,
) -> None:
    """Install manage-only four-player and combined-rating tournament routes."""

    def context(
        club_id: str,
        authorization: str | None,
        source: str,
        mutation_surface: str | None = None,
    ):
        if not is_admin_team_tournament_enabled():
            raise HTTPException(status_code=403, detail="Team tournaments are disabled.")
        if mutation_surface:
            try:
                require_tournament_admin_mutation_runtime(mutation_surface)
            except PermissionError as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        supabase = get_supabase_client()
        actor, _role = _resolve_manage_role_or_403(
            supabase=supabase,
            club_id=club_id,
            authorization=authorization,
            source=source,
        )
        return supabase, actor

    @app.get("/admin/clubs/{club_id}/tournaments/team-competition/status")
    def status(
        club_id: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        supabase, _actor = context(
            club_id, authorization, "next_team_tournament_status"
        )
        return build_admin_team_tournament_status(supabase, club_id=club_id)

    @app.get(
        "/admin/clubs/{club_id}/tournaments/admin/tournaments/"
        "{tournament_id}/team-competition"
    )
    def snapshot(
        club_id: str,
        tournament_id: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        supabase, _actor = context(
            club_id, authorization, "next_team_tournament_snapshot"
        )
        try:
            return get_admin_team_tournament_snapshot(
                supabase, club_id=club_id, tournament_id=tournament_id
            )
        except Exception as exc:
            _handle(exc)

    @app.post(
        "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/"
        "team-competition/events/{event_option_id}/config"
    )
    def config(
        club_id: str,
        tournament_id: str,
        event_option_id: str,
        payload: CompetitionConfigRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        supabase, actor = context(
            club_id, authorization, payload.source, "setup"
        )
        try:
            _confirm(payload.confirmation_text, "SAVE COMPETITION")
            return update_tournament_competition_config(
                supabase,
                club_id=club_id,
                tournament_id=tournament_id,
                event_option_id=event_option_id,
                expected_updated_at=payload.expected_updated_at,
                patch=payload.patch,
                actor_email=actor,
                idempotency_key=payload.idempotency_key,
            )
        except Exception as exc:
            _handle(exc)

    @app.post(
        "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/"
        "team-competition/rating-verifications"
    )
    def verification(
        club_id: str,
        tournament_id: str,
        payload: RatingVerificationRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        supabase, actor = context(
            club_id, authorization, payload.source, "registration"
        )
        try:
            _confirm(payload.confirmation_text, "VERIFY RATING")
            return upsert_rating_verification(
                supabase,
                club_id=club_id,
                tournament_id=tournament_id,
                event_option_id=payload.event_option_id,
                registration_id=payload.registration_id,
                rating=payload.rating,
                note=payload.note,
                expected_version=payload.expected_version,
                actor_email=actor,
                idempotency_key=payload.idempotency_key,
            )
        except Exception as exc:
            _handle(exc)

    @app.post(
        "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/"
        "team-competition/rating-reviews"
    )
    def review(
        club_id: str,
        tournament_id: str,
        payload: RatingReviewRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        supabase, actor = context(
            club_id, authorization, payload.source, "registration"
        )
        try:
            _confirm(payload.confirmation_text, "SAVE RATING REVIEW")
            return record_combined_rating_review(
                supabase,
                club_id=club_id,
                tournament_id=tournament_id,
                event_option_id=payload.event_option_id,
                selection_id=payload.selection_id,
                review_phase=payload.review_phase,
                override_state=payload.override_state,
                override_reason=payload.override_reason,
                expected_selection_updated_at=payload.expected_selection_updated_at,
                actor_email=actor,
                idempotency_key=payload.idempotency_key,
            )
        except Exception as exc:
            _handle(exc)

    @app.post(
        "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/"
        "team-competition/rating-reviews/close"
    )
    def close_reviews(
        club_id: str,
        tournament_id: str,
        payload: RatingCloseRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        supabase, actor = context(
            club_id, authorization, payload.source, "registration"
        )
        try:
            _confirm(payload.confirmation_text, "CLOSE RATING REVIEW")
            return close_combined_rating_reviews(
                supabase,
                club_id=club_id,
                tournament_id=tournament_id,
                event_option_id=payload.event_option_id,
                entries=payload.entries,
                actor_email=actor,
                idempotency_key=payload.idempotency_key,
            )
        except Exception as exc:
            _handle(exc)

    @app.post(
        "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/"
        "team-competition/teams"
    )
    def create_team(
        club_id: str,
        tournament_id: str,
        payload: FourPlayerTeamRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        supabase, actor = context(
            club_id, authorization, payload.source, "registration"
        )
        try:
            _confirm(payload.confirmation_text, "CREATE TEAM")
            return create_four_player_team(
                supabase,
                club_id=club_id,
                tournament_id=tournament_id,
                event_option_id=payload.event_option_id,
                team_name=payload.team_name,
                captain_registration_id=payload.captain_registration_id,
                members=payload.members,
                actor_email=actor,
                idempotency_key=payload.idempotency_key,
            )
        except Exception as exc:
            _handle(exc)

    @app.post(
        "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/"
        "team-competition/teams/{team_id}/invitations/reissue"
    )
    def reissue(
        club_id: str,
        tournament_id: str,
        team_id: str,
        payload: InviteReissueRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        supabase, actor = context(
            club_id, authorization, payload.source, "registration"
        )
        try:
            _confirm(payload.confirmation_text, "REISSUE INVITATION")
            return reissue_four_player_team_invitation(
                supabase,
                club_id=club_id,
                tournament_id=tournament_id,
                team_id=team_id,
                member_id=payload.member_id,
                expected_invitation_version=payload.expected_invitation_version,
                invited_email=payload.invited_email,
                actor_email=actor,
                idempotency_key=payload.idempotency_key,
            )
        except Exception as exc:
            _handle(exc)

    @app.post(
        "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/"
        "team-competition/teams/{team_id}/roster"
    )
    def roster(
        club_id: str,
        tournament_id: str,
        team_id: str,
        payload: RosterRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        supabase, actor = context(
            club_id, authorization, payload.source, "registration"
        )
        try:
            expected = (
                "WITHDRAW TEAM"
                if payload.action.strip().upper() == "WITHDRAW"
                else "REPLACE ROSTER"
            )
            _confirm(payload.confirmation_text, expected)
            return amend_four_player_team_roster(
                supabase,
                club_id=club_id,
                tournament_id=tournament_id,
                team_id=team_id,
                expected_team_version=payload.expected_team_version,
                action=payload.action,
                members=payload.members,
                reason=payload.reason,
                actor_email=actor,
                idempotency_key=payload.idempotency_key,
            )
        except Exception as exc:
            _handle(exc)

    @app.post(
        "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/"
        "team-competition/draws/{draw_id}/round-robin"
    )
    def round_robin(
        club_id: str,
        tournament_id: str,
        draw_id: str,
        payload: RoundRobinRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        supabase, actor = context(
            club_id, authorization, payload.source, "operations"
        )
        try:
            _confirm(payload.confirmation_text, "BUILD TEAM SCHEDULE")
            return replace_team_round_robin(
                supabase,
                club_id=club_id,
                tournament_id=tournament_id,
                event_option_id=payload.event_option_id,
                draw_id=draw_id,
                team_ids=payload.team_ids,
                expected_draw_updated_at=payload.expected_draw_updated_at,
                actor_email=actor,
                idempotency_key=payload.idempotency_key,
            )
        except Exception as exc:
            _handle(exc)

    @app.post(
        "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/"
        "team-competition/draws/{draw_id}/playoffs"
    )
    def playoffs(
        club_id: str,
        tournament_id: str,
        draw_id: str,
        payload: PlayoffRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        supabase, actor = context(
            club_id, authorization, payload.source, "operations"
        )
        try:
            _confirm(payload.confirmation_text, "BUILD TEAM PLAYOFFS")
            return create_team_playoffs(
                supabase,
                club_id=club_id,
                tournament_id=tournament_id,
                draw_id=draw_id,
                playoff_format=payload.playoff_format,
                expected_draw_updated_at=payload.expected_draw_updated_at,
                actor_email=actor,
                idempotency_key=payload.idempotency_key,
            )
        except Exception as exc:
            _handle(exc)

    @app.post(
        "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/"
        "team-competition/matchups/{matchup_id}/lineups"
    )
    def lineup(
        club_id: str,
        tournament_id: str,
        matchup_id: str,
        payload: LineupRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        supabase, actor = context(
            club_id, authorization, payload.source, "operations"
        )
        try:
            _confirm(payload.confirmation_text, "LOCK TEAM LINEUP")
            return lock_team_lineup(
                supabase,
                club_id=club_id,
                tournament_id=tournament_id,
                matchup_id=matchup_id,
                team_id=payload.team_id,
                mixed_pairing=payload.mixed_pairing,
                singles_tiebreak_player_id=payload.singles_tiebreak_player_id,
                expected_matchup_version=payload.expected_matchup_version,
                expected_lineup_version=payload.expected_lineup_version,
                actor_email=actor,
                idempotency_key=payload.idempotency_key,
            )
        except Exception as exc:
            _handle(exc)

    @app.post(
        "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/"
        "team-competition/games/{match_game_id}/score"
    )
    def score(
        club_id: str,
        tournament_id: str,
        match_game_id: str,
        payload: ScoreRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        supabase, actor = context(
            club_id, authorization, payload.source, "operations"
        )
        try:
            _confirm(payload.confirmation_text, "SAVE TEAM SCORE")
            return score_team_match_game(
                supabase,
                club_id=club_id,
                tournament_id=tournament_id,
                match_game_id=match_game_id,
                score_a=payload.score_a,
                score_b=payload.score_b,
                unusual_score_acknowledged=payload.unusual_score_acknowledged,
                expected_game_version=payload.expected_game_version,
                expected_matchup_version=payload.expected_matchup_version,
                actor_email=actor,
                idempotency_key=payload.idempotency_key,
            )
        except Exception as exc:
            _handle(exc)

    @app.post(
        "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/"
        "team-competition/games/{match_game_id}/reconcile"
    )
    def reconcile(
        club_id: str,
        tournament_id: str,
        match_game_id: str,
        payload: ReconcileRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        supabase, actor = context(
            club_id, authorization, payload.source, "operations"
        )
        try:
            _confirm(payload.confirmation_text, "RECONCILE TEAM SCORE")
            return reconcile_team_match_game(
                supabase,
                club_id=club_id,
                tournament_id=tournament_id,
                match_game_id=match_game_id,
                official_match_id=payload.official_match_id,
                expected_official_row_version=payload.expected_official_row_version,
                expected_game_version=payload.expected_game_version,
                expected_matchup_version=payload.expected_matchup_version,
                reason=payload.reason,
                actor_email=actor,
                idempotency_key=payload.idempotency_key,
            )
        except Exception as exc:
            _handle(exc)

    @app.post(
        "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/"
        "team-competition/draws/{draw_id}/podium"
    )
    def podium(
        club_id: str,
        tournament_id: str,
        draw_id: str,
        payload: PodiumRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        supabase, actor = context(
            club_id, authorization, payload.source, "operations"
        )
        try:
            _confirm(
                payload.confirmation_text,
                "PUBLISH TEAM PODIUM" if payload.publish else "SAVE TEAM PODIUM",
            )
            return replace_team_podium(
                supabase,
                club_id=club_id,
                tournament_id=tournament_id,
                draw_id=draw_id,
                expected_draw_updated_at=payload.expected_draw_updated_at,
                publish=payload.publish,
                reason=payload.reason,
                actor_email=actor,
                idempotency_key=payload.idempotency_key,
                podium=payload.podium,
            )
        except Exception as exc:
            _handle(exc)
