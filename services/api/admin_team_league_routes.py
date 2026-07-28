from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import (
    PERMISSION_MANAGE_MATCHES,
    has_permission,
    resolve_admin_role,
)
from jupr_app.domain.admin_activity_log import (
    build_activity_payload,
    write_admin_activity_log,
)
from jupr_app.services.team_league_service import (
    TeamLeagueConflictError,
    TeamLeagueRecoveryRequiredError,
    admin_team_league_waitlist_action,
    build_admin_team_league_schedule_preview,
    commit_admin_team_league_schedule,
    get_admin_team_league,
    inspect_admin_team_league_operation,
    list_admin_team_leagues,
    reconcile_admin_team_league_fixture,
    resolve_admin_team_league_operation,
    save_admin_team_league_settings,
    score_admin_team_league_fixture,
)
from services.api.auth import authenticate_bearer, auth_header
from services.api.staging_write_guard import (
    require_admin_team_league_write_or_403,
)
from services.api.team_league_feature import (
    require_team_leagues_enabled_or_403,
)


class TeamLeagueSettingsRequest(BaseModel):
    settings: dict[str, Any]
    expected_settings_version: int = Field(ge=0)
    idempotency_key: str = Field(min_length=8, max_length=160)
    confirmation_text: str = Field(min_length=1, max_length=80)
    source: str = Field(default="next_team_league_settings", max_length=160)


class TeamLeagueScheduleCommitRequest(BaseModel):
    phase: str = Field(pattern=r"^(regular|playoff)$")
    fixtures: list[dict[str, Any]] = Field(min_length=1, max_length=512)
    expected_schedule_version: int = Field(ge=0)
    expected_standings_version: int = Field(ge=0)
    expected_roster_version: int = Field(ge=0)
    confirmed_roster_fingerprint: str = Field(min_length=64, max_length=64)
    preview_fingerprint: str = Field(min_length=64, max_length=64)
    idempotency_key: str = Field(min_length=8, max_length=160)
    confirmation_text: str = Field(min_length=1, max_length=80)
    source: str = Field(default="next_team_league_schedule", max_length=160)


class TeamLeagueWaitlistActionRequest(BaseModel):
    action: str = Field(pattern=r"^(pair|withdraw)$")
    waitlist_ids: list[str] = Field(min_length=1, max_length=200)
    team_name: str = Field(default="", max_length=120)
    idempotency_key: str = Field(min_length=8, max_length=160)
    confirmation_text: str = Field(min_length=1, max_length=80)
    source: str = Field(default="next_team_league_waitlist", max_length=160)


class TeamLeagueFixtureScoreRequest(BaseModel):
    status: str = Field(pattern=r"^(complete|forfeit)$")
    team_a_score: int | None = Field(default=None, ge=0)
    team_b_score: int | None = Field(default=None, ge=0)
    winner_team_id: str = Field(min_length=1, max_length=80)
    team_a_player_ids: list[int] = Field(default_factory=list, max_length=2)
    team_b_player_ids: list[int] = Field(default_factory=list, max_length=2)
    score_note: str = Field(default="", max_length=500)
    idempotency_key: str = Field(min_length=8, max_length=160)
    confirmation_text: str = Field(min_length=1, max_length=80)
    source: str = Field(default="next_team_league_score", max_length=160)


class TeamLeagueReconcileRequest(BaseModel):
    idempotency_key: str = Field(min_length=8, max_length=160)
    confirmation_text: str = Field(min_length=1, max_length=80)
    source: str = Field(default="next_team_league_reconcile", max_length=160)


class TeamLeagueRecoveryRequest(BaseModel):
    resolution: str = Field(pattern=r"^(finalize|compensate)$")
    recovery_note: str = Field(min_length=5, max_length=500)
    confirmation_text: str = Field(min_length=1, max_length=80)
    source: str = Field(default="next_team_league_recovery", max_length=160)


def _role_or_403(
    supabase: Any,
    *,
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
    if not has_permission(resolution.role, PERMISSION_MANAGE_MATCHES):
        write_admin_activity_log(
            supabase,
            build_activity_payload(
                club_id=str(club_id),
                actor_email=user.email,
                actor_role=resolution.role,
                action_type="team_league_access_denied",
                entity_type="team_league",
                entity_id="admin",
                after_json={"reason": "insufficient_permission"},
                source_page=source,
                flagged_for_review=True,
            ),
        )
        raise HTTPException(status_code=403, detail="insufficient permission")
    return user.email, resolution.role


def _error(exc: Exception) -> None:
    if isinstance(exc, TeamLeagueConflictError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, TeamLeagueRecoveryRequiredError):
        raise HTTPException(
            status_code=503,
            detail={
                "message": str(exc),
                "operation_id": exc.operation_id,
                "recovery_required": True,
            },
        ) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    raise exc


def install_admin_team_league_routes(app, *, get_supabase_client) -> None:
    prefix = "/admin/clubs/{club_id}/league-manager/team-leagues"

    @app.get(prefix)
    def get_admin_team_leagues(
        club_id: str, authorization: str | None = auth_header()
    ) -> dict[str, Any]:
        require_team_leagues_enabled_or_403()
        supabase = get_supabase_client()
        _role_or_403(
            supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_team_league_list",
        )
        try:
            return list_admin_team_leagues(supabase, club_id=str(club_id))
        except Exception as exc:
            _error(exc)

    @app.get(f"{prefix}/{{league_name}}")
    def get_admin_team_league_detail(
        club_id: str,
        league_name: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        require_team_leagues_enabled_or_403()
        supabase = get_supabase_client()
        _role_or_403(
            supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_team_league_detail",
        )
        try:
            return get_admin_team_league(
                supabase, club_id=str(club_id), league_name=league_name
            )
        except Exception as exc:
            _error(exc)

    @app.put(
        "/admin/clubs/{club_id}/league-manager/team-leagues/{league_name}/settings"
    )
    def put_admin_team_league_settings(
        club_id: str,
        league_name: str,
        payload: TeamLeagueSettingsRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        require_team_leagues_enabled_or_403()
        require_admin_team_league_write_or_403()
        supabase = get_supabase_client()
        email, role = _role_or_403(
            supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return save_admin_team_league_settings(
                supabase,
                club_id=str(club_id),
                league_name=league_name,
                settings=payload.settings,
                expected_settings_version=payload.expected_settings_version,
                idempotency_key=payload.idempotency_key,
                confirmation_text=payload.confirmation_text,
                actor_email=email,
                actor_role=role,
                source=payload.source,
            )
        except Exception as exc:
            _error(exc)

    @app.post(
        "/admin/clubs/{club_id}/league-manager/team-leagues/"
        "{league_name}/schedule-preview/{phase}"
    )
    def post_admin_team_league_schedule_preview(
        club_id: str,
        league_name: str,
        phase: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        require_team_leagues_enabled_or_403()
        supabase = get_supabase_client()
        _role_or_403(
            supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_team_league_schedule_preview",
        )
        try:
            return build_admin_team_league_schedule_preview(
                supabase,
                club_id=str(club_id),
                league_name=league_name,
                phase=phase,
            )
        except Exception as exc:
            _error(exc)

    @app.post(
        "/admin/clubs/{club_id}/league-manager/team-leagues/{league_name}/schedule"
    )
    def post_admin_team_league_schedule(
        club_id: str,
        league_name: str,
        payload: TeamLeagueScheduleCommitRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        require_team_leagues_enabled_or_403()
        require_admin_team_league_write_or_403()
        supabase = get_supabase_client()
        email, role = _role_or_403(
            supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return commit_admin_team_league_schedule(
                supabase,
                club_id=str(club_id),
                league_name=league_name,
                phase=payload.phase,
                fixtures=payload.fixtures,
                expected_schedule_version=payload.expected_schedule_version,
                expected_standings_version=payload.expected_standings_version,
                expected_roster_version=payload.expected_roster_version,
                confirmed_roster_fingerprint_value=payload.confirmed_roster_fingerprint,
                preview_fingerprint=payload.preview_fingerprint,
                idempotency_key=payload.idempotency_key,
                confirmation_text=payload.confirmation_text,
                actor_email=email,
                actor_role=role,
                source=payload.source,
            )
        except Exception as exc:
            _error(exc)

    @app.post(
        "/admin/clubs/{club_id}/league-manager/team-leagues/"
        "{league_name}/waitlist-actions"
    )
    def post_admin_team_league_waitlist_action(
        club_id: str,
        league_name: str,
        payload: TeamLeagueWaitlistActionRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        require_team_leagues_enabled_or_403()
        require_admin_team_league_write_or_403()
        supabase = get_supabase_client()
        email, role = _role_or_403(
            supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return admin_team_league_waitlist_action(
                supabase,
                club_id=str(club_id),
                league_name=league_name,
                action=payload.action,
                waitlist_ids=payload.waitlist_ids,
                team_name=payload.team_name,
                idempotency_key=payload.idempotency_key,
                confirmation_text=payload.confirmation_text,
                actor_email=email,
                actor_role=role,
                source=payload.source,
            )
        except Exception as exc:
            _error(exc)

    @app.post(
        "/admin/clubs/{club_id}/league-manager/team-leagues/"
        "{league_name}/fixtures/{fixture_id}/score"
    )
    def post_admin_team_league_fixture_score(
        club_id: str,
        league_name: str,
        fixture_id: str,
        payload: TeamLeagueFixtureScoreRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        require_team_leagues_enabled_or_403()
        require_admin_team_league_write_or_403()
        supabase = get_supabase_client()
        email, role = _role_or_403(
            supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return score_admin_team_league_fixture(
                supabase,
                club_id=str(club_id),
                league_name=league_name,
                fixture_id=fixture_id,
                status=payload.status,
                team_a_score=payload.team_a_score,
                team_b_score=payload.team_b_score,
                winner_team_id=payload.winner_team_id,
                team_a_player_ids=payload.team_a_player_ids,
                team_b_player_ids=payload.team_b_player_ids,
                score_note=payload.score_note,
                idempotency_key=payload.idempotency_key,
                confirmation_text=payload.confirmation_text,
                actor_email=email,
                actor_role=role,
                source=payload.source,
            )
        except Exception as exc:
            _error(exc)

    @app.post(
        "/admin/clubs/{club_id}/league-manager/team-leagues/"
        "{league_name}/fixtures/{fixture_id}/reconcile"
    )
    def post_admin_team_league_fixture_reconcile(
        club_id: str,
        league_name: str,
        fixture_id: str,
        payload: TeamLeagueReconcileRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        require_team_leagues_enabled_or_403()
        require_admin_team_league_write_or_403()
        supabase = get_supabase_client()
        email, role = _role_or_403(
            supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return reconcile_admin_team_league_fixture(
                supabase,
                club_id=str(club_id),
                league_name=league_name,
                fixture_id=fixture_id,
                idempotency_key=payload.idempotency_key,
                confirmation_text=payload.confirmation_text,
                actor_email=email,
                actor_role=role,
                source=payload.source,
            )
        except Exception as exc:
            _error(exc)

    @app.get(f"{prefix}/operations/{{operation_id}}")
    def get_admin_team_league_operation(
        club_id: str,
        operation_id: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        require_team_leagues_enabled_or_403()
        supabase = get_supabase_client()
        _role_or_403(
            supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_team_league_recovery",
        )
        try:
            return inspect_admin_team_league_operation(
                supabase,
                club_id=str(club_id),
                operation_id=operation_id,
            )
        except Exception as exc:
            _error(exc)

    @app.post(
        "/admin/clubs/{club_id}/league-manager/team-leagues/"
        "operations/{operation_id}/resolve"
    )
    def post_admin_team_league_operation_resolution(
        club_id: str,
        operation_id: str,
        payload: TeamLeagueRecoveryRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        require_team_leagues_enabled_or_403()
        require_admin_team_league_write_or_403()
        supabase = get_supabase_client()
        email, role = _role_or_403(
            supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return resolve_admin_team_league_operation(
                supabase,
                club_id=str(club_id),
                operation_id=operation_id,
                resolution=payload.resolution,
                recovery_note=payload.recovery_note,
                confirmation_text=payload.confirmation_text,
                actor_email=email,
                actor_role=role,
                source=payload.source,
            )
        except Exception as exc:
            _error(exc)
