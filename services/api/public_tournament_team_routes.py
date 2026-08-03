from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from jupr_app.domain.tournament_registration_confirmation_tokens import (
    verify_registration_confirmation_token,
)
from jupr_app.services.public_tournament_team_service import (
    build_public_four_player_team_setup_recovery,
    build_public_team_invitation,
    build_public_team_tournament_index,
    build_public_team_tournament_results,
    create_public_four_player_team,
    public_team_tournament_runtime_ready,
    require_public_team_tournament_mutation_runtime,
    respond_public_team_invitation,
)
from jupr_app.services.admin_tournament_team_competition_service import (
    is_admin_team_tournament_enabled,
)
from jupr_app.services.production_tournament_guard import require_production_tournament_writes
from services.api.staging_write_guard import require_public_intake_or_403


class PublicTeamCreateRequest(BaseModel):
    tournament_id: str
    event_option_id: str
    team_name: str = Field(min_length=1, max_length=180)
    captain_registration_id: str
    confirmation_token: str = Field(min_length=24, max_length=4096)
    members: list[dict[str, Any]] = Field(min_length=4, max_length=4)
    idempotency_key: str = Field(min_length=8, max_length=160)
    website: str | None = None


class PublicTeamInvitationResolveRequest(BaseModel):
    token: str = Field(min_length=24, max_length=4096)
    website: str | None = None


class PublicTeamSetupRecoveryRequest(BaseModel):
    confirmation_token: str = Field(min_length=24, max_length=4096)
    website: str | None = None


class PublicTeamInvitationResponseRequest(PublicTeamInvitationResolveRequest):
    action: str
    registration_id: str
    idempotency_key: str = Field(min_length=8, max_length=160)


def _handle(exc: Exception) -> None:
    if isinstance(exc, ValueError):
        detail = str(exc)
        status = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status_code=status, detail=detail) from exc
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    raise exc


def _honeypot(value: str | None) -> None:
    if str(value or "").strip():
        raise HTTPException(status_code=400, detail="invalid submission")


def _require_team_mutation_runtime() -> None:
    try:
        require_public_team_tournament_mutation_runtime()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _require_team_feature() -> None:
    if not is_admin_team_tournament_enabled():
        raise HTTPException(status_code=404, detail="Team tournament feature not found.")


def install_public_tournament_team_routes(
    app,
    *,
    get_club,
    get_supabase_client,
    public_club_payload,
) -> None:
    @app.get("/clubs/{club_slug}/tournament-team-results")
    def results_index(
        club_slug: str,
        tournament_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        _require_team_feature()
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        try:
            result = build_public_team_tournament_index(
                get_supabase_client(),
                club_id=club_id,
                tournament_id=tournament_id,
            )
        except Exception as exc:
            _handle(exc)
        return {"club": public_club_payload(club, club_slug), **result}

    @app.get(
        "/clubs/{club_slug}/tournament-team-results/{tournament_id}/{draw_id}"
    )
    def results_detail(
        club_slug: str,
        tournament_id: str,
        draw_id: str,
    ) -> dict[str, Any]:
        _require_team_feature()
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        try:
            result = build_public_team_tournament_results(
                get_supabase_client(),
                club_id=club_id,
                tournament_id=tournament_id,
                draw_id=draw_id,
            )
        except Exception as exc:
            _handle(exc)
        return {"club": public_club_payload(club, club_slug), **result}

    @app.post("/clubs/{club_slug}/tournament-registration/four-player-team")
    def create_team(
        club_slug: str,
        payload: PublicTeamCreateRequest,
    ) -> dict[str, Any]:
        _require_team_feature()
        require_public_intake_or_403()
        try:
            require_production_tournament_writes()
        except (PermissionError, RuntimeError) as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        _require_team_mutation_runtime()
        _honeypot(payload.website)
        try:
            verify_registration_confirmation_token(
                payload.confirmation_token,
                expected_tournament_id=payload.tournament_id,
                expected_registration_id=payload.captain_registration_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if not public_team_tournament_runtime_ready():
            raise HTTPException(
                status_code=503,
                detail="Four-player team registration is temporarily unavailable.",
            )
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        try:
            result = create_public_four_player_team(
                get_supabase_client(),
                club_id=club_id,
                tournament_id=payload.tournament_id,
                event_option_id=payload.event_option_id,
                team_name=payload.team_name,
                captain_registration_id=payload.captain_registration_id,
                confirmation_token=payload.confirmation_token,
                members=payload.members,
                idempotency_key=payload.idempotency_key,
            )
        except Exception as exc:
            _handle(exc)
        return {"club": public_club_payload(club, club_slug), **result}

    @app.post(
        "/clubs/{club_slug}/tournament-registration/four-player-team/recover"
    )
    def recover_team_setup(
        club_slug: str,
        payload: PublicTeamSetupRecoveryRequest,
    ) -> dict[str, Any]:
        _require_team_feature()
        _honeypot(payload.website)
        # Reject malformed/expired bearer proof before resolving a club or
        # opening the database.  The service repeats verification against the
        # durable captain email before returning private setup details.
        try:
            verify_registration_confirmation_token(payload.confirmation_token)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if not public_team_tournament_runtime_ready():
            raise HTTPException(
                status_code=503,
                detail="Four-player team registration is temporarily unavailable.",
            )
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        try:
            result = build_public_four_player_team_setup_recovery(
                get_supabase_client(),
                club_id=club_id,
                confirmation_token=payload.confirmation_token,
            )
        except Exception as exc:
            _handle(exc)
        return {"club": public_club_payload(club, club_slug), **result}

    @app.post("/clubs/{club_slug}/tournament-team-invitation/resolve")
    def resolve_invitation(
        club_slug: str,
        payload: PublicTeamInvitationResolveRequest,
    ) -> dict[str, Any]:
        _require_team_feature()
        _honeypot(payload.website)
        if not public_team_tournament_runtime_ready():
            raise HTTPException(
                status_code=503, detail="Team invitations are temporarily unavailable."
            )
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        try:
            result = build_public_team_invitation(
                get_supabase_client(), club_id=club_id, token=payload.token
            )
        except Exception as exc:
            _handle(exc)
        return {"club": public_club_payload(club, club_slug), **result}

    @app.post("/clubs/{club_slug}/tournament-team-invitation/respond")
    def respond_invitation(
        club_slug: str,
        payload: PublicTeamInvitationResponseRequest,
    ) -> dict[str, Any]:
        _require_team_feature()
        require_public_intake_or_403()
        try:
            require_production_tournament_writes()
        except (PermissionError, RuntimeError) as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        _require_team_mutation_runtime()
        _honeypot(payload.website)
        if not public_team_tournament_runtime_ready():
            raise HTTPException(
                status_code=503, detail="Team invitations are temporarily unavailable."
            )
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        try:
            result = respond_public_team_invitation(
                get_supabase_client(),
                club_id=club_id,
                token=payload.token,
                action=payload.action,
                registration_id=payload.registration_id,
                idempotency_key=payload.idempotency_key,
            )
        except Exception as exc:
            _handle(exc)
        return {"club": public_club_payload(club, club_slug), **result}
