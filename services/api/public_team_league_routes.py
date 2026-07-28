from __future__ import annotations

from typing import Any, Callable
from urllib.parse import quote

from fastapi import HTTPException
from pydantic import BaseModel, Field

from jupr_app.config import get_next_web_base_url
from jupr_app.services.team_league_service import (
    TeamLeagueConflictError,
    TeamLeagueRecoveryRequiredError,
    confirm_public_team_league_partner,
    get_public_team_league,
    list_public_team_leagues,
    register_public_team_league,
)
from services.api.staging_write_guard import (
    require_public_team_league_write_or_403,
)
from services.api.team_league_feature import (
    require_team_leagues_enabled_or_403,
)


class PublicTeamLeagueRegistrationRequest(BaseModel):
    signup_type: str = Field(pattern=r"^(team|solo)$")
    player_id: int
    contact_email: str = Field(min_length=3, max_length=320)
    partner_player_id: int | None = None
    partner_email: str = Field(default="", max_length=320)
    team_name: str = Field(default="", max_length=120)
    note: str = Field(default="", max_length=500)
    idempotency_key: str = Field(min_length=8, max_length=160)
    confirmation_text: str = Field(min_length=1, max_length=80)


class PublicTeamLeaguePartnerConfirmationRequest(BaseModel):
    team_id: str = Field(min_length=1, max_length=80)
    token: str = Field(min_length=24, max_length=400)
    accept: bool
    idempotency_key: str = Field(min_length=8, max_length=160)


def _public_error(exc: Exception) -> None:
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
    if isinstance(exc, (ValueError, PermissionError)):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise exc


def install_public_team_league_routes(
    app,
    *,
    get_club: Callable[[str], dict[str, Any]],
    get_supabase_client,
    public_club_payload: Callable[[dict[str, Any], str], dict[str, Any]],
) -> None:
    @app.get("/clubs/{club_slug}/team-leagues")
    def get_public_team_leagues(club_slug: str) -> dict[str, Any]:
        require_team_leagues_enabled_or_403()
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        try:
            result = list_public_team_leagues(
                get_supabase_client(), club_id=club_id
            )
        except Exception as exc:
            _public_error(exc)
        return {
            "club": public_club_payload(club, club_slug),
            **result,
        }

    @app.get("/clubs/{club_slug}/team-leagues/{league_name}")
    def get_public_team_league_detail(
        club_slug: str, league_name: str
    ) -> dict[str, Any]:
        require_team_leagues_enabled_or_403()
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        try:
            result = get_public_team_league(
                get_supabase_client(),
                club_id=club_id,
                league_name=league_name,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            _public_error(exc)
        return {
            "club": public_club_payload(club, club_slug),
            **result,
        }

    @app.post("/clubs/{club_slug}/team-leagues/{league_name}/registrations")
    def post_public_team_league_registration(
        club_slug: str,
        league_name: str,
        payload: PublicTeamLeagueRegistrationRequest,
    ) -> dict[str, Any]:
        require_team_leagues_enabled_or_403()
        require_public_team_league_write_or_403()
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        public_base_url = (
            f"{get_next_web_base_url().rstrip('/')}/clubs/"
            f"{quote(str(club.get('slug') or club_slug).strip(), safe='-')}"
        )
        try:
            result = register_public_team_league(
                get_supabase_client(),
                club_id=club_id,
                league_name=league_name,
                signup_type=payload.signup_type,
                player_id=payload.player_id,
                contact_email=payload.contact_email,
                partner_player_id=payload.partner_player_id,
                partner_email=payload.partner_email,
                team_name=payload.team_name,
                note=payload.note,
                idempotency_key=payload.idempotency_key,
                confirmation_text=payload.confirmation_text,
                public_base_url=public_base_url,
                club_name=str(club.get("name") or club_slug),
            )
        except Exception as exc:
            _public_error(exc)
        return {
            "club": public_club_payload(club, club_slug),
            **result,
        }

    @app.post("/clubs/{club_slug}/team-leagues/partner-confirmations")
    def post_public_team_league_partner_confirmation(
        club_slug: str,
        payload: PublicTeamLeaguePartnerConfirmationRequest,
    ) -> dict[str, Any]:
        require_team_leagues_enabled_or_403()
        require_public_team_league_write_or_403()
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        try:
            result = confirm_public_team_league_partner(
                get_supabase_client(),
                club_id=club_id,
                team_id=payload.team_id,
                token=payload.token,
                accept=payload.accept,
                idempotency_key=payload.idempotency_key,
            )
        except Exception as exc:
            _public_error(exc)
        return {
            "club": public_club_payload(club, club_slug),
            **result,
        }
