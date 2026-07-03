from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field
from supabase import Client

from jupr_app.services.public_tournament_registration_service import (
    build_public_tournament_registration_confirmation,
    build_public_tournament_registration_page,
    submit_public_tournament_registration,
)


class PublicTournamentRegistrationRequest(BaseModel):
    tournament_id: str | None = None
    registration_slug: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    display_name: str | None = None
    email: str | None = None
    phone: str | None = None
    player_id: str | int | None = None
    dupr_id: str | None = None
    doubles_skill: float | None = None
    singles_skill: float | None = None
    age: int | None = None
    gender: str | None = None
    notes: str | None = None
    wants_partner_board_contact: bool = False
    terms_accepted: bool = False
    website: str | None = None
    selections: list[dict[str, Any]] = Field(default_factory=list)


def install_public_tournament_registration_routes(
    app,
    *,
    get_club,
    get_supabase_client,
    public_club_payload,
) -> None:
    """Register public tournament registration routes on the main FastAPI app."""

    @app.get("/clubs/{club_slug}/tournament-registration")
    def get_club_tournament_registration(
        club_slug: str,
        tournament_id: str | None = Query(default=None),
        registration_slug: str | None = Query(default=None),
    ) -> dict[str, Any]:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        page = build_public_tournament_registration_page(
            supabase,
            club_id=club_id,
            tournament_id=tournament_id,
            registration_slug=registration_slug,
        )
        return {"club": public_club_payload(club, club_slug), **page}

    @app.post("/clubs/{club_slug}/tournament-registration")
    def submit_club_tournament_registration(
        club_slug: str,
        payload: PublicTournamentRegistrationRequest,
    ) -> dict[str, Any]:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        try:
            result = submit_public_tournament_registration(
                supabase,
                club_id=club_id,
                payload=payload.dict(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"club": public_club_payload(club, club_slug), **result}

    @app.get("/clubs/{club_slug}/tournament-registration/confirmations/{registration_id}")
    def get_club_tournament_registration_confirmation(
        club_slug: str,
        registration_id: str,
        tournament_id: str | None = Query(default=None),
        registration_slug: str | None = Query(default=None),
    ) -> dict[str, Any]:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        confirmation = build_public_tournament_registration_confirmation(
            supabase,
            club_id=club_id,
            registration_id=registration_id,
            tournament_id=tournament_id,
            registration_slug=registration_slug,
        )
        if not confirmation:
            raise HTTPException(status_code=404, detail="registration confirmation not found")
        return {"club": public_club_payload(club, club_slug), **confirmation}
