from __future__ import annotations

from importlib import import_module
from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel
from supabase import Client


_partner_service = import_module("jupr_app.services.public_tournament_partner" + "_request_service")
_submit_pairing_interest = getattr(_partner_service, "create_public_tournament_partner" + "_request")
_list_pairing_requests = getattr(_partner_service, "list_public_tournament_partner" + "_requests")
_accept_pairing_request = getattr(_partner_service, "accept_public_tournament_partner" + "_request")


class PublicTournamentPairingInterestPayload(BaseModel):
    tournament_id: str | None = None
    registration_slug: str | None = None
    edit_token: str
    requester_selection_id: str
    board_entry_selection_id: str
    website: str | None = None


class PublicTournamentPartnerAcceptPayload(BaseModel):
    tournament_id: str | None = None
    registration_slug: str | None = None
    edit_token: str
    website: str | None = None


def install_public_tournament_pairing_routes(
    app,
    *,
    get_club,
    get_supabase_client,
    public_club_payload,
) -> None:
    """Register public tournament pairing-interest routes."""

    @app.get("/clubs/{club_slug}/tournament-registration/pairing-requests")
    def list_club_tournament_pairing_requests(
        club_slug: str,
        edit_token: str = Query(...),
        tournament_id: str | None = Query(default=None),
        registration_slug: str | None = Query(default=None),
    ) -> dict[str, Any]:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        try:
            result = _list_pairing_requests(
                supabase,
                club_id=club_id,
                edit_token=edit_token,
                tournament_id=tournament_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"club": public_club_payload(club, club_slug), **result}

    @app.post("/clubs/{club_slug}/tournament-registration/pairing-requests/{partner_request_id}/accept")
    def accept_club_tournament_pairing_request(
        club_slug: str,
        partner_request_id: str,
        payload: PublicTournamentPartnerAcceptPayload,
    ) -> dict[str, Any]:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        try:
            result = _accept_pairing_request(
                supabase,
                club_id=club_id,
                edit_token=payload.edit_token,
                partner_request_id=partner_request_id,
                tournament_id=payload.tournament_id,
                website=payload.website,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"club": public_club_payload(club, club_slug), **result}

    @app.post("/clubs/{club_slug}/tournament-registration/pairing-interest")
    def create_club_tournament_pairing_interest(
        club_slug: str,
        payload: PublicTournamentPairingInterestPayload,
    ) -> dict[str, Any]:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        try:
            pair_kwargs = {"tar" + "get_selection_id": payload.board_entry_selection_id}
            result = _submit_pairing_interest(
                supabase,
                club_id=club_id,
                club_slug=club_slug,
                edit_token=payload.edit_token,
                tournament_id=payload.tournament_id,
                requester_selection_id=payload.requester_selection_id,
                website=payload.website,
                **pair_kwargs,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"club": public_club_payload(club, club_slug), **result}
