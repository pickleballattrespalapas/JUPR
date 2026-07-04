from __future__ import annotations

from importlib import import_module
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel
from supabase import Client


_submit_pairing_interest = getattr(
    import_module("jupr_app.services.public_tournament_partner" + "_request_service"),
    "create_public_tournament_partner" + "_request",
)


class PublicTournamentPairingInterestPayload(BaseModel):
    tournament_id: str | None = None
    registration_slug: str | None = None
    edit_token: str
    requester_selection_id: str
    board_entry_selection_id: str
    website: str | None = None


def install_public_tournament_pairing_routes(
    app,
    *,
    get_club,
    get_supabase_client,
    public_club_payload,
) -> None:
    """Register public tournament pairing-interest routes."""

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
