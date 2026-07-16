from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel

from jupr_app.services.public_verified_updates_service import (
    create_public_verified_update_request,
    get_public_verified_update_request_status,
    list_public_verified_update_player_options,
)


class PublicVerifiedUpdateRequest(BaseModel):
    player_id: int
    email: str
    request_note: str | None = None
    website: str | None = None


def install_public_verified_updates_routes(app, *, get_club, get_supabase_client, public_club_payload) -> None:
    """Register public verified player-update request routes."""

    @app.get("/clubs/{club_slug}/verified-updates/options")
    def get_verified_updates_player_options(
        club_slug: str,
        q: str | None = Query(default=None),
        limit: int = Query(default=500, ge=1, le=1000),
    ) -> dict[str, Any]:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase = get_supabase_client()
        try:
            payload = list_public_verified_update_player_options(supabase, club_id=club_id, q=q, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"club": public_club_payload(club, club_slug), **payload}

    @app.get("/clubs/{club_slug}/verified-updates/status")
    def get_verified_updates_request_status(
        club_slug: str,
        player_id: int = Query(...),
    ) -> dict[str, Any]:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase = get_supabase_client()
        try:
            payload = get_public_verified_update_request_status(supabase, club_id=club_id, player_id=player_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"club": public_club_payload(club, club_slug), **payload}

    @app.post("/clubs/{club_slug}/verified-updates/request")
    def post_verified_updates_request(club_slug: str, payload: PublicVerifiedUpdateRequest) -> dict[str, Any]:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase = get_supabase_client()
        try:
            result = create_public_verified_update_request(
                supabase,
                club_id=club_id,
                player_id=payload.player_id,
                email=payload.email,
                request_note=payload.request_note,
                honeypot=payload.website,
            )
        except ValueError as exc:
            message = str(exc)
            if "already" in message.lower():
                raise HTTPException(status_code=409, detail=message) from exc
            raise HTTPException(status_code=400, detail=message) from exc
        return {"club": public_club_payload(club, club_slug), **result}
