from __future__ import annotations

from typing import Any

from supabase import Client

from jupr_app.services.public_challenge_ladder_service import build_public_challenge_ladder


def install_public_challenge_ladder_routes(
    app,
    *,
    get_club,
    get_supabase_client,
    public_club_payload,
) -> None:
    """Register public Challenge Ladder routes on the main FastAPI app."""

    @app.get("/clubs/{club_slug}/challenge-ladder")
    def get_club_challenge_ladder(club_slug: str) -> dict[str, Any]:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        ladder = build_public_challenge_ladder(supabase, club_id=club_id)
        return {"club": public_club_payload(club, club_slug), **ladder}
