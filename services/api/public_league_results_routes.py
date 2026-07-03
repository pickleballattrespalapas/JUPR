from __future__ import annotations

from typing import Any

from fastapi import Query
from supabase import Client

from jupr_app.services.public_league_results_service import build_public_league_results


def install_public_league_results_routes(
    app,
    *,
    get_club,
    get_supabase_client,
    public_club_payload,
) -> None:
    """Register public League Results routes on the main FastAPI app."""

    @app.get("/clubs/{club_slug}/league-results")
    def get_club_league_results(
        club_slug: str,
        league_name: str | None = Query(default=None),
    ) -> dict[str, Any]:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        result = build_public_league_results(supabase, club_id=club_id, league_name=league_name)
        return {"club": public_club_payload(club, club_slug), **result}
