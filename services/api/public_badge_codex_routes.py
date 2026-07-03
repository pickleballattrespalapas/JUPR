from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query
from supabase import Client

from jupr_app.services.public_badge_codex_service import (
    build_public_badge_codex,
    get_public_badge_earners,
)


def install_public_badge_codex_routes(
    app,
    *,
    get_club,
    get_supabase_client,
    public_club_payload,
) -> None:
    """Register public Badge Codex routes on the main FastAPI app."""

    @app.get("/clubs/{club_slug}/badges")
    def get_club_badge_codex(club_slug: str) -> dict[str, Any]:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        codex = build_public_badge_codex(supabase, club_id=club_id)
        return {"club": public_club_payload(club, club_slug), **codex}

    @app.get("/clubs/{club_slug}/badges/{badge_id}/earners")
    def get_club_badge_earners(
        club_slug: str,
        badge_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=25, ge=1, le=100),
    ) -> dict[str, Any]:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        try:
            earners = get_public_badge_earners(
                supabase,
                club_id=club_id,
                badge_id=badge_id,
                offset=int(offset),
                limit=int(limit),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"club": public_club_payload(club, club_slug), **earners}
