from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from supabase import Client

from jupr_app.services.public_match_explorer_service import (
    build_public_match_explorer_preview,
    get_public_match_explorer_contexts,
)

router = APIRouter()


def install_public_match_explorer_routes(
    app,
    *,
    get_club,
    get_supabase_client,
    public_club_payload,
) -> None:
    """Register public Match Explorer routes on the main FastAPI app."""

    @app.get("/clubs/{club_slug}/match-explorer")
    def get_club_match_explorer_context(club_slug: str) -> dict[str, Any]:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        contexts = get_public_match_explorer_contexts(supabase, club_id=club_id)
        return {"club": public_club_payload(club, club_slug), "contexts": contexts}

    @app.get("/clubs/{club_slug}/match-explorer/preview")
    def get_club_match_explorer_preview(
        club_slug: str,
        me: int = Query(..., ge=1),
        partner: int = Query(..., ge=1),
        opp1: int = Query(..., ge=1),
        opp2: int = Query(..., ge=1),
        context: str = Query(default="OVERALL"),
        score_you: int = Query(default=11, ge=0, le=99),
        score_opp: int = Query(default=9, ge=0, le=99),
    ) -> dict[str, Any]:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        try:
            preview = build_public_match_explorer_preview(
                supabase,
                club_id=club_id,
                me=int(me),
                partner=int(partner),
                opp1=int(opp1),
                opp2=int(opp2),
                context_name=str(context or "OVERALL"),
                score_you=int(score_you),
                score_opp=int(score_opp),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"club": public_club_payload(club, club_slug), "preview": preview}
