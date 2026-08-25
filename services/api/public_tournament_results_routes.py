from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query

from jupr_app.services.public_tournament_results_service import (
    build_public_tournament_index,
    build_public_tournament_results,
)


def _handle_public_results_error(exc: Exception) -> None:
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=404, detail="tournament results not found") from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    raise exc


def install_public_tournament_results_routes(
    app,
    *,
    get_club,
    get_supabase_client,
    public_club_payload,
) -> None:
    """Install patron-safe standard tournament discovery and results routes."""

    @app.get("/clubs/{club_slug}/tournaments")
    def tournament_index(
        club_slug: str,
        view: str = Query(default="current", pattern="^(current|past)$"),
    ) -> dict[str, Any]:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        try:
            result = build_public_tournament_index(
                get_supabase_client(),
                club_id=club_id,
                view=view,
            )
        except Exception as exc:
            _handle_public_results_error(exc)
        return {"club": public_club_payload(club, club_slug), **result}

    @app.get("/clubs/{club_slug}/tournament-results")
    def tournament_results(
        club_slug: str,
        tournament_id: str = Query(min_length=1, max_length=160),
    ) -> dict[str, Any]:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        try:
            result = build_public_tournament_results(
                get_supabase_client(),
                club_id=club_id,
                tournament_id=tournament_id,
            )
        except Exception as exc:
            _handle_public_results_error(exc)
        return {"club": public_club_payload(club, club_slug), **result}


__all__ = ["install_public_tournament_results_routes"]
