from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field
from supabase import Client

from jupr_app.domain.tournament_commerce import (
    TournamentCommerceValidationError,
)
from jupr_app.services.public_tournament_commerce_service import (
    TournamentCommerceConflictError,
    TournamentCommerceUnavailableError,
    build_public_tournament_commerce_catalog,
    is_tournament_commerce_enabled,
    quote_public_tournament_commerce,
)


class PublicTournamentCommerceQuoteRequest(BaseModel):
    tournament_id: str
    registration_id: str | None = None
    event_option_ids: list[str] = Field(default_factory=list, max_length=20)
    item_selections: list[dict[str, Any]] = Field(
        default_factory=list, max_length=50
    )
    website: str | None = None


def _dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _handle(exc: Exception) -> None:
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, TournamentCommerceConflictError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, TournamentCommerceValidationError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, TournamentCommerceUnavailableError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    raise exc


def _require_feature() -> None:
    if not is_tournament_commerce_enabled():
        raise HTTPException(
            status_code=404,
            detail="Tournament commerce is unavailable.",
        )


def install_public_tournament_commerce_routes(
    app,
    *,
    get_club,
    get_supabase_client,
    public_club_payload,
) -> None:
    """Install server-authoritative public catalog and quote routes."""

    @app.get("/clubs/{club_slug}/tournament-commerce")
    def get_tournament_commerce(
        club_slug: str,
        tournament_id: str = Query(...),
    ) -> dict[str, Any]:
        _require_feature()
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        try:
            catalog = build_public_tournament_commerce_catalog(
                supabase,
                club_id=club_id,
                tournament_id=tournament_id,
            )
        except Exception as exc:
            _handle(exc)
        return {
            "club": public_club_payload(club, club_slug),
            **catalog,
        }

    @app.post("/clubs/{club_slug}/tournament-commerce/quote")
    def post_tournament_commerce_quote(
        club_slug: str,
        payload: PublicTournamentCommerceQuoteRequest,
    ) -> dict[str, Any]:
        _require_feature()
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        request = _dump_model(payload)
        try:
            result = quote_public_tournament_commerce(
                supabase,
                club_id=club_id,
                tournament_id=payload.tournament_id,
                registration_id=payload.registration_id,
                request={
                    "event_option_ids": request["event_option_ids"],
                    "item_selections": request["item_selections"],
                    "website": request.get("website"),
                },
            )
        except Exception as exc:
            _handle(exc)
        public_result = {
            key: value
            for key, value in result.items()
            if key != "current_order"
        }
        return {
            "club": public_club_payload(club, club_slug),
            **public_result,
        }
