from __future__ import annotations

import os
from typing import Any

from fastapi import HTTPException, Query, Response
from pydantic import BaseModel, Field
from supabase import Client

from jupr_app.domain.tournament_registration_repo import (
    TournamentRegistrationEditConflictError,
    TournamentRegistrationImportedDrawError,
    TournamentRegistrationRelationshipLockedError,
)
from jupr_app.services.public_tournament_registration_edit_service import (
    PublicRegistrationEditUnavailableError,
    build_public_tournament_registration_edit_page,
    request_public_tournament_registration_edit_link,
    submit_public_tournament_registration_edit,
)
from jupr_app.services.public_tournament_registration_service import (
    DuplicateTournamentRegistrationError,
    build_public_tournament_registration_confirmation,
    build_public_tournament_registration_page,
    resolve_public_tournament_registration_profile,
    submit_public_tournament_registration,
)
from jupr_app.services.public_tournament_roster_service import build_public_tournament_roster_page


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


class PublicTournamentRegistrationEditRequest(PublicTournamentRegistrationRequest):
    edit_token: str
    expected_updated_at: str
    expected_selection_versions: list[dict[str, Any]] = Field(default_factory=list)


class PublicTournamentRegistrationEditLinkRequest(BaseModel):
    tournament_id: str | None = None
    registration_slug: str | None = None
    email: str
    website: str | None = None


class PublicTournamentRegistrationProfileResolutionRequest(BaseModel):
    tournament_id: str | None = None
    registration_slug: str | None = None
    first_name: str
    last_name: str
    email: str
    age: int
    gender: str
    website: str | None = None


def _dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _require_registration_edit_service_role() -> None:
    if not os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail="Public registration editing is temporarily unavailable because its server credential is not configured.",
        )


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

    @app.post("/clubs/{club_slug}/tournament-registration/edit-link/request")
    def request_club_tournament_registration_edit_link(
        club_slug: str,
        payload: PublicTournamentRegistrationEditLinkRequest,
    ) -> dict[str, Any]:
        _require_registration_edit_service_role()
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        try:
            result = request_public_tournament_registration_edit_link(
                supabase,
                club_id=club_id,
                club_slug=club_slug,
                email=payload.email,
                tournament_id=payload.tournament_id,
                registration_slug=payload.registration_slug,
                website=payload.website,
            )
        except PublicRegistrationEditUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"club": public_club_payload(club, club_slug), **result}

    @app.post("/clubs/{club_slug}/tournament-registration/profile-resolution")
    def resolve_club_tournament_registration_profile(
        club_slug: str,
        payload: PublicTournamentRegistrationProfileResolutionRequest,
    ) -> dict[str, Any]:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        try:
            result = resolve_public_tournament_registration_profile(
                supabase,
                club_id=club_id,
                payload=_dump_model(payload),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"club": public_club_payload(club, club_slug), **result}

    @app.get("/clubs/{club_slug}/tournament-registration/edit")
    def get_club_tournament_registration_edit(
        club_slug: str,
        response: Response,
        edit_token: str = Query(...),
        tournament_id: str | None = Query(default=None),
        registration_slug: str | None = Query(default=None),
    ) -> dict[str, Any]:
        response.headers["Cache-Control"] = "no-store, private"
        _require_registration_edit_service_role()
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        try:
            page = build_public_tournament_registration_edit_page(
                supabase,
                club_id=club_id,
                edit_token=edit_token,
                tournament_id=tournament_id,
                registration_slug=registration_slug,
            )
        except PublicRegistrationEditUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (TournamentRegistrationImportedDrawError, TournamentRegistrationEditConflictError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"club": public_club_payload(club, club_slug), **page}

    @app.post("/clubs/{club_slug}/tournament-registration/edit")
    def submit_club_tournament_registration_edit(
        club_slug: str,
        payload: PublicTournamentRegistrationEditRequest,
    ) -> dict[str, Any]:
        _require_registration_edit_service_role()
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        try:
            result = submit_public_tournament_registration_edit(
                supabase,
                club_id=club_id,
                club_slug=club_slug,
                edit_token=payload.edit_token,
                payload=_dump_model(payload),
            )
        except PublicRegistrationEditUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (
            TournamentRegistrationImportedDrawError,
            TournamentRegistrationEditConflictError,
            TournamentRegistrationRelationshipLockedError,
        ) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"club": public_club_payload(club, club_slug), **result}

    @app.get("/clubs/{club_slug}/tournament-roster")
    def get_club_tournament_roster(
        club_slug: str,
        tournament_id: str | None = Query(default=None),
        registration_slug: str | None = Query(default=None),
    ) -> dict[str, Any]:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        page = build_public_tournament_roster_page(
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
            request_payload = _dump_model(payload)
            request_payload["_require_demographics"] = True
            result = submit_public_tournament_registration(
                supabase,
                club_id=club_id,
                club_slug=club_slug,
                payload=request_payload,
            )
        except DuplicateTournamentRegistrationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"club": public_club_payload(club, club_slug), **result}

    @app.get("/clubs/{club_slug}/tournament-registration/confirmation")
    def get_club_tournament_registration_confirmation(
        club_slug: str,
        confirmation_token: str = Query(...),
    ) -> dict[str, Any]:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        try:
            confirmation = build_public_tournament_registration_confirmation(
                supabase,
                club_id=club_id,
                confirmation_token=confirmation_token,
            )
        except ValueError:
            confirmation = None
        if not confirmation:
            raise HTTPException(status_code=404, detail="registration confirmation not found")
        return {"club": public_club_payload(club, club_slug), **confirmation}
