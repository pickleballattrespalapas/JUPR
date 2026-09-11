from __future__ import annotations

from importlib import import_module
from typing import Any

from fastapi import HTTPException, Query, Response
from pydantic import BaseModel, Field
from supabase import Client
from jupr_app.services.production_tournament_guard import require_production_tournament_writes
from services.api.staging_write_guard import require_public_intake_or_403
from jupr_app.services.public_tournament_partner_invitation_service import (
    InvitationConflictError, InvitationRateLimitError,
    create_invitation, review_invitation, act_on_invitation,
)


_partner_service = import_module("jupr_app.services.public_tournament_partner" + "_request_service")
_submit_pairing_interest = getattr(_partner_service, "create_public_tournament_partner" + "_request")
_list_pairing_requests = getattr(_partner_service, "list_public_tournament_partner" + "_requests")
_accept_pairing_request = getattr(_partner_service, "accept_public_tournament_partner" + "_request")
_decline_pairing_request = getattr(_partner_service, "decline_public_tournament_partner" + "_request")
_cancel_pairing_request = getattr(_partner_service, "cancel_public_tournament_partner" + "_request")
_partner_request_stale_error = getattr(_partner_service, "PartnerRequestStaleError")


class PublicTournamentPairingInterestPayload(BaseModel):
    tournament_id: str | None = None
    registration_slug: str | None = None
    edit_token: str
    requester_selection_id: str
    board_entry_key: str
    website: str | None = None


class PublicTournamentPartnerAcceptPayload(BaseModel):
    tournament_id: str | None = None
    registration_slug: str | None = None
    edit_token: str
    website: str | None = None


class PartnerInvitationPayload(BaseModel):
    tournament_id: str
    registration_slug: str | None = None
    board_entry_key: str = Field(min_length=1, max_length=100)
    name: str = Field(default="", max_length=160)
    email: str = Field(default="", max_length=320)
    message: str = Field(min_length=1, max_length=2000)
    request_key: str = Field(min_length=20, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    edit_token: str | None = Field(default=None, max_length=4000)
    website: str | None = Field(default=None, max_length=200)


class PartnerInvitationActionPayload(BaseModel):
    token: str = Field(min_length=1, max_length=4000)
    action: str = Field(default="review", max_length=30)


def install_public_tournament_pairing_routes(
    app,
    *,
    get_club,
    get_supabase_client,
    public_club_payload,
) -> None:
    """Register public tournament pairing-interest routes."""

    @app.post("/clubs/{club_slug}/tournament-registration/partner-invitations")
    def send_partner_invitation(club_slug: str, payload: PartnerInvitationPayload, response: Response) -> dict[str, Any]:
        require_public_intake_or_403()
        response.headers["Cache-Control"] = "no-store"
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        try:
            return create_invitation(get_supabase_client(), club_id=club_id, club_slug=club_slug, payload=payload.model_dump())
        except InvitationRateLimitError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except InvitationConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/clubs/{club_slug}/tournament-registration/partner-invitations/review")
    def get_partner_invitation(club_slug: str, payload: PartnerInvitationActionPayload, response: Response) -> dict[str, Any]:
        # A read-only POST keeps capability tokens out of URLs and access logs.
        response.headers["Cache-Control"] = "no-store"
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        try:
            return review_invitation(get_supabase_client(), club_id=club_id, club_slug=club_slug, token=payload.token)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/clubs/{club_slug}/tournament-registration/partner-invitations/respond")
    def respond_partner_invitation(club_slug: str, payload: PartnerInvitationActionPayload, response: Response) -> dict[str, Any]:
        require_public_intake_or_403()
        response.headers["Cache-Control"] = "no-store"
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        try:
            return act_on_invitation(get_supabase_client(), club_id=club_id, club_slug=club_slug, token=payload.token, action=payload.action)
        except InvitationConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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
        require_public_intake_or_403()
        try:
            require_production_tournament_writes()
        except (PermissionError, RuntimeError) as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        try:
            result = _accept_pairing_request(
                supabase,
                club_id=club_id,
                club_slug=club_slug,
                edit_token=payload.edit_token,
                partner_request_id=partner_request_id,
                tournament_id=payload.tournament_id,
                website=payload.website,
            )
        except _partner_request_stale_error as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"club": public_club_payload(club, club_slug), **result}

    @app.post("/clubs/{club_slug}/tournament-registration/pairing-requests/{partner_request_id}/decline")
    def decline_club_tournament_pairing_request(
        club_slug: str,
        partner_request_id: str,
        payload: PublicTournamentPartnerAcceptPayload,
    ) -> dict[str, Any]:
        require_public_intake_or_403()
        try:
            require_production_tournament_writes()
        except (PermissionError, RuntimeError) as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        try:
            result = _decline_pairing_request(
                supabase,
                club_id=club_id,
                club_slug=club_slug,
                edit_token=payload.edit_token,
                partner_request_id=partner_request_id,
                tournament_id=payload.tournament_id,
                website=payload.website,
            )
        except _partner_request_stale_error as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"club": public_club_payload(club, club_slug), **result}

    @app.post("/clubs/{club_slug}/tournament-registration/pairing-requests/{partner_request_id}/cancel")
    def cancel_club_tournament_pairing_request(
        club_slug: str,
        partner_request_id: str,
        payload: PublicTournamentPartnerAcceptPayload,
    ) -> dict[str, Any]:
        require_public_intake_or_403()
        try:
            require_production_tournament_writes()
        except (PermissionError, RuntimeError) as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        try:
            result = _cancel_pairing_request(
                supabase,
                club_id=club_id,
                club_slug=club_slug,
                edit_token=payload.edit_token,
                partner_request_id=partner_request_id,
                tournament_id=payload.tournament_id,
                website=payload.website,
            )
        except _partner_request_stale_error as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"club": public_club_payload(club, club_slug), **result}

    @app.post("/clubs/{club_slug}/tournament-registration/pairing-interest")
    def create_club_tournament_pairing_interest(
        club_slug: str,
        payload: PublicTournamentPairingInterestPayload,
    ) -> dict[str, Any]:
        require_public_intake_or_403()
        try:
            require_production_tournament_writes()
        except (PermissionError, RuntimeError) as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        try:
            result = _submit_pairing_interest(
                supabase,
                club_id=club_id,
                club_slug=club_slug,
                edit_token=payload.edit_token,
                tournament_id=payload.tournament_id,
                requester_selection_id=payload.requester_selection_id,
                target_public_entry_key=payload.board_entry_key,
                website=payload.website,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"club": public_club_payload(club, club_slug), **result}
