from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel
from supabase import Client

from jupr_app.services.public_support_intake_service import (
    SupportIntakeRateLimitError,
    create_public_support_intake_request,
)
from services.api.staging_write_guard import require_public_intake_or_403


class PublicSupportIntakeRequest(BaseModel):
    request_type: str = "general_support"
    requester_name: str | None = None
    requester_email: str | None = None
    player_name: str | None = None
    player_id: int | str | None = None
    match_id: str | int | None = None
    tournament_id: str | None = None
    subject: str | None = None
    description: str | None = None
    requested_action: str | None = None
    evidence_url: str | None = None
    consent_to_contact: bool = False
    website: str | None = None
    source: str = "next_public_support_intake"


def _dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def install_public_support_intake_routes(
    app,
    *,
    get_club,
    get_supabase_client,
    public_club_payload,
) -> None:
    """Register public support/data-correction/privacy intake routes."""

    @app.post("/clubs/{club_slug}/support/intake")
    def post_club_public_support_intake(
        club_slug: str,
        payload: PublicSupportIntakeRequest,
    ) -> dict[str, Any]:
        require_public_intake_or_403()
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        supabase: Client = get_supabase_client()
        try:
            result = create_public_support_intake_request(
                supabase,
                club_id=club_id,
                club_slug=club_slug,
                payload=_dump_model(payload),
                source=payload.source,
            )
        except SupportIntakeRateLimitError as exc:
            raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "3600"}) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"club": public_club_payload(club, club_slug), **result}
