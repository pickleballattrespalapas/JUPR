"""Unauthenticated, narrow event check-in; no public access to hub operations."""
from typing import Any
from uuid import UUID

from fastapi import HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from jupr_app.services.round_robin_hub_registration_service import (
    CheckinRateLimitError, public_hub_registration, submit_hub_checkin,
)
from services.api.staging_write_guard import require_public_intake_or_403


class HubCheckinRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=2, max_length=120)
    session_key: str = Field(min_length=1, max_length=160)


def _error(exc: Exception) -> None:
    if isinstance(exc, LookupError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, CheckinRateLimitError):
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "60"}) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise HTTPException(status_code=503, detail="Check-in is temporarily unavailable. Please retry or ask the event organizer.") from exc


def install_public_round_robin_hub_routes(app, *, get_supabase_client) -> None:
    @app.get("/public/round-robin-hubs/{hub_id}/registration")
    def registration(hub_id: UUID, response: Response) -> dict[str, Any]:
        response.headers["Cache-Control"] = "no-store"
        try:
            return public_hub_registration(get_supabase_client(), hub_id=str(hub_id))
        except Exception as exc:
            _error(exc)

    @app.post("/public/round-robin-hubs/{hub_id}/checkins")
    def checkin(hub_id: UUID, payload: HubCheckinRequest, response: Response) -> dict[str, Any]:
        response.headers["Cache-Control"] = "no-store"
        require_public_intake_or_403()
        try:
            return submit_hub_checkin(get_supabase_client(), hub_id=str(hub_id), **payload.model_dump())
        except Exception as exc:
            _error(exc)
