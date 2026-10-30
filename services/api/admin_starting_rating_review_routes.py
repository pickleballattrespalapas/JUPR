"""Authenticated staff review for starting estimates before rated play."""
from __future__ import annotations

from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import PERMISSION_MANAGE_PLAYERS, has_permission, resolve_admin_role
from jupr_app.services.starting_rating_review_service import (
    StartingRatingReviewConflict, confirm_starting_ratings, get_starting_rating_review,
)
from services.api.auth import authenticate_bearer, auth_header
from services.api.staging_write_guard import require_league_manager_write_or_403


class StartingRatingReviewPreviewRequest(BaseModel):
    player_ids: list[int] = Field(default_factory=list, max_length=500)
    match_format: Literal["doubles", "singles"] = "doubles"
    requested_ratings: dict[str, str | None] = Field(default_factory=dict)


class StartingRatingReviewConfirmation(BaseModel):
    player_id: int = Field(gt=0)
    rating_jupr: float = Field(ge=1, le=7, allow_inf_nan=False)
    expected_state_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_rating: str | None = Field(default=None, max_length=120)


class StartingRatingReviewConfirmRequest(BaseModel):
    match_format: Literal["doubles", "singles"] = "doubles"
    reviews: list[StartingRatingReviewConfirmation] = Field(min_length=1, max_length=500)


def install_admin_starting_rating_review_routes(app, *, get_supabase_client) -> None:
    def authorize(db, club_id, authorization):
        user = authenticate_bearer(authorization)
        role = resolve_admin_role(supabase=db, club_id=club_id, email=user.email,
                                  user_id=user.user_id, allowlist=set()).role
        if not has_permission(role, PERMISSION_MANAGE_PLAYERS):
            raise HTTPException(status_code=403, detail="Player-management permission is required.")
        return user.email, role

    @app.post("/admin/clubs/{club_id}/starting-rating-review/preview")
    def preview(club_id: str, payload: StartingRatingReviewPreviewRequest, authorization: str | None = auth_header()):
        db = get_supabase_client()
        authorize(db, club_id, authorization)
        try:
            return get_starting_rating_review(db, club_id=club_id, player_ids=payload.player_ids,
                match_format=payload.match_format, requested_ratings=payload.requested_ratings)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/admin/clubs/{club_id}/starting-rating-review/confirm")
    def confirm(club_id: str, payload: StartingRatingReviewConfirmRequest, authorization: str | None = auth_header()):
        db = get_supabase_client()
        email, role = authorize(db, club_id, authorization)
        require_league_manager_write_or_403()
        try:
            return confirm_starting_ratings(db, club_id=club_id, match_format=payload.match_format,
                reviews=[row.model_dump() for row in payload.reviews], actor_email=email, actor_role=role)
        except StartingRatingReviewConflict as exc:
            raise HTTPException(status_code=409, detail={"code": "STARTING_RATING_REVIEW_CONFLICT", "message": str(exc)}) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
