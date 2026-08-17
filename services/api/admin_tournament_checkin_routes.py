from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import PERMISSION_MANAGE_TOURNAMENTS
from jupr_app.services.admin_tournament_checkin_service import (
    StaleTournamentCheckInError,
    TournamentCheckInIdempotencyConflictError,
    build_admin_tournament_checkin_snapshot,
    update_admin_tournament_checkin,
)
from jupr_app.services.admin_tournament_service import is_admin_tournament_admin_enabled
from services.api.admin_tournament_routes import (
    _handle,
    _resolve_tournament_role_or_403,
)
from services.api.auth import auth_header


class AdminTournamentCheckInUpdateRequest(BaseModel):
    expected_updated_at: str | None = Field(default=None, max_length=120)
    attendance_status: Literal["EXPECTED", "CHECKED_IN", "ABSENT"]
    operation_key: UUID
    waiver_verified: bool = False
    approved_substitute_player_id: int | None = Field(default=None, ge=1)
    approved_substitute_name: str | None = Field(default=None, max_length=160)
    notes: str | None = Field(default=None, max_length=1000)


def install_admin_tournament_checkin_routes(app, *, get_supabase_client) -> None:
    """Install the authenticated, server-authoritative event-day workspace."""

    @app.get("/admin/clubs/{club_id}/tournament-live/tournaments/{tournament_id}/check-in")
    def get_admin_tournament_check_in(
        club_id: str,
        tournament_id: str,
        day_id: str | None = Query(default=None, min_length=1, max_length=160),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_tournament_check_in",
            required_permissions=(PERMISSION_MANAGE_TOURNAMENTS,),
            require_all=True,
        )
        try:
            return build_admin_tournament_checkin_snapshot(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                registration_day_id=day_id,
            )
        except Exception as exc:
            _handle(exc)

    @app.put(
        "/admin/clubs/{club_id}/tournament-live/tournaments/{tournament_id}/check-in/{registration_id}"
    )
    def put_admin_tournament_check_in(
        club_id: str,
        tournament_id: str,
        registration_id: str,
        payload: AdminTournamentCheckInUpdateRequest,
        day_id: str = Query(min_length=1, max_length=160),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_tournament_check_in",
            required_permissions=(PERMISSION_MANAGE_TOURNAMENTS,),
            require_all=True,
        )
        try:
            if (
                payload.approved_substitute_player_id is None
                and str(payload.approved_substitute_name or "").strip()
            ):
                raise ValueError(
                    "Select an active club player as the approved substitute; a typed name is not authoritative."
                )
            return update_admin_tournament_checkin(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                registration_id=str(registration_id),
                registration_day_id=day_id,
                expected_updated_at=payload.expected_updated_at,
                attendance_status=payload.attendance_status,
                operation_key=str(payload.operation_key),
                waiver_verified=payload.waiver_verified,
                approved_substitute_player_id=payload.approved_substitute_player_id,
                approved_substitute_name=payload.approved_substitute_name,
                notes=payload.notes,
                actor_email=actor_email,
                actor_role=actor_role,
            )
        except StaleTournamentCheckInError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except TournamentCheckInIdempotencyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            _handle(exc)


__all__ = ["install_admin_tournament_checkin_routes"]
