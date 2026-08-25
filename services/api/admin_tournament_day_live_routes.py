from __future__ import annotations

from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from jupr_app.domain.admin.roles import (
    PERMISSION_ENTER_SCORES,
    PERMISSION_MANAGE_TOURNAMENTS,
)
from jupr_app.services.admin_tournament_day_live_service import (
    build_admin_tournament_day_live_snapshot,
    execute_admin_tournament_day_live_command,
    reconcile_admin_tournament_day_live_operation,
)
from jupr_app.services.admin_tournament_service import is_admin_tournament_admin_enabled
from services.api.admin_tournament_routes import _handle, _resolve_tournament_role_or_403
from services.api.auth import auth_header


class AdminTournamentDayLiveExpected(BaseModel):
    day_run_version: str | int
    state_fingerprint: str = Field(min_length=64, max_length=64)
    draw_version: str | int | None = None
    game_version: str | None = Field(default=None, max_length=120)
    court_version: str | int | None = None
    queue_version: str | int | None = None
    queue_entry_version: str | int | None = None


class AdminTournamentDayLivePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draw_id: str | None = Field(default=None, max_length=160)
    advance_count: int | None = Field(default=None, ge=4, le=6)
    game_id: str | None = Field(default=None, max_length=160)
    score_a: int | None = None
    score_b: int | None = None
    unusual_score_acknowledgement: bool | None = None
    result_type: Literal["FORFEIT", "NO_SHOW", "RETIREMENT"] | None = None
    winner_team_id: str | None = Field(default=None, max_length=160)
    result_note: str | None = Field(default=None, max_length=500)


class AdminTournamentDayLiveCommandRequest(BaseModel):
    action: Literal[
        "activate_day",
        "activate_draw",
        "pause_draw",
        "resume_draw",
        "auto_fill_courts",
        "score_and_release",
        "correct_completed_score",
        "record_non_played_result",
        "generate_playoffs",
        "close_day",
    ]
    client_idempotency_key: str = Field(min_length=32, max_length=64)
    confirmation_text: str = Field(min_length=1, max_length=120)
    expected: AdminTournamentDayLiveExpected
    payload: AdminTournamentDayLivePayload


class AdminTournamentDayLiveReconcileRequest(BaseModel):
    confirmation_text: str = Field(min_length=1, max_length=120)


def _dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def install_admin_tournament_day_live_routes(app, *, get_supabase_client) -> None:
    """Register the service-role-only tournament-day workspace API."""

    @app.get(
        "/admin/clubs/{club_id}/tournament-live/tournaments/{tournament_id}/days/{day_id}/snapshot"
    )
    def get_admin_tournament_day_live_snapshot(
        club_id: str,
        tournament_id: str,
        day_id: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_tournament_day_live_snapshot",
            required_permissions=(PERMISSION_MANAGE_TOURNAMENTS, PERMISSION_ENTER_SCORES),
            require_all=False,
        )
        try:
            return build_admin_tournament_day_live_snapshot(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                registration_day_id=day_id,
            )
        except ValueError as exc:
            if "not found" in str(exc).lower():
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            _handle(exc)
        except Exception as exc:
            _handle(exc)

    @app.post(
        "/admin/clubs/{club_id}/tournament-live/tournaments/{tournament_id}/days/{day_id}/commands"
    )
    def post_admin_tournament_day_live_command(
        club_id: str,
        tournament_id: str,
        day_id: str,
        payload: AdminTournamentDayLiveCommandRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        action = str(payload.action)
        if action in {
            "score_and_release",
            "correct_completed_score",
            "record_non_played_result",
        }:
            required_permissions = (PERMISSION_ENTER_SCORES,)
        else:
            required_permissions = (PERMISSION_MANAGE_TOURNAMENTS,)
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_tournament_day_live_command",
            required_permissions=required_permissions,
            require_all=True,
        )
        try:
            return execute_admin_tournament_day_live_command(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                registration_day_id=day_id,
                request=_dump_model(payload),
                actor_email=actor_email,
                actor_role=actor_role,
            )
        except Exception as exc:
            _handle(exc)

    @app.post(
        "/admin/clubs/{club_id}/tournament-live/tournaments/{tournament_id}/days/{day_id}/operations/{operation_key}/reconcile"
    )
    def post_admin_tournament_day_live_reconcile(
        club_id: str,
        tournament_id: str,
        day_id: str,
        operation_key: str,
        payload: AdminTournamentDayLiveReconcileRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_tournament_day_live_reconcile",
            required_permissions=(PERMISSION_MANAGE_TOURNAMENTS, PERMISSION_ENTER_SCORES),
            require_all=False,
        )
        try:
            return reconcile_admin_tournament_day_live_operation(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                registration_day_id=day_id,
                operation_key=str(operation_key),
                confirmation_text=payload.confirmation_text,
                actor_email=actor_email,
                actor_role=actor_role,
            )
        except Exception as exc:
            _handle(exc)


__all__ = ["install_admin_tournament_day_live_routes"]
