from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import (
    PERMISSION_ENTER_SCORES,
    PERMISSION_MANAGE_MATCHES,
    PERMISSION_MANAGE_TOURNAMENTS,
)
from jupr_app.services.admin_tournament_live_service import (
    build_admin_tournament_live_snapshot,
    build_admin_tournament_live_status,
    execute_admin_tournament_live_command,
    reconcile_admin_tournament_live_operation,
)
from jupr_app.services.admin_tournament_service import is_admin_tournament_admin_enabled
from services.api.admin_tournament_routes import _handle, _resolve_tournament_role_or_403
from services.api.auth import auth_header


class AdminTournamentLiveRowVersion(BaseModel):
    id: str = Field(min_length=1, max_length=160)
    updated_at: str = Field(min_length=1, max_length=120)


class AdminTournamentLiveCommandRequest(BaseModel):
    command: str = Field(min_length=1, max_length=80)
    expected_state_fingerprint: str = Field(min_length=64, max_length=64)
    idempotency_key: str = Field(min_length=32, max_length=64)
    confirmation_text: str = Field(default="", max_length=120)
    expected_draw_updated_at: str = Field(min_length=1, max_length=120)
    expected_game_updated_at: str | None = Field(default=None, max_length=120)
    expected_team_versions: list[AdminTournamentLiveRowVersion] | None = None
    expected_source_game_versions: list[AdminTournamentLiveRowVersion] | None = None
    game_id: str | None = Field(default=None, max_length=160)
    score_a: int | None = None
    score_b: int | None = None
    advance_count: int | None = None
    playoff_winner_bonus_elo: float | None = None


class AdminTournamentLiveReconcileRequest(BaseModel):
    confirmation_text: str = Field(default="", max_length=120)


def _dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def install_admin_tournament_live_routes(app, *, get_supabase_client) -> None:
    """Register the draw-scoped in-play runner separately from JUPR Live."""

    @app.get("/admin/clubs/{club_id}/tournament-live/status")
    def get_admin_tournament_live_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_tournament_admin_enabled() else None
        return build_admin_tournament_live_status(supabase, club_id=str(club_id))

    @app.get("/admin/clubs/{club_id}/tournament-live/tournaments/{tournament_id}/snapshot")
    def get_admin_tournament_live_snapshot(
        club_id: str,
        tournament_id: str,
        draw_id: str | None = Query(default=None, max_length=160),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_tournament_live_snapshot",
            required_permissions=(PERMISSION_MANAGE_TOURNAMENTS, PERMISSION_ENTER_SCORES),
            require_all=False,
        )
        try:
            return build_admin_tournament_live_snapshot(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                draw_id=str(draw_id) if draw_id else None,
            )
        except ValueError as exc:
            if "not found" in str(exc).lower():
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            _handle(exc)
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournament-live/tournaments/{tournament_id}/draws/{draw_id}/commands")
    def post_admin_tournament_live_command(
        club_id: str,
        tournament_id: str,
        draw_id: str,
        payload: AdminTournamentLiveCommandRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        command = str(payload.command or "").strip().lower()
        if command == "save_score":
            required_permissions = (PERMISSION_ENTER_SCORES,)
            require_all = True
        elif command == "publish_official_matches":
            required_permissions = (PERMISSION_MANAGE_TOURNAMENTS, PERMISSION_MANAGE_MATCHES)
            require_all = True
        else:
            required_permissions = (PERMISSION_MANAGE_TOURNAMENTS,)
            require_all = True
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_tournament_live_command",
            required_permissions=required_permissions,
            require_all=require_all,
        )
        try:
            return execute_admin_tournament_live_command(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                draw_id=str(draw_id),
                request=_dump_model(payload),
                actor_email=actor_email,
                actor_role=actor_role,
            )
        except Exception as exc:
            _handle(exc)

    @app.post(
        "/admin/clubs/{club_id}/tournament-live/tournaments/{tournament_id}/draws/{draw_id}/operations/{operation_key}/reconcile"
    )
    def post_admin_tournament_live_reconcile(
        club_id: str,
        tournament_id: str,
        draw_id: str,
        operation_key: str,
        payload: AdminTournamentLiveReconcileRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_tournament_live_reconcile",
            required_permissions=(PERMISSION_MANAGE_TOURNAMENTS, PERMISSION_ENTER_SCORES),
            require_all=False,
        )
        try:
            return reconcile_admin_tournament_live_operation(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                draw_id=str(draw_id),
                operation_key=str(operation_key),
                confirmation_text=payload.confirmation_text,
                actor_email=actor_email,
                actor_role=actor_role,
            )
        except Exception as exc:
            _handle(exc)


__all__ = ["install_admin_tournament_live_routes"]
