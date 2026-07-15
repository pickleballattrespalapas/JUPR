from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import PERMISSION_MANAGE_TOURNAMENTS, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_tournament_bulk_service import bulk_update_admin_tournament_registrations
from jupr_app.services.admin_tournament_delete_service import delete_admin_tournament_draft
from jupr_app.services.admin_tournament_draw_service import create_admin_tournament_draw
from jupr_app.services.admin_tournament_game_service import generate_admin_tournament_round_robin_games
from jupr_app.services.admin_tournament_ops_service import get_admin_tournament_ops_snapshot
from jupr_app.services.admin_tournament_service import (
    build_admin_tournament_status,
    get_admin_tournament_detail,
    is_admin_tournament_admin_enabled,
    list_admin_tournaments,
    update_admin_tournament_registration,
    update_admin_tournament_selection,
)
from jupr_app.services.admin_tournament_status_service import apply_admin_tournament_status_action
from jupr_app.services.admin_tournament_team_service import replace_admin_tournament_draw_teams
from services.api.auth import authenticate_bearer, auth_header


class AdminTournamentRegistrationUpdateRequest(BaseModel):
    registration_status: str | None = None
    payment_status: str | None = None
    notes: str | None = None
    confirmation_text: str = ""
    source: str = "next_tournament_admin_registration_update"


class AdminTournamentRegistrationBulkUpdateRequest(BaseModel):
    registration_ids: list[str] = Field(default_factory=list)
    registration_status: str | None = None
    payment_status: str | None = None
    append_note: str | None = None
    confirmation_text: str = ""
    source: str = "next_tournament_admin_registration_bulk_update"


class AdminTournamentStatusActionRequest(BaseModel):
    action: str
    confirmation_text: str = ""
    source: str = "next_tournament_admin_status_action"


class AdminTournamentDeleteDraftRequest(BaseModel):
    confirmation_text: str = ""
    source: str = "next_tournament_admin_delete_draft"


class AdminTournamentDrawCreateRequest(BaseModel):
    registration_day_id: str | None = None
    event_option_id: str | None = None
    name: str | None = None
    confirmation_text: str = ""
    source: str = "next_tournament_admin_create_draw"


class AdminTournamentDrawTeamRow(BaseModel):
    team_number: int
    player1_id: int
    player2_id: int | None = None
    seed: int | None = None
    source: str | None = "MANUAL"
    notes: str | None = None


class AdminTournamentDrawTeamsReplaceRequest(BaseModel):
    teams: list[AdminTournamentDrawTeamRow] = Field(default_factory=list)
    confirmation_text: str = ""
    source: str = "next_tournament_admin_replace_teams"


class AdminTournamentRoundRobinGenerateRequest(BaseModel):
    confirmation_text: str = ""
    source: str = "next_tournament_admin_generate_round_robin"


class AdminTournamentSelectionUpdateRequest(BaseModel):
    event_option_id: str | None = None
    partner_mode: str | None = None
    partner_name: str | None = None
    partner_email: str | None = None
    partner_phone: str | None = None
    partner_note: str | None = None
    confirmation_text: str = ""
    source: str = "next_tournament_admin_selection_update"


def _dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def _resolve_tournament_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(
        supabase=supabase,
        club_id=str(club_id),
        email=user.email,
        user_id=user.user_id,
        allowlist=set(),
    )
    if not has_permission(role_resolution.role, PERMISSION_MANAGE_TOURNAMENTS):
        denied_payload = build_activity_payload(
            club_id=str(club_id),
            actor_email=user.email,
            actor_role=role_resolution.role,
            action_type="admin_tournament_denied",
            entity_type="tournament_admin",
            entity_id="tournament_admin",
            after_json={"source_client": "fastapi/nextjs", "reason": "insufficient_permission"},
            source_page=source,
            flagged_for_review=True,
        )
        write_admin_activity_log(supabase, denied_payload)
        raise HTTPException(status_code=403, detail="insufficient permission")
    return user.email, role_resolution.role


def install_admin_tournament_routes(app, *, get_supabase_client) -> None:
    """Register guarded Tournament Admin routes for the Next admin pilot."""

    @app.get("/admin/clubs/{club_id}/tournaments/admin/status")
    def get_admin_tournament_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_tournament_admin_enabled() else None
        return build_admin_tournament_status(supabase, club_id=str(club_id))

    @app.get("/admin/clubs/{club_id}/tournaments/admin/tournaments")
    def get_admin_tournaments(
        club_id: str,
        include_archived: bool = Query(default=False),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_tournament_admin_list",
        )
        try:
            return list_admin_tournaments(supabase, club_id=str(club_id), include_archived=bool(include_archived))
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/ops")
    def get_admin_tournament_ops(
        club_id: str,
        tournament_id: str,
        draw_id: str | None = Query(default=None),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_tournament_admin_ops",
        )
        try:
            return get_admin_tournament_ops_snapshot(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                draw_id=draw_id,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws")
    def post_admin_tournament_draw(
        club_id: str,
        tournament_id: str,
        payload: AdminTournamentDrawCreateRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return create_admin_tournament_draw(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                registration_day_id=payload.registration_day_id,
                event_option_id=payload.event_option_id,
                name=payload.name,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.put("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/teams")
    def put_admin_tournament_draw_teams(
        club_id: str,
        tournament_id: str,
        draw_id: str,
        payload: AdminTournamentDrawTeamsReplaceRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            teams = [_dump_model(team) for team in payload.teams]
            return replace_admin_tournament_draw_teams(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                draw_id=str(draw_id),
                teams=teams,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/games/round-robin")
    def post_admin_tournament_round_robin_games(
        club_id: str,
        tournament_id: str,
        draw_id: str,
        payload: AdminTournamentRoundRobinGenerateRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return generate_admin_tournament_round_robin_games(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                draw_id=str(draw_id),
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.patch("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/status-action")
    def patch_admin_tournament_status_action(
        club_id: str,
        tournament_id: str,
        payload: AdminTournamentStatusActionRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return apply_admin_tournament_status_action(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                action=payload.action,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/delete-draft")
    def post_admin_tournament_delete_draft(
        club_id: str,
        tournament_id: str,
        payload: AdminTournamentDeleteDraftRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return delete_admin_tournament_draft(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}")
    def get_admin_tournament(
        club_id: str,
        tournament_id: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_tournament_admin_detail",
        )
        try:
            return get_admin_tournament_detail(supabase, club_id=str(club_id), tournament_id=str(tournament_id))
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.patch("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/registrations/bulk")
    def patch_admin_tournament_registrations_bulk(
        club_id: str,
        tournament_id: str,
        payload: AdminTournamentRegistrationBulkUpdateRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        patch = _dump_model(payload)
        source = str(patch.pop("source", payload.source))
        confirmation_text = str(patch.pop("confirmation_text", payload.confirmation_text))
        registration_ids = list(patch.pop("registration_ids", payload.registration_ids) or [])
        try:
            return bulk_update_admin_tournament_registrations(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                registration_ids=registration_ids,
                patch=patch,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=confirmation_text,
                source=source,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.patch("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/registrations/{registration_id}")
    def patch_admin_tournament_registration(
        club_id: str,
        tournament_id: str,
        registration_id: str,
        payload: AdminTournamentRegistrationUpdateRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        patch = _dump_model(payload)
        source = str(patch.pop("source", payload.source))
        confirmation_text = str(patch.pop("confirmation_text", payload.confirmation_text))
        try:
            return update_admin_tournament_registration(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                registration_id=str(registration_id),
                patch=patch,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=confirmation_text,
                source=source,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.patch("/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/selections/{selection_id}")
    def patch_admin_tournament_selection(
        club_id: str,
        tournament_id: str,
        selection_id: str,
        payload: AdminTournamentSelectionUpdateRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_admin_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_tournament_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        patch = _dump_model(payload)
        source = str(patch.pop("source", payload.source))
        confirmation_text = str(patch.pop("confirmation_text", payload.confirmation_text))
        try:
            return update_admin_tournament_selection(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                selection_id=str(selection_id),
                patch=patch,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=confirmation_text,
                source=source,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
