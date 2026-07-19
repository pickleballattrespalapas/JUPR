from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import PERMISSION_ENTER_SCORES, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_match_uploader_service import (
    build_admin_match_uploader_round_robin_preview,
    build_admin_match_uploader_status,
    create_admin_match_uploader_players,
    is_admin_match_uploader_enabled,
    submit_admin_match_uploader_batch,
)
from jupr_app.services.admin_player_updates_service import auto_send_player_updates_for_match_payloads
from jupr_app.services.admin_singles_match_service import submit_admin_singles_match
from services.api.auth import authenticate_bearer, auth_header


class AdminMatchUploaderBatchRequest(BaseModel):
    matches: list[dict[str, Any]] = Field(default_factory=list)
    source: str = "next_match_uploader"


class AdminMatchUploaderSinglesRequest(BaseModel):
    date: str | None = None
    league: str = "Singles"
    week_tag: str = "Singles"
    t1_p1: int
    t2_p1: int
    score_t1: int
    score_t2: int
    rating_scope: str | None = None
    source: str = "next_match_uploader_singles"


class AdminMatchUploaderRoundRobinCourtRequest(BaseModel):
    court: int | None = None
    format_type: str
    player_names: list[str] = Field(default_factory=list)
    names: str | None = None
    players_text: str | None = None


class AdminMatchUploaderRoundRobinPreviewRequest(BaseModel):
    courts: list[AdminMatchUploaderRoundRobinCourtRequest] = Field(default_factory=list)
    custom_schedule: str = ""
    schedule_mode: str = "full"
    source: str = "next_match_uploader_round_robin_preview"


class AdminMatchUploaderNewPlayerRequest(BaseModel):
    name: str
    starting_jupr: float = Field(default=3.5, ge=1.0, le=7.0)


class AdminMatchUploaderCreatePlayersRequest(BaseModel):
    players: list[AdminMatchUploaderNewPlayerRequest] = Field(default_factory=list)
    source: str = "next_match_uploader_new_players"


def _dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _resolve_score_entry_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(
        supabase=supabase,
        club_id=str(club_id),
        email=user.email,
        user_id=user.user_id,
        allowlist=set(),
    )
    if not has_permission(role_resolution.role, PERMISSION_ENTER_SCORES):
        denied_payload = build_activity_payload(
            club_id=str(club_id),
            actor_email=user.email,
            actor_role=role_resolution.role,
            action_type="submit_match_uploader_batch_denied",
            entity_type="matches",
            entity_id="batch",
            after_json={"source_client": "fastapi/nextjs", "reason": "insufficient_permission"},
            source_page=source,
            flagged_for_review=True,
        )
        write_admin_activity_log(supabase, denied_payload)
        raise HTTPException(status_code=403, detail="insufficient permission")
    return user.email, role_resolution.role


def _handle_write_error(exc: Exception) -> None:
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise exc


def install_admin_match_uploader_routes(app, *, get_supabase_client) -> None:
    """Register guarded Match Uploader routes for the Next admin pilot."""

    @app.get("/admin/clubs/{club_id}/match-uploader/status")
    def get_admin_match_uploader_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_match_uploader_enabled() else None
        status = build_admin_match_uploader_status(supabase, club_id=str(club_id))
        status["singles_submit_endpoint"] = "/admin/clubs/{club_id}/match-uploader/singles"
        return status

    @app.post("/admin/clubs/{club_id}/match-uploader/round-robin/preview")
    def post_admin_match_uploader_round_robin_preview(
        club_id: str,
        payload: AdminMatchUploaderRoundRobinPreviewRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_uploader_enabled():
            raise HTTPException(status_code=403, detail="Next Match Uploader is disabled.")
        supabase = get_supabase_client()
        _resolve_score_entry_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return build_admin_match_uploader_round_robin_preview(
                supabase,
                club_id=str(club_id),
                courts=[_dump_model(court) for court in payload.courts],
                custom_schedule=payload.custom_schedule,
                schedule_mode=payload.schedule_mode,
                source=payload.source,
            )
        except Exception as exc:
            _handle_write_error(exc)

    @app.post("/admin/clubs/{club_id}/match-uploader/players")
    def post_admin_match_uploader_players(
        club_id: str,
        payload: AdminMatchUploaderCreatePlayersRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_uploader_enabled():
            raise HTTPException(status_code=403, detail="Next Match Uploader is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_score_entry_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return create_admin_match_uploader_players(
                supabase,
                club_id=str(club_id),
                players=[_dump_model(player) for player in payload.players],
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
            )
        except Exception as exc:
            _handle_write_error(exc)

    @app.post("/admin/clubs/{club_id}/match-uploader/singles")
    def post_admin_match_uploader_singles(
        club_id: str,
        payload: AdminMatchUploaderSinglesRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_uploader_enabled():
            raise HTTPException(status_code=403, detail="Next Match Uploader is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_score_entry_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            result = submit_admin_singles_match(
                supabase,
                club_id=str(club_id),
                match=_dump_model(payload),
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
            )
            result["match_write_committed"] = True
            result["recovery"] = {
                "match_log_route": "/admin/match-log",
                "replay_history_route": "/admin/replay-history",
                "operator_rule": "Verify the new singles row in Match Log before retrying any request whose outcome is unclear.",
            }
            return result
        except Exception as exc:
            _handle_write_error(exc)

    @app.post("/admin/clubs/{club_id}/match-uploader/batch")
    def post_admin_match_uploader_batch(
        club_id: str,
        payload: AdminMatchUploaderBatchRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_match_uploader_enabled():
            raise HTTPException(status_code=403, detail="Next Match Uploader is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_score_entry_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            result = submit_admin_match_uploader_batch(
                supabase,
                club_id=str(club_id),
                matches=payload.matches,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
            )
        except Exception as exc:
            _handle_write_error(exc)

        # The match service has returned a committed domain result. Email is a
        # post-commit handoff and must never turn that success into an ambiguous
        # HTTP 500 that tempts an operator to submit duplicate matches.
        result["match_write_committed"] = True
        result["recovery"] = {
            "match_log_route": "/admin/match-log",
            "player_updates_route": "/admin/player-updates",
            "replay_history_route": "/admin/replay-history",
            "operator_rule": "Never resubmit committed matches solely because player-update email failed. Verify Match Log, then retry email from Player Updates.",
        }
        try:
            result["auto_player_updates"] = auto_send_player_updates_for_match_payloads(
                supabase,
                club_id=str(club_id),
                match_payloads=payload.matches,
                source=payload.source,
            )
        except Exception as exc:
            result["auto_player_updates"] = {
                "mode": "error",
                "reason": "Match rows committed, but the post-batch player-update email handoff failed.",
                "error_code": type(exc).__name__,
                "attempted": 0,
                "sent": 0,
                "skipped": 0,
                "errors": 1,
            }
            warnings = result.setdefault("warnings", [])
            if not isinstance(warnings, list):
                warnings = []
                result["warnings"] = warnings
            warnings.append(
                "Matches are committed. Do not resubmit them; use Player Updates to retry email delivery."
            )
        return result
