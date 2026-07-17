from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import PERMISSION_MANAGE_MATCHES, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_challenge_ladder_service import (
    accept_admin_challenge_ladder_challenge,
    build_admin_challenge_ladder_status,
    create_admin_challenge_ladder_challenge,
    get_admin_challenge_ladder_dashboard,
    is_admin_challenge_ladder_enabled,
    preview_admin_challenge_ladder_result_for_challenge,
    record_admin_challenge_ladder_forfeit,
    record_admin_challenge_ladder_result,
    start_admin_challenge_ladder_clock,
    update_admin_challenge_ladder_challenge,
)
from services.api.auth import authenticate_bearer, auth_header


class ChallengeUpdateRequest(BaseModel):
    status: str
    admin_note: str | None = None
    confirmation_text: str = ""
    source: str = "next_challenge_ladder_admin"


class ChallengeCreateRequest(BaseModel):
    challenger_id: int
    defender_id: int
    tier_id: str
    ledger_ref: str | None = None
    override: bool = False
    start_clock: bool = False
    confirmation_text: str = ""
    source: str = "next_challenge_ladder_admin_create"


class ChallengeSimpleConfirmationRequest(BaseModel):
    confirmation_text: str = ""
    source: str = "next_challenge_ladder_admin"


class ChallengeForfeitRequest(BaseModel):
    forfeited_by_id: int
    admin_note: str | None = None
    confirmation_text: str = ""
    source: str = "next_challenge_ladder_forfeit"


class ChallengeResultRequest(BaseModel):
    partner_a_challenger_id: int
    partner_a_defender_id: int
    partner_b_challenger_id: int
    partner_b_defender_id: int
    match_a_games: list[list[int]] = Field(default_factory=list)
    match_b_games: list[list[int]] = Field(default_factory=list)
    match_date: str = ""
    winner_override: str = "computed"
    publish_official_matches: bool = True
    confirmation_text: str = ""
    source: str = "next_challenge_ladder_result"


class ChallengeResultPreviewRequest(BaseModel):
    partner_a_challenger_id: int
    partner_a_defender_id: int
    partner_b_challenger_id: int
    partner_b_defender_id: int
    match_a_games: list[list[int]] = Field(default_factory=list)
    match_b_games: list[list[int]] = Field(default_factory=list)
    match_date: str = ""
    winner_override: str = "computed"
    publish_official_matches: bool = True
    source: str = "next_challenge_ladder_result_preview"


def _resolve_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(supabase=supabase, club_id=str(club_id), email=user.email, user_id=user.user_id, allowlist=set())
    if not has_permission(role_resolution.role, PERMISSION_MANAGE_MATCHES):
        denied = build_activity_payload(
            club_id=str(club_id), actor_email=user.email, actor_role=role_resolution.role, action_type="admin_challenge_ladder_denied", entity_type="challenge_ladder", entity_id="challenge_ladder",
            after_json={"source_client": "fastapi/nextjs", "reason": "insufficient_permission"}, source_page=source, flagged_for_review=True,
        )
        write_admin_activity_log(supabase, denied)
        raise HTTPException(status_code=403, detail="insufficient permission")
    return user.email, role_resolution.role


def _handle(exc: Exception) -> None:
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise exc


def install_admin_challenge_ladder_routes(app, *, get_supabase_client) -> None:
    """Register guarded Challenge Ladder Admin routes."""

    @app.get("/admin/clubs/{club_id}/challenge-ladder/status")
    def get_admin_challenge_ladder_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_challenge_ladder_enabled() else None
        return build_admin_challenge_ladder_status(supabase, club_id=str(club_id))

    @app.get("/admin/clubs/{club_id}/challenge-ladder/dashboard")
    def get_admin_challenge_ladder_dashboard_route(club_id: str, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_challenge_ladder_admin_dashboard")
        try:
            return get_admin_challenge_ladder_dashboard(supabase, club_id=str(club_id))
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/challenge-ladder/challenges")
    def post_admin_challenge_ladder_challenge(club_id: str, payload: ChallengeCreateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return create_admin_challenge_ladder_challenge(
                supabase,
                club_id=str(club_id),
                challenger_id=payload.challenger_id,
                defender_id=payload.defender_id,
                tier_id=payload.tier_id,
                ledger_ref=payload.ledger_ref,
                override=payload.override,
                start_clock=payload.start_clock,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/start-clock")
    def post_admin_challenge_ladder_start_clock(club_id: str, challenge_id: int, payload: ChallengeSimpleConfirmationRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return start_admin_challenge_ladder_clock(supabase, club_id=str(club_id), challenge_id=int(challenge_id), actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source)
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/accept")
    def post_admin_challenge_ladder_accept(club_id: str, challenge_id: int, payload: ChallengeSimpleConfirmationRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return accept_admin_challenge_ladder_challenge(supabase, club_id=str(club_id), challenge_id=int(challenge_id), actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source)
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/forfeit")
    def post_admin_challenge_ladder_forfeit(club_id: str, challenge_id: int, payload: ChallengeForfeitRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return record_admin_challenge_ladder_forfeit(supabase, club_id=str(club_id), challenge_id=int(challenge_id), forfeited_by_id=payload.forfeited_by_id, admin_note=payload.admin_note, actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source)
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/result")
    def post_admin_challenge_ladder_result(club_id: str, challenge_id: int, payload: ChallengeResultRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return record_admin_challenge_ladder_result(
                supabase,
                club_id=str(club_id),
                challenge_id=int(challenge_id),
                partner_a_challenger_id=payload.partner_a_challenger_id,
                partner_a_defender_id=payload.partner_a_defender_id,
                partner_b_challenger_id=payload.partner_b_challenger_id,
                partner_b_defender_id=payload.partner_b_defender_id,
                match_a_games=payload.match_a_games,
                match_b_games=payload.match_b_games,
                match_date=payload.match_date,
                winner_override=payload.winner_override,
                publish_official_matches=payload.publish_official_matches,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}/result/preview")
    def post_admin_challenge_ladder_result_preview(club_id: str, challenge_id: int, payload: ChallengeResultPreviewRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return preview_admin_challenge_ladder_result_for_challenge(
                supabase,
                club_id=str(club_id),
                challenge_id=int(challenge_id),
                partner_a_challenger_id=payload.partner_a_challenger_id,
                partner_a_defender_id=payload.partner_a_defender_id,
                partner_b_challenger_id=payload.partner_b_challenger_id,
                partner_b_defender_id=payload.partner_b_defender_id,
                match_a_games=payload.match_a_games,
                match_b_games=payload.match_b_games,
                match_date=payload.match_date,
                winner_override=payload.winner_override,
                publish_official_matches=payload.publish_official_matches,
            )
        except Exception as exc:
            _handle(exc)

    @app.patch("/admin/clubs/{club_id}/challenge-ladder/challenges/{challenge_id}")
    def patch_admin_challenge_ladder_challenge(club_id: str, challenge_id: int, payload: ChallengeUpdateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_challenge_ladder_enabled():
            raise HTTPException(status_code=403, detail="Next Challenge Ladder Admin is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return update_admin_challenge_ladder_challenge(
                supabase,
                club_id=str(club_id),
                challenge_id=int(challenge_id),
                status=payload.status,
                admin_note=payload.admin_note,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)
