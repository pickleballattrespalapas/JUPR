from __future__ import annotations

import os
from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import PERMISSION_MANAGE_MATCHES, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_league_awards_service import (
    archive_admin_league_awards,
    close_admin_league_and_award,
    freeze_admin_league_awards,
    get_admin_league_awards_wizard,
    mint_admin_league_awards,
    persist_admin_league_awards_preview,
    preview_admin_league_awards,
    require_admin_league_awards_write,
    save_admin_league_awards_config,
    save_admin_league_award_overrides,
)
from jupr_app.services.admin_league_live_service import (
    LeagueLiveConflictError,
    LeagueLivePersistenceError,
    build_admin_league_live_round_plan,
    build_admin_league_live_status,
    create_admin_league_live_session,
    get_admin_league_live_session,
    is_admin_league_live_submit_enabled,
    list_admin_league_live_sessions,
    save_admin_league_live_round,
    suggest_admin_league_live_roster,
    update_admin_league_live_session_snapshot,
)
from jupr_app.services.admin_league_live_submit_service import (
    build_admin_league_live_export,
    compensate_admin_league_live_round_publish,
    create_admin_league_live_guest,
    list_admin_league_live_publish_operations,
    reconcile_admin_league_live_round_publish,
    submit_admin_league_live_round_publish,
)
from jupr_app.services.admin_league_manager_create_service import (
    create_admin_league_manager_draft,
    duplicate_admin_league_manager_draft,
)
from jupr_app.services.admin_league_print_service import (
    build_admin_league_printout,
    build_admin_top_players_printable,
)
from jupr_app.services.admin_league_manager_lifecycle_service import transition_admin_league_manager_lifecycle
from jupr_app.services.admin_league_manager_roster_service import update_admin_league_manager_roster_membership
from jupr_app.services.admin_league_manager_roster_batch_service import (
    update_admin_league_manager_roster_batch,
)
from jupr_app.services.admin_league_manager_service import (
    build_admin_league_manager_status,
    build_admin_league_schedule_preview,
    get_admin_league_manager_detail,
    is_admin_league_manager_enabled,
    list_admin_league_manager_leagues,
)
from jupr_app.services.admin_league_manager_update_service import (
    normalize_admin_league_schedule_config,
    update_admin_league_manager_settings,
)
from services.api.auth import authenticate_bearer, auth_header
from services.api.staging_write_guard import require_league_manager_write_or_403


class AdminLeagueManagerSettingsUpdateRequest(BaseModel):
    description: str | None = Field(default=None, max_length=2000)
    status: str | None = None
    k_factor: int | None = None
    min_games: int | None = None
    schedule_config: dict[str, Any] | None = None
    court_board_defaults: dict[str, Any] | None = None
    rules_config: dict[str, Any] | None = None
    awards_config: dict[str, Any] | None = None
    event_tags: dict[str, Any] | None = None
    confirmation_text: str = ""
    source: str = "next_league_manager_settings_update"


class AdminLeagueManagerCreateRequest(BaseModel):
    league_name: str = Field(min_length=1, max_length=120)
    match_format: str = Field(default="doubles", pattern=r"^(doubles|singles)$")
    description: str = Field(default="", max_length=2000)
    min_games: int = Field(default=6, ge=0, le=1000)
    k_factor: int = Field(default=32, ge=1, le=128)
    confirmation_text: str = ""
    source: str = "next_league_manager_create"


class AdminLeagueManagerDuplicateRequest(BaseModel):
    target_league_name: str = Field(min_length=1, max_length=120)
    confirmation_text: str = ""
    source: str = "next_league_manager_duplicate"


class AdminLeagueManagerLifecycleRequest(BaseModel):
    action: str = Field(min_length=1, max_length=40)
    confirmation_text: str = ""
    source: str = "next_league_manager_lifecycle"


class AdminLeagueManagerSchedulePreviewRequest(BaseModel):
    schedule_config: dict[str, Any]


class AdminLeagueManagerRosterMembershipRequest(BaseModel):
    action: str
    starting_rating: float | None = None
    confirmation_text: str = ""
    source: str = "next_league_manager_roster_update"


class AdminLeagueManagerRosterBatchRequest(BaseModel):
    action: str = Field(pattern=r"^(activate|deactivate)$")
    player_ids: list[int] = Field(min_length=1, max_length=500)
    starting_rating: float | None = None
    idempotency_key: str = Field(min_length=8, max_length=160)
    confirmation_text: str = ""
    source: str = "next_league_manager_bulk_roster_editor"


class AdminLeagueAwardsCloseRequest(BaseModel):
    award_badges: bool = True
    confirmation_text: str = ""
    idempotency_key: str = Field(default="", max_length=160)
    source: str = "next_league_manager_awards_close"


class AdminLeagueAwardsActionRequest(BaseModel):
    idempotency_key: str = Field(min_length=8, max_length=160)
    confirmation_text: str = Field(default="", max_length=80)
    source: str = Field(default="next_league_manager_awards_action", max_length=120)


class AdminLeagueAwardsConfigRequest(BaseModel):
    awards_config: dict[str, Any]
    expected_config_version: int = Field(ge=0)
    source: str = Field(
        default="next_league_manager_awards_config", max_length=120
    )


class AdminLeagueAwardOverrideItem(BaseModel):
    award_key: str | None = Field(default=None, min_length=3, max_length=240)
    category_key: str = Field(min_length=1, max_length=80)
    rank: int = Field(default=1, ge=1, le=3)
    player_id: int
    reason: str = Field(default="", max_length=500)


class AdminLeagueAwardOverridesRequest(BaseModel):
    idempotency_key: str = Field(min_length=8, max_length=160)
    preview_fingerprint: str = Field(min_length=64, max_length=64)
    overrides: list[AdminLeagueAwardOverrideItem] = Field(default_factory=list, max_length=60)
    source: str = Field(default="next_league_manager_awards_overrides", max_length=120)


class AdminLeagueLiveSessionCreateRequest(BaseModel):
    league_name: str
    week_tag: str = "Week 1"
    total_rounds: int = Field(default=1, ge=1, le=50)
    current_round: int = Field(default=1, ge=1, le=50)
    roster: list[dict[str, Any]] = Field(default_factory=list)
    courts: list[dict[str, Any]] = Field(default_factory=list)
    bench_player_ids: list[int] = Field(default_factory=list)
    bench_override_reason: str | None = Field(default=None, max_length=500)
    notes: str | None = None
    confirmation_text: str = ""
    source: str = "next_league_live_session_create"


class AdminLeagueLiveSessionSnapshotRequest(BaseModel):
    status: str | None = None
    week_tag: str | None = None
    total_rounds: int | None = Field(default=None, ge=1, le=50)
    current_round: int | None = Field(default=None, ge=1, le=50)
    roster: list[dict[str, Any]] | None = None
    courts: list[dict[str, Any]] | None = None
    bench_player_ids: list[int] | None = None
    bench_override_reason: str | None = Field(default=None, max_length=500)
    notes: str | None = None
    expected_updated_at: str = Field(min_length=1, max_length=120)
    confirmation_text: str = ""
    source: str = "next_league_live_session_snapshot"


class AdminLeagueLiveRoundSaveRequest(BaseModel):
    round_label: str | None = None
    match_date: str | None = None
    preview: dict[str, Any] | None = None
    matches: list[dict[str, Any]] = Field(default_factory=list)
    movement_overrides: list[dict[str, Any]] = Field(default_factory=list)
    override_reason: str | None = Field(default=None, max_length=500)
    roster_change: dict[str, Any] | None = None
    bench_player_ids: list[int] = Field(default_factory=list)
    bench_override_reason: str | None = Field(default=None, max_length=500)
    expected_updated_at: str = Field(min_length=1, max_length=120)
    expected_operation_key: str = Field(min_length=32, max_length=128)
    submitted_match_count: int | None = None
    submitted_match_ids: list[Any] = Field(default_factory=list)
    courts: list[dict[str, Any]] = Field(default_factory=list)
    advance_after_save: bool = True
    confirmation_text: str = ""
    source: str = "next_league_live_round_save"


class AdminLeagueLiveRosterSuggestionRequest(BaseModel):
    roster: list[dict[str, Any]] = Field(default_factory=list)
    court_sizes: list[int] = Field(default_factory=list)
    prefer_keep_player_ids: list[int] = Field(default_factory=list)
    bench_player_ids: list[int] = Field(default_factory=list)
    bench_override_reason: str | None = Field(default=None, max_length=500)
    round_number: int = Field(default=1, ge=1, le=50)


class AdminLeagueLiveRoundPlanRequest(BaseModel):
    expected_updated_at: str = Field(min_length=1, max_length=120)
    matches: list[dict[str, Any]] = Field(default_factory=list)
    courts: list[dict[str, Any]] = Field(default_factory=list)
    movement_overrides: list[dict[str, Any]] = Field(default_factory=list)
    override_reason: str | None = Field(default=None, max_length=500)
    roster_change: dict[str, Any] | None = None
    bench_player_ids: list[int] = Field(default_factory=list)
    bench_override_reason: str | None = Field(default=None, max_length=500)


class AdminLeagueLiveRoundPublishRequest(BaseModel):
    round_label: str | None = Field(default=None, max_length=80)
    match_date: str = Field(min_length=1, max_length=40)
    preview: dict[str, Any] | None = None
    matches: list[dict[str, Any]] = Field(default_factory=list, min_length=1, max_length=200)
    expected_match_count: int = Field(ge=1, le=200)
    movement_overrides: list[dict[str, Any]] = Field(default_factory=list)
    override_reason: str | None = Field(default=None, max_length=500)
    roster_change: dict[str, Any] | None = None
    bench_player_ids: list[int] = Field(default_factory=list)
    bench_override_reason: str | None = Field(default=None, max_length=500)
    expected_updated_at: str = Field(min_length=1, max_length=120)
    expected_operation_key: str = Field(min_length=64, max_length=64)
    idempotency_key: str = Field(min_length=8, max_length=160)
    courts: list[dict[str, Any]] = Field(default_factory=list)
    confirmation_text: str = Field(default="", max_length=80)
    source: str = Field(default="next_league_live_round_submit", max_length=120)


class AdminLeagueLiveRoundReconcileRequest(BaseModel):
    confirmation_text: str = Field(default="", max_length=80)
    source: str = Field(default="next_league_live_round_reconcile", max_length=120)


class AdminLeagueLiveRoundCompensateRequest(BaseModel):
    recovery_reference: str = Field(min_length=8, max_length=240)
    reason: str = Field(min_length=10, max_length=500)
    confirmation_text: str = Field(default="", max_length=80)
    source: str = Field(default="next_league_live_round_compensate", max_length=120)


class AdminLeagueLiveGuestRequest(BaseModel):
    guest_name: str = Field(min_length=2, max_length=160)
    starting_jupr: float = Field(ge=1.0, le=7.0)
    reason: str = Field(min_length=10, max_length=500)
    expected_updated_at: str = Field(min_length=1, max_length=120)
    idempotency_key: str = Field(min_length=8, max_length=160)
    confirmation_text: str = Field(default="", max_length=80)
    source: str = Field(default="next_league_live_guest_create", max_length=120)


def _dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def _resolve_league_manager_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(
        supabase=supabase,
        club_id=str(club_id),
        email=user.email,
        user_id=user.user_id,
        allowlist=set(),
    )
    if not has_permission(role_resolution.role, PERMISSION_MANAGE_MATCHES):
        denied_payload = build_activity_payload(
            club_id=str(club_id),
            actor_email=user.email,
            actor_role=role_resolution.role,
            action_type="admin_league_manager_denied",
            entity_type="league_manager",
            entity_id="league_manager",
            after_json={"source_client": "fastapi/nextjs", "reason": "insufficient_permission"},
            source_page=source,
            flagged_for_review=True,
        )
        write_admin_activity_log(supabase, denied_payload)
        raise HTTPException(status_code=403, detail="insufficient permission")
    return user.email, role_resolution.role


def _require_league_live_service_role_or_503() -> None:
    if not os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail="League Live requires SUPABASE_SERVICE_ROLE_KEY on FastAPI; browser and anonymous keys are not accepted.",
        )


def _handle_common(exc: Exception) -> None:
    if isinstance(exc, LeagueLiveConflictError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, LeagueLivePersistenceError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise exc


def _require_league_manager_service_role() -> None:
    if not os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail="League Manager mutations require SUPABASE_SERVICE_ROLE_KEY on FastAPI.",
        )


def _require_league_awards_write_or_403() -> None:
    try:
        require_admin_league_awards_write()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def install_admin_league_manager_routes(app, *, get_supabase_client) -> None:
    """Register guarded League Manager routes for the Next admin pilot."""

    @app.get("/admin/clubs/{club_id}/league-manager/status")
    def get_admin_league_manager_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_league_manager_enabled() else None
        return build_admin_league_manager_status(supabase, club_id=str(club_id))

    @app.get("/admin/clubs/{club_id}/league-manager/live/status")
    def get_admin_league_live_status(club_id: str) -> dict[str, Any]:
        service_role_configured = bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip())
        payload = build_admin_league_live_status(None, club_id=str(club_id))
        payload["service_role_configured"] = service_role_configured
        if payload.get("enabled") and not service_role_configured:
            payload.update(
                {
                    "enabled": False,
                    "status": "service_role_required",
                    "sessions_endpoint": None,
                    "roster_suggestion_endpoint": None,
                    "round_plan_endpoint": None,
                    "submit_enabled": False,
                    "round_submit_endpoint": None,
                    "round_reconcile_endpoint": None,
                    "round_compensate_endpoint": None,
                    "guest_endpoint": None,
                    "export_endpoint": None,
                }
            )
            payload.setdefault("warnings", []).append("SUPABASE_SERVICE_ROLE_KEY is required on FastAPI for League Live.")
        return payload

    @app.post("/admin/clubs/{club_id}/league-manager/live/roster-suggestion")
    def post_admin_league_live_roster_suggestion(
        club_id: str,
        payload: AdminLeagueLiveRosterSuggestionRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_league_live_service_role_or_503()
        supabase = get_supabase_client()
        _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_league_live_roster_suggestion",
        )
        try:
            return suggest_admin_league_live_roster(
                supabase,
                club_id=str(club_id),
                roster=payload.roster,
                court_sizes=payload.court_sizes,
                prefer_keep_player_ids=payload.prefer_keep_player_ids,
                bench_player_ids=payload.bench_player_ids,
                bench_override_reason=payload.bench_override_reason,
                round_number=payload.round_number,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.get("/admin/clubs/{club_id}/league-manager/live-sessions")
    def get_admin_league_live_sessions(
        club_id: str,
        status: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_league_live_service_role_or_503()
        supabase = get_supabase_client()
        _resolve_league_manager_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_league_live_sessions_list")
        try:
            return list_admin_league_live_sessions(supabase, club_id=str(club_id), status=status, limit=limit)
        except Exception as exc:
            _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/league-manager/live-sessions")
    def post_admin_league_live_session(
        club_id: str,
        payload: AdminLeagueLiveSessionCreateRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_league_live_service_role_or_503()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return create_admin_league_live_session(
                supabase,
                club_id=str(club_id),
                league_name=payload.league_name,
                week_tag=payload.week_tag,
                total_rounds=payload.total_rounds,
                current_round=payload.current_round,
                roster=payload.roster,
                courts=payload.courts,
                bench_player_ids=payload.bench_player_ids,
                bench_override_reason=payload.bench_override_reason,
                notes=payload.notes,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.get("/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}")
    def get_admin_league_live_session_detail(
        club_id: str,
        session_id: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_league_live_service_role_or_503()
        supabase = get_supabase_client()
        _resolve_league_manager_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_league_live_session_detail")
        try:
            detail = get_admin_league_live_session(supabase, club_id=str(club_id), session_id=str(session_id))
            detail["publish_operations"] = (
                list_admin_league_live_publish_operations(
                    supabase,
                    club_id=str(club_id),
                    session_id=str(session_id),
                )
                if is_admin_league_live_submit_enabled()
                else []
            )
            return detail
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            _handle_common(exc)

    @app.patch("/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/snapshot")
    def patch_admin_league_live_session_snapshot(
        club_id: str,
        session_id: str,
        payload: AdminLeagueLiveSessionSnapshotRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_league_live_service_role_or_503()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        patch = _dump_model(payload)
        confirmation_text = str(patch.pop("confirmation_text", payload.confirmation_text))
        source = str(patch.pop("source", payload.source))
        try:
            return update_admin_league_live_session_snapshot(
                supabase,
                club_id=str(club_id),
                session_id=str(session_id),
                patch=patch,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=confirmation_text,
                source=source,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}/plan")
    def post_admin_league_live_round_plan(
        club_id: str,
        session_id: str,
        round_number: int,
        payload: AdminLeagueLiveRoundPlanRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_league_live_service_role_or_503()
        supabase = get_supabase_client()
        _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_league_live_round_plan",
        )
        try:
            return build_admin_league_live_round_plan(
                supabase,
                club_id=str(club_id),
                session_id=str(session_id),
                round_number=int(round_number),
                expected_updated_at=payload.expected_updated_at,
                matches=payload.matches,
                courts=payload.courts,
                movement_overrides=payload.movement_overrides,
                override_reason=payload.override_reason,
                roster_change=payload.roster_change,
                bench_player_ids=payload.bench_player_ids,
                bench_override_reason=payload.bench_override_reason,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}")
    def post_admin_league_live_round(
        club_id: str,
        session_id: str,
        round_number: int,
        payload: AdminLeagueLiveRoundSaveRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_league_live_service_role_or_503()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return save_admin_league_live_round(
                supabase,
                club_id=str(club_id),
                session_id=str(session_id),
                round_number=int(round_number),
                round_label=payload.round_label,
                match_date=payload.match_date,
                preview=payload.preview,
                matches=payload.matches,
                movement_overrides=payload.movement_overrides,
                override_reason=payload.override_reason,
                roster_change=payload.roster_change,
                bench_player_ids=payload.bench_player_ids,
                bench_override_reason=payload.bench_override_reason,
                expected_updated_at=payload.expected_updated_at,
                expected_operation_key=payload.expected_operation_key,
                submitted_match_count=payload.submitted_match_count,
                submitted_match_ids=payload.submitted_match_ids,
                courts=payload.courts,
                advance_after_save=payload.advance_after_save,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}/submit")
    def post_admin_league_live_round_publish(
        club_id: str,
        session_id: str,
        round_number: int,
        payload: AdminLeagueLiveRoundPublishRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_league_live_service_role_or_503()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return submit_admin_league_live_round_publish(
                supabase,
                club_id=str(club_id),
                session_id=str(session_id),
                round_number=int(round_number),
                round_label=payload.round_label,
                match_date=payload.match_date,
                preview=payload.preview,
                matches=payload.matches,
                expected_match_count=payload.expected_match_count,
                movement_overrides=payload.movement_overrides,
                override_reason=payload.override_reason,
                roster_change=payload.roster_change,
                bench_player_ids=payload.bench_player_ids,
                bench_override_reason=payload.bench_override_reason,
                expected_updated_at=payload.expected_updated_at,
                expected_operation_key=payload.expected_operation_key,
                idempotency_key=payload.idempotency_key,
                courts=payload.courts,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}/reconcile")
    def post_admin_league_live_round_reconcile(
        club_id: str,
        session_id: str,
        round_number: int,
        payload: AdminLeagueLiveRoundReconcileRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_league_live_service_role_or_503()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return reconcile_admin_league_live_round_publish(
                supabase,
                club_id=str(club_id),
                session_id=str(session_id),
                round_number=int(round_number),
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}/compensate")
    def post_admin_league_live_round_compensate(
        club_id: str,
        session_id: str,
        round_number: int,
        payload: AdminLeagueLiveRoundCompensateRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_league_live_service_role_or_503()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return compensate_admin_league_live_round_publish(
                supabase,
                club_id=str(club_id),
                session_id=str(session_id),
                round_number=int(round_number),
                recovery_reference=payload.recovery_reference,
                reason=payload.reason,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/guests")
    def post_admin_league_live_guest(
        club_id: str,
        session_id: str,
        payload: AdminLeagueLiveGuestRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_league_live_service_role_or_503()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return create_admin_league_live_guest(
                supabase,
                club_id=str(club_id),
                session_id=str(session_id),
                guest_name=payload.guest_name,
                starting_jupr=payload.starting_jupr,
                reason=payload.reason,
                expected_updated_at=payload.expected_updated_at,
                idempotency_key=payload.idempotency_key,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.get("/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/export")
    def get_admin_league_live_export(
        club_id: str,
        session_id: str,
        kind: str = Query(default="matches"),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_league_live_service_role_or_503()
        supabase = get_supabase_client()
        _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_league_live_export",
        )
        try:
            return build_admin_league_live_export(
                supabase,
                club_id=str(club_id),
                session_id=str(session_id),
                export_kind=kind,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.get("/admin/clubs/{club_id}/league-manager/leagues")
    def get_admin_league_manager_leagues(
        club_id: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        supabase = get_supabase_client()
        _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_league_manager_list",
        )
        try:
            return list_admin_league_manager_leagues(supabase, club_id=str(club_id))
        except Exception as exc:
            _handle_common(exc)

    @app.get("/admin/clubs/{club_id}/league-manager/top-players-printable")
    def get_admin_top_players_printable(
        club_id: str,
        limit: int = Query(default=50, ge=1, le=200),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        supabase = get_supabase_client()
        _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_top_players_printable",
        )
        try:
            return build_admin_top_players_printable(
                supabase,
                club_id=str(club_id),
                limit=int(limit),
            )
        except Exception as exc:
            _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/league-manager/leagues")
    def post_admin_league_manager_league(
        club_id: str,
        payload: AdminLeagueManagerCreateRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        require_league_manager_write_or_403()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        _require_league_manager_service_role()
        try:
            return create_admin_league_manager_draft(
                supabase,
                club_id=str(club_id),
                league_name=payload.league_name,
                description=payload.description,
                min_games=payload.min_games,
                k_factor=payload.k_factor,
                match_format=payload.match_format,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/league-manager/leagues/{league_name}/duplicate")
    def post_admin_league_manager_league_duplicate(
        club_id: str,
        league_name: str,
        payload: AdminLeagueManagerDuplicateRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        require_league_manager_write_or_403()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        _require_league_manager_service_role()
        try:
            return duplicate_admin_league_manager_draft(
                supabase,
                club_id=str(club_id),
                source_league_name=str(league_name),
                target_league_name=payload.target_league_name,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/league-manager/leagues/{league_name}/lifecycle")
    def post_admin_league_manager_league_lifecycle(
        club_id: str,
        league_name: str,
        payload: AdminLeagueManagerLifecycleRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        require_league_manager_write_or_403()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        _require_league_manager_service_role()
        try:
            return transition_admin_league_manager_lifecycle(
                supabase,
                club_id=str(club_id),
                league_name=str(league_name),
                action=payload.action,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.get("/admin/clubs/{club_id}/league-manager/leagues/{league_name}")
    def get_admin_league_manager_league_detail(
        club_id: str,
        league_name: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        supabase = get_supabase_client()
        _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_league_manager_detail",
        )
        try:
            return get_admin_league_manager_detail(supabase, club_id=str(club_id), league_name=str(league_name))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/league-manager/leagues/{league_name}/schedule/preview")
    def post_admin_league_manager_schedule_preview(
        club_id: str,
        league_name: str,
        payload: AdminLeagueManagerSchedulePreviewRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        supabase = get_supabase_client()
        _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_league_manager_schedule_preview",
        )
        try:
            schedule_config = normalize_admin_league_schedule_config(payload.schedule_config)
            return build_admin_league_schedule_preview(schedule_config, league_name=str(league_name))
        except Exception as exc:
            _handle_common(exc)

    @app.get("/admin/clubs/{club_id}/league-manager/leagues/{league_name}/printout")
    def get_admin_league_manager_printout(
        club_id: str,
        league_name: str,
        week_num: int | None = Query(default=None, ge=1, le=1000),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        supabase = get_supabase_client()
        _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_league_manager_printout",
        )
        try:
            return build_admin_league_printout(
                supabase,
                club_id=str(club_id),
                league_name=str(league_name),
                week_num=week_num,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.get("/admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards")
    def get_admin_league_awards_wizard_state(
        club_id: str,
        league_name: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        supabase = get_supabase_client()
        _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_league_manager_awards_wizard",
        )
        try:
            return get_admin_league_awards_wizard(supabase, club_id=str(club_id), league_name=str(league_name))
        except Exception as exc:
            _handle_common(exc)

    @app.get("/admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards/preview")
    def get_admin_league_awards_preview(
        club_id: str,
        league_name: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        supabase = get_supabase_client()
        _resolve_league_manager_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_league_manager_awards_preview")
        try:
            return preview_admin_league_awards(supabase, club_id=str(club_id), league_name=str(league_name))
        except Exception as exc:
            _handle_common(exc)

    @app.put("/admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards/config")
    def put_admin_league_awards_config(
        club_id: str,
        league_name: str,
        payload: AdminLeagueAwardsConfigRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_league_awards_write_or_403()
        require_league_manager_write_or_403()
        if not is_admin_league_manager_enabled():
            raise HTTPException(
                status_code=403, detail="Next League Manager is disabled."
            )
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return save_admin_league_awards_config(
                supabase,
                club_id=str(club_id),
                league_name=str(league_name),
                awards_config=payload.awards_config,
                expected_config_version=payload.expected_config_version,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards/freeze")
    def post_admin_league_awards_freeze(
        club_id: str,
        league_name: str,
        payload: AdminLeagueAwardsActionRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_league_awards_write_or_403()
        require_league_manager_write_or_403()
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return freeze_admin_league_awards(
                supabase,
                club_id=str(club_id),
                league_name=str(league_name),
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                idempotency_key=payload.idempotency_key,
                source=payload.source,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards/preview")
    def post_admin_league_awards_preview(
        club_id: str,
        league_name: str,
        payload: AdminLeagueAwardsActionRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_league_awards_write_or_403()
        require_league_manager_write_or_403()
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return persist_admin_league_awards_preview(
                supabase,
                club_id=str(club_id),
                league_name=str(league_name),
                actor_email=actor_email,
                actor_role=actor_role,
                idempotency_key=payload.idempotency_key,
                source=payload.source,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards/overrides")
    def post_admin_league_awards_overrides(
        club_id: str,
        league_name: str,
        payload: AdminLeagueAwardOverridesRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_league_awards_write_or_403()
        require_league_manager_write_or_403()
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return save_admin_league_award_overrides(
                supabase,
                club_id=str(club_id),
                league_name=str(league_name),
                overrides=[_dump_model(item) for item in payload.overrides],
                preview_fingerprint=payload.preview_fingerprint,
                actor_email=actor_email,
                actor_role=actor_role,
                idempotency_key=payload.idempotency_key,
                source=payload.source,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards/mint")
    def post_admin_league_awards_mint(
        club_id: str,
        league_name: str,
        payload: AdminLeagueAwardsActionRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_league_awards_write_or_403()
        require_league_manager_write_or_403()
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return mint_admin_league_awards(
                supabase,
                club_id=str(club_id),
                league_name=str(league_name),
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                idempotency_key=payload.idempotency_key,
                source=payload.source,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards/archive")
    def post_admin_league_awards_archive(
        club_id: str,
        league_name: str,
        payload: AdminLeagueAwardsActionRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_league_awards_write_or_403()
        require_league_manager_write_or_403()
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return archive_admin_league_awards(
                supabase,
                club_id=str(club_id),
                league_name=str(league_name),
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                idempotency_key=payload.idempotency_key,
                source=payload.source,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/league-manager/leagues/{league_name}/awards/close")
    def post_admin_league_awards_close(
        club_id: str,
        league_name: str,
        payload: AdminLeagueAwardsCloseRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_league_awards_write_or_403()
        require_league_manager_write_or_403()
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return close_admin_league_and_award(
                supabase,
                club_id=str(club_id),
                league_name=str(league_name),
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                award_badges=payload.award_badges,
                source=payload.source,
                idempotency_key=payload.idempotency_key,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.patch("/admin/clubs/{club_id}/league-manager/leagues/{league_name}")
    def patch_admin_league_manager_settings(
        club_id: str,
        league_name: str,
        payload: AdminLeagueManagerSettingsUpdateRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        require_league_manager_write_or_403()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        _require_league_manager_service_role()
        patch = _dump_model(payload)
        source = str(patch.pop("source", payload.source))
        confirmation_text = str(patch.pop("confirmation_text", payload.confirmation_text))
        try:
            return update_admin_league_manager_settings(
                supabase,
                club_id=str(club_id),
                league_name=str(league_name),
                patch=patch,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=confirmation_text,
                source=source,
            )
        except Exception as exc:
            _handle_common(exc)

    @app.post("/admin/clubs/{club_id}/league-manager/leagues/{league_name}/roster/batch")
    def post_admin_league_manager_roster_batch(
        club_id: str,
        league_name: str,
        payload: AdminLeagueManagerRosterBatchRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_league_manager_enabled():
            raise HTTPException(
                status_code=403, detail="Next League Manager is disabled."
            )
        require_league_manager_write_or_403()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        _require_league_manager_service_role()
        try:
            result = update_admin_league_manager_roster_batch(
                supabase,
                club_id=str(club_id),
                league_name=str(league_name),
                action=payload.action,
                player_ids=payload.player_ids,
                starting_rating=payload.starting_rating,
                idempotency_key=payload.idempotency_key,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
            result["detail"] = get_admin_league_manager_detail(
                supabase,
                club_id=str(club_id),
                league_name=str(league_name),
            )
            return result
        except Exception as exc:
            _handle_common(exc)

    @app.patch("/admin/clubs/{club_id}/league-manager/leagues/{league_name}/roster/{player_id}")
    def patch_admin_league_manager_roster_membership(
        club_id: str,
        league_name: str,
        player_id: int,
        payload: AdminLeagueManagerRosterMembershipRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_league_manager_enabled():
            raise HTTPException(status_code=403, detail="Next League Manager is disabled.")
        require_league_manager_write_or_403()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_league_manager_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        _require_league_manager_service_role()
        try:
            return update_admin_league_manager_roster_membership(
                supabase,
                club_id=str(club_id),
                league_name=str(league_name),
                player_id=player_id,
                action=payload.action,
                starting_rating=payload.starting_rating,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle_common(exc)
