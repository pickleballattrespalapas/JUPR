from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import PERMISSION_MANAGE_TOURNAMENTS, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_tournament_setup_service import (
    build_admin_tournament_setup_status,
    get_admin_tournament_setup_detail,
    is_admin_tournament_setup_enabled,
    list_admin_tournament_setup_tournaments,
    preview_admin_tournament_age_split,
    publish_admin_tournament_setup,
    review_admin_tournament_setup_impact,
    save_admin_tournament_setup_draft,
    update_admin_tournament_setup_settings,
)
from jupr_app.services.admin_tournament_shell_create_service import (
    CONFIRM_CREATE,
    create_admin_tournament_shell,
    get_tournament_shell_creation_state_fingerprint,
    reconcile_admin_tournament_shell_creation,
    tournament_shell_absent_state_fingerprint,
)
from jupr_app.services.admin_tournament_guarded_operation import (
    StaleTournamentAdminStateError,
    TournamentAdminRecoveryRequiredError,
    require_tournament_admin_mutation_runtime,
    run_tournament_admin_guarded_operation,
    tournament_admin_guarded_runtime_enabled,
)
from services.api.auth import authenticate_bearer, auth_header


class TournamentSetupSettingsRequest(BaseModel):
    registration_slug: str | None = None
    locale: str | None = "en"
    registration_status: str | None = None
    registration_open_at: str | None = None
    registration_close_at: str | None = None
    waitlist_enabled: bool | None = None
    partner_board_enabled: bool | None = None
    rules_markdown: str | None = None
    refund_policy_markdown: str | None = None
    weather_policy_markdown: str | None = None
    sponsor_markdown: str | None = None
    location_name: str | None = None
    venue_address: str | None = None
    venue_directions: str | None = None
    venue_courts_json: list[dict[str, Any]] | None = None
    timezone: str | None = None
    sponsors_json: list[dict[str, Any]] | None = None
    confirmation_text: str = ""
    expected_state_fingerprint: str | None = None
    source: str = "next_tournament_setup_settings"


class TournamentShellCreateRequest(BaseModel):
    tournament_id: str = Field(min_length=36, max_length=36)
    idempotency_key: str = Field(min_length=36, max_length=64)
    name: str = Field(min_length=1, max_length=180)
    start_date: str | None = Field(default=None, max_length=40)
    end_date: str | None = Field(default=None, max_length=40)
    confirmation_text: str = ""
    source: str = "next_tournament_setup_create_shell"


class TournamentSetupDraftRequest(BaseModel):
    days: list[dict[str, Any]] = Field(default_factory=list)
    event_families: list[dict[str, Any]] = Field(default_factory=list)
    event_options: list[dict[str, Any]] = Field(default_factory=list)
    basics: dict[str, Any] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)
    saved_step: str | None = "next_setup"
    confirmation_text: str = ""
    expected_state_fingerprint: str | None = None
    source: str = "next_tournament_setup_draft"


class TournamentSetupPublishRequest(BaseModel):
    days: list[dict[str, Any]] = Field(default_factory=list)
    event_families: list[dict[str, Any]] = Field(default_factory=list)
    event_options: list[dict[str, Any]] = Field(default_factory=list)
    builder_event_options: list[dict[str, Any]] = Field(default_factory=list)
    basics: dict[str, Any] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)
    confirmation_text: str = ""
    expected_state_fingerprint: str | None = None
    reviewed_impact_fingerprint: str | None = None
    source: str = "next_tournament_setup_publish"


class TournamentSetupImpactRequest(BaseModel):
    days: list[dict[str, Any]] = Field(default_factory=list)
    event_families: list[dict[str, Any]] = Field(default_factory=list)
    event_options: list[dict[str, Any]] = Field(default_factory=list)
    builder_event_options: list[dict[str, Any]] = Field(default_factory=list)
    basics: dict[str, Any] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)
    expected_state_fingerprint: str
    source: str = "next_tournament_setup_impact_review"


class TournamentAgeSplitPreviewRequest(BaseModel):
    event_family: str = Field(min_length=1, max_length=180)
    participant_type: str | None = Field(default=None, max_length=40)
    policy: dict[str, Any] = Field(default_factory=dict)
    event_options: list[dict[str, Any]] = Field(default_factory=list)
    source: str = "next_tournament_age_split_preview"


def _dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def _resolve_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(supabase=supabase, club_id=str(club_id), email=user.email, user_id=user.user_id, allowlist=set())
    if not has_permission(role_resolution.role, PERMISSION_MANAGE_TOURNAMENTS):
        write_admin_activity_log(
            supabase,
            build_activity_payload(
                club_id=str(club_id),
                actor_email=user.email,
                actor_role=role_resolution.role,
                action_type="admin_tournament_setup_denied",
                entity_type="tournament_setup",
                entity_id="tournament_setup",
                after_json={"source_client": "fastapi/nextjs", "reason": "insufficient_permission"},
                source_page=source,
                flagged_for_review=True,
            ),
        )
        raise HTTPException(status_code=403, detail="insufficient permission")
    return user.email, role_resolution.role


def _handle(exc: Exception) -> None:
    if isinstance(exc, (StaleTournamentAdminStateError, TournamentAdminRecoveryRequiredError)):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise exc


def _require_confirmation(actual: str, expected: str) -> None:
    if str(actual or "").strip().upper() != expected:
        raise ValueError(f"Type {expected} to confirm this Tournament Setup mutation.")


def _require_canonical_uuid(value: str, *, field: str) -> str:
    text = str(value or "").strip()
    try:
        parsed = UUID(text)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{field} must be a valid UUID.") from exc
    if str(parsed) != text.lower():
        raise ValueError(f"{field} must use canonical UUID format.")
    return str(parsed)


def install_admin_tournament_setup_routes(app, *, get_supabase_client) -> None:
    """Register guarded Tournament Setup Manager routes for the Next staging surface."""

    @app.get("/admin/clubs/{club_id}/tournaments/setup/status")
    def get_setup_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_tournament_setup_enabled() else None
        return build_admin_tournament_setup_status(supabase, club_id=str(club_id))

    @app.get("/admin/clubs/{club_id}/tournaments/setup/tournaments")
    def get_setup_tournaments(club_id: str, include_archived: bool = Query(default=True), authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_setup_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Setup is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_tournament_setup_list")
        try:
            return list_admin_tournament_setup_tournaments(supabase, club_id=str(club_id), include_archived=bool(include_archived))
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/setup/tournaments")
    def post_create_tournament_shell(
        club_id: str,
        payload: TournamentShellCreateRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_setup_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Setup is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            _require_confirmation(payload.confirmation_text, CONFIRM_CREATE)
            tournament_id = _require_canonical_uuid(
                payload.tournament_id,
                field="tournament_id",
            )
            idempotency_key = _require_canonical_uuid(
                payload.idempotency_key,
                field="idempotency_key",
            )
            mutation_payload = {
                "tournament_id": tournament_id,
                "name": payload.name,
                "start_date": payload.start_date,
                "end_date": payload.end_date,
            }
            preflight = lambda: create_admin_tournament_shell(
                supabase,
                club_id=str(club_id),
                **mutation_payload,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
                dry_run=True,
            )
            require_tournament_admin_mutation_runtime("setup")
            mutate = lambda: create_admin_tournament_shell(
                supabase,
                club_id=str(club_id),
                **mutation_payload,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
            if not tournament_admin_guarded_runtime_enabled("setup"):
                return mutate()
            expected_state = tournament_shell_absent_state_fingerprint(
                club_id=str(club_id),
                tournament_id=tournament_id,
            )
            return run_tournament_admin_guarded_operation(
                supabase,
                club_id=str(club_id),
                surface="setup",
                action="tournament_setup_shell_create",
                entity_type="tournament",
                entity_id=tournament_id,
                lock_scope=tournament_id,
                expected_state=expected_state,
                current_state=lambda: get_tournament_shell_creation_state_fingerprint(
                    supabase,
                    club_id=str(club_id),
                    tournament_id=tournament_id,
                ),
                payload=mutation_payload,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                preflight=preflight,
                reconcile=lambda _operation: reconcile_admin_tournament_shell_creation(
                    supabase,
                    club_id=str(club_id),
                    tournament_id=tournament_id,
                    name=payload.name,
                    start_date=payload.start_date,
                    end_date=payload.end_date,
                ),
                mutate=mutate,
                idempotency_key=idempotency_key,
            )
        except Exception as exc:
            _handle(exc)

    @app.get("/admin/clubs/{club_id}/tournaments/setup/tournaments/{tournament_id}")
    def get_setup_detail(club_id: str, tournament_id: str, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_setup_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Setup is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_tournament_setup_detail")
        try:
            return get_admin_tournament_setup_detail(supabase, club_id=str(club_id), tournament_id=str(tournament_id))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/setup/tournaments/{tournament_id}/age-split-preview")
    def post_age_split_preview(
        club_id: str,
        tournament_id: str,
        payload: TournamentAgeSplitPreviewRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_tournament_setup_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Setup is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return preview_admin_tournament_age_split(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                event_family=payload.event_family,
                participant_type=payload.participant_type,
                policy=payload.policy,
                event_options=payload.event_options,
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/setup/tournaments/{tournament_id}/impact")
    def post_setup_impact(club_id: str, tournament_id: str, payload: TournamentSetupImpactRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_setup_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Setup is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return review_admin_tournament_setup_impact(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                days=payload.days,
                event_options=payload.event_options,
                event_families=payload.event_families,
                builder_event_options=payload.builder_event_options,
                basics=payload.basics,
                settings=payload.settings,
                expected_state_fingerprint=payload.expected_state_fingerprint,
            )
        except Exception as exc:
            _handle(exc)

    @app.patch("/admin/clubs/{club_id}/tournaments/setup/tournaments/{tournament_id}/settings")
    def patch_setup_settings(club_id: str, tournament_id: str, payload: TournamentSetupSettingsRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_setup_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Setup is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        patch = _dump_model(payload)
        source = str(patch.pop("source", payload.source))
        confirmation_text = str(patch.pop("confirmation_text", payload.confirmation_text))
        expected_state = str(patch.pop("expected_state_fingerprint", payload.expected_state_fingerprint) or "")
        try:
            _require_confirmation(confirmation_text, "SAVE SETUP")
            preflight = lambda: update_admin_tournament_setup_settings(supabase, club_id=str(club_id), tournament_id=str(tournament_id), patch=patch, actor_email=actor_email, actor_role=actor_role, confirmation_text=confirmation_text, source=source, dry_run=True)
            require_tournament_admin_mutation_runtime("setup")
            mutate = lambda: update_admin_tournament_setup_settings(supabase, club_id=str(club_id), tournament_id=str(tournament_id), patch=patch, actor_email=actor_email, actor_role=actor_role, confirmation_text=confirmation_text, source=source)
            if not tournament_admin_guarded_runtime_enabled("setup"):
                return mutate()
            return run_tournament_admin_guarded_operation(
                supabase,
                club_id=str(club_id),
                surface="setup",
                action="tournament_setup_settings",
                entity_type="tournament_setup",
                entity_id=str(tournament_id),
                lock_scope=str(tournament_id),
                expected_state=expected_state,
                current_state=lambda: str(get_admin_tournament_setup_detail(supabase, club_id=str(club_id), tournament_id=str(tournament_id)).get("state_fingerprint") or ""),
                payload=patch,
                actor_email=actor_email,
                actor_role=actor_role,
                source=source,
                preflight=preflight,
                mutate=mutate,
            )
        except Exception as exc:
            _handle(exc)

    @app.put("/admin/clubs/{club_id}/tournaments/setup/tournaments/{tournament_id}/draft")
    def put_setup_draft(club_id: str, tournament_id: str, payload: TournamentSetupDraftRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_setup_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Setup is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            _require_confirmation(payload.confirmation_text, "SAVE SETUP DRAFT")
            preflight = lambda: save_admin_tournament_setup_draft(supabase, club_id=str(club_id), tournament_id=str(tournament_id), days=payload.days, event_families=payload.event_families, event_options=payload.event_options, basics=payload.basics, settings=payload.settings, saved_step=payload.saved_step, actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source, dry_run=True)
            require_tournament_admin_mutation_runtime("setup")
            mutate = lambda: save_admin_tournament_setup_draft(supabase, club_id=str(club_id), tournament_id=str(tournament_id), days=payload.days, event_families=payload.event_families, event_options=payload.event_options, basics=payload.basics, settings=payload.settings, saved_step=payload.saved_step, actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, source=payload.source)
            if not tournament_admin_guarded_runtime_enabled("setup"):
                return mutate()
            return run_tournament_admin_guarded_operation(
                supabase,
                club_id=str(club_id),
                surface="setup",
                action="tournament_setup_draft",
                entity_type="tournament_setup",
                entity_id=str(tournament_id),
                lock_scope=str(tournament_id),
                expected_state=str(payload.expected_state_fingerprint or ""),
                current_state=lambda: str(get_admin_tournament_setup_detail(supabase, club_id=str(club_id), tournament_id=str(tournament_id)).get("state_fingerprint") or ""),
                payload={"days": payload.days, "event_families": payload.event_families, "event_options": payload.event_options, "basics": payload.basics, "settings": payload.settings, "saved_step": payload.saved_step},
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                preflight=preflight,
                mutate=mutate,
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/tournaments/setup/tournaments/{tournament_id}/publish")
    def post_setup_publish(club_id: str, tournament_id: str, payload: TournamentSetupPublishRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_tournament_setup_enabled():
            raise HTTPException(status_code=403, detail="Next Tournament Setup is disabled.")
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            _require_confirmation(payload.confirmation_text, "PUBLISH SETUP")
            if tournament_admin_guarded_runtime_enabled("setup") and not str(payload.reviewed_impact_fingerprint or "").strip():
                raise ValueError("Review publish impact before publishing Tournament Setup.")
            preflight = lambda: publish_admin_tournament_setup(supabase, club_id=str(club_id), tournament_id=str(tournament_id), days=payload.days, event_options=payload.event_options, event_families=payload.event_families, builder_event_options=payload.builder_event_options, basics=payload.basics, settings=payload.settings, actor_email=actor_email, actor_role=actor_role, confirmation_text=payload.confirmation_text, expected_state_fingerprint=payload.expected_state_fingerprint, reviewed_impact_fingerprint=payload.reviewed_impact_fingerprint, source=payload.source, dry_run=True)
            require_tournament_admin_mutation_runtime("setup")
            mutate = lambda: publish_admin_tournament_setup(
                supabase,
                club_id=str(club_id),
                tournament_id=str(tournament_id),
                days=payload.days,
                event_options=payload.event_options,
                event_families=payload.event_families,
                builder_event_options=payload.builder_event_options,
                basics=payload.basics,
                settings=payload.settings,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                expected_state_fingerprint=payload.expected_state_fingerprint,
                reviewed_impact_fingerprint=payload.reviewed_impact_fingerprint,
                source=payload.source,
            )
            if not tournament_admin_guarded_runtime_enabled("setup"):
                return mutate()
            return run_tournament_admin_guarded_operation(
                supabase,
                club_id=str(club_id),
                surface="setup",
                action="tournament_setup_publish",
                entity_type="tournament_setup",
                entity_id=str(tournament_id),
                lock_scope=str(tournament_id),
                expected_state=str(payload.expected_state_fingerprint or ""),
                current_state=lambda: str(get_admin_tournament_setup_detail(supabase, club_id=str(club_id), tournament_id=str(tournament_id)).get("state_fingerprint") or ""),
                payload={
                    "days": payload.days,
                    "event_families": payload.event_families,
                    "event_options": payload.event_options,
                    "builder_event_options": payload.builder_event_options,
                    "basics": payload.basics,
                    "settings": payload.settings,
                    "reviewed_impact_fingerprint": payload.reviewed_impact_fingerprint,
                    "activate_draft_on_publish": True,
                },
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                preflight=preflight,
                mutate=mutate,
            )
        except Exception as exc:
            _handle(exc)
