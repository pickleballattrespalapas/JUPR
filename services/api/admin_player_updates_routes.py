from __future__ import annotations

import os
from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import PERMISSION_MANAGE_SUBSCRIPTIONS, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_player_updates_service import (
    build_admin_player_updates_status,
    is_admin_player_updates_enabled,
    run_admin_player_update_range,
)
from jupr_app.domain.notifications.player_profile_update_repo import StaleCommunicationsStateError
from jupr_app.services.admin_communications_service import (
    build_communications_workspace,
    deactivate_active_subscription,
    delete_outbox_rows,
    preview_player_digest,
    queue_player_digests,
    replace_active_subscription,
    retry_outbox_rows,
    send_selected_outbox_rows,
)
from services.api.auth import authenticate_bearer, auth_header


class AdminPlayerUpdateRangeRequest(BaseModel):
    start_date: str
    end_date: str
    only_players_with_matches: bool = True
    send_now: bool = True
    confirmation_text: str = ""
    source: str = "next_player_updates_admin_range"


class CommunicationsRowRef(BaseModel):
    id: str
    expected_row_version: int = Field(ge=1)


class PlayerDigestPreviewRequest(BaseModel):
    player_id: int
    start_date: str
    end_date: str


class PlayerDigestQueueRequest(BaseModel):
    start_date: str
    end_date: str
    player_id: int | None = None
    only_players_with_matches: bool = True
    confirmation_text: str = ""
    operation_key: str
    source: str = "next_player_updates_queue"


class OutboxSelectionRequest(BaseModel):
    items: list[CommunicationsRowRef] = Field(default_factory=list)
    confirmation_text: str = ""
    operation_key: str | None = None
    source: str = "next_player_updates_outbox"


class SubscriptionReplaceRequest(BaseModel):
    expected_row_version: int = Field(ge=1)
    new_email: str
    request_note: str | None = None
    admin_note: str | None = None
    confirmation_text: str = ""
    operation_key: str
    source: str = "next_player_updates_replace"


class SubscriptionDeactivateRequest(BaseModel):
    expected_row_version: int = Field(ge=1)
    confirmation_text: str = ""
    source: str = "next_player_updates_deactivate"


def _require_service_role() -> None:
    if not os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail="Player Updates Admin requires SUPABASE_SERVICE_ROLE_KEY on FastAPI. This secret must never be configured in the browser deployment.",
        )


def _resolve_player_updates_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(
        supabase=supabase,
        club_id=str(club_id),
        email=user.email,
        user_id=user.user_id,
        allowlist=set(),
    )
    if not has_permission(role_resolution.role, PERMISSION_MANAGE_SUBSCRIPTIONS):
        denied_payload = build_activity_payload(
            club_id=str(club_id),
            actor_email=user.email,
            actor_role=role_resolution.role,
            action_type="admin_player_updates_denied",
            entity_type="player_updates",
            entity_id="player_updates",
            after_json={"source_client": "fastapi/nextjs", "reason": "insufficient_permission"},
            source_page=source,
            flagged_for_review=True,
        )
        write_admin_activity_log(supabase, denied_payload)
        raise HTTPException(status_code=403, detail="insufficient permission")
    return user.email, role_resolution.role


def _handle(exc: Exception) -> None:
    if isinstance(exc, StaleCommunicationsStateError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise exc


def install_admin_player_updates_routes(app, *, get_supabase_client) -> None:
    """Register guarded Player Updates Admin routes for the Next admin pilot."""

    @app.get("/admin/clubs/{club_id}/player-updates/status")
    def get_admin_player_updates_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_player_updates_enabled() else None
        return build_admin_player_updates_status(supabase, club_id=str(club_id))

    @app.post("/admin/clubs/{club_id}/player-updates/send-range")
    def post_admin_player_updates_send_range(
        club_id: str,
        payload: AdminPlayerUpdateRangeRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_player_updates_enabled():
            raise HTTPException(status_code=403, detail="Next Player Updates Admin is disabled.")
        _require_service_role()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_player_updates_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return run_admin_player_update_range(
                supabase,
                club_id=str(club_id),
                start_date=payload.start_date,
                end_date=payload.end_date,
                only_players_with_matches=payload.only_players_with_matches,
                send_now=payload.send_now,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)

    @app.get("/admin/clubs/{club_id}/player-updates/workspace")
    def get_admin_player_updates_workspace(
        club_id: str,
        start_date: str = Query(...),
        end_date: str = Query(...),
        outbox_status: str | None = Query(default=None),
        limit: int = Query(default=500, ge=1, le=1000),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        if not is_admin_player_updates_enabled():
            raise HTTPException(status_code=403, detail="Next Player Updates Admin is disabled.")
        _require_service_role()
        supabase = get_supabase_client()
        _resolve_player_updates_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_player_updates_workspace",
        )
        try:
            return build_communications_workspace(
                supabase,
                club_id=str(club_id),
                start_date=start_date,
                end_date=end_date,
                outbox_status=outbox_status,
                limit=limit,
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/player-updates/digests/preview")
    def post_admin_player_digest_preview(
        club_id: str,
        payload: PlayerDigestPreviewRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_service_role()
        supabase = get_supabase_client()
        _resolve_player_updates_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source="next_player_updates_digest_preview",
        )
        try:
            return preview_player_digest(
                supabase,
                club_id=str(club_id),
                player_id=payload.player_id,
                start_date=payload.start_date,
                end_date=payload.end_date,
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/player-updates/digests/queue")
    def post_admin_player_digest_queue(
        club_id: str,
        payload: PlayerDigestQueueRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_service_role()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_player_updates_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return queue_player_digests(
                supabase,
                club_id=str(club_id),
                start_date=payload.start_date,
                end_date=payload.end_date,
                player_id=payload.player_id,
                only_players_with_matches=payload.only_players_with_matches,
                confirmation_text=payload.confirmation_text,
                operation_key=payload.operation_key,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/player-updates/outbox/send")
    def post_admin_player_updates_outbox_send(
        club_id: str,
        payload: OutboxSelectionRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_service_role()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_player_updates_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return send_selected_outbox_rows(
                supabase,
                club_id=str(club_id),
                items=[item.model_dump() for item in payload.items],
                confirmation_text=payload.confirmation_text,
                operation_key=str(payload.operation_key or ""),
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/player-updates/outbox/retry")
    def post_admin_player_updates_outbox_retry(
        club_id: str,
        payload: OutboxSelectionRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_service_role()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_player_updates_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return retry_outbox_rows(
                supabase,
                club_id=str(club_id),
                items=[item.model_dump() for item in payload.items],
                confirmation_text=payload.confirmation_text,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/player-updates/outbox/delete")
    def post_admin_player_updates_outbox_delete(
        club_id: str,
        payload: OutboxSelectionRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_service_role()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_player_updates_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return delete_outbox_rows(
                supabase,
                club_id=str(club_id),
                items=[item.model_dump() for item in payload.items],
                confirmation_text=payload.confirmation_text,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/player-updates/subscriptions/{subscription_id}/replace")
    def post_admin_player_updates_subscription_replace(
        club_id: str,
        subscription_id: str,
        payload: SubscriptionReplaceRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_service_role()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_player_updates_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return replace_active_subscription(
                supabase,
                club_id=str(club_id),
                subscription_id=str(subscription_id),
                expected_row_version=payload.expected_row_version,
                new_email=payload.new_email,
                request_note=payload.request_note,
                admin_note=payload.admin_note,
                confirmation_text=payload.confirmation_text,
                operation_key=payload.operation_key,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/player-updates/subscriptions/{subscription_id}/deactivate")
    def post_admin_player_updates_subscription_deactivate(
        club_id: str,
        subscription_id: str,
        payload: SubscriptionDeactivateRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_service_role()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_player_updates_role_or_403(
            supabase=supabase,
            club_id=str(club_id),
            authorization=authorization,
            source=payload.source,
        )
        try:
            return deactivate_active_subscription(
                supabase,
                club_id=str(club_id),
                subscription_id=str(subscription_id),
                expected_row_version=payload.expected_row_version,
                confirmation_text=payload.confirmation_text,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)
