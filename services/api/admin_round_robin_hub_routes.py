"""Authenticated club operations for the shared Round Robin Hub."""
from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from services.api.auth import auth_header
from services.api.admin_play_generator_routes import _handle, _require_write_gate, _resolve_role_or_403
from jupr_app.services.admin_round_robin_hub_service import (
    command_round_robin_hub, create_round_robin_hub, get_round_robin_hub, list_round_robin_hubs,
)


class HubCourtRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=80)
    dedicated_session_key: str | None = Field(default=None, max_length=160)
    open: bool = True

    @field_validator("id", "label")
    @classmethod
    def nonblank(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("Enter a court name.")
        return value


class HubCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=160)
    courts: list[HubCourtRequest] = Field(min_length=1, max_length=40)
    idempotency_key: str = Field(min_length=8, max_length=160)

    @field_validator("title")
    @classmethod
    def nonblank(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("Enter an event name.")
        return value


class HubCommandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal[
        "add_group", "remove_group", "configure_courts", "assign_next_court", "assign_game",
        "reserve_game_for_court", "cancel_reservation", "move_game", "release_game",
        "score_and_release", "mark_played", "pause_player", "resume_player", "advance_round",
        "finish_hub", "rename_hub",
        "auto_fill_courts",
    ]
    payload: dict[str, Any] = Field(default_factory=dict, max_length=20)
    expected_version: int = Field(ge=1)
    idempotency_key: str = Field(min_length=8, max_length=160)

    @field_validator("payload")
    @classmethod
    def bounded_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        import json
        if len(json.dumps(value)) > 20000:
            raise ValueError("The hub change is too large.")
        return value


def _error(exc: Exception) -> None:
    if isinstance(exc, LookupError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _handle(exc)


def install_admin_round_robin_hub_routes(app, *, get_supabase_client) -> None:
    @app.get("/admin/clubs/{club_id}/round-robin-hubs")
    def list_hubs(club_id: str, authorization: str | None = auth_header()) -> dict[str, Any]:
        db = get_supabase_client()
        _resolve_role_or_403(supabase=db, club_id=club_id, authorization=authorization, source="round_robin_hub_list")
        try:
            return list_round_robin_hubs(db, club_id=club_id)
        except Exception as exc:
            _error(exc)

    @app.get("/admin/clubs/{club_id}/round-robin-hubs/{hub_id}")
    def detail(club_id: str, hub_id: UUID, authorization: str | None = auth_header()) -> dict[str, Any]:
        db = get_supabase_client()
        _resolve_role_or_403(supabase=db, club_id=club_id, authorization=authorization, source="round_robin_hub_detail")
        try:
            return get_round_robin_hub(db, club_id=club_id, hub_id=str(hub_id))
        except Exception as exc:
            _error(exc)

    @app.post("/admin/clubs/{club_id}/round-robin-hubs")
    def create(club_id: str, payload: HubCreateRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        _require_write_gate()
        db = get_supabase_client()
        email, _ = _resolve_role_or_403(supabase=db, club_id=club_id, authorization=authorization, source="round_robin_hub_create")
        try:
            return create_round_robin_hub(db, club_id=club_id, title=payload.title,
                courts=[court.model_dump() for court in payload.courts],
                idempotency_key=payload.idempotency_key, actor_email=email)
        except Exception as exc:
            _error(exc)

    @app.post("/admin/clubs/{club_id}/round-robin-hubs/{hub_id}/commands")
    def command(club_id: str, hub_id: UUID, payload: HubCommandRequest,
                authorization: str | None = auth_header()) -> dict[str, Any]:
        _require_write_gate()
        db = get_supabase_client()
        email, _ = _resolve_role_or_403(supabase=db, club_id=club_id, authorization=authorization, source="round_robin_hub_command")
        try:
            return command_round_robin_hub(db, club_id=club_id, hub_id=str(hub_id),
                action=payload.action, payload=payload.payload, expected_version=payload.expected_version,
                idempotency_key=payload.idempotency_key, actor_email=email)
        except Exception as exc:
            _error(exc)
