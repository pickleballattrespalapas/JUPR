"""Authenticated club operations for the shared Round Robin Hub."""
from __future__ import annotations

from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from services.api.auth import auth_header
from services.api.generator_request_models import require_complete_score_pair
from services.api.admin_play_generator_routes import _handle, _require_write_gate, _resolve_role_or_403
from jupr_app.services.admin_round_robin_hub_service import (
    command_round_robin_hub, create_round_robin_hub, get_round_robin_hub, list_round_robin_hubs,
)


class HubCourtRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(strict=True, min_length=1, max_length=80)
    label: str = Field(strict=True, min_length=1, max_length=80)
    dedicated_session_key: str | None = Field(default=None, strict=True, max_length=160)
    open: bool = Field(default=True, strict=True)

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


HubIdentifier = Annotated[str, Field(strict=True, min_length=1, max_length=160, pattern=r"\S")]
HubScore = Annotated[int, Field(strict=True, ge=0, le=99)]
HubProfile = Annotated[int, Field(strict=True, ge=1)]


class HubCommandPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HubSessionPayload(HubCommandPayload):
    session_key: HubIdentifier


class HubMatchPayload(HubSessionPayload):
    match_id: HubIdentifier
    round_number: Annotated[int, Field(strict=True, ge=1)] | None = None


class HubAssignedMatchPayload(HubMatchPayload):
    court_id: HubIdentifier


class HubScorePayload(HubAssignedMatchPayload):
    score_a: HubScore | None
    score_b: HubScore | None

    @model_validator(mode="after")
    def complete_pair(self) -> "HubScorePayload":
        require_complete_score_pair(self.score_a, self.score_b, self.match_id)
        return self


class HubResolveGamePayload(HubMatchPayload):
    court_id: HubIdentifier | None = None
    resolution: Literal["not_played"]
    reason: str = Field(default="", strict=True, max_length=300)


class HubCourtAssignmentPayload(HubCommandPayload):
    court_id: HubIdentifier
    session_key: HubIdentifier | None = None
    match_id: HubIdentifier | None = None
    round_number: Annotated[int, Field(strict=True, ge=1)] | None = None

    @model_validator(mode="after")
    def match_reference(self) -> "HubCourtAssignmentPayload":
        if (self.session_key is None) != (self.match_id is None):
            raise ValueError("Provide both the session and game reference.")
        return self


class HubMoveGamePayload(HubCourtAssignmentPayload):
    target_court_id: HubIdentifier


class HubCourtsPayload(HubCommandPayload):
    courts: list[HubCourtRequest] = Field(min_length=1, max_length=40)


class HubIdentityPayload(HubCommandPayload):
    identity: HubIdentifier


class HubParticipantPayload(HubSessionPayload):
    participant_id: HubIdentifier


class HubAddPlayerPayload(HubSessionPayload):
    name: HubIdentifier
    player_id: HubProfile | None = None


class HubTransferPlayerPayload(HubCommandPayload):
    source_session_key: HubIdentifier
    destination_session_key: HubIdentifier
    participant_id: HubIdentifier
    name: HubIdentifier | None = None


class HubIdentityResolutionPayload(HubCommandPayload):
    resolution: Literal["same", "different", "all_different"]
    participants: list[HubParticipantPayload] = Field(default_factory=list, max_length=160)

    @model_validator(mode="after")
    def distinct_participants(self) -> "HubIdentityResolutionPayload":
        refs = {(item.session_key, item.participant_id) for item in self.participants}
        if len(refs) != len(self.participants) or (self.resolution != "all_different" and len(refs) < 2):
            raise ValueError("Choose at least two distinct player entries.")
        return self


class HubEditPlayersPayload(HubMatchPayload):
    court_id: HubIdentifier | None = None
    participant_id: HubIdentifier
    replacement_participant_id: HubIdentifier | None = None
    name: HubIdentifier | None = None
    player_id: HubProfile | None = None

    @model_validator(mode="after")
    def replacement(self) -> "HubEditPlayersPayload":
        if self.replacement_participant_id:
            if self.name is not None or self.player_id is not None:
                raise ValueError("Choose an existing player or add a new player, not both.")
        elif self.name is None:
            raise ValueError("Enter the replacement player's name.")
        return self


class HubTitlePayload(HubCommandPayload):
    title: HubIdentifier


class HubRegistrationPayload(HubCommandPayload):
    open: bool = Field(strict=True)


class HubReviewCheckinPayload(HubCommandPayload):
    checkin_id: HubIdentifier
    disposition: Literal["add", "existing", "dismiss", "restore"]
    participant_id: HubIdentifier | None = None
    name: HubIdentifier | None = None
    player_id: HubProfile | None = None

    @model_validator(mode="after")
    def existing_player(self) -> "HubReviewCheckinPayload":
        if self.disposition == "existing" and self.participant_id is None:
            raise ValueError("Choose an existing player.")
        return self


HUB_PAYLOAD_MODELS = {
    "add_group": HubSessionPayload, "remove_group": HubSessionPayload,
    "configure_courts": HubCourtsPayload, "assign_next_court": HubMatchPayload,
    "assign_game": HubAssignedMatchPayload, "reserve_game_for_court": HubAssignedMatchPayload,
    "cancel_reservation": HubCourtAssignmentPayload, "release_game": HubCourtAssignmentPayload,
    "move_game": HubMoveGamePayload, "score_and_release": HubScorePayload,
    "mark_played": HubAssignedMatchPayload, "resolve_game": HubResolveGamePayload,
    "pause_player": HubIdentityPayload, "resume_player": HubIdentityPayload,
    "advance_round": HubSessionPayload, "finish_group": HubSessionPayload,
    "finish_hub": HubCommandPayload, "auto_fill_courts": HubCommandPayload,
    "rename_hub": HubTitlePayload, "resolve_identities": HubIdentityResolutionPayload,
    "add_player": HubAddPlayerPayload, "remove_player": HubParticipantPayload,
    "rejoin_player": HubParticipantPayload, "transfer_player": HubTransferPlayerPayload,
    "set_registration": HubRegistrationPayload, "review_checkin": HubReviewCheckinPayload,
    "process_checkins": HubCommandPayload, "edit_match_players": HubEditPlayersPayload,
}


class HubCommandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal[
        "add_group", "remove_group", "configure_courts", "assign_next_court", "assign_game",
        "reserve_game_for_court", "cancel_reservation", "move_game", "release_game",
        "score_and_release", "mark_played", "pause_player", "resume_player", "advance_round",
        "finish_hub", "finish_group", "rename_hub",
        "auto_fill_courts", "resolve_identities", "add_player", "remove_player",
        "set_registration", "review_checkin", "process_checkins", "edit_match_players",
        "transfer_player", "rejoin_player", "resolve_game",
    ]
    payload: dict[str, Any] = Field(default_factory=dict, max_length=20)
    expected_version: int = Field(strict=True, ge=1)
    idempotency_key: str = Field(min_length=8, max_length=160)

    @field_validator("payload")
    @classmethod
    def bounded_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        import json
        if len(json.dumps(value)) > 20000:
            raise ValueError("The hub change is too large.")
        return value


    @model_validator(mode="after")
    def validate_action_payload(self) -> "HubCommandRequest":
        # Preserve the original JSON for exact idempotency fingerprints.
        HUB_PAYLOAD_MODELS[self.action].model_validate(self.payload)
        return self


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
