from __future__ import annotations

from typing import Any
from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from jupr_app.services.generator_sharing_service import (
    get_shared_generator, manage_generator_sharing, mutate_shared_generator, sharing_row,
)
from services.api.public_play_generator_routes import (
    PublicGeneratorMutationRequest, PublicGeneratorScoresRequest, PublicGeneratorSkipRequest,
    PublicGeneratorRosterRequest, PublicGeneratorPlayoffRequest,
)


class GeneratorSharingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: str = Field(pattern=r"^(view|handoff|revoke)$")
    expected_version: str | int
    edit_token: str | None = Field(default=None, max_length=128)


class SharedMutation(PublicGeneratorMutationRequest):
    expected_version: str = Field(min_length=1, max_length=64)


class SharedScores(PublicGeneratorScoresRequest):
    expected_version: str = Field(min_length=1, max_length=64)


class SharedSkip(PublicGeneratorSkipRequest):
    expected_version: str = Field(min_length=1, max_length=64)


class SharedRoster(PublicGeneratorRosterRequest):
    expected_version: str = Field(min_length=1, max_length=64)


class SharedPlayoff(PublicGeneratorPlayoffRequest):
    expected_version: str = Field(min_length=1, max_length=64)


def install_shared_play_generator_routes(app, *, get_club, get_supabase_client,
        require_public_writes, require_service_role, require_admin_writes,
        requester_hash, raise_public_error) -> None:
    def context(slug: str):
        club = get_club(slug)
        return str(club.get("id") or club.get("club_id") or slug), get_supabase_client()

    def handle(exc: Exception):
        if isinstance(exc, LookupError):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raise_public_error(exc)
        raise exc

    @app.post("/clubs/{club_slug}/play-generators/sessions/{session_key}/sharing")
    def manage_public_sharing(club_slug: str, session_key: str, payload: GeneratorSharingRequest):
        require_public_writes()
        require_service_role()
        club_id, supabase = context(club_slug)
        try:
            return manage_generator_sharing(supabase, club_id=club_id, session_key=session_key,
                action=payload.action, expected_version=payload.expected_version, owner_token=payload.edit_token)
        except Exception as exc:
            handle(exc)

    @app.get("/clubs/{club_slug}/shared-play-generators/sessions/{session_key}")
    def get_session(club_slug: str, session_key: str):
        require_service_role()
        club_id, supabase = context(club_slug)
        try:
            return get_shared_generator(supabase, club_id=club_id, session_key=session_key)
        except Exception as exc:
            handle(exc)

    def mutate(slug: str, key: str, payload, request: Request, action: str, number: int | None = None):
        require_public_writes()
        require_service_role()
        club_id, supabase = context(slug)
        try:
            row = sharing_row(supabase, club_id=club_id, session_key=key)
            if row["state"]["mode"] == "admin_play_generator":
                require_admin_writes()
            body = payload.model_dump(exclude={"edit_token", "expected_version", "idempotency_key"})
            if number is not None:
                body["round_number"] = number
            return mutate_shared_generator(supabase, club_id=club_id, session_key=key,
                edit_token=payload.edit_token, expected_version=payload.expected_version,
                idempotency_key=payload.idempotency_key, requester_hash=requester_hash(request),
                action=action, body=body)
        except HTTPException:
            raise
        except Exception as exc:
            handle(exc)

    @app.patch("/clubs/{club_slug}/shared-play-generators/sessions/{session_key}/rounds/{round_number}/scores")
    def scores(club_slug: str, session_key: str, round_number: int, payload: SharedScores, request: Request):
        return mutate(club_slug, session_key, payload, request, "scores", round_number)

    @app.post("/clubs/{club_slug}/shared-play-generators/sessions/{session_key}/rounds/{round_number}/skip")
    def skip(club_slug: str, session_key: str, round_number: int, payload: SharedSkip, request: Request):
        return mutate(club_slug, session_key, payload, request, "skip", round_number)

    @app.post("/clubs/{club_slug}/shared-play-generators/sessions/{session_key}/rounds/{round_number}/played")
    def played(club_slug: str, session_key: str, round_number: int, payload: SharedMutation, request: Request):
        return mutate(club_slug, session_key, payload, request, "played", round_number)

    @app.post("/clubs/{club_slug}/shared-play-generators/sessions/{session_key}/rounds/{round_number}/reopen")
    def reopen(club_slug: str, session_key: str, round_number: int, payload: SharedMutation, request: Request):
        return mutate(club_slug, session_key, payload, request, "reopen", round_number)

    @app.post("/clubs/{club_slug}/shared-play-generators/sessions/{session_key}/rounds/{round_number}/select")
    def select(club_slug: str, session_key: str, round_number: int, payload: SharedMutation, request: Request):
        return mutate(club_slug, session_key, payload, request, "select", round_number)

    @app.post("/clubs/{club_slug}/shared-play-generators/sessions/{session_key}/advance")
    def advance(club_slug: str, session_key: str, payload: SharedMutation, request: Request):
        return mutate(club_slug, session_key, payload, request, "advance")

    @app.post("/clubs/{club_slug}/shared-play-generators/sessions/{session_key}/complete")
    def complete(club_slug: str, session_key: str, payload: SharedMutation, request: Request):
        return mutate(club_slug, session_key, payload, request, "complete")

    @app.post("/clubs/{club_slug}/shared-play-generators/sessions/{session_key}/roster")
    def roster(club_slug: str, session_key: str, payload: SharedRoster, request: Request):
        return mutate(club_slug, session_key, payload, request, "roster")

    @app.post("/clubs/{club_slug}/shared-play-generators/sessions/{session_key}/playoff")
    def playoff(club_slug: str, session_key: str, payload: SharedPlayoff, request: Request):
        return mutate(club_slug, session_key, payload, request, "playoff")
