from __future__ import annotations

from typing import Any, Callable

from fastapi import Query, Request, Response
from pydantic import BaseModel, Field

from jupr_app.services.public_play_generator_service import (
    advance_public_play_generator_session,
    build_public_play_generator_export,
    build_public_play_generator_status,
    complete_public_play_generator_session,
    create_public_play_generator_session,
    get_public_play_generator_session,
    list_public_play_generator_sessions,
    mutate_public_play_generator_roster,
    preview_public_play_generator,
    save_public_play_generator_round,
    skip_public_play_generator_round,
)


class PublicGeneratorPreviewRequest(BaseModel):
    generator_kind: str = Field(pattern=r"^(round_robin|ladder)$")
    play_format: str = Field(pattern=r"^(singles|doubles)$")
    title: str = Field(default="Play session", max_length=160)
    participant_names: list[str] = Field(min_length=2, max_length=40)
    participant_player_ids: dict[str, int] = Field(default_factory=dict, max_length=40)
    total_rounds: int = Field(default=3, ge=1, le=50)
    court_count: int = Field(default=0, ge=0, le=20)
    standings_sort: str = Field(default="wins", pattern=r"^(wins|points|differential)$")


class PublicGeneratorStartRequest(PublicGeneratorPreviewRequest):
    preview_fingerprint: str | None = Field(default=None, max_length=128)
    idempotency_key: str = Field(min_length=8, max_length=160)


class PublicGeneratorMutationRequest(BaseModel):
    edit_token: str = Field(min_length=1, max_length=128)
    expected_version: int = Field(ge=1)
    idempotency_key: str = Field(min_length=8, max_length=160)


class PublicGeneratorScorePayload(BaseModel):
    match_id: str = Field(min_length=1, max_length=160)
    score_a: int | None = Field(default=None, ge=0, le=99)
    score_b: int | None = Field(default=None, ge=0, le=99)


class PublicGeneratorScoresRequest(PublicGeneratorMutationRequest):
    scores: list[PublicGeneratorScorePayload] = Field(min_length=1, max_length=1000)


class PublicGeneratorSkipRequest(PublicGeneratorMutationRequest):
    reason: str = Field(default="", max_length=300)


class PublicGeneratorRosterRequest(PublicGeneratorMutationRequest):
    action: str = Field(pattern=r"^(add|remove|substitute|reorder)$")
    participant_id: str | None = Field(default=None, max_length=160)
    name: str | None = Field(default=None, max_length=160)
    player_id: int | None = None
    substitute_scope: str = Field(default="rest", pattern=r"^(round|rest)$")
    roster_order: list[str] = Field(default_factory=list, max_length=40)


def _model_payload(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[attr-defined]
    return model.dict()


def install_public_play_generator_routes(
    app,
    *,
    get_club: Callable[[str], dict[str, Any]],
    get_supabase_client: Callable[[], Any],
    public_club_payload: Callable[[dict[str, Any], str], dict[str, Any]],
    require_public_writes: Callable[[], None],
    require_service_role: Callable[[], None],
    requester_hash: Callable[[Request], str],
    raise_public_error: Callable[[Exception], None],
    public_writes_enabled: Callable[[], bool],
    service_role_configured: Callable[[], bool],
) -> None:
    def context(club_slug: str) -> tuple[dict[str, Any], str, Any]:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        return club, club_id, get_supabase_client()

    @app.get("/clubs/{club_slug}/play-generators/status")
    def get_public_generator_status(club_slug: str) -> dict[str, Any]:
        club = get_club(club_slug)
        club_id = str(club.get("id") or club.get("club_id") or club_slug)
        ready = bool(public_writes_enabled() and service_role_configured())
        try:
            supabase = get_supabase_client() if service_role_configured() else None
        except Exception:
            supabase = None
        status = build_public_play_generator_status(
            supabase,
            club_id=club_id,
            writes_enabled=ready,
        )
        return {"club": public_club_payload(club, club_slug), **status}

    @app.post("/clubs/{club_slug}/play-generators/preview")
    def post_public_generator_preview(
        club_slug: str,
        payload: PublicGeneratorPreviewRequest,
    ) -> dict[str, Any]:
        require_service_role()
        club, club_id, supabase = context(club_slug)
        try:
            result = preview_public_play_generator(
                supabase,
                club_id=club_id,
                generator_kind=payload.generator_kind,
                play_format=payload.play_format,
                title=payload.title,
                participant_names=payload.participant_names,
                participant_player_ids=payload.participant_player_ids,
                total_rounds=payload.total_rounds,
                court_count=payload.court_count,
                standings_sort=payload.standings_sort,
            )
        except Exception as exc:
            raise_public_error(exc)
            raise
        return {"club": public_club_payload(club, club_slug), **result}

    @app.get("/clubs/{club_slug}/play-generators/sessions")
    def get_public_generator_sessions(
        club_slug: str,
        generator_kind: str | None = Query(default=None, pattern=r"^(round_robin|ladder)$"),
        limit: int = Query(default=50, ge=1, le=100),
    ) -> dict[str, Any]:
        require_service_role()
        club, club_id, supabase = context(club_slug)
        try:
            result = list_public_play_generator_sessions(
                supabase,
                club_id=club_id,
                generator_kind=generator_kind,
                limit=limit,
            )
        except Exception as exc:
            raise_public_error(exc)
            raise
        return {"club": public_club_payload(club, club_slug), **result}

    @app.post("/clubs/{club_slug}/play-generators/sessions")
    def post_public_generator_session(
        club_slug: str,
        payload: PublicGeneratorStartRequest,
        request: Request,
    ) -> dict[str, Any]:
        require_public_writes()
        require_service_role()
        club, club_id, supabase = context(club_slug)
        try:
            result = create_public_play_generator_session(
                supabase,
                club_id=club_id,
                generator_kind=payload.generator_kind,
                play_format=payload.play_format,
                title=payload.title,
                participant_names=payload.participant_names,
                participant_player_ids=payload.participant_player_ids,
                total_rounds=payload.total_rounds,
                court_count=payload.court_count,
                preview_fingerprint=payload.preview_fingerprint,
                standings_sort=payload.standings_sort,
                idempotency_key=payload.idempotency_key,
                requester_hash=requester_hash(request),
            )
        except Exception as exc:
            raise_public_error(exc)
            raise
        return {"club": public_club_payload(club, club_slug), **result}

    @app.get("/clubs/{club_slug}/play-generators/sessions/{session_key}")
    def get_public_generator_session(
        club_slug: str,
        session_key: str,
    ) -> dict[str, Any]:
        require_service_role()
        club, club_id, supabase = context(club_slug)
        try:
            result = get_public_play_generator_session(
                supabase,
                club_id=club_id,
                session_key=session_key,
            )
        except Exception as exc:
            raise_public_error(exc)
            raise
        return {"club": public_club_payload(club, club_slug), **result}

    @app.patch("/clubs/{club_slug}/play-generators/sessions/{session_key}/rounds/{round_number}/scores")
    def patch_public_generator_round_scores(
        club_slug: str,
        session_key: str,
        round_number: int,
        payload: PublicGeneratorScoresRequest,
        request: Request,
    ) -> dict[str, Any]:
        require_public_writes()
        require_service_role()
        club, club_id, supabase = context(club_slug)
        try:
            result = save_public_play_generator_round(
                supabase,
                club_id=club_id,
                session_key=session_key,
                round_number=round_number,
                scores=[_model_payload(row) for row in payload.scores],
                edit_token=payload.edit_token,
                expected_version=payload.expected_version,
                idempotency_key=payload.idempotency_key,
                requester_hash=requester_hash(request),
            )
        except Exception as exc:
            raise_public_error(exc)
            raise
        return {"club": public_club_payload(club, club_slug), **result}

    @app.post("/clubs/{club_slug}/play-generators/sessions/{session_key}/rounds/{round_number}/skip")
    def post_public_generator_round_skip(
        club_slug: str,
        session_key: str,
        round_number: int,
        payload: PublicGeneratorSkipRequest,
        request: Request,
    ) -> dict[str, Any]:
        require_public_writes()
        require_service_role()
        club, club_id, supabase = context(club_slug)
        try:
            result = skip_public_play_generator_round(
                supabase,
                club_id=club_id,
                session_key=session_key,
                round_number=round_number,
                reason=payload.reason,
                edit_token=payload.edit_token,
                expected_version=payload.expected_version,
                idempotency_key=payload.idempotency_key,
                requester_hash=requester_hash(request),
            )
        except Exception as exc:
            raise_public_error(exc)
            raise
        return {"club": public_club_payload(club, club_slug), **result}

    @app.post("/clubs/{club_slug}/play-generators/sessions/{session_key}/advance")
    def post_public_generator_advance(
        club_slug: str,
        session_key: str,
        payload: PublicGeneratorMutationRequest,
        request: Request,
    ) -> dict[str, Any]:
        require_public_writes()
        require_service_role()
        club, club_id, supabase = context(club_slug)
        try:
            result = advance_public_play_generator_session(
                supabase,
                club_id=club_id,
                session_key=session_key,
                edit_token=payload.edit_token,
                expected_version=payload.expected_version,
                idempotency_key=payload.idempotency_key,
                requester_hash=requester_hash(request),
            )
        except Exception as exc:
            raise_public_error(exc)
            raise
        return {"club": public_club_payload(club, club_slug), **result}

    @app.post("/clubs/{club_slug}/play-generators/sessions/{session_key}/roster")
    def post_public_generator_roster(
        club_slug: str,
        session_key: str,
        payload: PublicGeneratorRosterRequest,
        request: Request,
    ) -> dict[str, Any]:
        require_public_writes()
        require_service_role()
        club, club_id, supabase = context(club_slug)
        try:
            result = mutate_public_play_generator_roster(
                supabase,
                club_id=club_id,
                session_key=session_key,
                action=payload.action,
                participant_id=payload.participant_id,
                name=payload.name,
                player_id=payload.player_id,
                substitute_scope=payload.substitute_scope,
                roster_order=payload.roster_order,
                edit_token=payload.edit_token,
                expected_version=payload.expected_version,
                idempotency_key=payload.idempotency_key,
                requester_hash=requester_hash(request),
            )
        except Exception as exc:
            raise_public_error(exc)
            raise
        return {"club": public_club_payload(club, club_slug), **result}

    @app.post("/clubs/{club_slug}/play-generators/sessions/{session_key}/complete")
    def post_public_generator_complete(
        club_slug: str,
        session_key: str,
        payload: PublicGeneratorMutationRequest,
        request: Request,
    ) -> dict[str, Any]:
        require_public_writes()
        require_service_role()
        club, club_id, supabase = context(club_slug)
        try:
            result = complete_public_play_generator_session(
                supabase,
                club_id=club_id,
                session_key=session_key,
                edit_token=payload.edit_token,
                expected_version=payload.expected_version,
                idempotency_key=payload.idempotency_key,
                requester_hash=requester_hash(request),
            )
        except Exception as exc:
            raise_public_error(exc)
            raise
        return {"club": public_club_payload(club, club_slug), **result}

    @app.get("/clubs/{club_slug}/play-generators/sessions/{session_key}/export")
    def export_public_generator_session(
        club_slug: str,
        session_key: str,
        format: str = Query(default="csv", pattern=r"^(csv|json)$"),
    ) -> Response:
        require_service_role()
        _club, club_id, supabase = context(club_slug)
        try:
            export = build_public_play_generator_export(
                supabase,
                club_id=club_id,
                session_key=session_key,
                export_format=format,
            )
        except Exception as exc:
            raise_public_error(exc)
            raise
        return Response(
            content=str(export["content"]),
            media_type=str(export["media_type"]),
            headers={"Content-Disposition": f'attachment; filename="{export["filename"]}"'},
        )
