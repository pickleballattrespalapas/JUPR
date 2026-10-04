from __future__ import annotations

from datetime import date
from typing import Literal
from uuid import UUID

from fastapi import HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from jupr_app.domain.event_seasons import new_season_template, next_event_id
from jupr_app.services.event_season_service import admin_event_href, event_history, load_source, source_summary
from services.api.auth import auth_header
from services.api.club_site_routes import published_site, site_administrator

EventKind = Literal["interclub", "league", "tournament"]


class SeasonIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    request_id: UUID
    fingerprint: str = Field(pattern=r"^[a-f0-9]{32}$")
    series_name: str = Field(min_length=1, max_length=180)
    current_label: str = Field(min_length=1, max_length=120)
    label: str = Field(min_length=1, max_length=120)


class StartSeason(SeasonIdentity):
    name: str = Field(min_length=1, max_length=120)
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def ordered(self):
        if self.end_date < self.start_date:
            raise ValueError("The end date must be on or after the start date.")
        return self


class LinkSeason(SeasonIdentity):
    past_source_id: str = Field(min_length=1, max_length=180)
    past_fingerprint: str = Field(pattern=r"^[a-f0-9]{32}$")
    before_source_id: str = Field(min_length=1, max_length=180)


def valid_source(kind, source):
    if not source or len(source) > 180:
        raise HTTPException(422, "Choose an event.")
    if kind != "league":
        try:
            return str(UUID(source))
        except ValueError as exc:
            raise HTTPException(422, "Choose a valid event.") from exc
    return source


def _error(exc):
    if isinstance(exc, HTTPException):
        raise exc
    code = str(getattr(exc, "code", ""))
    if code in {"PT409", "40001"}:
        raise HTTPException(409, getattr(exc, "message", "The event changed. Reload and review it again.")) from exc
    if code == "23505":
        raise HTTPException(409, "A season with this name or label already exists. Check History before continuing.") from exc
    if code == "42501" or isinstance(exc, PermissionError):
        raise HTTPException(403, "Club administrator access and enabled event operations are required.") from exc
    if code == "P0002" or isinstance(exc, LookupError):
        raise HTTPException(404, "This event history is not available.") from exc
    if code in {"22023", "23514"} or isinstance(exc, ValueError):
        raise HTTPException(422, str(exc) if isinstance(exc, ValueError) else "Check the season names, labels and dates.") from exc
    raise HTTPException(503, "Could not confirm this action. Reload History to check whether the season was created before retrying.") from exc


def _write_guard(kind, source):
    if kind == "league":
        from services.api.staging_write_guard import require_league_manager_write_or_403, require_admin_team_league_write_or_403
        require_league_manager_write_or_403()
        if source["event"].get("league_type") == "Team":
            require_admin_team_league_write_or_403()
    elif kind == "tournament":
        from jupr_app.services.admin_tournament_guarded_operation import require_tournament_admin_mutation_runtime
        require_tournament_admin_mutation_runtime("setup")


def install_event_season_routes(app, *, get_supabase_client):
    @app.get("/admin/clubs/{club_id}/event-seasons")
    def admin_history(club_id: str, response: Response, kind: EventKind, event: str = Query(min_length=1, max_length=180), authorization: str | None = auth_header()):
        response.headers["Cache-Control"] = "no-store"
        site_administrator(get_supabase_client, authorization, club_id)
        try:
            db = get_supabase_client()
            source_id = valid_source(kind, event)
            if kind == "interclub":
                seasons = db.table("pcs_interclub_seasons").select("organizer_club_id").eq("id", source_id).limit(1).execute().data or []
                if seasons and seasons[0]["organizer_club_id"] != club_id:
                    own = db.table("pcs_interclub_participations").select("status").eq("season_id", source_id).eq("club_id", club_id).eq("status", "accepted").limit(1).execute().data or []
                    if not own:
                        raise LookupError()
                    history = event_history(db, club_id=seasons[0]["organizer_club_id"], kind=kind, source_id=source_id)
                    current = next(row for row in history["seasons"] if row["selected"])
                    return {**history, "current": {**current, "fingerprint": "", "admin_href": f"/admin/interclub/registrations?season={source_id}"},
                            "current_label": current["label"], "can_start": False, "past_candidates": [],
                            "reason": "The season organizer manages new seasons and history links."}
            return event_history(db, club_id=club_id, kind=kind, source_id=source_id, admin=True)
        except Exception as exc:
            _error(exc)

    @app.get("/admin/clubs/{club_id}/event-seasons/source")
    def source_preview(club_id: str, response: Response, kind: EventKind, event: str = Query(min_length=1, max_length=180), authorization: str | None = auth_header()):
        response.headers["Cache-Control"] = "no-store"
        site_administrator(get_supabase_client, authorization, club_id)
        try:
            source_id = valid_source(kind, event)
            source = load_source(get_supabase_client(), club_id, kind, source_id)
            if not source:
                raise LookupError()
            return source_summary(kind, source_id, source)
        except Exception as exc:
            _error(exc)

    @app.post("/admin/clubs/{club_id}/event-seasons/start")
    def start(club_id: str, body: StartSeason, kind: EventKind, event: str = Query(min_length=1, max_length=180), authorization: str | None = auth_header()):
        user = site_administrator(get_supabase_client, authorization, club_id)
        db = get_supabase_client()
        try:
            source_id = valid_source(kind, event)
            source = load_source(db, club_id, kind, source_id)
            if not source:
                raise LookupError()
            _write_guard(kind, source)
            new_id = next_event_id(club_id, str(body.request_id))
            template = new_season_template(kind, source, name=body.name, start_date=body.start_date.isoformat(), end_date=body.end_date.isoformat(), new_id=new_id)
            result = db.rpc("pcs_start_event_season", {"p_actor_id": user.user_id, "p_actor_email": user.email, "p_club_id": club_id,
                "p_kind": kind, "p_source_id": source_id, "p_request_id": str(body.request_id), "p_input": body.model_dump(mode="json", exclude={"request_id"}),
                "p_new_id": new_id, "p_template": template}).execute().data
            return {**result, "admin_href": admin_event_href(kind, result["source_id"], draft=True)}
        except Exception as exc:
            _error(exc)

    @app.post("/admin/clubs/{club_id}/event-seasons/link")
    def link(club_id: str, body: LinkSeason, kind: EventKind, event: str = Query(min_length=1, max_length=180), authorization: str | None = auth_header()):
        user = site_administrator(get_supabase_client, authorization, club_id)
        db = get_supabase_client()
        try:
            source_id = valid_source(kind, event)
            valid_source(kind, body.past_source_id)
            source = load_source(db, club_id, kind, source_id)
            if not source:
                raise LookupError()
            _write_guard(kind, source)
            return db.rpc("pcs_link_event_season", {"p_actor_id": user.user_id, "p_actor_email": user.email, "p_club_id": club_id,
                "p_kind": kind, "p_source_id": source_id, "p_request_id": str(body.request_id),
                "p_input": body.model_dump(mode="json", exclude={"request_id"})}).execute().data
        except Exception as exc:
            _error(exc)

    @app.get("/public/clubs/{slug}/event-history")
    def public_history(slug: str, response: Response, kind: Literal["league", "tournament"], event: str = Query(min_length=1, max_length=180)):
        response.headers["Cache-Control"] = "no-store"
        db = get_supabase_client()
        site = published_site(db, slug)
        if kind == "tournament":
            try:
                UUID(event)
            except ValueError:
                settings = db.table("tournament_registration_settings").select("tournament_id").eq("registration_slug", event).limit(1).execute().data or []
                if not settings:
                    raise HTTPException(404, "This event history is not available.")
                event = settings[0]["tournament_id"]
        try:
            return event_history(db, club_id=site["club_id"], kind=kind, source_id=valid_source(kind, event), slug=slug)
        except Exception as exc:
            _error(exc)

    @app.get("/public/interclub/{season_id}/history")
    def interclub_history(season_id: UUID, response: Response):
        response.headers["Cache-Control"] = "no-store"
        db = get_supabase_client()
        seasons = db.table("pcs_interclub_seasons").select("organizer_club_id").eq("id", str(season_id)).limit(1).execute().data or []
        if not seasons:
            raise HTTPException(404, "This event history is not available.")
        try:
            return event_history(db, club_id=seasons[0]["organizer_club_id"], kind="interclub", source_id=str(season_id))
        except Exception as exc:
            _error(exc)
