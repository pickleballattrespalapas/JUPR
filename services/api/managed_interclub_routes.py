"""Single-organizer seasons and revocable, meet-scoped host links."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import secrets
from typing import Literal
from uuid import UUID

from fastapi import HTTPException, Response
from pydantic import Field

from jupr_app.domain import managed_interclub as managed
from jupr_app.domain import interclub_competition as competition
from services.api.auth import auth_header
from services.api.club_site_routes import site_administrator
from services.api.interclub_competition_models import CompetitionDocument
from services.api.interclub_result_views import result_rows
from services.api.staging_write_guard import require_league_manager_write_or_403, require_public_intake_or_403

SEASONS = "pcs_managed_interclub_seasons"
MEETS = "pcs_managed_interclub_meets"
SEASON_FIELDS = "id,organizer_club_id,revision,details,published,source_season_id,updated_at"
MEET_FIELDS = "id,season_id,revision,details,state,document,approved_document,host_expires_at,updated_at"
PRIVATE_HEADERS = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer", "X-Robots-Tag": "noindex, nofollow"}


class Revision(managed.Model):
    expected_revision: int = Field(ge=0)


class SaveSeason(Revision):
    details: managed.SeasonDetails


class SaveMeet(Revision):
    season_revision: int = Field(ge=1)
    details: managed.MeetDetails


class SaveScores(Revision):
    document: CompetitionDocument


class Action(Revision):
    reason: str = Field(default="", max_length=500)


class HostRead(managed.Model):
    token: str = Field(min_length=40, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")


class HostWrite(HostRead):
    expected_revision: int = Field(ge=1)
    action: Literal["scores", "submit"]
    document: CompetitionDocument | None = None


def _rows(db, table, fields="*", **where):
    query = db.table(table).select(fields)
    for key, value in where.items():
        query = query.eq(key, value)
    return query.execute().data or []


def _one(db, table, fields="*", **where):
    rows = _rows(db, table, fields, **where)
    if not rows:
        raise HTTPException(404, "Season or meet unavailable.", headers=PRIVATE_HEADERS)
    return rows[0]


def _value(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ValueError as exc:
        raise HTTPException(422, str(exc), headers=PRIVATE_HEADERS) from exc


def _write(db, *, season_id, action, revision, payload=None, user=None, club_id=None, meet_id=None, token_hash=None):
    try:
        return db.rpc("pcs_managed_interclub_write", {
            "p_actor_id": user.user_id if user else None, "p_actor_email": user.email if user else None,
            "p_club_id": club_id, "p_season_id": str(season_id), "p_meet_id": str(meet_id) if meet_id else None,
            "p_action": action, "p_revision": revision, "p_payload": payload or {}, "p_token_hash": token_hash,
        }).execute().data
    except Exception as exc:
        code = str(getattr(exc, "code", ""))
        status, message = {
            "42501": (403, "This access is no longer available. Ask the organizer for a new host link." if token_hash else "Organizer access required."),
            "PT409": (409, "This meet or season changed. Reload its saved version before continuing."),
            "40001": (409, "This meet or season changed. Reload its saved version before continuing."),
            "23505": (409, "This record already exists. Reload before continuing."),
            "P0002": (404, "Season or meet unavailable."),
            "22023": (422, "Check the meet state, schedule and correction reason before continuing."),
        }.get(code, (503, "Could not confirm the save. Reload to check the saved scores before retrying."))
        raise HTTPException(status, message, headers=PRIVATE_HEADERS) from exc


def context(db, season):
    meets = _rows(db, MEETS, MEET_FIELDS, season_id=season["id"])
    meets.sort(key=lambda m: (m["details"]["starts_at"], m["id"]))
    return {"season": season, "meets": meets, **managed.standings(season, meets)}


def host_context(season, meet):
    # Explicit projection: no captain contacts, other meet records or link hash.
    details = season["details"]
    clubs = [c for c in managed.safe_clubs(details) if c["id"] in meet["details"]["club_ids"]]
    return {"season": {"id": season["id"], "details": {k: details.get(k) for k in ("name", "timezone", "rules", "divisions")}},
            "clubs": clubs, "meet": {k: meet[k] for k in ("id", "season_id", "revision", "details", "state", "document", "updated_at")}}


def public_context(db, season):
    data = context(db, season)
    details, meets = season["details"], data["meets"]
    docs = [m["approved_document"] for m in meets if m.get("approved_document")]
    results = result_rows(docs)
    catalog = {p["entry_id"]: {"id": p["entry_id"], "name": p["name"], "club_id": t["club_id"]}
               for m in meets for t in m["details"]["teams"] for p in [*t["roster"], *t.get("reserves", [])]}
    used = {pid for row in results for pairing in row["pairings"] for game in pairing["games"] for side in ("a", "b") for pid in game.get(f"players_{side}", [])}
    doc = {**{k: details.get(k) or "" for k in ("name", "start_date", "end_date", "timezone", "divisions")},
           "clubs": managed.safe_clubs(details), "rules_text": details.get("rules", ""), "scoring_version": 1,
           "meets": [{"id": m["id"], **{k: m["details"][k] for k in ("host_club_id", "club_ids", "starts_at", "duration_minutes", "courts")},
                      "competition_phase": m["details"]["phase"]} for m in meets],
           "results": [], "competition_results": results, "players": [catalog[pid] for pid in sorted(used) if pid in catalog]}
    return {"id": season["id"], "document": doc,
            "standings": [{"division": division, "rows": data["standings"]["divisions"].get(division, [])} for division in details["divisions"]],
            "club_cup": data["club_cup"], "qualification": data["standings"]["qualification"], "published_at": season["updated_at"]}


def _host(db, token):
    digest = hashlib.sha256(token.encode()).hexdigest()
    rows = _rows(db, MEETS, MEET_FIELDS, host_token_hash=digest)
    if not rows or not rows[0].get("host_expires_at") or datetime.fromisoformat(rows[0]["host_expires_at"].replace("Z", "+00:00")) <= datetime.now(timezone.utc):
        raise HTTPException(404, "This host link has expired or was disabled. Ask the organizer for a new link.", headers=PRIVATE_HEADERS)
    meet = rows[0]
    return digest, _one(db, SEASONS, SEASON_FIELDS, id=meet["season_id"]), meet


def _score_payload(meet, body, official=False):
    if meet["state"] != "draft" or not meet.get("document"):
        raise HTTPException(409, "This meet is not open for score entry. Reload its current status.", headers=PRIVATE_HEADERS)
    incoming = body.document.model_dump(mode="json") if body.document is not None else meet["document"]
    document = _value(managed.score_document, meet["document"], incoming, meet["details"]["teams"], official=official)
    return {"document": document}


def import_legacy(db, club_id, season_id):
    """Copy organizer-visible competition facts; leave originals and ratings intact."""
    source = _one(db, "pcs_interclub_seasons", "id,organizer_club_id,details", id=season_id, organizer_club_id=club_id)
    details = source["details"]
    clubs = db.table("clubs").select("id,name").in_("id", details["club_ids"]).execute().data or []
    rosters = _rows(db, "pcs_interclub_current_rosters", "id,meet_id,club_id,division,roster,withdrawn", season_id=season_id)
    def players(roster):
        result = []
        for p in roster:
            gender = str(p.get("gender", "")).lower()
            if gender in {"f", "female", "woman", "women"}: gender = "female"
            elif gender in {"m", "male", "man", "men"}: gender = "male"
            else: raise HTTPException(422, f"Confirm {p.get('name', 'a player')}'s gender in the original roster before importing.")
            result.append(managed.Player(entry_id=p["entry_id"], name=p["name"], gender=gender, starting_rating=p.get("starting_rating")).model_dump(mode="json"))
        return result
    managed_clubs = []
    for club in clubs:
        by_division = {}
        for row in rosters:
            if row["club_id"] == club["id"] and not row["withdrawn"]:
                group = by_division.setdefault(row["division"], {})
                for p in players(row["roster"]): group[p["entry_id"]] = p
        managed_clubs.append({**club, "captain": "", "contact": "", "rosters": [{"division": d, "roster": list(ps.values())} for d, ps in by_division.items()]})
    batches = _rows(db, "pcs_interclub_competition_batches", "meet_id,document,approved_document,state,phase", season_id=season_id)
    catalog = {(c["id"], p["entry_id"]): p for c in managed_clubs for r in c["rosters"] for p in r["roster"]}
    used = {}
    for batch in batches:
        for document in (batch.get("document"), batch.get("approved_document")):
            for encounter in (document or {}).get("encounters", []):
                for side in ("a", "b"):
                    key = (batch["meet_id"], encounter[f"club_{side}"], encounter["division"])
                    group = used.setdefault(key, set())
                    for pairing in encounter["pairings"]:
                        group.update(pairing[f"players_{side}"])
                        for game in pairing["games"]: group.update(game.get(f"players_{side}", []))
    # Historical substitutes may no longer be on a current lineup. Resolve only
    # the entry/profile names actually used in the season's saved score sheets.
    missing = {entry_id for (_, club, _), ids in used.items() for entry_id in ids if (club, entry_id) not in catalog}
    if missing:
        entries = db.table("pcs_interclub_entries").select("id,club_id,player_id,starting_rating").eq("season_id", season_id).in_("id", sorted(missing)).execute().data or []
        for club in managed_clubs:
            selected = [e for e in entries if e["club_id"] == club["id"]]
            if not selected: continue
            profiles = db.table("players").select("id,name,gender").eq("club_id", club["id"]).in_("id", [e["player_id"] for e in selected]).execute().data or []
            by_id = {str(p["id"]): p for p in profiles}
            for entry in selected:
                profile = by_id.get(str(entry["player_id"]))
                if profile:
                    catalog[(club["id"], entry["id"])] = players([{**profile, "entry_id": entry["id"], "starting_rating": entry["starting_rating"]}])[0]
    meets = []
    for old in _rows(db, "pcs_interclub_meets", "id,host_club_id,club_ids,starts_at,duration_minutes,courts,competition_phase", season_id=season_id):
        found = [b for b in batches if b["meet_id"] == old["id"]]
        if len(found) > 1:
            raise HTTPException(422, "This season has multiple competition phases in one meet. Keep its existing workspace for those historical results.")
        batch = found[0] if found else None
        teams = [{"club_id": t["club_id"], "division": t["division"], "roster": players(t["roster"]), "reserves": []} for t in rosters if t["meet_id"] == old["id"] and not t["withdrawn"]]
        for (meet_id, club_id, division), ids in used.items():
            if meet_id != old["id"]: continue
            team = next((t for t in teams if t["club_id"] == club_id and t["division"] == division), None)
            if team is None:
                team = {"club_id": club_id, "division": division, "roster": [], "reserves": []}; teams.append(team)
            known = {p["entry_id"] for p in team["roster"]}
            for entry_id in sorted(ids - known):
                player = catalog.get((club_id, entry_id))
                if not player: raise HTTPException(422, "A historical player could not be resolved. The original season remains available; no results were moved.")
                team["reserves"].append(deepcopy(player))
                club = next(c for c in managed_clubs if c["id"] == club_id)
                roster = next((r for r in club["rosters"] if r["division"] == division), None)
                if roster is None: roster = {"division": division, "roster": []}; club["rosters"].append(roster)
                if not any(p["entry_id"] == entry_id for p in roster["roster"]): roster["roster"].append(deepcopy(player))
        plan = {k: old[k] for k in ("host_club_id", "club_ids", "starts_at", "duration_minutes", "courts")}
        plan.update(phase=batch["document"]["phase"] if batch else old.get("competition_phase") or "regular", format=batch["document"]["format"] if batch else "gender", teams=teams)
        if batch:
            managed.score_document(batch["document"], batch["document"], teams, official=batch["state"] in {"submitted", "approved"})
            if batch.get("approved_document"):
                managed.score_document(batch["approved_document"], batch["approved_document"], teams, official=True)
        meets.append({"id": old["id"], "details": managed.MeetDetails.model_validate(plan).model_dump(mode="json"),
                      "document": batch["document"] if batch else None, "approved_document": batch.get("approved_document") if batch else None,
                      "state": batch["state"] if batch else "planning"})
    new_details = managed.SeasonDetails(**{k: details[k] for k in ("name", "start_date", "end_date", "timezone", "divisions")}, clubs=managed_clubs).model_dump(mode="json")
    return {"details": new_details, "meets": meets}


def install_managed_interclub_routes(app, *, get_supabase_client):
    root = "/admin/clubs/{club_id}/interclub/managed"
    def admin(club_id, authorization, season_id=None):
        user = site_administrator(get_supabase_client, authorization, club_id)
        db = get_supabase_client()
        season = _one(db, SEASONS, SEASON_FIELDS, id=str(season_id), organizer_club_id=club_id) if season_id else None
        return user, db, season

    @app.get(root)
    def seasons(club_id: str, response: Response, authorization: str | None = auth_header()):
        response.headers.update(PRIVATE_HEADERS)
        _, db, _ = admin(club_id, authorization)
        rows = _rows(db, SEASONS, SEASON_FIELDS, organizer_club_id=club_id)
        rows.sort(key=lambda s: s["updated_at"], reverse=True)
        return {"seasons": rows}

    @app.get(root + "/{season_id}")
    def detail(club_id: str, season_id: UUID, response: Response, authorization: str | None = auth_header()):
        response.headers.update(PRIVATE_HEADERS)
        _, db, season = admin(club_id, authorization, season_id)
        return context(db, season)

    @app.put(root + "/{season_id}")
    def save_season(club_id: str, season_id: UUID, body: SaveSeason, response: Response, authorization: str | None = auth_header()):
        response.headers.update(PRIVATE_HEADERS); require_league_manager_write_or_403()
        user, db, _ = admin(club_id, authorization)
        existing = _rows(db, SEASONS, SEASON_FIELDS, id=str(season_id), organizer_club_id=club_id)
        if existing:
            plans = _rows(db, MEETS, MEET_FIELDS, season_id=str(season_id))
            new = body.details.model_dump(mode="json")
            # Historical roster snapshots are independent of today's pool.
            for meet in plans:
                if not set(meet["details"]["club_ids"]) <= {c["id"] for c in new["clubs"]} or not {t["division"] for t in meet["details"]["teams"]} <= set(new["divisions"]):
                    raise HTTPException(422, "Keep clubs and divisions already used in a scheduled meet.")
        saved = _write(db, season_id=season_id, action="season", revision=body.expected_revision,
                       payload={"details": body.details.model_dump(mode="json")}, user=user, club_id=club_id)
        return context(db, saved)

    @app.post(root + "/{season_id}/import")
    def import_season(club_id: str, season_id: UUID, response: Response, authorization: str | None = auth_header()):
        response.headers.update(PRIVATE_HEADERS); require_league_manager_write_or_403()
        user, db, _ = admin(club_id, authorization)
        existing = _rows(db, SEASONS, SEASON_FIELDS, id=str(season_id), organizer_club_id=club_id)
        if existing: return context(db, existing[0])
        payload = _value(import_legacy, db, club_id, str(season_id))
        saved = _write(db, season_id=season_id, action="import", revision=0, payload=payload, user=user, club_id=club_id)
        return context(db, saved)

    @app.post(root + "/{season_id}/visibility/{action}")
    def visibility(club_id: str, season_id: UUID, action: Literal["publish", "unpublish"], body: Revision, response: Response, authorization: str | None = auth_header()):
        response.headers.update(PRIVATE_HEADERS); require_league_manager_write_or_403()
        user, db, season = admin(club_id, authorization, season_id)
        if action == "publish" and len(season["details"]["clubs"]) < 2:
            raise HTTPException(422, "Add at least two participating clubs before publishing.")
        saved = _write(db, season_id=season_id, action=action, revision=body.expected_revision, user=user, club_id=club_id)
        return context(db, saved)

    @app.put(root + "/{season_id}/meets/{meet_id}")
    def save_meet(club_id: str, season_id: UUID, meet_id: UUID, body: SaveMeet, response: Response, authorization: str | None = auth_header()):
        response.headers.update(PRIVATE_HEADERS); require_league_manager_write_or_403()
        user, db, season = admin(club_id, authorization, season_id)
        others = _rows(db, MEETS, MEET_FIELDS, season_id=str(season_id))
        plan = _value(managed.validate_meet, body.details, season["details"], others, str(meet_id))
        _write(db, season_id=season_id, meet_id=meet_id, action="plan", revision=body.expected_revision,
               payload={"details": plan, "season_revision": body.season_revision}, user=user, club_id=club_id)
        return context(db, _one(db, SEASONS, SEASON_FIELDS, id=str(season_id)))

    @app.put(root + "/{season_id}/meets/{meet_id}/scores")
    def save_scores(club_id: str, season_id: UUID, meet_id: UUID, body: SaveScores, response: Response, authorization: str | None = auth_header()):
        response.headers.update(PRIVATE_HEADERS); require_league_manager_write_or_403()
        user, db, _ = admin(club_id, authorization, season_id)
        meet = _one(db, MEETS, MEET_FIELDS, id=str(meet_id), season_id=str(season_id))
        _write(db, season_id=season_id, meet_id=meet_id, action="scores", revision=body.expected_revision,
               payload=_score_payload(meet, body), user=user, club_id=club_id)
        return context(db, _one(db, SEASONS, SEASON_FIELDS, id=str(season_id)))

    @app.post(root + "/{season_id}/meets/{meet_id}/{action}")
    def meet_action(club_id: str, season_id: UUID, meet_id: UUID, action: Literal["generate", "submit", "approve", "reopen", "share", "revoke"], body: Action, response: Response, authorization: str | None = auth_header()):
        response.headers.update(PRIVATE_HEADERS); require_league_manager_write_or_403()
        user, db, season = admin(club_id, authorization, season_id)
        meet = _one(db, MEETS, MEET_FIELDS, id=str(meet_id), season_id=str(season_id))
        payload, token = {}, None
        if action == "generate": payload["document"] = _value(managed.generate, str(meet_id), meet["details"])
        elif action in {"submit", "approve"}:
            if not meet.get("document"): raise HTTPException(422, "Generate the meet schedule first.")
            _value(managed.score_document, meet["document"], meet["document"], meet["details"]["teams"], official=True)
        elif action == "reopen": payload["reason"] = body.reason
        elif action == "share":
            token = secrets.token_urlsafe(32)
            expiry = datetime.now(timezone.utc) + timedelta(days=90)
            if season["details"].get("end_date"):
                expiry = max(expiry, datetime.fromisoformat(season["details"]["end_date"]).replace(tzinfo=timezone.utc) + timedelta(days=30))
            payload = {"token_hash": hashlib.sha256(token.encode()).hexdigest(), "expires_at": expiry.isoformat()}
        _write(db, season_id=season_id, meet_id=meet_id, action=action, revision=body.expected_revision, payload=payload, user=user, club_id=club_id)
        result = context(db, _one(db, SEASONS, SEASON_FIELDS, id=str(season_id)))
        if token: result["host_token"] = token
        return result

    @app.post("/public/interclub-host/read")
    def host_read(body: HostRead, response: Response):
        response.headers.update(PRIVATE_HEADERS)
        db = get_supabase_client(); _, season, meet = _host(db, body.token)
        return host_context(season, meet)

    @app.post("/public/interclub-host/write")
    def host_write(body: HostWrite, response: Response):
        response.headers.update(PRIVATE_HEADERS); require_public_intake_or_403()
        db = get_supabase_client(); digest, season, meet = _host(db, body.token)
        if body.action == "scores" and body.document is None:
            raise HTTPException(422, "Include the scores to save.", headers=PRIVATE_HEADERS)
        # Submit uses the saved revision only, never an unacknowledged draft body.
        if body.action == "submit" and body.document is not None:
            raise HTTPException(422, "Save scores before submitting the meet.", headers=PRIVATE_HEADERS)
        payload = _score_payload(meet, body, official=body.action == "submit")
        _write(db, season_id=season["id"], meet_id=meet["id"], action=body.action, revision=body.expected_revision,
               payload=payload if body.action == "scores" else {}, token_hash=digest)
        # Recheck revocation before returning anything after the write.
        _, season, meet = _host(db, body.token)
        return host_context(season, meet)

    @app.get("/public/managed-interclub/{season_id}")
    def public_league(season_id: UUID, response: Response):
        response.headers.update(PRIVATE_HEADERS)
        db = get_supabase_client()
        season = _one(db, SEASONS, SEASON_FIELDS, id=str(season_id), published=True)
        return public_context(db, season)
