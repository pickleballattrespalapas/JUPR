"""Paper-first meet operations with exact-revision approval and isolated club access."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from fastapi import HTTPException
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from jupr_app.domain.admin.staff_policy import ADMIN_ROLES, permits
from jupr_app.domain import interclub_competition as engine
from services.api.admin_auth_routes import require_admin_assignments
from services.api.auth import auth_header
from services.api.interclub_competition_models import CompetitionDocument
from services.api.interclub_registration_routes import SEASON_FIELDS, MEET_FIELDS as REGISTRATION_MEET_FIELDS, TEAM_FIELDS, safe_roster
from services.api.interclub_registration_phase import registration_season, registration_state, require_season_phase

MEET_FIELDS = REGISTRATION_MEET_FIELDS

Phase = Literal["regular", "final", "qualifier"]
BATCH_FIELDS = "id,season_id,meet_id,phase,revision,state,document,roster_sources,ratings_status,ratings_error,approved_document,approved_revision,approved_at,updated_at"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Revision(StrictModel):
    expected_revision: int = Field(ge=0)


class CreateMeet(StrictModel):
    host_club_id: str = Field(min_length=1, max_length=100)
    club_ids: list[str] = Field(min_length=2, max_length=32)
    starts_at: AwareDatetime
    roster_deadline: AwareDatetime
    courts: int = Field(default=4, ge=1, le=100)
    duration_minutes: int = Field(default=180, ge=30, le=180)
    competition_phase: Phase = "regular"

    @model_validator(mode="after")
    def valid(self):
        if len(set(self.club_ids)) != len(self.club_ids):
            raise ValueError("Choose each club only once.")
        if self.competition_phase == "regular" and len(self.club_ids) > 4:
            raise ValueError("A regular meet has two to four clubs.")
        if self.roster_deadline <= datetime.now(timezone.utc) or self.roster_deadline > self.starts_at:
            raise ValueError("Choose a future roster deadline no later than the meet.")
        return self


class Generate(Revision):
    format: Literal["gender", "mixed", "mlp"] = "gender"
    division: str | None = Field(default=None, max_length=20)
    club_a: str | None = Field(default=None, max_length=100)
    club_b: str | None = Field(default=None, max_length=100)


class Save(Revision):
    document: CompetitionDocument


class Reopen(Revision):
    reason: str = Field(min_length=3, max_length=500)


class Reschedule(Reopen):
    starts_at: AwareDatetime
    roster_deadline: AwareDatetime

    @model_validator(mode="after")
    def dates(self):
        now = datetime.now(timezone.utc)
        if self.roster_deadline <= now or self.starts_at < self.roster_deadline:
            raise ValueError("Choose a future roster deadline no later than the replay date.")
        return self


def _rows(db, table, fields, **where):
    query = db.table(table).select(fields)
    for key, value in where.items():
        query = query.eq(key, value)
    return query.execute().data or []


def approved_documents(db, season_id):
    """Only approved canonical revisions may feed public results or rating projections."""
    return [row["approved_document"] for row in _rows(db, "pcs_interclub_competition_batches", "approved_document", season_id=str(season_id)) if row.get("approved_document")]


def public_competition(db, season_id, clubs=None):
    docs = approved_documents(db, season_id)
    standings = engine.league_standings(docs, clubs=clubs)
    # Entry IDs and names are public roster facts. Internal source revisions,
    # club-local player IDs, contacts, audit identities and errors are not.
    public_docs = deepcopy(docs)
    for document in public_docs:
        for encounter in document["encounters"]:
            for pairing in encounter["pairings"]:
                for game in pairing["games"]:
                    game.pop("injury_reason", None)
    return {"documents": public_docs, "standings": standings, "club_cup": engine.club_cup(docs, clubs=clubs)}


def _rpc(db, params, name="pcs_write_interclub_competition"):
    try:
        return db.rpc(name, params).execute().data
    except Exception as exc:
        code = str(getattr(exc, "code", ""))
        if code == "PT409":
            code = "40001"
        status, message = {
            "42501": (403, "Only the assigned host staff or league organizer can run this meet; approval and corrections require the organizer."),
            "40001": (409, "This meet, lineup or result revision changed. Reload before continuing."),
            "22023": (422, "Check the approved player pool, deadline ratings, lineups and complete meet scores."),
            "22P02": (422, "Choose valid season players and score identities."),
            "23505": (409, "This result or player assignment already exists. Reload before continuing."),
            "P0002": (404, "Meet or competition workspace unavailable."),
            "PT423": (423, "Meet planning is locked until season registration closes for all clubs."),
        }.get(code, (503, "Could not confirm the update. Reload before retrying."))
        raise HTTPException(status, message) from exc


def _validate(document, *, official=False):
    try:
        return engine.validate_document(document, official=official)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _approved_teams(db, season_id, meet_id):
    rows = _rows(db, "pcs_interclub_current_rosters", TEAM_FIELDS, season_id=str(season_id), meet_id=str(meet_id))
    return [row for row in rows if not row.get("withdrawn") and row.get("status") in {"eligible", "exception_approved"}]


def _sources(teams):
    return sorted([{"team_id": row["id"], "revision": row["revision"]} for row in teams], key=lambda row: row["team_id"])


def _eligible_players(db, season_id, meet):
    """Return selectable identities only; pool contacts never leave club ownership."""
    cutoff = datetime.fromisoformat(meet["roster_deadline"].replace("Z", "+00:00"))
    locked = cutoff <= datetime.now(timezone.utc)
    if locked:
        db.rpc("pcs_lock_interclub_meet_eligibility", {"p_meet_id": meet["id"]}).execute()
    snapshots = _rows(db, "pcs_interclub_meet_eligibility_snapshots", "entry_id,club_id,rating,gender,deadline", meet_id=meet["id"]) if locked else []
    snapshots = {r["entry_id"]: r for r in snapshots if datetime.fromisoformat(r["deadline"].replace("Z", "+00:00")) == cutoff}
    members = _rows(db, "pcs_interclub_pool_members", "id,club_id,player_id,name,approval_status,status", season_id=str(season_id))
    active = {r["id"]: r for r in members if r["approval_status"] == "approved" and r["status"] == "active" and r["club_id"] in meet["club_ids"]}
    entries = _rows(db, "pcs_interclub_entries", "id,club_id,player_id,pool_member_id,starting_rating", season_id=str(season_id))
    eligible = {}
    for club_id in meet["club_ids"]:
        selected = [row for row in entries if row["club_id"] == club_id and (row["id"] in snapshots if locked else row["pool_member_id"] in active)]
        if not selected:
            continue
        players = db.table("players").select("id,name,gender").eq("club_id", club_id).in_("id", [r["player_id"] for r in selected]).execute().data or []
        by_id = {str(p["id"]): p for p in players}
        for entry in selected:
            player = by_id.get(str(entry["player_id"]))
            if not player:
                continue
            snapshot = snapshots.get(entry["id"])
            rating = snapshot["rating"] if snapshot else db.rpc("pcs_interclub_rating_at", {"p_season_id": str(season_id), "p_entry_id": entry["id"], "p_cutoff": datetime.now(timezone.utc).isoformat()}).execute().data
            if rating is None:
                continue
            eligible.setdefault(club_id, []).append({"entry_id": entry["id"], "name": player["name"], "gender": snapshot["gender"] if snapshot else player["gender"],
                "starting_rating": entry["starting_rating"], "eligibility_rating": float(rating), "rating_deadline": meet["roster_deadline"], "rating_locked": locked})
    return eligible


def _display_players(db, season_id, meet, saved):
    if not saved:
        return []
    referenced = set()
    for encounter in saved["document"]["encounters"]:
        for pairing in encounter["pairings"]:
            for row in [pairing, *pairing["games"]]:
                for side in ("a", "b"):
                    referenced.update(row.get(f"players_{side}", []))
    if not referenced:
        return []
    entries = db.table("pcs_interclub_entries").select("id,club_id,player_id").eq("season_id", str(season_id)).in_("id", sorted(referenced)).execute().data or []
    result = []
    for club_id in meet["club_ids"]:
        selected = [row for row in entries if row["club_id"] == club_id]
        if not selected:
            continue
        players = db.table("players").select("id,name,gender").eq("club_id", club_id).in_("id", [r["player_id"] for r in selected]).execute().data or []
        by_id = {str(row["id"]): row for row in players}
        for entry in selected:
            player = by_id.get(str(entry["player_id"]))
            if player:
                result.append({"entry_id": entry["id"], "club_id": club_id, "name": player["name"], "gender": player["gender"]})
    return result


def _pairings(document):
    return {(encounter["id"], pairing["kind"]): pairing for encounter in document["encounters"] for pairing in encounter["pairings"]}


def _prepared_for_lineup_changes(document):
    return all(
        (g["status"] == "pending" and g.get("a") is None and g.get("b") is None)
        or (g["status"] in {"forfeit", "double_forfeit"} and (not p["players_a"] or not p["players_b"]))
        for e in document["encounters"] for p in e["pairings"] for g in p["games"]
    )


def _fixed_schedule(previous, current, current_deadline):
    """Scores are editable; an opponent or backdated eligibility snapshot is not."""
    before = {(row["id"], row["division"], row["club_a"], row["club_b"]) for row in previous["encounters"]}
    after = {(row["id"], row["division"], row["club_a"], row["club_b"]) for row in current["encounters"]}
    if before != after or previous["format"] != current["format"]:
        raise HTTPException(422, "Keep the generated schedule. Create a new draft schedule before entering results to change opponents.")
    old_pairs, new_pairs = _pairings(previous), _pairings(current)
    if old_pairs.keys() != new_pairs.keys():
        raise HTTPException(422, "Keep every scheduled doubles pairing in this meet.")
    prepared = _prepared_for_lineup_changes(previous)
    if prepared:
        permitted = {}
        chosen = {}
        for encounter in previous["encounters"]:
            for side in ("a", "b"):
                key = (encounter["division"], encounter[f"club_{side}"])
                permitted.setdefault(key, set()).update(player for pairing in encounter["pairings"] for player in pairing[f"players_{side}"])
        for encounter in current["encounters"]:
            for side in ("a", "b"):
                key = (encounter["division"], encounter[f"club_{side}"])
                players = {player for pairing in encounter["pairings"] for player in pairing[f"players_{side}"]}
                if players != permitted[key]:
                    raise HTTPException(422, "Arrange pairings using the four players on the prepared meet roster.")
                for pairing in encounter["pairings"]:
                    pairing_key = (*key, pairing["kind"])
                    lineup = frozenset(pairing[f"players_{side}"])
                    if pairing_key in chosen and chosen[pairing_key] != lineup:
                        raise HTTPException(422, "Keep the same starting doubles pairs against every opponent at this meet.")
                    chosen[pairing_key] = lineup
    for key, old in old_pairs.items():
        new = new_pairs[key]
        if not prepared and (old["players_a"] != new["players_a"] or old["players_b"] != new["players_b"]):
            raise HTTPException(422, "Keep the prepared lineup. Record an injury replacement on its actual game, or refresh replay lineups.")
        if old["id"] != new["id"] or [g["id"] for g in old["games"]] != [g["id"] for g in new["games"]]:
            raise HTTPException(422, "Keep the generated pairing and game identities when correcting scores.")
        if old.get("eligibility_deadline") != new.get("eligibility_deadline"):
            raise HTTPException(422, "A pairing's eligibility deadline is set by the meet or official reschedule.")
        deadline = old.get("eligibility_deadline")
        if deadline and datetime.fromisoformat(deadline.replace("Z", "+00:00")) != datetime.fromisoformat(current_deadline.replace("Z", "+00:00")) and old != new:
            raise HTTPException(422, "Completed pairings from before the reschedule remain official and cannot be edited in the replay.")


def install_interclub_competition_routes(app, *, get_supabase_client):
    def actor(club_id, authorization):
        db = get_supabase_client()
        user, assignments = require_admin_assignments(get_supabase_client=lambda: db, authorization=authorization, requested_club_id=club_id)
        return db, user, assignments

    def access(db, club_id, season_id):
        rows = _rows(db, "pcs_interclub_seasons", SEASON_FIELDS, id=str(season_id))
        if not rows:
            raise HTTPException(404, "Season unavailable.")
        season = rows[0]
        participations = _rows(db, "pcs_interclub_participations", "club_id,status", season_id=str(season_id))
        accepted = {row["club_id"] for row in participations if row["status"] == "accepted"}
        if season["organizer_club_id"] != club_id and club_id not in accepted:
            raise HTTPException(404, "Season unavailable for this club.")
        return season, accepted

    def meet_access(db, club_id, season, meet_id, assignments):
        rows = _rows(db, "pcs_interclub_meet_workspaces", MEET_FIELDS, season_id=season["id"], id=str(meet_id))
        if not rows or (club_id != season["organizer_club_id"] and club_id != rows[0]["host_club_id"] and club_id not in rows[0]["club_ids"]):
            raise HTTPException(404, "Meet unavailable for this club.")
        meet = rows[0]
        admin = any(row["role"] in ADMIN_ROLES for row in assignments)
        operator = any(row["role"] == "operator" and permits(row.get("scopes", []), "leagues", {str(meet_id), season["id"]}) for row in assignments)
        if not (admin or operator):
            raise HTTPException(403, "Meet staff access required.")
        organizer = club_id == season["organizer_club_id"] and admin
        can_manage = (club_id == meet["host_club_id"] or club_id == season["organizer_club_id"]) and (admin or operator)
        return meet, organizer, can_manage

    def batch(db, season_id, meet_id, phase):
        rows = _rows(db, "pcs_interclub_competition_batches", BATCH_FIELDS, season_id=str(season_id), meet_id=str(meet_id), phase=phase)
        return rows[0] if rows else None

    def context(club_id, season_id, meet_id, phase, authorization, *, mutation=False, organizer_only=False):
        db, user, assignments = actor(club_id, authorization)
        season, accepted = access(db, club_id, season_id)
        require_season_phase(season)
        meet, organizer, can_manage = meet_access(db, club_id, season, meet_id, assignments)
        if (meet.get("competition_phase") or "regular") != phase:
            raise HTTPException(422, "Choose the competition phase scheduled for this meet.")
        if mutation and not can_manage or organizer_only and not organizer:
            raise HTTPException(403, "League organizer access required." if organizer_only else "The host or league organizer runs this meet.")
        return db, user, season, accepted, meet, organizer, can_manage, batch(db, season_id, meet_id, phase)

    def write(db, user, club_id, season_id, meet_id, phase, action, revision, *, document=None, sources=None, reason=None, starts_at=None, deadline=None, qualification_sources=None):
        return _rpc(db, dict(p_actor_id=user.user_id, p_actor_email=user.email, p_club_id=club_id,
                            p_season_id=str(season_id), p_meet_id=str(meet_id), p_phase=phase,
                            p_action=action, p_revision=revision, p_document=document, p_roster_sources=sources,
                            p_reason=reason, p_starts_at=starts_at, p_deadline=deadline, p_qualification_sources=qualification_sources))

    def qualification_guard(db, season_id, document):
        if document["phase"] == "regular":
            return None
        rows = _rows(db, "pcs_interclub_competition_batches", "id,approved_revision,approved_document", season_id=str(season_id))
        approved = [row for row in rows if row.get("approved_document")]
        qualification_documents = [row["approved_document"] for row in approved if not (document["phase"] == "qualifier" and row["approved_document"]["meet_id"] == document["meet_id"] and row["approved_document"]["phase"] == "qualifier")]
        qualifications = engine.league_standings(qualification_documents).get("qualification", {})
        for encounter in document["encounters"]:
            qualification = qualifications.get(encounter["division"], {})
            selected = {encounter["club_a"], encounter["club_b"]}
            if document["phase"] == "final":
                allowed = set(qualification.get("qualifiers", []))
                if qualification.get("status") != "ready" or len(allowed) != 2 or selected != allowed:
                    raise HTTPException(409, "Standings changed or qualification needs a playoff. Reload the qualified finalists.")
            else:
                allowed = set(qualification.get("playoff_required", []))
                if not selected.issubset(allowed):
                    raise HTTPException(409, "Only clubs tied across the qualifying boundary need this playoff.")
        return sorted([{"id": row["id"], "revision": row["approved_revision"]} for row in approved], key=lambda row: row["id"])

    def require_revision(saved, revision):
        if not saved or saved["revision"] != revision:
            raise HTTPException(409, "Reload this competition workspace before continuing.")

    @app.get("/admin/clubs/{club_id}/interclub/competition")
    def seasons(club_id: str, authorization: str | None = auth_header()):
        db, _, assignments = actor(club_id, authorization)
        own = _rows(db, "pcs_interclub_participations", "season_id,club_id,status,revision", club_id=club_id)
        organized = _rows(db, "pcs_interclub_seasons", SEASON_FIELDS, organizer_club_id=club_id)
        ids = [row["season_id"] for row in own if row["status"] == "accepted"]
        invited = db.table("pcs_interclub_seasons").select(SEASON_FIELDS).in_("id", ids).execute().data if ids else []
        candidates = {row["id"]: row for row in organized + (invited or [])}
        if not any(row["role"] in ADMIN_ROLES for row in assignments):
            allowed = set()
            for sid, season in candidates.items():
                for meet in _rows(db, "pcs_interclub_meet_workspaces", MEET_FIELDS, season_id=sid):
                    if club_id not in {meet["host_club_id"], season["organizer_club_id"]}:
                        continue
                    try:
                        meet_access(db, club_id, season, meet["id"], assignments)
                        allowed.add(sid)
                    except HTTPException:
                        continue
            candidates = {sid: row for sid, row in candidates.items() if sid in allowed}
        participation = {row["season_id"]: row for row in own}
        return {"seasons": [{**registration_season(row), "participation": participation.get(sid)} for sid, row in candidates.items()]}

    @app.get("/admin/clubs/{club_id}/interclub/competition/{season_id}")
    def workspace(club_id: str, season_id: UUID, authorization: str | None = auth_header()):
        db, _, assignments = actor(club_id, authorization)
        season, accepted = access(db, club_id, season_id)
        organizer = club_id == season["organizer_club_id"] and any(row["role"] in ADMIN_ROLES for row in assignments)
        if not registration_state(season)["meet_planning_open"]:
            return {"season": registration_season(season), "clubs": [], "meets": [], "batches": [],
                    "is_organizer": organizer, "standings": {}, "club_cup": {}, "qualifying": {}}
        meets = _rows(db, "pcs_interclub_meet_workspaces", MEET_FIELDS, season_id=str(season_id))
        if club_id != season["organizer_club_id"]:
            meets = [meet for meet in meets if club_id in meet["club_ids"] or club_id == meet["host_club_id"]]
        permitted = []
        for meet in meets:
            try:
                meet_access(db, club_id, season, meet["id"], assignments)
                permitted.append(meet)
            except HTTPException as exc:
                if exc.status_code != 403:
                    raise
        ids = {meet["id"] for meet in permitted}
        batches = _rows(db, "pcs_interclub_competition_batches", BATCH_FIELDS, season_id=str(season_id))
        clubs = db.table("clubs").select("id,name,slug").in_("id", sorted(accepted | {season["organizer_club_id"]})).execute().data or []
        published = [row["approved_document"] for row in batches if row.get("approved_document")]
        standings = engine.league_standings(published, clubs=clubs)
        hidden = {meet["id"] for meet in permitted if club_id not in {meet["host_club_id"], season["organizer_club_id"]} and datetime.fromisoformat(meet["roster_deadline"].replace("Z", "+00:00")) > datetime.now(timezone.utc)}
        return {"season": registration_season(season), "clubs": clubs, "meets": permitted, "batches": [row for row in batches if row["meet_id"] in ids - hidden],
                "is_organizer": organizer, "standings": standings, "club_cup": engine.club_cup(published, clubs=clubs),
                "qualifying": standings.get("qualification", {})}

    @app.post("/admin/clubs/{club_id}/interclub/competition/{season_id}/meets")
    def create_meet(club_id: str, season_id: UUID, body: CreateMeet, authorization: str | None = auth_header()):
        db, user, assignments = actor(club_id, authorization)
        season, accepted = access(db, club_id, season_id)
        if season["organizer_club_id"] != club_id or not any(row["role"] in ADMIN_ROLES for row in assignments):
            raise HTTPException(403, "Only the organizer can schedule meets.")
        require_season_phase(season)
        if not set(body.club_ids).issubset(accepted) or body.host_club_id not in accepted:
            raise HTTPException(422, "Choose accepted clubs and an accepted host club.")
        meet = _rpc(db, {"p_actor_id": user.user_id, "p_actor_email": user.email, "p_club_id": club_id,
                        "p_season_id": str(season_id), "p_meet": body.model_dump(mode="json")}, "pcs_create_interclub_competition_meet")
        return {"meet": meet}

    @app.get("/admin/clubs/{club_id}/interclub/competition/{season_id}/meets/{meet_id}/{phase}")
    def detail(club_id: str, season_id: UUID, meet_id: UUID, phase: Phase, authorization: str | None = auth_header()):
        db, _, season, accepted, meet, organizer, can_manage, saved = context(club_id, season_id, meet_id, phase, authorization)
        hidden = not can_manage and datetime.fromisoformat(meet["roster_deadline"].replace("Z", "+00:00")) > datetime.now(timezone.utc)
        teams = [safe_roster(row, own_club=False) for row in _approved_teams(db, season_id, meet_id) if row["club_id"] in accepted and (not hidden or row["club_id"] == club_id)]
        visible_meet = {**meet, "club_ids": [club_id]} if hidden else meet
        eligible = _eligible_players(db, season_id, visible_meet)
        saved = None if hidden else saved
        return {"season": registration_season(season), "meet": meet, "batch": saved, "teams": teams, "eligible_players": eligible,
                "can_manage": can_manage, "is_organizer": organizer, "lineups_hidden": hidden, "display_players": _display_players(db, season_id, meet, saved)}

    @app.post("/admin/clubs/{club_id}/interclub/competition/{season_id}/meets/{meet_id}/{phase}/generate")
    def generate(club_id: str, season_id: UUID, meet_id: UUID, phase: Phase, body: Generate, authorization: str | None = auth_header()):
        db, user, season, accepted, meet, organizer, _, saved = context(club_id, season_id, meet_id, phase, authorization, mutation=True, organizer_only=phase != "regular")
        if saved and (saved["state"] != "draft" or any(game["status"] != "pending" for e in saved["document"]["encounters"] for p in e["pairings"] for game in p["games"])):
            raise HTTPException(409, "This schedule already contains scores. Continue its existing result batch.")
        teams = [row for row in _approved_teams(db, season_id, meet_id) if row["club_id"] in accepted]
        try:
            if phase == "regular":
                if body.format == "mlp":
                    raise ValueError("Regular meets use gender or mixed doubles.")
                document = engine.generate_round_robin(str(meet_id), teams, format=body.format, played_at=None, courts=meet["courts"])
            else:
                if not all([body.division, body.club_a, body.club_b]) or body.club_a == body.club_b:
                    raise ValueError("Choose a skill level and two different qualifying clubs.")
                qualification = engine.league_standings(approved_documents(db, season_id)).get("qualification", {}).get(body.division, {})
                selected = {body.club_a, body.club_b}
                allowed = set(qualification.get("qualifiers" if phase == "final" else "playoff_required", []))
                if phase == "final" and (qualification.get("status") != "ready" or len(allowed) != 2):
                    raise ValueError("Resolve the qualifying playoff before preparing the championship final.")
                if not selected.issubset(allowed):
                    raise ValueError("Choose the qualified finalists or the clubs requiring a qualification playoff.")
                by_club = {row["club_id"]: row for row in teams if row["division"] == body.division}
                if not selected.issubset(by_club):
                    raise ValueError("Both clubs must submit an eligible four-player lineup for this meet and skill level.")
                teams = [by_club[body.club_a], by_club[body.club_b]]
                document = engine.generate_championship(str(meet_id), body.division, teams[0], teams[1], phase=phase, played_at=None)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        for encounter in document["encounters"]:
            for pairing in encounter["pairings"]:
                pairing["eligibility_deadline"] = meet["roster_deadline"]
        sources = _sources(teams)
        if phase != "regular" and saved:
            if any(e["division"] == body.division and (phase == "final" or {e["club_a"], e["club_b"]} == {body.club_a, body.club_b}) for e in saved["document"]["encounters"]):
                raise HTTPException(409, "That skill level or qualifying pair already has a generated matchup.")
            for encounter in document["encounters"]:
                encounter["rotation"] = max(e["rotation"] for e in saved["document"]["encounters"]) + 1
            document["encounters"] = saved["document"]["encounters"] + document["encounters"]
            sources = sorted({s["team_id"]: s for s in saved["roster_sources"] + sources}.values(), key=lambda s: s["team_id"])
        result = write(db, user, club_id, season_id, meet_id, phase, "generate", body.expected_revision,
                       document=_validate(document), sources=sources, qualification_sources=qualification_guard(db, season_id, document))
        return {"batch": result}

    @app.put("/admin/clubs/{club_id}/interclub/competition/{season_id}/meets/{meet_id}/{phase}")
    def save(club_id: str, season_id: UUID, meet_id: UUID, phase: Phase, body: Save, authorization: str | None = auth_header()):
        db, user, _, _, meet, _, _, saved = context(club_id, season_id, meet_id, phase, authorization, mutation=True)
        require_revision(saved, body.expected_revision)
        document = _validate(body.document.model_dump(mode="json"))
        if document["meet_id"] != str(meet_id) or document["phase"] != phase:
            raise HTTPException(422, "Results must belong to this meet and competition phase.")
        if document["weather"] == "rescheduled" and saved["document"]["weather"] != "rescheduled":
            raise HTTPException(422, "Use the reschedule action to set the new date and roster deadline.")
        _fixed_schedule(saved["document"], document, meet["roster_deadline"])
        try:
            for encounter in document["encounters"]:
                for pairing in encounter["pairings"]:
                    for row in [pairing, *pairing["games"]]:
                        for side in ("a", "b"):
                            for entry_id in row.get(f"players_{side}", []):
                                UUID(entry_id)
        except ValueError as exc:
            raise HTTPException(422, "Choose players from the approved season entries.") from exc
        return {"batch": write(db, user, club_id, season_id, meet_id, phase, "save", body.expected_revision, document=document, sources=saved["roster_sources"])}

    @app.post("/admin/clubs/{club_id}/interclub/competition/{season_id}/meets/{meet_id}/{phase}/refresh-lineups")
    def refresh_lineups(club_id: str, season_id: UUID, meet_id: UUID, phase: Phase, body: Revision, authorization: str | None = auth_header()):
        db, user, _, accepted, meet, _, _, saved = context(club_id, season_id, meet_id, phase, authorization, mutation=True)
        require_revision(saved, body.expected_revision)
        is_replay = saved["document"]["weather"] == "rescheduled"
        before_play = datetime.fromisoformat(meet["starts_at"].replace("Z", "+00:00")) > datetime.now(timezone.utc) and _prepared_for_lineup_changes(saved["document"])
        if phase != "regular" or (not is_replay and not before_play):
            raise HTTPException(422, "Refresh lineups before play begins, or for unfinished pairings on the rescheduled date.")
        teams = [row for row in _approved_teams(db, season_id, meet_id) if row["club_id"] in accepted]
        by_club = {(row["division"], row["club_id"]): row for row in teams}
        document = deepcopy(saved["document"])
        for encounter in document["encounters"]:
            keys = [(encounter["division"], encounter["club_a"]), (encounter["division"], encounter["club_b"])]
            unfinished = [pairing for pairing in encounter["pairings"] if not is_replay or any(g["status"] == "pending" for g in pairing["games"])]
            if not unfinished:
                continue
            if not all(key in by_club for key in keys):
                raise HTTPException(422, "Both clubs must submit eligible replay lineups for each unfinished skill-level matchup.")
            try:
                fresh = engine.generate_round_robin(str(meet_id), [by_club[key] for key in keys], format=document["format"], played_at=None, courts=meet["courts"])
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            fresh_enc = fresh["encounters"][0]
            reversed_sides = fresh_enc["club_a"] != encounter["club_a"]
            for pairing in unfinished:
                if any((g["status"] != "pending" or g.get("a") is not None or g.get("b") is not None)
                       and not (not is_replay and g["status"] in {"forfeit", "double_forfeit"} and (not pairing["players_a"] or not pairing["players_b"])) for g in pairing["games"]):
                    raise HTTPException(409, "Play has begun; only injury substitutions between games are allowed.")
                replacement = next(p for p in fresh_enc["pairings"] if p["kind"] == pairing["kind"])
                pairing["players_a"] = replacement["players_b" if reversed_sides else "players_a"]
                pairing["players_b"] = replacement["players_a" if reversed_sides else "players_b"]
                for game, fresh_game in zip(pairing["games"], replacement["games"]):
                    winner = fresh_game.get("winner")
                    if reversed_sides and winner in {"a", "b"}:
                        winner = "b" if winner == "a" else "a"
                    game.update(status=fresh_game["status"], winner=winner, a=None, b=None, players_a=[], players_b=[], played_at=None, injury_reason=None)
        return {"batch": write(db, user, club_id, season_id, meet_id, phase, "refresh_lineups", body.expected_revision,
                               document=_validate(document), sources=_sources(teams))}

    @app.post("/admin/clubs/{club_id}/interclub/competition/{season_id}/meets/{meet_id}/{phase}/submit")
    def submit(club_id: str, season_id: UUID, meet_id: UUID, phase: Phase, body: Revision, authorization: str | None = auth_header()):
        db, user, _, _, _, _, _, saved = context(club_id, season_id, meet_id, phase, authorization, mutation=True)
        require_revision(saved, body.expected_revision)
        _validate(saved["document"], official=True)
        return {"batch": write(db, user, club_id, season_id, meet_id, phase, "submit", body.expected_revision, qualification_sources=qualification_guard(db, season_id, saved["document"]))}

    @app.post("/admin/clubs/{club_id}/interclub/competition/{season_id}/meets/{meet_id}/{phase}/approve")
    def approve(club_id: str, season_id: UUID, meet_id: UUID, phase: Phase, body: Revision, authorization: str | None = auth_header()):
        db, user, _, _, _, _, _, saved = context(club_id, season_id, meet_id, phase, authorization, mutation=True, organizer_only=True)
        require_revision(saved, body.expected_revision)
        _validate(saved["document"], official=True)
        approved = write(db, user, club_id, season_id, meet_id, phase, "approve", body.expected_revision, qualification_sources=qualification_guard(db, season_id, saved["document"]))
        from jupr_app.services.interclub_rating_service import process_interclub_ratings
        rating_result = process_interclub_ratings(db, approved)
        current = batch(db, season_id, meet_id, phase) or approved
        return {"batch": current, "ratings": rating_result}

    @app.post("/admin/clubs/{club_id}/interclub/competition/{season_id}/meets/{meet_id}/{phase}/retry-ratings")
    def retry_ratings(club_id: str, season_id: UUID, meet_id: UUID, phase: Phase, body: Revision, authorization: str | None = auth_header()):
        db, _, _, _, _, _, _, saved = context(club_id, season_id, meet_id, phase, authorization, mutation=True, organizer_only=True)
        require_revision(saved, body.expected_revision)
        if saved["state"] != "approved":
            raise HTTPException(409, "Approve this exact result revision before applying ratings.")
        from jupr_app.services.interclub_rating_service import process_interclub_ratings
        result = process_interclub_ratings(db, saved)
        return {"batch": batch(db, season_id, meet_id, phase) or saved, "ratings": result}

    @app.post("/admin/clubs/{club_id}/interclub/competition/{season_id}/meets/{meet_id}/{phase}/reopen")
    def reopen(club_id: str, season_id: UUID, meet_id: UUID, phase: Phase, body: Reopen, authorization: str | None = auth_header()):
        db, user, _, _, _, _, _, saved = context(club_id, season_id, meet_id, phase, authorization, mutation=True, organizer_only=True)
        require_revision(saved, body.expected_revision)
        return {"batch": write(db, user, club_id, season_id, meet_id, phase, "reopen", body.expected_revision, reason=body.reason)}

    @app.post("/admin/clubs/{club_id}/interclub/competition/{season_id}/meets/{meet_id}/{phase}/reschedule")
    def reschedule(club_id: str, season_id: UUID, meet_id: UUID, phase: Phase, body: Reschedule, authorization: str | None = auth_header()):
        db, user, _, _, _, _, _, saved = context(club_id, season_id, meet_id, phase, authorization, mutation=True, organizer_only=True)
        require_revision(saved, body.expected_revision)
        if phase != "regular":
            raise HTTPException(422, "Championship and qualification matches must finish as a full MLP matchup.")
        try:
            document = engine.prepare_reschedule(saved["document"], played_at=None, eligibility_deadline=body.roster_deadline.isoformat())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"batch": write(db, user, club_id, season_id, meet_id, phase, "reschedule", body.expected_revision,
                               document=document, reason=body.reason, starts_at=body.starts_at.isoformat(), deadline=body.roster_deadline.isoformat())}
