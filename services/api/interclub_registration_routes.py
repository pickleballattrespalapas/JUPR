"""Club-owned participation/meet rosters and organizer-only eligibility decisions."""
from typing import Literal
from uuid import UUID

from fastapi import HTTPException, Query
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, PositiveInt, model_validator

from jupr_app.domain.admin.staff_policy import ADMIN_ROLES
from services.api.admin_auth_routes import require_admin_assignments
from services.api.auth import auth_header

SEASON_FIELDS = "id,organizer_club_id,source_revision,details,rules,opened_at"
PARTICIPATION_FIELDS = "season_id,club_id,status,revision,updated_at"
MEET_FIELDS = "id,season_id,plan_index,host_club_id,club_ids,starts_at,duration_minutes,courts,roster_deadline,revision,roster_open,deadline_editable"
TEAM_FIELDS = "id,season_id,meet_id,club_id,division,name,revision,withdrawn,created_at,updated_at,roster,issues,status,late_change,submitted_at,decision_reason,decided_at"
HISTORY_FIELDS = "team_id,revision,name,roster,issues,status,late_change,submitted_at,decision_reason,decided_at"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class DivisionRule(StrictModel):
    min_rating: float | None = Field(default=None, ge=1, le=7)
    max_rating: float | None = Field(default=None, ge=1, le=7)
    women_required: int | None = Field(default=None, ge=0, le=4)

    @model_validator(mode="after")
    def ordered(self):
        if self.min_rating is not None and self.max_rating is not None and self.min_rating > self.max_rating:
            raise ValueError("Minimum rating cannot exceed maximum rating.")
        return self


class OpenRegistration(StrictModel):
    expected_revision: int = Field(ge=1)
    rules: dict[str, DivisionRule] = Field(min_length=1, max_length=8)


class ParticipationUpdate(StrictModel):
    expected_revision: int = Field(ge=1)
    action: Literal["accept", "decline", "cancel", "reinvite"]


class RosterUpdate(StrictModel):
    expected_meet_revision: int = Field(ge=1)
    expected_revision: int = Field(ge=0)
    name: str = Field(min_length=1, max_length=80)
    division: str = Field(min_length=1, max_length=20)
    player_ids: list[PositiveInt] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def distinct(self):
        if len(set(self.player_ids)) != 4:
            raise ValueError("Choose four different players.")
        return self


class RosterRevision(StrictModel):
    expected_meet_revision: int = Field(ge=1)
    expected_revision: int = Field(ge=1)


class MeetDeadlineUpdate(StrictModel):
    expected_revision: int = Field(ge=1)
    roster_deadline: AwareDatetime


class EligibilityDecision(RosterRevision):
    approve: bool
    reason: str = Field(min_length=1, max_length=500)


def safe_roster(row, *, own_club: bool):
    # Even a future private column in the DB must not leak through snapshots.
    fields = set((TEAM_FIELDS + "," + HISTORY_FIELDS).split(","))
    result = {k: v for k, v in row.items() if k in fields}
    allowed = ["entry_id", "name", "starting_rating", "gender"] + (["player_id"] if own_club else [])
    result["roster"] = [{k: entry.get(k) for k in allowed} for entry in row.get("roster", [])]
    result["issues"] = [{k: issue.get(k) for k in ("code", "message")} for issue in row.get("issues", [])]
    return result


def call_rpc(db, name, params):
    try:
        return db.rpc(name, params).execute().data
    except Exception as exc:
        code = getattr(exc, "code", "")
        if code == "42501":
            raise HTTPException(403, "Your club account cannot perform this action. Check its administrator access and season invitation.") from exc
        if code == "40001":
            raise HTTPException(409, "This invitation, meet or roster changed, or the meet has started. Reload before continuing.") from exc
        if code == "23505":
            raise HTTPException(409, "A player or team name is already used by another team in this club, meet and division.") from exc
        if code == "22023":
            raise HTTPException(422, "Check the rules, meet deadline and four active club players. Each new season player needs a starting club rating.") from exc
        if code == "P0002":
            raise HTTPException(404, "Season, meet, invitation or team unavailable.") from exc
        raise HTTPException(503, "Could not confirm the update. Reload before retrying.") from exc


def install_interclub_registration_routes(app, *, get_supabase_client):
    def administrator(club_id, authorization):
        db = get_supabase_client()
        user, assignments = require_admin_assignments(get_supabase_client=lambda: db, authorization=authorization, requested_club_id=club_id)
        if not any(row["role"] in ADMIN_ROLES for row in assignments):
            raise HTTPException(403, "Club administrator access required.")
        return db, user

    def access(db, club_id, season_id):
        rows = db.table("pcs_interclub_seasons").select(SEASON_FIELDS).eq("id", str(season_id)).limit(1).execute().data or []
        if not rows:
            raise HTTPException(404, "Season unavailable.")
        season = rows[0]
        own = db.table("pcs_interclub_participations").select(PARTICIPATION_FIELDS).eq("season_id", str(season_id)).eq("club_id", club_id).limit(1).execute().data or []
        if season["organizer_club_id"] != club_id and not own:
            raise HTTPException(404, "Season unavailable for this club.")
        return season, own[0] if own else None

    def actor_params(user, club_id, season_id):
        return dict(p_actor_id=user.user_id, p_actor_email=user.email, p_club_id=club_id, p_season_id=str(season_id))

    @app.get("/admin/clubs/{club_id}/interclub/registrations")
    def registrations(club_id: str, authorization: str | None = auth_header()):
        db, _ = administrator(club_id, authorization)
        own = db.table("pcs_interclub_participations").select(PARTICIPATION_FIELDS).eq("club_id", club_id).order("updated_at", desc=True).limit(100).execute().data or []
        organized = db.table("pcs_interclub_seasons").select(SEASON_FIELDS).eq("organizer_club_id", club_id).order("opened_at", desc=True).limit(100).execute().data or []
        invited = db.table("pcs_interclub_seasons").select(SEASON_FIELDS).in_("id", [p["season_id"] for p in own]).execute().data if own else []
        by_id = {s["id"]: s for s in [*organized, *(invited or [])]}
        by_season = {p["season_id"]: p for p in own}
        return {"seasons": [{**s, "participation": by_season.get(s["id"])} for s in sorted(by_id.values(), key=lambda s: s["opened_at"], reverse=True)]}

    @app.post("/admin/clubs/{club_id}/interclub/registrations/{season_id}/open")
    def open_registration(club_id: str, season_id: UUID, payload: OpenRegistration, authorization: str | None = auth_header()):
        db, user = administrator(club_id, authorization)
        season = call_rpc(db, "pcs_open_interclub_meet_registration", {**actor_params(user, club_id, season_id),
            "p_revision": payload.expected_revision, "p_rules": {d: r.model_dump() for d, r in payload.rules.items()}})
        return {"season": {key: season.get(key) for key in SEASON_FIELDS.split(",")}}

    @app.get("/admin/clubs/{club_id}/interclub/registrations/{season_id}")
    def detail(club_id: str, season_id: UUID, team_offset: int = Query(default=0, ge=0, le=100000), authorization: str | None = auth_header()):
        db, _ = administrator(club_id, authorization)
        season, own = access(db, club_id, season_id)
        organizer = season["organizer_club_id"] == club_id
        participations = db.table("pcs_interclub_participations").select(PARTICIPATION_FIELDS).eq("season_id", str(season_id)).execute().data or [] if organizer else ([own] if own else [])
        query = db.table("pcs_interclub_current_rosters").select(TEAM_FIELDS).eq("season_id", str(season_id)).is_("meet_id", "null")
        if not organizer:
            query = query.eq("club_id", club_id)
        teams = query.order("id").range(team_offset, team_offset+100).execute().data or []
        clubs = db.table("clubs").select("id,name,slug").in_("id", list(set(season["details"]["club_ids"] + [season["organizer_club_id"]]))).execute().data or []
        meet_query = db.table("pcs_interclub_meet_workspaces").select(MEET_FIELDS).eq("season_id", str(season_id))
        if not organizer:
            meet_query = meet_query.contains("club_ids", [club_id])
        meets = meet_query.order("starts_at").order("id").limit(100).execute().data or []
        return {"season": season, "meets": meets, "is_organizer": organizer, "own_participation": own, "participations": participations, "clubs": clubs,
                "teams": [safe_roster(row, own_club=row["club_id"] == club_id) for row in teams[:100]],
                "next_team_offset": team_offset+100 if len(teams)>100 else None}

    @app.post("/admin/clubs/{club_id}/interclub/registrations/{season_id}/participations/{target_club_id}")
    def respond(club_id: str, season_id: UUID, target_club_id: str, payload: ParticipationUpdate, authorization: str | None = auth_header()):
        db, user = administrator(club_id, authorization)
        row = call_rpc(db, "pcs_interclub_participation", {**actor_params(user, club_id, season_id), "p_target_club_id": target_club_id,
                       "p_revision": payload.expected_revision, "p_action": payload.action})
        return {"participation": {key: row.get(key) for key in PARTICIPATION_FIELDS.split(",")}}

    def meet_access(db, club_id, season, meet_id):
        rows = db.table("pcs_interclub_meet_workspaces").select(MEET_FIELDS).eq("id", str(meet_id)).eq("season_id", season["id"]).limit(1).execute().data or []
        if not rows or (season["organizer_club_id"] != club_id and club_id not in rows[0]["club_ids"]):
            raise HTTPException(404, "Meet unavailable for this club.")
        return rows[0]

    @app.get("/admin/clubs/{club_id}/interclub/registrations/{season_id}/meets/{meet_id}")
    def meet_detail(club_id: str, season_id: UUID, meet_id: UUID, team_offset: int = Query(default=0, ge=0, le=100000), authorization: str | None = auth_header()):
        db, _ = administrator(club_id, authorization)
        season, _ = access(db, club_id, season_id)
        meet = meet_access(db, club_id, season, meet_id)
        query = db.table("pcs_interclub_current_rosters").select(TEAM_FIELDS).eq("season_id", str(season_id)).eq("meet_id", str(meet_id))
        if season["organizer_club_id"] != club_id:
            query = query.eq("club_id", club_id)
        teams = query.order("id").range(team_offset, team_offset+100).execute().data or []
        return {"meet": meet, "teams": [safe_roster(row, own_club=row["club_id"] == club_id) for row in teams[:100]],
                "next_team_offset": team_offset+100 if len(teams)>100 else None}

    @app.put("/admin/clubs/{club_id}/interclub/registrations/{season_id}/meets/{meet_id}/deadline")
    def meet_deadline(club_id: str, season_id: UUID, meet_id: UUID, payload: MeetDeadlineUpdate, authorization: str | None = auth_header()):
        db, user = administrator(club_id, authorization)
        row = call_rpc(db, "pcs_set_interclub_meet_deadline", {**actor_params(user, club_id, season_id),
            "p_meet_id": str(meet_id), "p_revision": payload.expected_revision, "p_deadline": payload.roster_deadline.isoformat()})
        return {"meet": {key: row.get(key) for key in MEET_FIELDS.split(",") if key not in ("roster_open", "deadline_editable")}}

    @app.get("/admin/clubs/{club_id}/interclub/registrations/{season_id}/meets/{meet_id}/players")
    def meet_players(club_id: str, season_id: UUID, meet_id: UUID, q: str = Query(default="", max_length=80), offset: int = Query(default=0, ge=0, le=100000), authorization: str | None = auth_header()):
        db, _ = administrator(club_id, authorization)
        season, own = access(db, club_id, season_id)
        meet = meet_access(db, club_id, season, meet_id)
        if not meet["roster_open"]:
            raise HTTPException(409, "This meet has started. Its rosters are now history.")
        return player_choices(db, club_id, season_id, own, q, offset)

    @app.get("/admin/clubs/{club_id}/interclub/registrations/{season_id}/players")
    def players(club_id: str, season_id: UUID, q: str = Query(default="", max_length=80), offset: int = Query(default=0, ge=0, le=100000), authorization: str | None = auth_header()):
        db, _ = administrator(club_id, authorization)
        _, own = access(db, club_id, season_id)
        return player_choices(db, club_id, season_id, own, q, offset)

    def player_choices(db, club_id, season_id, own, q, offset):
        if not own or own["status"] != "accepted":
            raise HTTPException(403, "Accept your club's season invitation before choosing players.")
        query = db.table("players").select("id,name,rating,gender").eq("club_id", club_id).eq("active", True)
        if q.strip():
            term = q.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            query = query.ilike("name", f"%{term}%")
        rows = query.order("name").order("id").range(offset, offset+100).execute().data or []
        entries = db.table("pcs_interclub_entries").select("player_id,starting_rating").eq("season_id", str(season_id)).eq("club_id", club_id).in_("player_id", [r["id"] for r in rows[:100]]).execute().data if rows else []
        seeds = {str(e["player_id"]): e["starting_rating"] for e in entries or []}
        return {"players": [{"id": str(row["id"]), "name": row["name"], "starting_rating": seeds.get(str(row["id"]), float(row["rating"])/400 if row.get("rating") is not None else None)} for row in rows[:100]],
                "next_offset": offset+100 if len(rows)>100 else None}

    @app.put("/admin/clubs/{club_id}/interclub/registrations/{season_id}/teams/{team_id}")
    @app.post("/admin/clubs/{club_id}/interclub/registrations/{season_id}/teams/{team_id}/withdraw")
    @app.post("/admin/clubs/{club_id}/interclub/registrations/{season_id}/teams/{team_id}/eligibility")
    def retired_season_roster(club_id: str, season_id: UUID, team_id: UUID, authorization: str | None = auth_header()):
        administrator(club_id, authorization)
        raise HTTPException(409, "Rosters now belong to individual meets. Reload and choose a meet.")

    @app.put("/admin/clubs/{club_id}/interclub/registrations/{season_id}/meets/{meet_id}/teams/{team_id}")
    def save_roster(club_id: str, season_id: UUID, meet_id: UUID, team_id: UUID, payload: RosterUpdate, authorization: str | None = auth_header()):
        db, user = administrator(club_id, authorization)
        saved = call_rpc(db, "pcs_save_interclub_meet_roster", {**actor_params(user, club_id, season_id), "p_meet_id": str(meet_id), "p_meet_revision": payload.expected_meet_revision, "p_team_id": str(team_id),
            "p_revision": payload.expected_revision, "p_name": payload.name, "p_division": payload.division, "p_player_ids": payload.player_ids})
        return {"team": safe_roster({**saved["team"], **saved["roster"]}, own_club=True)}

    @app.post("/admin/clubs/{club_id}/interclub/registrations/{season_id}/meets/{meet_id}/teams/{team_id}/withdraw")
    def withdraw(club_id: str, season_id: UUID, meet_id: UUID, team_id: UUID, payload: RosterRevision, authorization: str | None = auth_header()):
        db, user = administrator(club_id, authorization)
        saved = call_rpc(db, "pcs_save_interclub_meet_roster", {**actor_params(user, club_id, season_id), "p_meet_id": str(meet_id), "p_meet_revision": payload.expected_meet_revision, "p_team_id": str(team_id),
            "p_revision": payload.expected_revision, "p_name": "", "p_division": "", "p_player_ids": [], "p_withdraw": True})
        return {"team": safe_roster({**saved["team"], **saved["roster"]}, own_club=True)}

    @app.post("/admin/clubs/{club_id}/interclub/registrations/{season_id}/meets/{meet_id}/teams/{team_id}/eligibility")
    def decide(club_id: str, season_id: UUID, meet_id: UUID, team_id: UUID, payload: EligibilityDecision, authorization: str | None = auth_header()):
        db, user = administrator(club_id, authorization)
        saved = call_rpc(db, "pcs_review_interclub_meet_roster", {**actor_params(user, club_id, season_id), "p_meet_id": str(meet_id), "p_meet_revision": payload.expected_meet_revision, "p_team_id": str(team_id),
            "p_revision": payload.expected_revision, "p_approve": payload.approve, "p_reason": payload.reason})
        return {"team": safe_roster({**saved["team"], **saved["roster"]}, own_club=saved["team"]["club_id"] == club_id)}

    @app.get("/admin/clubs/{club_id}/interclub/registrations/{season_id}/meets/{meet_id}/teams/{team_id}/history")
    @app.get("/admin/clubs/{club_id}/interclub/registrations/{season_id}/teams/{team_id}/history")
    def history(club_id: str, season_id: UUID, team_id: UUID, meet_id: UUID | None = None, authorization: str | None = auth_header()):
        db, _ = administrator(club_id, authorization)
        season, _ = access(db, club_id, season_id)
        teams = db.table("pcs_interclub_teams").select("id,club_id,meet_id").eq("id", str(team_id)).eq("season_id", str(season_id)).limit(1).execute().data or []
        if meet_id is not None:
            meet_access(db, club_id, season, meet_id)
        if not teams or (meet_id is not None and teams[0]["meet_id"] != str(meet_id)) or (season["organizer_club_id"] != club_id and teams[0]["club_id"] != club_id):
            raise HTTPException(404, "Team unavailable for this club.")
        rows = db.table("pcs_interclub_roster_versions").select(HISTORY_FIELDS).eq("team_id", str(team_id)).order("revision", desc=True).limit(50).execute().data or []
        return {"history": [safe_roster(row, own_club=teams[0]["club_id"] == club_id) for row in rows]}
