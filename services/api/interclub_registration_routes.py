"""Club-owned participation/meet rosters and organizer-only eligibility decisions."""
from datetime import datetime, timezone
import json
from typing import Literal
from uuid import UUID

from fastapi import HTTPException, Query
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, PositiveInt, ValidationError, model_validator

from jupr_app.domain.admin.staff_policy import ADMIN_ROLES
from services.api.admin_auth_routes import require_admin_assignments
from services.api.auth import auth_header
from services.api.interclub_models import DivisionRule, SeasonDraft, canonical_southern_bcs_rules
from jupr_app.services.interclub_registration_phase import REGISTRATION_FIELDS
from services.api.interclub_registration_phase import registration_season, registration_state, require_season_phase

SEASON_FIELDS = "id,organizer_club_id,source_revision,details,rules,opened_at," + REGISTRATION_FIELDS
PARTICIPATION_FIELDS = "season_id,club_id,status,revision,updated_at"
MEET_FIELDS = "id,season_id,plan_index,host_club_id,club_ids,starts_at,duration_minutes,courts,roster_deadline,revision,roster_open,deadline_editable,competition_phase"
TEAM_FIELDS = "id,season_id,meet_id,club_id,division,name,revision,withdrawn,created_at,updated_at,roster,issues,status,late_change,submitted_at,decision_reason,decided_at"
HISTORY_FIELDS = "team_id,revision,name,roster,issues,status,late_change,submitted_at,decision_reason,decided_at"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class OpenRegistration(StrictModel):
    expected_revision: int = Field(ge=1)
    rules: dict[str, DivisionRule] = Field(min_length=1, max_length=8)


class ParticipationUpdate(StrictModel):
    expected_revision: int = Field(ge=1)
    action: Literal["accept", "decline", "cancel", "reinvite"]


class RegistrationWindowUpdate(StrictModel):
    expected_revision: int = Field(ge=0)
    opens_at: AwareDatetime
    closes_at: AwareDatetime

    @model_validator(mode="after")
    def ordered(self):
        if self.closes_at <= self.opens_at:
            raise ValueError("Registration must close after it opens.")
        return self


class RosterUpdate(StrictModel):
    expected_meet_revision: int = Field(ge=1)
    expected_revision: int = Field(ge=0)
    name: str = Field(min_length=1, max_length=80)
    division: str = Field(min_length=1, max_length=20)
    player_ids: list[PositiveInt] = Field(min_length=2, max_length=4)
    missing_pairing_forfeit: bool = False

    @model_validator(mode="after")
    def distinct(self):
        expected = 2 if self.missing_pairing_forfeit else 4
        if len(self.player_ids) != expected or len(set(self.player_ids)) != expected:
            raise ValueError("Choose two different players and confirm the missing pairing forfeit, or choose a full four-player team.")
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
    allowed = ["entry_id", "name", "starting_rating", "eligibility_rating", "rating_deadline", "rating_locked", "gender"] + (["player_id"] if own_club else [])
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
        if code == "PT423":
            raise HTTPException(423, "This action is locked by the season registration window. Reload the league workspace for its current dates.") from exc
        if code in {"40001", "PT409"}:
            raise HTTPException(409, "This invitation, meet or roster changed, or the meet has started. Reload before continuing.") from exc
        if code == "23505":
            raise HTTPException(409, "A player or team name is already used by another team in this club, meet and division.") from exc
        if code == "22023":
            raise HTTPException(422, "Choose approved season-pool players in the correct skill level: a full team of two women and two men, or two players with the missing pairing declared as a forfeit.") from exc
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
        return {"seasons": [{**registration_season(s), "participation": by_season.get(s["id"])} for s in sorted(by_id.values(), key=lambda s: s["opened_at"], reverse=True)]}

    @app.post("/admin/clubs/{club_id}/interclub/registrations/{season_id}/open")
    def open_registration(club_id: str, season_id: UUID, payload: OpenRegistration, authorization: str | None = auth_header()):
        db, user = administrator(club_id, authorization)
        rows = db.table("pcs_interclub_drafts").select("id,revision,draft").eq("id", str(season_id)).eq("organizer_club_id", club_id).limit(1).execute().data or []
        if not rows:
            raise HTTPException(404, "Season setup unavailable for this organizer.")
        saved = rows[0]
        if saved["revision"] != payload.expected_revision:
            raise HTTPException(409, "This setup changed. Reload it before opening invitations.")
        try:
            draft = SeasonDraft.model_validate(saved["draft"])
        except ValidationError as exc:
            messages = [error["msg"].removeprefix("Value error, ") for error in exc.errors()]
            raise HTTPException(422, "Complete season setup before opening invitations: " + "; ".join(messages[:3])) from exc
        if len(draft.club_ids) < 2:
            raise HTTPException(422, "Select at least two participating clubs before opening invitations.")
        if not any(meet.starts_at > datetime.now(timezone.utc) for meet in draft.meets):
            raise HTTPException(422, "Schedule at least one upcoming meet before opening invitations.")
        rules = {d: r.model_dump() for d, r in payload.rules.items()}
        if set(rules) != set(draft.divisions):
            raise HTTPException(422, "Review eligibility rules for each selected division.")
        # Earlier saved drafts allowed arbitrary bands and team composition.
        # Normalize these historical form values to the confirmed league rules.
        rules = {d: rule.model_dump() for d, rule in canonical_southern_bcs_rules(draft.divisions).items()}
        season = call_rpc(db, "pcs_open_interclub_meet_registration", {**actor_params(user, club_id, season_id),
            "p_revision": payload.expected_revision, "p_rules": rules})
        return {"season": registration_season({key: season.get(key) for key in SEASON_FIELDS.split(",")})}

    @app.put("/admin/clubs/{club_id}/interclub/registrations/{season_id}/registration-window")
    def update_registration_window(club_id: str, season_id: UUID, payload: RegistrationWindowUpdate, authorization: str | None = auth_header()):
        db, user = administrator(club_id, authorization)
        rows = db.table("pcs_interclub_seasons").select(SEASON_FIELDS).eq("id", str(season_id)).limit(1).execute().data or []
        if not rows:
            raise HTTPException(404, "Season unavailable.")
        season = rows[0]
        if season["organizer_club_id"] != club_id:
            raise HTTPException(403, "Only the league commissioner can set registration dates for all clubs.")
        if int(season.get("registration_revision") or 0) != payload.expected_revision:
            raise HTTPException(409, "Registration dates changed. Reload before saving.")
        try:
            updated = db.rpc("pcs_set_interclub_registration_window", {**actor_params(user, club_id, season_id),
                "p_revision": payload.expected_revision, "p_opens_at": payload.opens_at.isoformat(), "p_closes_at": payload.closes_at.isoformat()}).execute().data
        except Exception as exc:
            code = getattr(exc, "code", "")
            if code == "42501":
                raise HTTPException(403, "Only the league commissioner can set registration dates for all clubs.") from exc
            if code in {"40001", "PT409"}:
                raise HTTPException(409, "Registration dates changed. Reload before saving.") from exc
            if code == "PT423":
                raise HTTPException(423, "The registration window is currently locked. Reload the season before changing its dates.") from exc
            if code == "22023":
                raise HTTPException(422, "Registration must close after it opens and no later than the first meet.") from exc
            if code == "P0002":
                raise HTTPException(404, "Season unavailable.") from exc
            raise HTTPException(503, "Could not confirm the registration dates. Reload before retrying.") from exc
        return {"season": registration_season({**season, **{key: updated.get(key) for key in REGISTRATION_FIELDS.split(",")}})}

    @app.get("/admin/clubs/{club_id}/interclub/registrations/{season_id}")
    def detail(club_id: str, season_id: UUID, team_offset: int = Query(default=0, ge=0, le=100000), authorization: str | None = auth_header()):
        db, _ = administrator(club_id, authorization)
        season, own = access(db, club_id, season_id)
        organizer = season["organizer_club_id"] == club_id
        participations = db.table("pcs_interclub_participations").select(PARTICIPATION_FIELDS).eq("season_id", str(season_id)).execute().data or [] if organizer else ([own] if own else [])
        clubs = db.table("clubs").select("id,name,slug").in_("id", list(set(season["details"]["club_ids"] + [season["organizer_club_id"]]))).execute().data or []
        if not registration_state(season)["meet_planning_open"]:
            calendar = db.table("pcs_interclub_meet_workspaces").select("id,starts_at,host_club_id,club_ids").eq("season_id", str(season_id))
            if not organizer:
                calendar = calendar.contains("club_ids", json.dumps([club_id]))
            schedule = calendar.order("starts_at").order("id").limit(100).execute().data or []
            return {"season": registration_season(season), "meets": [], "is_organizer": organizer, "own_participation": own,
                    "participations": participations, "clubs": clubs, "teams": [], "next_team_offset": None,
                    "meet_schedule": schedule, "first_meet_at": schedule[0]["starts_at"] if schedule else None}
        query = db.table("pcs_interclub_current_rosters").select(TEAM_FIELDS).eq("season_id", str(season_id)).is_("meet_id", "null")
        if not organizer:
            query = query.eq("club_id", club_id)
        teams = query.order("id").range(team_offset, team_offset+100).execute().data or []
        meet_query = db.table("pcs_interclub_meet_workspaces").select(MEET_FIELDS).eq("season_id", str(season_id))
        if not organizer:
            # club_ids is JSONB. A Python list is encoded by PostgREST's client
            # as a PostgreSQL array literal, which is invalid for this column.
            meet_query = meet_query.contains("club_ids", json.dumps([club_id]))
        meets = meet_query.order("starts_at").order("id").limit(100).execute().data or []
        return {"season": registration_season(season), "meets": meets, "is_organizer": organizer, "own_participation": own, "participations": participations, "clubs": clubs,
                "meet_schedule": [{key: meet[key] for key in ("id", "starts_at", "host_club_id", "club_ids")} for meet in meets],
                "first_meet_at": meets[0]["starts_at"] if meets else None,
                "teams": [safe_roster(row, own_club=row["club_id"] == club_id) for row in teams[:100]],
                "next_team_offset": team_offset+100 if len(teams)>100 else None}

    @app.post("/admin/clubs/{club_id}/interclub/registrations/{season_id}/participations/{target_club_id}")
    def respond(club_id: str, season_id: UUID, target_club_id: str, payload: ParticipationUpdate, authorization: str | None = auth_header()):
        db, user = administrator(club_id, authorization)
        row = call_rpc(db, "pcs_interclub_participation", {**actor_params(user, club_id, season_id), "p_target_club_id": target_club_id,
                       "p_revision": payload.expected_revision, "p_action": payload.action})
        return {"participation": {key: row.get(key) for key in PARTICIPATION_FIELDS.split(",")}}

    def meet_access(db, club_id, season, meet_id):
        require_season_phase(season)
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
        season, _ = access(db, club_id, season_id)
        require_season_phase(season)
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
        return player_choices(db, club_id, season_id, own, q, offset, meet=meet)

    @app.get("/admin/clubs/{club_id}/interclub/registrations/{season_id}/players")
    def players(club_id: str, season_id: UUID, q: str = Query(default="", max_length=80), offset: int = Query(default=0, ge=0, le=100000), authorization: str | None = auth_header()):
        db, _ = administrator(club_id, authorization)
        _, own = access(db, club_id, season_id)
        return player_choices(db, club_id, season_id, own, q, offset)

    def player_choices(db, club_id, season_id, own, q, offset, meet=None):
        if not own or own["status"] != "accepted":
            raise HTTPException(403, "Accept your club's season invitation before choosing players.")
        query = db.table("players").select("id,name,rating,gender").eq("club_id", club_id).eq("active", True)
        meet_ratings = {}
        if meet is not None:
            members = db.table("pcs_interclub_pool_members").select("player_id").eq("season_id", str(season_id)).eq("club_id", club_id).eq("status", "active").eq("approval_status", "approved").execute().data or []
            member_ids = [row["player_id"] for row in members if row.get("player_id") is not None]
            if not member_ids:
                return {"players": [], "next_offset": None}
            query = query.in_("id", member_ids)
            values = call_rpc(db, "pcs_interclub_meet_player_ratings", {"p_season_id": str(season_id), "p_meet_id": meet["id"], "p_club_id": club_id})
            meet_ratings = {str(row["player_id"]): row for row in (values or [])}
            if not meet_ratings:
                return {"players": [], "next_offset": None}
            query = query.in_("id", [int(value) for value in meet_ratings])
        if q.strip():
            term = q.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            query = query.ilike("name", f"%{term}%")
        rows = query.order("name").order("id").range(offset, offset+100).execute().data or []
        entries = db.table("pcs_interclub_entries").select("player_id,starting_rating").eq("season_id", str(season_id)).eq("club_id", club_id).in_("player_id", [r["id"] for r in rows[:100]]).execute().data if rows else []
        seeds = {str(e["player_id"]): e["starting_rating"] for e in entries or []}
        return {"players": [{"id": str(row["id"]), "name": row["name"], "starting_rating": seeds.get(str(row["id"]), float(row["rating"])/400 if row.get("rating") is not None else None), **({key: meet_ratings[str(row["id"])].get(key) for key in ("entry_id", "eligibility_rating", "rating_deadline", "rating_locked", "gender")} if meet is not None else {})} for row in rows[:100]],
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
        season, _ = access(db, club_id, season_id)
        require_season_phase(season)
        saved = call_rpc(db, "pcs_save_interclub_meet_roster", {**actor_params(user, club_id, season_id), "p_meet_id": str(meet_id), "p_meet_revision": payload.expected_meet_revision, "p_team_id": str(team_id),
            "p_revision": payload.expected_revision, "p_name": payload.name, "p_division": payload.division, "p_player_ids": payload.player_ids})
        return {"team": safe_roster({**saved["team"], **saved["roster"]}, own_club=True)}

    @app.post("/admin/clubs/{club_id}/interclub/registrations/{season_id}/meets/{meet_id}/teams/{team_id}/withdraw")
    def withdraw(club_id: str, season_id: UUID, meet_id: UUID, team_id: UUID, payload: RosterRevision, authorization: str | None = auth_header()):
        db, user = administrator(club_id, authorization)
        season, _ = access(db, club_id, season_id)
        require_season_phase(season)
        saved = call_rpc(db, "pcs_save_interclub_meet_roster", {**actor_params(user, club_id, season_id), "p_meet_id": str(meet_id), "p_meet_revision": payload.expected_meet_revision, "p_team_id": str(team_id),
            "p_revision": payload.expected_revision, "p_name": "", "p_division": "", "p_player_ids": [], "p_withdraw": True})
        return {"team": safe_roster({**saved["team"], **saved["roster"]}, own_club=True)}

    @app.post("/admin/clubs/{club_id}/interclub/registrations/{season_id}/meets/{meet_id}/teams/{team_id}/eligibility")
    def decide(club_id: str, season_id: UUID, meet_id: UUID, team_id: UUID, payload: EligibilityDecision, authorization: str | None = auth_header()):
        db, user = administrator(club_id, authorization)
        season, _ = access(db, club_id, season_id)
        require_season_phase(season)
        saved = call_rpc(db, "pcs_review_interclub_meet_roster", {**actor_params(user, club_id, season_id), "p_meet_id": str(meet_id), "p_meet_revision": payload.expected_meet_revision, "p_team_id": str(team_id),
            "p_revision": payload.expected_revision, "p_approve": payload.approve, "p_reason": payload.reason})
        return {"team": safe_roster({**saved["team"], **saved["roster"]}, own_club=saved["team"]["club_id"] == club_id)}

    @app.get("/admin/clubs/{club_id}/interclub/registrations/{season_id}/meets/{meet_id}/teams/{team_id}/history")
    @app.get("/admin/clubs/{club_id}/interclub/registrations/{season_id}/teams/{team_id}/history")
    def history(club_id: str, season_id: UUID, team_id: UUID, meet_id: UUID | None = None, authorization: str | None = auth_header()):
        db, _ = administrator(club_id, authorization)
        season, _ = access(db, club_id, season_id)
        require_season_phase(season)
        teams = db.table("pcs_interclub_teams").select("id,club_id,meet_id").eq("id", str(team_id)).eq("season_id", str(season_id)).limit(1).execute().data or []
        if meet_id is not None:
            meet_access(db, club_id, season, meet_id)
        if not teams or (meet_id is not None and teams[0]["meet_id"] != str(meet_id)) or (season["organizer_club_id"] != club_id and teams[0]["club_id"] != club_id):
            raise HTTPException(404, "Team unavailable for this club.")
        rows = db.table("pcs_interclub_roster_versions").select(HISTORY_FIELDS).eq("team_id", str(team_id)).order("revision", desc=True).limit(50).execute().data or []
        return {"history": [safe_roster(row, own_club=teams[0]["club_id"] == club_id) for row in rows]}
