"""Organizer-owned interclub records; participating clubs are labels, never tenants.

This workflow reuses the competition engine and its scoring rules. It has no
club-account, email, membership or cross-club rating side effects.
"""
from copy import deepcopy
from datetime import date, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from jupr_app.domain import interclub_competition as engine
from services.api.interclub_models import Division


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class Player(Model):
    entry_id: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=120)
    gender: Literal["female", "male"]
    starting_rating: float | None = Field(default=None, gt=0, le=8)


class Roster(Model):
    division: Division
    roster: list[Player] = Field(default_factory=list, max_length=40)


class Club(Model):
    id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=120)
    captain: str = Field(default="", max_length=120)
    contact: str = Field(default="", max_length=200)
    rosters: list[Roster] = Field(default_factory=list, max_length=8)


class SeasonDetails(Model):
    name: str = Field(min_length=1, max_length=120)
    start_date: date | None = None
    end_date: date | None = None
    timezone: str = "America/Mazatlan"
    divisions: list[Division] = Field(default_factory=lambda: ["3.5", "4.0"], min_length=1, max_length=8)
    clubs: list[Club] = Field(default_factory=list, max_length=32)
    rules: str = Field(default="", max_length=10000)

    @model_validator(mode="after")
    def valid(self):
        try:
            ZoneInfo(self.timezone)
        except (KeyError, ValueError):
            raise ValueError("Choose a valid season timezone.")
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("The season end must follow its start.")
        if len(set(self.divisions)) != len(self.divisions):
            raise ValueError("Choose each division once.")
        ids, names, identities = set(), set(), {}
        for club in self.clubs:
            if club.id in ids or club.name.casefold() in names:
                raise ValueError("Use a different name for each participating club.")
            ids.add(club.id); names.add(club.name.casefold())
            divisions = set()
            for roster in club.rosters:
                if roster.division not in self.divisions or roster.division in divisions:
                    raise ValueError("Use each offered division once per club.")
                divisions.add(roster.division)
                seen = set()
                for player in roster.roster:
                    if player.entry_id in seen:
                        raise ValueError("A player appears twice in a roster.")
                    seen.add(player.entry_id)
                    identity = (club.id, player.name, player.gender)
                    if player.entry_id in identities and identities[player.entry_id] != identity:
                        raise ValueError("A player identity cannot represent two people or clubs.")
                    identities[player.entry_id] = identity
        return self


class Team(Roster):
    club_id: str = Field(min_length=1, max_length=100)
    reserves: list[Player] = Field(default_factory=list, max_length=40)


class MeetDetails(Model):
    host_club_id: str = Field(min_length=1, max_length=100)
    club_ids: list[str] = Field(min_length=2, max_length=4)
    starts_at: AwareDatetime
    duration_minutes: int = Field(default=180, ge=30, le=720)
    courts: int = Field(default=4, ge=1, le=100)
    phase: Literal["regular", "final", "qualifier"] = "regular"
    format: Literal["gender", "mixed", "mlp"] = "gender"
    teams: list[Team] = Field(default_factory=list, max_length=32)


def validate_meet(details: MeetDetails, season: dict, others: list[dict], meet_id: str) -> dict:
    clubs = {club["id"]: club for club in season["clubs"]}
    if len(set(details.club_ids)) != len(details.club_ids) or not set(details.club_ids) <= clubs.keys():
        raise ValueError("Choose two to four different participating clubs.")
    if details.host_club_id not in details.club_ids:
        raise ValueError("The host must be one of this meet's clubs.")
    start, end = details.starts_at, details.starts_at + timedelta(minutes=details.duration_minutes)
    zone = ZoneInfo(season["timezone"])
    if season.get("start_date") and start.astimezone(zone).date() < date.fromisoformat(season["start_date"]):
        raise ValueError("The meet starts before the season.")
    if season.get("end_date") and end.astimezone(zone).date() > date.fromisoformat(season["end_date"]):
        raise ValueError("The meet ends after the season.")
    for other in others:
        if other["id"] == meet_id:
            continue
        plan = MeetDetails.model_validate(other["details"])
        if start < plan.starts_at + timedelta(minutes=plan.duration_minutes) and end > plan.starts_at and set(details.club_ids) & set(plan.club_ids):
            raise ValueError("A participating club already has a meet at this time.")
    seen, identities = set(), {}
    for team in details.teams:
        if team.club_id not in details.club_ids or team.division not in season["divisions"]:
            raise ValueError("Every lineup must belong to this meet and an offered division.")
        key = (team.club_id, team.division)
        if key in seen:
            raise ValueError("Use one lineup per club and division.")
        seen.add(key)
        pool = {p["entry_id"]: p for r in clubs[team.club_id]["rosters"] for p in r["roster"]}
        if len({p.entry_id for p in [*team.roster, *team.reserves]}) != len(team.roster) + len(team.reserves):
            raise ValueError("A player can appear only once in a meet lineup and its substitutes.")
        for player in [*team.roster, *team.reserves]:
            if player.entry_id not in pool or any(getattr(player, k) != pool[player.entry_id].get(k) for k in ("name", "gender", "starting_rating")):
                raise ValueError("Choose lineup players from the saved club roster. Save new players first.")
            if player.entry_id in identities and identities[player.entry_id] != team.club_id:
                raise ValueError("A player cannot represent two clubs.")
            identities[player.entry_id] = team.club_id
            if player.starting_rating is not None and team.division not in {"Open", "4.5/Open"} and player.starting_rating >= float(team.division) + .5:
                raise ValueError(f"{player.name}'s rating is too high for {team.division}.")
    if (details.phase == "regular") == (details.format == "mlp"):
        raise ValueError("Regular meets use gender or mixed doubles; championship meets use MLP.")
    return details.model_dump(mode="json")


def generate(meet_id: str, details: dict) -> dict:
    if not details["teams"]:
        raise ValueError("Select lineups for at least two clubs before generating the schedule.")
    if details["phase"] == "regular":
        return engine.generate_round_robin(meet_id, details["teams"], format=details["format"], courts=details["courts"], schedule_mode="staggered")
    divisions = {}
    for team in details["teams"]:
        divisions.setdefault(team["division"], []).append(team)
    encounters = []
    for division, teams in divisions.items():
        if len(teams) != 2:
            raise ValueError("Choose two championship teams per division.")
        doc = engine.generate_championship(meet_id, division, *teams, phase=details["phase"])
        encounters.extend(doc["encounters"])
    # Sequential MLP matchups on Court 1 avoid overlapping players and courts.
    for index, encounter in enumerate(encounters, 1):
        encounter["rotation"] = index
        for pairing in encounter["pairings"]:
            pairing["court"] = 1
    return engine.validate_document({"meet_id": meet_id, "phase": details["phase"], "format": "mlp", "encounters": encounters})


def has_scores(document: dict | None) -> bool:
    return bool(document and any(g.get("a") is not None or g.get("b") is not None or g["status"] not in {"pending", "forfeit", "double_forfeit"}
                                for e in document["encounters"] for p in e["pairings"] for g in p["games"]))


def score_document(previous: dict, incoming: dict, teams: list[dict], *, official=False) -> dict:
    """Accept score/incident edits, never replacement match identities or lineups.

The host can use approved roster substitutes already supplied with this meet.
Season setup, draw generation and changing starting lineups stay with Joe.
"""
    new = engine.validate_document(incoming, official=official)
    old = engine.validate_document(previous)
    def skeleton(doc):
        result = deepcopy(doc)
        result.pop("weather", None)
        for encounter in result["encounters"]:
            encounter.pop("tiebreak", None)
            for pairing in encounter["pairings"]:
                pairing["games"] = [{"id": g["id"]} for g in pairing["games"]]
        return result
    if skeleton(old) != skeleton(new):
        raise ValueError("The host link can edit scores only. The organizer controls the schedule and starting lineups.")
    rosters = {(t["club_id"], t["division"]): {p["entry_id"]: p for p in [*t["roster"], *t.get("reserves", [])]} for t in teams}
    for encounter in new["encounters"]:
        for pairing in encounter["pairings"]:
            for game in pairing["games"]:
                for side in ("a", "b"):
                    ids = game[f"players_{side}"] or pairing[f"players_{side}"]
                    players = rosters.get((encounter[f"club_{side}"], encounter["division"]), {})
                    if not set(ids) <= players.keys():
                        raise ValueError("A replacement must be on this meet's roster. Contact the organizer to update it.")
                    genders = [players[i]["gender"] for i in ids]
                    if ids and ((pairing["kind"] == "women" and genders.count("female") != 2) or
                                (pairing["kind"] == "men" and genders.count("male") != 2) or
                                (pairing["kind"].startswith("mixed") and sorted(genders) != ["female", "male"])):
                        raise ValueError("The replacement must preserve the pairing's gender composition.")
    return new


def safe_clubs(details: dict) -> list[dict]:
    return [{"id": c["id"], "name": c["name"]} for c in details["clubs"]]


def standings(season: dict, meets: list[dict]) -> dict:
    docs = [m["approved_document"] for m in meets if m.get("approved_document")]
    clubs = safe_clubs(season["details"])
    return {"standings": engine.league_standings(docs, clubs=clubs), "club_cup": engine.club_cup(docs, clubs=clubs)}
