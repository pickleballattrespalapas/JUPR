"""Season honors must follow official appearances, results and organizer review."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from jupr_app.domain.interclub_awards import final_results, performance_awards, season_awards
from jupr_app.services.interclub_awards_service import public_interclub_trophies, player_interclub_trophies, player_interclub_awards, public_interclub_honors
from services.api import interclub_awards_routes as routes


def finished_season():
    def game(status, a, b):
        return {"status": status, "a": 11, "b": 7, "players_a": a, "players_b": b}
    regular = {"meet_id": "regular", "phase": "regular", "weather": "normal", "division": "3.5",
               "club_a": "alpha", "club_b": "beta", "pairings": [{"games": [
                   game("retired", ["original", "partner"], ["opponent", "other"]),
                   game("completed", ["sub", "partner"], ["opponent", "other"]),
                   game("forfeit", ["never"], []),
               ]}], "tiebreak": None}
    final = {"meet_id": "final", "phase": "final", "weather": "normal", "division": "3.5",
             "club_a": "alpha", "club_b": "beta", "outcome": {"winner": "b", "games_a": 0, "games_b": 3},
             "pairings": [{"games": [game("completed", ["sub", "partner"], ["opponent", "other"])]} for _ in range(3)]
                         + [{"games": [game("not_needed", ["never"], ["unused"]) ]}], "tiebreak": None}
    return {"name": "Coastal season", "scoring_version": 1, "season_complete": True,
            "meets": [{"id": "regular"}, {"id": "final"}],
            "clubs": [{"id": c, "name": c.title()} for c in ("alpha", "beta", "absent")],
            "players": [{"id": p, "club_id": c, "name": p.title()} for c, ps in
                        (("alpha", ["original", "partner", "sub"]), ("beta", ["opponent", "other"])) for p in ps],
            "competition_results": [regular, final], "competition_standings": [], "qualification": [],
            "club_cup": {"status": "complete", "champions": ["alpha"], "standings": []}}


def recipients(awards, kind, recipient_type="player"):
    return {a["entry_id"] if recipient_type == "player" else a["club_id"] for a in awards
            if a["award_key"] == kind and a["recipient_type"] == recipient_type}


def test_actual_players_receive_participation_and_winning_final_or_cup_honors():
    doc = finished_season()
    awards = season_awards("season", doc)
    assert recipients(awards, "participation") == {"original", "sub", "partner", "opponent", "other"}
    assert recipients(awards, "division_champion") == {"opponent", "other"}
    assert recipients(awards, "club_cup_champion") == {"original", "sub", "partner"}
    assert not recipients(awards, "participation", "club")
    assert next(a["title"] for a in awards if a["award_key"] == "division_champion") == "Coastal season 3.5 Division Champion"
    assert next(a["title"] for a in awards if a["award_key"] == "club_cup_champion") == "Coastal season League Champion"
    assert recipients(awards, "division_champion", "club") == {"beta"}
    assert recipients(awards, "club_cup_champion", "club") == {"alpha"}
    final = final_results(doc)
    assert final["players"] == 5 and final["meets"] == 2
    assert final["divisions"][0] == {"division": "3.5", "meet_id": "final", "winner": "beta", "runner_up": "alpha",
                                      "games_won": 3, "games_lost": 0, "tiebreak": None, "players": ["opponent", "other"]}
    assert len({a["id"] for a in awards}) == len(awards)
    assert season_awards("season", deepcopy(doc)) == awards


def test_singles_tiebreak_and_joint_cup_winners_receive_honors():
    doc = finished_season()
    doc["club_cup"]["champions"] = ["alpha", "beta"]
    final = doc["competition_results"][-1]
    final["outcome"].update(games_a=2, games_b=2)
    final["tiebreak"] = {"status": "completed", "a": 19, "b": 21, "players_a": ["original"], "players_b": ["singles"]}
    doc["players"].append({"id": "singles", "club_id": "beta", "name": "Singles player"})
    awards = season_awards("season", doc)
    assert "singles" in recipients(awards, "participation") & recipients(awards, "division_champion")
    assert recipients(awards, "club_cup_champion", "club") == {"alpha", "beta"}
    assert final_results(doc)["divisions"][0]["tiebreak"] == {"winner_score": 21, "runner_up_score": 19}


@pytest.mark.parametrize("change", ["extra_meet", "rescheduled", "unfinished_cup", "completion_flag"])
def test_unfinished_season_cannot_award(change):
    doc = finished_season()
    if change == "extra_meet": doc["meets"].append({"id": "tomorrow"})
    elif change == "rescheduled": doc["competition_results"][0]["weather"] = "rescheduled"
    elif change == "unfinished_cup": doc["club_cup"]["status"] = "provisional"
    else: doc["season_complete"] = False
    assert not final_results(doc)["complete"]
    with pytest.raises(ValueError, match="Approve every scheduled meet"):
        season_awards("season", doc)


def test_correction_replaces_winners_preserves_participation_identity_and_rejects_wrong_club():
    doc = finished_season()
    before = season_awards("season", doc)
    doc["competition_results"][-1]["outcome"].update(winner="a", games_a=3, games_b=0)
    after = season_awards("season", doc)
    assert recipients(after, "division_champion") == {"original", "sub", "partner"}
    assert {a["id"] for a in before if a["award_key"] == "participation"} == {a["id"] for a in after if a["award_key"] == "participation"}
    doc["players"][0]["club_id"] = "beta"
    with pytest.raises(ValueError, match="season player record"):
        season_awards("season", doc)


class Query:
    def __init__(self, rows): self.rows = deepcopy(rows); self.bounds = None
    def select(self, *args, **kwargs): return self
    def eq(self, key, value): self.rows = [r for r in self.rows if r.get(key) == value]; return self
    def in_(self, key, values): self.rows = [r for r in self.rows if r.get(key) in values]; return self
    def order(self, *args, **kwargs): return self
    def limit(self, value): self.rows = self.rows[:value]; return self
    def range(self, start, end): self.bounds = (start, end); return self
    def execute(self): return SimpleNamespace(data=deepcopy(self.rows if self.bounds is None else self.rows[self.bounds[0]:self.bounds[1]+1]))


@pytest.fixture
def client(monkeypatch):
    sid = str(uuid4()); doc = finished_season()
    season = {"id": sid, "organizer_club_id": "alpha"}
    batches = [{"season_id": sid, "meet_id": m["id"], "state": "approved", "revision": 3, "approved_revision": 3, "ratings_status": "completed"} for m in doc["meets"]]
    tables = {"pcs_interclub_competition_batches": batches, "pcs_interclub_award_sets": [], "pcs_interclub_publications": [], "pcs_public_interclub_awards": []}
    calls = []
    db = SimpleNamespace(table=lambda name: Query(tables[name]), rpc=lambda name, params: (calls.append((name, params)) or SimpleNamespace(execute=lambda: SimpleNamespace(data={"revision": 1}))))
    monkeypatch.setattr(routes, "site_administrator", lambda *args: SimpleNamespace(user_id=str(uuid4()), email="staff@example.invalid"))
    monkeypatch.setattr(routes, "season_context", lambda *args: (season, doc["clubs"], doc["meets"]))
    monkeypatch.setattr(routes, "reviewed_publication", lambda *args: (deepcopy(doc), {"approved": [3]}, hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()))
    app = FastAPI(); routes.install_interclub_awards_routes(app, get_supabase_client=lambda: db)
    return TestClient(app), f"/admin/clubs/alpha/interclub/{sid}/awards", SimpleNamespace(doc=doc, season=season, tables=tables, calls=calls, db=db)


def post_body(preview):
    return {key: preview[key] for key in ("revision", "publication_revision", "preview_fingerprint")}


def test_api_uses_server_computed_recipients_and_exact_review(client):
    api, path, state = client
    preview = api.get(path).json()
    assert preview["ready"] and not preview["current"]
    assert api.post(path, json=post_body(preview)).status_code == 200
    name, args = state.calls[-1]
    assert name == "pcs_award_interclub_season" and args["p_awards"] == preview["awards"]
    assert args["p_document"] == preview["preview"]["document"]
    assert api.post(path, json={**post_body(preview), "awards": []}).status_code == 422
    state.doc["club_cup"]["champions"] = ["beta"]
    assert api.post(path, json=post_body(preview)).status_code == 409
    assert len(state.calls) == 1


@pytest.mark.parametrize("change", ["draft", "stale_approval", "rating_error", "missing_meet"])
def test_open_correction_pending_ratings_or_missing_batch_block_issuance(client, change):
    api, path, state = client
    batch = state.tables["pcs_interclub_competition_batches"][0]
    if change == "draft": batch["state"] = "draft"
    elif change == "stale_approval": batch["revision"] += 1
    elif change == "rating_error": batch["ratings_status"] = "failed"
    else: state.tables["pcs_interclub_competition_batches"].pop()
    preview = api.get(path).json()
    assert not preview["ready"] and preview["problems"]
    assert api.post(path, json=post_body(preview)).status_code == 409
    assert not state.calls


def test_organizer_access_and_unpublished_awards_stay_private(client, monkeypatch):
    api, path, state = client
    assert api.get(path.replace("/alpha/", "/beta/")).status_code == 403
    def deny(*args): raise HTTPException(403, "Club administrator access required.")
    monkeypatch.setattr(routes, "site_administrator", deny)
    assert api.get(path).status_code == 403
    assert api.get(f"/public/interclub/{state.season['id']}/awards").status_code == 404
    assert not state.calls


def test_public_projection_is_club_and_player_scoped_private_fields_removed_and_paginated():
    rows = [{"id": str(i), "season_id": "season", "club_id": "alpha", "entry_id": str(i), "player_id": 1,
             "award_key": "participation", "division": "", "title": "Participant", "recipient_type": "player", "recipient_name": "Player",
             "season_name": "Season", "earned_at": "2026-09-30T00:00:00Z", "email": "private@example.invalid"} for i in range(1001)]
    rows += [{**rows[0], "id": "other", "club_id": "beta"}, {**rows[0], "id": "club", "entry_id": None, "player_id": None, "recipient_type": "club", "award_key": "club_cup_champion"}]
    db = SimpleNamespace(table=lambda name: Query(rows if name == "pcs_public_interclub_awards" else []))
    own = public_interclub_trophies(db, club_id="alpha", player_id=1)
    assert len(own) == 1001
    assert all("player_id" not in row and "email" not in row and "entry_id" not in row for row in own)
    assert own[0]["results_href"] == "/interclub/season/final-results"
    assert [row["id"] for row in public_interclub_trophies(db, club_id="alpha")] == ["club"]
    assert player_interclub_trophies(db, club_id="alpha", player_id=1)[0]["placement"] is None


def regular_season():
    doc = finished_season()
    doc["season_complete"] = False
    doc["club_cup"]["status"] = "provisional"
    doc["competition_results"] = [doc["competition_results"][0]]
    result = doc["competition_results"][0]
    result["id"] = "alpha-beta"
    pair = result["pairings"][0]
    pair["kind"] = "women"
    pair["games"] = [{"status": "completed", "a": 11, "b": 7, "players_a": ["original", "partner"],
                      "players_b": ["opponent", "other"], "played_at": "2026-09-30T15:00:00Z"} for _ in range(3)]
    return doc


def test_two_one_win_sweep_and_undefeated_are_distinct_and_repeatable():
    doc = regular_season()
    doc["competition_results"][0]["pairings"][0]["games"][2].update(a=7, b=11)
    awards = performance_awards("season", doc)
    assert recipients(awards, "matchup_win") == {"original", "partner"}
    assert not recipients(awards, "matchup_sweep") and not recipients(awards, "undefeated_meet")
    doc["competition_results"][0]["pairings"][0]["games"][2].update(a=11, b=7)
    second = deepcopy(doc["competition_results"][0]); second["meet_id"] = "second"
    doc["competition_results"].append(second)
    awards = performance_awards("season", doc)
    for kind in ("matchup_win", "matchup_sweep", "undefeated_meet"):
        assert recipients(awards, kind) == {"original", "partner"}
        assert len([a for a in awards if a["award_key"] == kind]) == 4
    assert len({a["id"] for a in awards}) == len(awards)
    doc["competition_results"].reverse()
    assert performance_awards("season", doc) == awards
    assert not {a["id"] for a in performance_awards("another-season", doc)} & {a["id"] for a in awards}


def test_substitutes_earn_only_their_actual_wins_and_retirement_uses_official_winner():
    doc = regular_season()
    games = doc["competition_results"][0]["pairings"][0]["games"]
    for game in games[1:]: game["players_a"] = ["sub", "partner"]
    awards = performance_awards("season", doc)
    assert recipients(awards, "matchup_win") == {"sub", "partner"}
    assert recipients(awards, "matchup_sweep") == {"partner"}
    assert recipients(awards, "undefeated_meet") == {"original", "sub", "partner"}
    games[0].update(status="retired", a=8, b=2, winner="b")
    awards = performance_awards("season", doc)
    assert recipients(awards, "matchup_win") == {"sub", "partner"}
    assert not recipients(awards, "matchup_sweep")
    assert recipients(awards, "undefeated_meet") == {"sub"}


@pytest.mark.parametrize("status", ["forfeit", "double_forfeit", "unplayed", "not_needed", "pending"])
def test_unplayed_lineups_cannot_earn_achievements(status):
    doc = regular_season()
    for game in doc["competition_results"][0]["pairings"][0]["games"]: game.update(status=status, winner="a")
    assert performance_awards("season", doc) == []


def test_undefeated_meet_includes_other_pairings_divisions_and_singles():
    doc = regular_season()
    other = deepcopy(doc["competition_results"][0]); other.update(id="other-division", division="4.0", phase="final")
    other["pairings"] = [{"kind": "mixed_1", "games": [other["pairings"][0]["games"][0]]}]
    other["tiebreak"] = {"status": "completed", "a": 19, "b": 21, "players_a": ["original"], "players_b": ["opponent"]}
    doc["competition_results"].append(other)
    awards = performance_awards("season", doc)
    assert recipients(awards, "undefeated_meet") == {"partner"}
    assert len([a for a in awards if a["award_key"] == "matchup_win"]) == 2


def test_division_champions_include_regular_players_only_in_divisions_they_played():
    doc = finished_season()
    doc["competition_results"][-1]["outcome"]["winner"] = "a"
    original = doc["competition_results"][0]
    original["pairings"][0]["games"][0]["players_a"] = ["partner"]
    higher = deepcopy(original); higher["division"] = "4.0"
    higher["pairings"][0]["games"][0]["players_a"] = ["original"]
    doc["competition_results"].append(higher)
    awards = season_awards("season", doc)
    assert recipients(awards, "division_champion") == {"partner", "sub"}
    assert "original" in recipients(awards, "club_cup_champion")


def test_published_achievements_reconcile_without_waiting_for_season_close_and_stay_player_scoped():
    doc = regular_season()
    tables = {"pcs_public_interclub_awards": [], "pcs_interclub_entries": [
        {"id": "original", "season_id": "season", "club_id": "alpha", "player_id": 1},
        {"id": "opponent", "season_id": "season", "club_id": "beta", "player_id": 1}],
        "pcs_interclub_publications": [{"season_id": "season", "published": doc, "published_at": "2026-09-30T16:00:00Z"}]}
    db = SimpleNamespace(table=lambda name: Query(tables[name]))
    public = public_interclub_honors(db, club_id="alpha", player_id=1)
    assert {a["award_key"] for a in public} == {"participation", "matchup_win", "matchup_sweep", "undefeated_meet"}
    assert all(a["recipient_key"] == "original" and a["earned_at"] == "2026-09-30T15:00:00Z" for a in public)
    assert all("player_id" not in a and "entry_id" not in a for a in public)
    profile = player_interclub_awards(db, club_id="alpha", player_id=1)
    assert len(profile["badges"]) == 3 and len(profile["trophies"]) == 1
    assert profile["badges"][0]["count"] == 1 and profile["badges"][0]["achievements"][0]["results_href"].endswith("?view=results")
    tables["pcs_interclub_publications"][0]["published"] = deepcopy(doc)
    games = tables["pcs_interclub_publications"][0]["published"]["competition_results"][0]["pairings"][0]["games"]
    for game in games: game.update(a=5, b=11)
    assert not player_interclub_awards(db, club_id="alpha", player_id=1)["badges"]
    assert len(player_interclub_awards(db, club_id="beta", player_id=1)["badges"]) == 3
    tables["pcs_interclub_publications"][0]["published"] = None
    assert player_interclub_awards(db, club_id="alpha", player_id=1) == {"badges": [], "trophies": []}
