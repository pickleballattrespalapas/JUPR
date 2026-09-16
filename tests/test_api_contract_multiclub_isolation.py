"""Exercise real club-scoped services with deliberately overlapping fixture names."""
from copy import deepcopy

import pytest

from scripts.prepare_multiclub_isolation_fixture import CLUBS, LEAGUE, build_fixture, render_sql
from services.api.interclub_models import SeasonDraft
from jupr_app.services.admin_player_editor_service import (
    get_admin_player_editor_detail, list_admin_player_editor_players, update_admin_player_editor_player,
)
from jupr_app.services.admin_league_manager_service import get_admin_league_manager_detail
from tests.test_admin_player_editor_service import FakeSupabase


@pytest.fixture
def database(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", "1")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    tables = {"players": [], "league_ratings": [], "leagues_metadata": [], "matches": [], "admin_activity_log": []}
    for n, club in enumerate(build_fixture()["clubs"]):
        for i, row in enumerate(club["players"]):
            pid = n * 100 + i + 1
            tables["players"].append({**row, "id": pid, "club_id": club["id"], "inactive_at": None})
            tables["league_ratings"].append({"id": pid, "player_id": pid, "club_id": club["id"], "league_name": LEAGUE,
                "rating": row["rating"], "starting_rating": row["starting_rating"], "wins": row["wins"], "losses": row["losses"],
                "matches_played": row["matches_played"], "is_active": row["active"]})
        tables["leagues_metadata"].append({"id": n+1, "club_id": club["id"], "league_name": LEAGUE,
            "league_type": "Individual", "is_active": True, "status": "active", "description": club["code"]})
    return FakeSupabase(tables)


@pytest.mark.parametrize("n", range(3))
def test_duplicate_names_have_separate_directories_and_league_rosters(database, n):
    cid = CLUBS[n][0]
    listing = list_admin_player_editor_players(database, club_id=cid)
    assert listing["count"] == 24
    assert {p["club_id"] for p in listing["players"]} == {cid}
    detail = get_admin_player_editor_detail(database, club_id=cid, player_id=n*100+1)
    assert detail["player"]["name"] == "Alex Rivera [TEST]"
    assert detail["player"]["starting_jupr"] == pytest.approx(3.15+.05*n)
    league = get_admin_league_manager_detail(database, club_id=cid, league_name=LEAGUE)
    assert league["league"]["description"] == CLUBS[n][2]
    assert len(league["roster"]) == 24
    assert {int(p["player_id"]) for p in league["roster"]} == set(range(n*100+1,n*100+25))


def test_foreign_player_id_cannot_be_read_or_changed(database):
    before = deepcopy(database.storage)
    with pytest.raises(ValueError, match="not found"):
        get_admin_player_editor_detail(database, club_id=CLUBS[0][0], player_id=101)
    with pytest.raises(ValueError, match="not found"):
        update_admin_player_editor_player(database, club_id=CLUBS[0][0], player_id=101,
            patch={"rating_jupr": 6.5}, expected_state_fingerprint="0"*64,
            idempotency_key="isolation-foreign-player", actor_email="test@example.invalid", actor_role="administrator")
    assert database.storage["players"] == before["players"]
    assert database.storage["league_ratings"] == before["league_ratings"]


def test_editing_shared_name_changes_only_selected_club(database):
    detail = get_admin_player_editor_detail(database, club_id=CLUBS[0][0], player_id=1)
    protected = deepcopy([p for p in database.storage["players"] if p["club_id"] != CLUBS[0][0]])
    update_admin_player_editor_player(database, club_id=CLUBS[0][0], player_id=1,
        patch={"rating_jupr": 3.75}, expected_state_fingerprint=detail["player"]["state_fingerprint"],
        idempotency_key="isolation-own-player", actor_email="test@example.invalid", actor_role="administrator")
    assert database.storage["players"][0]["rating"] == 1500
    assert [p for p in database.storage["players"] if p["club_id"] != CLUBS[0][0]] == protected


def test_fixture_histories_balance_and_meet_substitutions_are_independent():
    f = build_fixture()
    SeasonDraft.model_validate(f["draft"])
    for c in f["clubs"]:
        assert sum(p["matches_played"] for p in c["players"]) == 4*len(c["matches"])
        assert sum(p["wins"] for p in c["players"]) == 2*len(c["matches"])
        assert all(p["wins"]+p["losses"] == p["matches_played"] for p in c["players"])
        assert len({m[17] for m in c["matches"]}) == len(c["matches"])
        for r in c["rosters"]:
            players = [c["players"][i] for i in r["slots"]]
            assert sum(p["gender"]=="female" for p in players) == 2
            assert all(p["active"] and p["rating"]/400 <= f["draft"]["registration_rules"][r["division"]]["max_rating"] for p in players)
        assert c["rosters"][0]["slots"] != c["rosters"][2]["slots"]


def test_fixture_generation_rejects_any_other_project():
    with pytest.raises(ValueError, match="staging project"):
        render_sql("another-project")
