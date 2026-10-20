from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from jupr_app.domain import player_activity
from jupr_app.domain.match_processing import process_matches
from jupr_app.domain.singles_match_processing import process_singles_matches
from tests.test_match_processing_badges import _Supabase


NOW = datetime(2026, 10, 20, 20, tzinfo=timezone.utc)
ACTIVITY_FIELDS = {"last_game_at", "inactive_at", "active"}


@pytest.fixture
def fixture_db(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW.astimezone(tz or timezone.utc)

    monkeypatch.setattr(player_activity, "datetime", Clock)
    db = _Supabase()
    original_execute = db.execute

    def execute(query):
        # PostgREST accepts the literal "null" for IS NULL. Mirror that filter
        # in the shared in-memory query fixture before exercising CAS.
        query._filters = [(op, key, None if op == "is" and value == "null" else value)
                          for op, key, value in query._filters]
        return original_execute(query)

    monkeypatch.setattr(db, "execute", execute)
    db.tables["players"] = [
        {"id": i, "club_id": "club", "name": f"Player {i}", "rating": 1400.,
         "wins": 3, "losses": 2, "matches_played": 5, "active": False,
         "inactive_at": "2026-09-01T00:00:00+00:00", "last_game_at": None,
         "singles_rating": None, "singles_wins": None, "singles_losses": None,
         "singles_matches_played": None, "singles_last_game_at": None,
         "singles_replay_baseline": {"rating": 1200., "wins": 0, "losses": 0, "matches_played": 0}}
        for i in range(1, 6)
    ]
    return db


def match(format, scope="unrated", *, date=None):
    return {"match_format": format, "league": "Social", "rating_scope": scope,
            "date": (date or NOW - timedelta(hours=1)).isoformat(),
            "t1_p1": 1, "t2_p1": 3,
            "t1_p2": 2 if format == "doubles" else None,
            "t2_p2": 4 if format == "doubles" else None,
            "score_t1": 11, "score_t2": 7}


def process(db, format, matches, *, plan):
    handler = process_singles_matches if format == "singles" else process_matches
    return handler(matches, supabase=db, club_id="club", name_to_id={},
                   df_players_all=pd.DataFrame(deepcopy(db.tables["players"])),
                   df_leagues=pd.DataFrame(), df_meta=pd.DataFrame(),
                   build_write_plan_only=plan)


@pytest.mark.parametrize("format", ["doubles", "singles"])
@pytest.mark.parametrize("plan", [True, False], ids=["atomic-plan", "legacy-writer"])
def test_unrated_game_records_activity_only_for_scored_participants(fixture_db, format, plan):
    db = fixture_db
    before = deepcopy(db.tables["players"])
    scored = match(format)
    empty = {**scored, "t1_p1": 5, "score_t1": 0, "score_t2": 0}
    result = process(db, format, [scored, empty], plan=plan)
    assert result["inserted"] == 1
    assert result["skipped_empty"] == 1
    played_ids = {1, 3} if format == "singles" else {1, 2, 3, 4}
    if plan:
        assert db.tables["players"] == before
        updates = result["write_plan"]["player_updates"]
        assert {u["player_id"] for u in updates} == played_ids
        assert result["side_effect_context"]["affected_player_ids"] == []
        for update in updates:
            assert update["activity_only"] is True
            for key, value in update["expected"].items():
                if key not in ACTIVITY_FIELDS:
                    assert update["after"][key] == value
            next(p for p in db.tables["players"] if p["id"] == update["player_id"]).update(update["after"])
    for previous, current in zip(before, db.tables["players"]):
        assert {k: v for k, v in current.items() if k not in ACTIVITY_FIELDS} == {
            k: v for k, v in previous.items() if k not in ACTIVITY_FIELDS}
        if current["id"] in played_ids:
            assert current["last_game_at"] == scored["date"]
            assert current["active"] is True and current["inactive_at"] is None
        else:
            assert current == previous


@pytest.mark.parametrize("format", ["doubles", "singles"])
def test_mixed_rating_scopes_advance_activity_without_counting_social_game(fixture_db, format):
    rated_time = NOW - timedelta(days=1)
    social_time = NOW - timedelta(hours=1)
    result = process(fixture_db, format, [match(format, "overall_only", date=rated_time),
                                       match(format, date=social_time)], plan=True)
    for update in result["write_plan"]["player_updates"]:
        after = update["after"]
        assert update["activity_only"] is False
        assert after["last_game_at"] == social_time.isoformat()
        if format == "singles":
            assert after["singles_matches_played"] == 1
            assert after["singles_last_game_at"] == rated_time.isoformat()
        else:
            assert after["matches_played"] == 6


@pytest.mark.parametrize("format", ["doubles", "singles"])
@pytest.mark.parametrize("race_point", ["after-match-insert", "before-activity-cas"])
def test_legacy_social_activity_preserves_concurrent_game_and_manual_exclusion(
    fixture_db, monkeypatch, format, race_point,
):
    db = fixture_db
    newer_game = (NOW - timedelta(minutes=10)).isoformat()
    exclusion = NOW.isoformat()
    original_execute = db.execute
    raced = False

    def concurrent_update(query):
        nonlocal raced
        insert = query.table == "matches" and query._op == "insert"
        update = query.table == "players" and query._op == "update"
        if not raced and ((race_point == "after-match-insert" and insert)
                          or (race_point == "before-activity-cas" and update)):
            raced = True
            for player in db.tables["players"]:
                player.update(last_game_at=newer_game, active=False, inactive_at=exclusion)
        return original_execute(query)

    monkeypatch.setattr(db, "execute", concurrent_update)
    result = process(db, format, [match(format)], plan=False)
    assert result["inserted"] == 1 and raced
    for player in db.tables["players"]:
        assert player["last_game_at"] == newer_game
        assert player["active"] is False
        assert player["inactive_at"] == exclusion
