"""Revision-keyed reuse must preserve results across writes and request variants."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from jupr_app.services import leaderboard_service as service
from tests.test_leaderboard_seasons import fixture, match


def leaderboard_config(db):
    # Production keeps the leaderboard document in its dedicated table;
    # staging currently embeds it in the club-site document.
    if "club_leaderboard_settings" in db.store:
        return db.store["club_leaderboard_settings"][0]["published"]
    return db.store["pcs_club_sites"][0]["published"]["leaderboard"]


def versioned_db():
    db = fixture(ongoing=True)
    db.store["badge_program_state"] = [{"club_id": "club-1", "revision": 1}]
    config = leaderboard_config(db)
    config["cards"] = ["longest_win_streak", "most_upsets"]
    return db


def bump(db):
    db.store["badge_program_state"][0]["revision"] += 1


def test_stat_variants_reuse_history_and_metrics_but_not_player_status(monkeypatch):
    db = versioned_db()
    calls = []
    calculate = service.compute_overall_metrics
    monkeypatch.setattr(service, "compute_overall_metrics", lambda *a, **kw: (calls.append(1), calculate(*a, **kw))[1])
    service.build_public_leaderboard(db, club_id="club-1", stat="longest_win_streak")
    query_count = len(db.match_queries)
    service.build_public_leaderboard(db, club_id="club-1", stat="most_upsets", status="inactive")
    assert len(db.match_queries) == query_count
    assert len(calls) == 1
    # Even fresh eligibility inputs changed outside a simulated revision trigger
    # cannot reuse an earlier player's active state from calculated rows.
    db.store["players"][0]["active"] = False
    payload = service.build_public_leaderboard(db, club_id="club-1", stat="most_matches")
    assert 1 not in [row["player_id"] for row in payload["stat_ranking"]["entries"]]


@pytest.mark.parametrize("mutation", ["insert", "edit", "delete", "soft_delete"])
def test_revision_invalidates_history_without_waiting_for_ttl(mutation):
    db = versioned_db()
    before = service.build_public_leaderboard(db, club_id="club-1", stat="most_wins")
    count = len(db.match_queries)
    if mutation == "insert":
        db.store["matches"].append(match(9, "2024-09-19T12:00:00Z"))
    elif mutation == "edit":
        db.store["matches"][1].update(score_t1=1, score_t2=11)
    elif mutation == "delete":
        db.store["matches"].pop(1)
    else:
        db.store["matches"][1]["deleted_at"] = "2024-09-20T00:00:00Z"
    bump(db)
    fresh = service.build_public_leaderboard(db, club_id="club-1", stat="most_wins")
    assert len(db.match_queries) > count
    assert fresh["stat_ranking"] != before["stat_ranking"]
    # Compare the cacheable response with a fresh uncached baseline.
    db.store["badge_program_state"] = []
    assert fresh == service.build_public_leaderboard(db, club_id="club-1", stat="most_wins")


def test_revision_unavailable_never_serves_prior_cache(monkeypatch):
    db = versioned_db()
    service.build_public_leaderboard(db, club_id="club-1", stat="most_matches")
    count = len(db.match_queries)
    db.store["matches"].append(match(9, "2024-09-19T12:00:00Z"))
    monkeypatch.setattr(service, "_leaderboard_source_revision", lambda *_: None)
    payload = service.build_public_leaderboard(db, club_id="club-1", stat="most_matches")
    assert len(db.match_queries) > count
    assert next(row for row in payload["stat_ranking"]["entries"] if row["player_id"] == 1)["matches_played"] == 4


def test_revision_race_does_not_cache_mixed_history(monkeypatch):
    db = versioned_db()
    original = service._overall_matches
    loads = []
    def race(*args, **kwargs):
        result = original(*args, **kwargs)
        loads.append(1)
        if len(loads) == 1:
            db.store["matches"].append(match(9, "2024-09-19T12:00:00Z"))
            bump(db)
        return result
    monkeypatch.setattr(service, "_overall_matches", race)
    payload = service.build_public_leaderboard(db, club_id="club-1", stat="most_matches")
    assert len(loads) == 2
    assert next(row for row in payload["stat_ranking"]["entries"] if row["player_id"] == 1)["matches_played"] == 4


def test_future_game_becomes_eligible_without_revision_change():
    db = versioned_db()
    now = datetime(2024, 9, 19, 12, tzinfo=timezone.utc)
    db.store["matches"] = [match(1, (now + timedelta(seconds=10)).isoformat())]
    period = leaderboard_config(db)["seasons"][0]
    rows = service._overall_rows(db.store["players"], club_id="club-1")
    def load(when):
        return service._overall_statistics(db, club_id="club-1", period=period,
                                           base_rows=rows, extended=True,
                                           partnership_minimum=0, now=when)
    before = load(now)
    count = len(db.match_queries)
    after = load(now + timedelta(seconds=11))
    assert not before[0]
    assert len(after[0]) == 1
    assert len(db.match_queries) == count
    assert before[2] != after[2]


def test_shared_origin_clients_invalidate_cache_from_database_revision():
    first = versioned_db()
    first.supabase_url = "https://test-origin.invalid"
    service.build_public_leaderboard(first, club_id="club-1", stat="most_matches")
    second = versioned_db()
    second.supabase_url = first.supabase_url
    second.store = deepcopy(first.store)
    service.build_public_leaderboard(second, club_id="club-1", stat="most_matches")
    assert not second.match_queries
    second.store["matches"].append(match(9, "2024-09-19T12:00:00Z"))
    bump(second)
    changed = service.build_public_leaderboard(second, club_id="club-1", stat="most_matches")
    assert second.match_queries
    assert next(row for row in changed["stat_ranking"]["entries"] if row["player_id"] == 1)["matches_played"] == 4
