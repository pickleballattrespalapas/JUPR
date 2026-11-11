from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from jupr_app.domain.leaderboard_metrics import EVENT_WIN_CARD_KEYS
from jupr_app.domain.live_beta_engine import round_robin_result_fingerprint
from jupr_app.domain.moneyball_expectations import settlement, standings
from jupr_app.services.event_winner_service import event_winner_highlights
from jupr_app.services.leaderboard_service import build_public_leaderboard, LeaderboardDataUnavailable
from tests.test_leaderboard_seasons import fixture
from tests.test_weekly_report import DB, Query


PERIOD = {"id": "2026", "name": "2026–27", "start_date": "2026-09-15", "end_date": "2027-09-14", "timezone": "America/Mazatlan"}
NOW = datetime(2026, 11, 11, 4, tzinfo=timezone.utc)
PLAYERS = [{"id": i, "name": name, "club_id": "club-1", "active": True, "last_game_at": NOW.isoformat(), "matches_played": 0}
           for i, name in enumerate(["Alex", "Bea", "Cam", "Dee", "Eve", "Fran", "Gus", "Hal"], 1)]


class NestedQuery(Query):
    def in_(self, key, values):
        if "->" in key:
            def value(row):
                for part in key.replace("->>", "->").split("->"):
                    row = (row or {}).get(part)
                return row
            self.data = [r for r in self.data if value(r) in values]
            return self
        self.data = [r for r in self.data if str(r.get(key)) in {str(v) for v in values}]
        return self


class WinnerDB(DB):
    def __init__(self, **tables):
        super().__init__(**tables)
        self.reads = []

    def table(self, name):
        self.reads.append(name)
        return NestedQuery(self.tables.get(name, []))


def event():
    games = [
        {"sideA": ["p1", "p2"], "sideB": ["p3", "p4"]},
        {"sideA": ["p1", "p3"], "sideB": ["p2", "p4"]},
        {"sideA": ["p1", "p4"], "sideB": ["p2", "p3"]},
    ]
    return {"type": "round_robin", "generatorKind": "round_robin", "status": "completed", "name": "Evening RR", "sourceEventUid": "rr-1",
            "participants": [{"id": f"p{i}", "name": p["name"], "player_id": i} for i, p in enumerate(PLAYERS[:4], 1)],
            "rounds": [{"number": i, "status": "saved", "matches": [{**game, "id": f"g{i}", "scoreA": 11, "scoreB": 5}]} for i, game in enumerate(games, 1)]}


def session(ev=None, **overrides):
    return {"id": "s1", "club_id": "club-1", "session_key": "key", "status": "completed", "version": 1,
            "state": {"mode": "admin_play_generator", "page_state": {"event": ev or event()},
                      "generator_submission": {"status": "approved", "match_date": "2026-10-13"}}, **overrides}


def winners(db, *, keys=EVENT_WIN_CARD_KEYS, period=PERIOD, status="all", options=None):
    return event_winner_highlights(db, club_id="club-1", period=period, now=NOW, players=PLAYERS,
                                   keys=keys, status=status, card_options=options or {})


def test_completed_approved_unrated_events_count_without_badges_or_rated_games():
    db = WinnerDB(live_sessions=[session()])
    original = deepcopy(db.tables)
    result = winners(db)
    assert result["round_robin_wins"][0]["player_id"] == 1
    assert result["round_robin_wins"][0]["metric_display"] == "1 title"
    assert not result["moneyball_wins"] and not result["ladder_court_wins"]
    assert db.tables == original
    assert not any("badge" in table for table in db.reads)


@pytest.mark.parametrize("change", ["pending", "rejected", "processing", "unfinished", "private", "foreign", "future", "prior", "unlinked", "duplicate_link", "invalid", "missing_score"])
def test_ineligible_sources_never_promote_a_runner_up(change):
    row = session()
    ev = row["state"]["page_state"]["event"]
    if change in {"pending", "rejected", "processing"}: row["state"]["generator_submission"]["status"] = change
    if change == "unfinished": ev["status"] = "active"
    if change == "private": row["state"]["mode"] = "private"
    if change == "foreign": row["club_id"] = "other"
    if change == "future": row["state"]["generator_submission"]["match_date"] = "2026-11-12"
    if change == "prior": row["state"]["generator_submission"]["match_date"] = "2026-09-14"
    if change == "unlinked": ev["participants"][0]["player_id"] = None
    if change == "duplicate_link": ev["participants"][1]["player_id"] = 1
    if change == "invalid": ev["rounds"][0]["matches"][0]["scoreA"] = True
    if change == "missing_score": ev["rounds"][0]["matches"][0]["scoreB"] = None
    assert winners(WinnerDB(live_sessions=[row]))["round_robin_wins"] == []


def test_reviewed_date_and_player_mapping_are_authoritative():
    row = session()
    row["state"]["generator_submission"]["review"] = {"request": {"match_date": "2026-10-14", "player_ids": {"p1": 8}}}
    assert winners(WinnerDB(live_sessions=[row]))["round_robin_wins"][0]["player_id"] == 8
    row["state"]["generator_submission"]["review"]["request"]["match_date"] = "2026-09-14"
    assert not winners(WinnerDB(live_sessions=[row]))["round_robin_wins"]


def test_round_robin_tie_requires_current_admin_decision():
    ev = event()
    ev["rounds"] = ev["rounds"][:1]
    assert not winners(WinnerDB(live_sessions=[session(ev)]))["round_robin_wins"]
    ev["round_robin_winner"] = {"participant_id": "p2", "display_fingerprint": round_robin_result_fingerprint(ev)}
    assert winners(WinnerDB(live_sessions=[session(ev)]))["round_robin_wins"][0]["player_id"] == 2
    ev["rounds"][0]["matches"][0]["scoreA"] = 12
    assert not winners(WinnerDB(live_sessions=[session(ev)]))["round_robin_wins"]


def test_saved_event_and_generator_count_once_and_use_normalized_links():
    ev = event()
    people = [{"id": f"person{i}", "event_id": "saved", "participant_key": f"p{i}", "linked_player_id": i} for i in range(1, 5)]
    records = [{"id": f"result{i}", "event_id": "saved", "match_key": g["id"], "score_t1": 11, "score_t2": 5,
                **dict(zip(("t1_p1_participant_id", "t1_p2_participant_id", "t2_p1_participant_id", "t2_p2_participant_id"),
                           [f"person{pid[1:]}" for pid in g["sideA"] + g["sideB"]]))}
               for i, r in enumerate(ev["rounds"], 1) for g in r["matches"]]
    saved = {"id": "saved", "club_id": "club-1", "source_event_uid": "rr-1", "event_date": "2026-10-13", "event_type": "round_robin", "status": "saved", "raw_event_json": ev}
    db = WinnerDB(live_sessions=[session(ev)], live_events=[saved], live_event_participants=people, live_event_matches=records)
    assert winners(db)["round_robin_wins"][0]["metric_value"] == 1
    # A corrected normalized result changes the winner; stale raw JSON cannot win.
    for g in db.tables["live_event_matches"]: g.update(score_t1=5, score_t2=11)
    assert not winners(db)["round_robin_wins"]  # Three-way tie; no inherited win.
    db.tables["live_events"][0]["status"] = "rejected"
    assert not winners(db)["round_robin_wins"]


def test_moneyball_uses_frozen_scoring_and_is_separate_from_round_robin():
    ev = json.loads(Path("tests/fixtures/moneyball_v2_frozen.json").read_text())["event"]
    expected = settlement(ev, standings(ev))["winners"]
    assert expected
    people = deepcopy(PLAYERS)
    for i, p in enumerate(ev["participants"], 1): p["player_id"] = i
    result = event_winner_highlights(WinnerDB(live_sessions=[session(ev)]), club_id="club-1", period=PERIOD, players=people,
                                    keys=EVENT_WIN_CARD_KEYS, status="all", card_options={}, now=NOW)
    mapping = {str(p["id"]): p["player_id"] for p in ev["participants"]}
    assert {r["player_id"] for r in result["moneyball_wins"]} == {mapping[str(w["participantId"])] for w in expected}
    assert not result["round_robin_wins"]
    ev["status"] = "active"
    assert not winners(WinnerDB(live_sessions=[session(ev)]))["moneyball_wins"]


def league_tables():
    ev = event()
    games = [{"id": i, "club_id": "club-1", "league": "Sunday", "date": "2026-10-13T12:00:00+00:00", "score_t1": 11, "score_t2": 5,
              **dict(zip(("t1_p1", "t1_p2", "t2_p1", "t2_p2"), [int(pid[1:]) for pid in g["sideA"] + g["sideB"]]))}
             for i, r in enumerate(ev["rounds"], 1) for g in r["matches"]]
    row = {"id": "round", "club_id": "club-1", "session_id": "league", "status": "submitted", "round_number": 1,
           "match_date": "2026-10-13", "submitted_match_ids": [1, 2, 3], "matches_json": [{**g, "court": 5} for g in games]}
    return {"matches": games, "league_live_rounds": [row], "league_live_sessions": [{"id": "league", "club_id": "club-1", "league_name": "Sunday"}]}


def test_ladder_counts_each_completed_court_round_and_all_tied_first_places():
    db = WinnerDB(**league_tables())
    db.tables["league_live_rounds"].append({**deepcopy(db.tables["league_live_rounds"][0]), "id": "round2", "round_number": 2})
    result = winners(db)["ladder_court_wins"]
    assert [(r["player_id"], r["metric_value"]) for r in result] == [(1, 2)]
    for g in db.tables["matches"]: g.update(score_t1=5, score_t2=11)
    result = winners(db)["ladder_court_wins"]
    assert {r["player_id"] for r in result} == {2, 3, 4}
    assert all(r["rank"] == 1 and r["metric_value"] == 2 for r in result)


@pytest.mark.parametrize("date", ["2026-10-13", "2026-10-13T00:00:00+00:00", "2026-10-13T00:00:00Z", "2026-10-13T19:00:00+00:00", "2026-10-14T01:00:00+00:00"])
def test_ladder_uses_played_date_for_legacy_dates_and_real_local_timestamps(date):
    tables = league_tables()
    for game in tables["matches"]:
        game["date"] = date
    result = winners(WinnerDB(**tables))["ladder_court_wins"]
    assert [(r["player_id"], r["metric_value"]) for r in result] == [(1, 1)]


def test_ladder_legacy_midnight_results_respect_season_boundaries():
    tables = league_tables()
    for game in tables["matches"]:
        game["date"] = "2026-10-13T00:00:00+00:00"
    assert winners(WinnerDB(**tables), period={**PERIOD, "start_date": "2026-10-13"})["ladder_court_wins"]
    assert not winners(WinnerDB(**tables), period={**PERIOD, "start_date": "2026-10-14"})["ladder_court_wins"]
    assert not winners(WinnerDB(**tables), period={**PERIOD, "end_date": "2026-10-12"})["ladder_court_wins"]


def test_public_ladder_card_and_detail_include_legacy_courts():
    sb = fixture()
    sb.store["players"] = deepcopy(PLAYERS)
    settings = sb.store["club_leaderboard_settings"][0]["published"]
    settings.update(cards=["ladder_court_wins"], seasons=[PERIOD], default_season_id="2026")
    tables = league_tables()
    for game in tables["matches"]:
        game["date"] = "2026-10-13T00:00:00+00:00"
    db = WinnerDB(**{**sb.store, **tables})
    card = build_public_leaderboard(db, club_id="club-1", status="all")["highlights"]["ladder_court_wins"]
    detail = build_public_leaderboard(db, club_id="club-1", status="all", stat="ladder_court_wins")["stat_ranking"]["entries"]
    assert [(r["player_id"], r["metric_value"]) for r in card] == [(1, 1)]
    assert [(r["player_id"], r["metric_value"]) for r in detail] == [(1, 1)]


@pytest.mark.parametrize("title", ["Moneyball Oct 10", "Moneyball October 13", "Money ball — October 13", " moneyBALL November 7 "])
def test_legacy_moneyball_titles_use_recorded_rr_rules_in_moneyball_category(title):
    ev = event()
    ev.update(name=title, standingsSort="points", playFormat="singles")
    # The recorded points rule awards Bea; the wins rule would award Alex.
    games = [r["matches"][0] for r in ev["rounds"]]
    for game, a, b, score_a, score_b in zip(games, ["p1", "p1", "p2"], ["p2", "p3", "p3"], [11, 11, 25], [0, 0, 23]):
        game.update(sideA=[a], sideB=[b], scoreA=score_a, scoreB=score_b)
    result = winners(WinnerDB(live_sessions=[session(ev)]))
    assert [(r["player_id"], r["metric_value"]) for r in result["moneyball_wins"]] == [(2, 1)]
    assert not result["round_robin_wins"]
    assert winners(WinnerDB(live_sessions=[session(ev)]), keys={"moneyball_wins"})["moneyball_wins"]


def test_ordinary_rr_and_invalid_frozen_moneyball_never_use_legacy_fallback():
    ev = event()
    ev["name"] = "Evening round robin after Moneyball"
    result = winners(WinnerDB(live_sessions=[session(ev)]))
    assert result["round_robin_wins"] and not result["moneyball_wins"]
    ev.update(name="Moneyball October 13", eventFormat="moneyball_expectations")
    result = winners(WinnerDB(live_sessions=[session(ev)]))
    assert not result["round_robin_wins"] and not result["moneyball_wins"]


@pytest.mark.parametrize("change", ["deleted", "missing", "roster", "date", "league", "draft", "repeat"])
def test_ladder_ignores_incomplete_or_stale_courts(change):
    tables = league_tables()
    if change == "deleted": tables["matches"][0]["deleted_at"] = "2026-10-14"
    if change == "missing": tables["matches"].pop()
    if change == "roster": tables["matches"][0]["t1_p1"] = 8
    if change == "date": tables["matches"][0]["date"] = "2026-10-14T12:00:00Z"
    if change == "league": tables["matches"][0]["league"] = "Other"
    if change == "draft": tables["league_live_rounds"][0]["status"] = "draft"
    if change == "repeat":
        for game in tables["matches"][1:]:
            for slot in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"): game[slot] = tables["matches"][0][slot]
        for g, snapshot in zip(tables["matches"], tables["league_live_rounds"][0]["matches_json"]): snapshot.update(g)
    assert not winners(WinnerDB(**tables))["ladder_court_wins"]


def test_generator_ladder_uses_completed_courts_and_avoids_overall_rr_credit():
    ev = event()
    games = [r["matches"][0] for r in ev["rounds"]]
    ev.update(generatorKind="ladder", type="league", rounds=[{"number": 1, "status": "saved", "courts": [{"courtNumber": 2, "matches": games}]},
                                                            {"number": 2, "status": "draft", "courts": [{"courtNumber": 2, "matches": games}]}])
    result = winners(WinnerDB(live_sessions=[session(ev)]))
    assert result["ladder_court_wins"][0]["metric_value"] == 1
    assert not result["round_robin_wins"]


def test_session_snapshot_refreshes_after_version_change():
    db = WinnerDB(live_sessions=[session()])
    assert winners(db)["round_robin_wins"][0]["player_id"] == 1
    for r in db.tables["live_sessions"][0]["state"]["page_state"]["event"]["rounds"]:
        r["matches"][0].update(scoreA=5, scoreB=11)
    db.tables["live_sessions"][0]["version"] += 1
    assert not winners(db)["round_robin_wins"]


def test_official_scores_override_snapshot_and_deleted_or_changed_rosters_do_not_count():
    tables = league_tables()
    row = session()
    mapping = {f"g{i}": f"context{i}" for i in range(1, 4)}
    row["state"]["official_publish"] = {"published_match_ids": list(mapping), "match_context_by_live_id": mapping}
    for i, g in enumerate(tables["matches"], 1): g["context_id"] = f"context{i}"
    db = WinnerDB(live_sessions=[row], matches=tables["matches"])
    assert winners(db)["round_robin_wins"][0]["player_id"] == 1
    for g in db.tables["matches"]: g.update(score_t1=5, score_t2=11)
    assert not winners(db)["round_robin_wins"]
    for g in db.tables["matches"]: g.update(score_t1=11, score_t2=5)
    db.tables["matches"][0]["deleted_at"] = "2026-10-14"
    assert not winners(db)["round_robin_wins"]
    db.tables["matches"][0].update(deleted_at=None, t1_p1=8)
    assert not winners(db)["round_robin_wins"]


def test_public_cards_and_full_rankings_share_counts_and_keep_season_search_and_depth():
    sb = fixture()
    sb.store["players"] = deepcopy(PLAYERS)
    settings = sb.store["club_leaderboard_settings"][0]["published"]
    settings.update(cards=sorted(EVENT_WIN_CARD_KEYS), seasons=[PERIOD], default_season_id="2026", min_games=99,
                    card_options={"round_robin_wins": {"depth": 1, "minimum": 1}})
    second = session(event(), id="s2")
    second["state"]["page_state"]["event"]["sourceEventUid"] = "rr-2"
    second["state"]["page_state"]["event"]["participants"][0]["player_id"] = 8
    sb.store["live_sessions"] = [session(), second]
    # This fixture implements the same query methods as the public service.
    db = WinnerDB(**sb.store)
    cards = build_public_leaderboard(db, club_id="club-1", status="all")
    assert len(cards["highlights"]["round_robin_wins"]) == 1
    full = build_public_leaderboard(db, club_id="club-1", status="all", stat="round_robin_wins", search="Hal", limit=1)
    ranking = full["stat_ranking"]
    assert ranking["entries"][0]["player_id"] == 8
    assert ranking["entries"][0]["rank"] == 1
    assert ranking["total_qualified"] == 2 and ranking["pagination"]["total"] == 1
    assert ranking["min_games"] == 0 and ranking["minimum_metric"] == "round_robin_wins"
    assert full["period"] == PERIOD


def test_existing_cards_do_not_load_event_sources_and_event_outages_fail_visibly():
    db = WinnerDB(**fixture().store)
    build_public_leaderboard(db, club_id="club-1")
    assert "live_sessions" not in db.reads and "live_events" not in db.reads
    original = db.table
    def fail(name):
        if name == "live_events": raise RuntimeError("unavailable")
        return original(name)
    db.table = fail
    with pytest.raises(LeaderboardDataUnavailable, match="event-winner"):
        build_public_leaderboard(db, club_id="club-1", stat="round_robin_wins")
