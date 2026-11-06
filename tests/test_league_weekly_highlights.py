from types import SimpleNamespace

import pytest

from jupr_app.services.league_weekly_highlights_service import (
    HighlightsConflict, build_weekly_highlights, save_weekly_highlights, stat_candidates,
)
from tests.test_public_league_results_service import FakeSupabase


def game(mid, a=11, b=9, **extra):
    return {"id": mid, "date": f"2026-10-25T12:{mid:02d}:00Z", "week_num": 2, "week_tag": "Week 2",
            "club_id": "club", "league": "Open", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4,
            "score_t1": a, "score_t2": b, "deleted_at": None,
            **{f"{slot}{suffix}": value for slot in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")
               for suffix, value in (("_r", 1200), ("_r_end", 1204))}, **extra}


NAMES = {1: "Alex", 2: "Blair", 3: "Casey", 4: "Devon"}


def test_actual_metrics_and_qualification():
    cards = {row["id"]: row for row in stat_candidates([game(i) for i in range(1, 5)], NAMES)}
    assert cards["unbeaten:1"]["metric"] == "4–0"
    assert cards["streak:1"]["metric"] == "4 straight wins"
    assert cards["rating_gain:1"]["metric"] == "+0.040 JUPR"
    assert cards["point_difference:1"]["metric"] == "+8 points"
    assert cards["close_wins:1"]["metric"] == "4 close wins"
    assert "unbeaten:3" not in cards
    assert "point_difference:3" not in cards
    short = {row["id"] for row in stat_candidates([game(1), game(2)], NAMES)}
    assert "unbeaten:1" not in short and "win_pct:1" not in short and "streak:1" not in short


def test_streak_breaks_and_missing_ratings_are_not_invented():
    cards = {row["id"]: row for row in stat_candidates([game(1), game(2, 5, 11), game(3), game(4, t1_p1_r=None), game(5)], NAMES)}
    assert cards["streak:1"]["value"] == 3
    assert "unbeaten:1" not in cards
    assert "rating_gain:1" not in cards
    assert cards["rating_gain:2"]["value"] == .05
    invalid = [game(1, 0, 0), game(2, -1, 11), game(3, t1_p1=2), game(4, 3.5, 11)]
    assert stat_candidates(invalid, NAMES) == []


def build(db, matches, week=2, **extra):
    return build_weekly_highlights(db, club_id="club", league_name="Open", week_num=week,
                                  matches=matches, players={pid: {"name": name} for pid, name in NAMES.items()}, **extra)


def test_selection_is_week_club_league_scoped_ordered_and_publicly_sanitized():
    config = {"club_id": "club", "league_name": "Open", "week_num": 2, "mode": "curated",
              "selected_ids": ["point_difference:2", "unbeaten:1"], "revision": 3, "updated_by": "private@example.com"}
    db = FakeSupabase({"league_weekly_highlights": [
        {**config, "club_id": "elsewhere", "selected_ids": []},
        {**config, "league_name": "Other", "selected_ids": []},
        {**config, "week_num": 1, "selected_ids": []}, config,
    ]})
    games = [game(i) for i in range(1, 5)] + [game(5, week_num=1), game(6, deleted_at="deleted")]
    result = build(db, games)
    assert [row["id"] for row in result["cards"]] == ["point_difference:2", "unbeaten:1"]
    assert result["cards"][1]["metric"] == "4–0"
    assert set(result) == {"week_num", "mode", "cards"}
    assert "private@example.com" not in str(result)
    assert build(db, games, week=1)["cards"] == []  # intentional empty does not fall back
    games[0]["score_t1"] = 1
    refreshed = build(db, games, include_candidates=True)
    assert refreshed["unavailable_ids"] == ["point_difference:2", "unbeaten:1"]
    assert refreshed["cards"] == []


def test_complete_courts_use_current_scores_and_reject_changed_or_missing_games():
    games = [game(1), game(2, t1_p2=3, t2_p1=2), game(3, t1_p2=4, t2_p2=2)]
    snapshot = [{**g, "court": 1} for g in games]
    db = FakeSupabase({
        "league_live_sessions": [{"id": "session", "club_id": "club", "league_name": "Open", "week_tag": "Week 2"}],
        "league_live_rounds": [{"id": "round", "club_id": "club", "session_id": "session", "status": "submitted", "round_number": 1,
                                "match_date": "2026-10-25", "matches_json": snapshot, "submitted_match_ids": [1, 2, 3]}],
    })
    def courts(source):
        return next(group["candidates"] for group in build(db, source, include_candidates=True)["categories"] if group["id"] == "court")
    assert courts(games)[0]["name"] == "Alex"
    assert "3–0" in courts(games)[0]["detail"]
    assert courts(games[:-1]) == []
    assert courts([{**games[0], "t1_p1": 99}, *games[1:]]) == []
    assert courts([{**g, "score_t1": 4} for g in games])[0]["name"] != "Alex"


def test_save_rejects_wrong_week_unqualified_choices_operator_and_conflicts(monkeypatch):
    from jupr_app.services import admin_league_results_service as admin
    db = FakeSupabase({})
    editor = build(db, [game(i) for i in range(1, 5)], include_candidates=True)
    monkeypatch.setattr(admin, "build_admin_league_results", lambda *_a, **_k: {"selected_week": 2, "weekly_highlight_selection": editor})
    calls = []
    db.rpc = lambda name, args: calls.append((name, args)) or SimpleNamespace(execute=lambda: SimpleNamespace(data={"revision": 1, "mode": args["p_mode"], "selected_ids": args["p_selected_ids"]}))
    args = dict(club_id="club", league_name="Open", week_num=2, mode="curated", selected_ids=["unbeaten:1"], expected_revision=0,
                actor_email="admin@example.com", actor_role="club_owner")
    assert save_weekly_highlights(db, **args)["revision"] == 1
    assert calls[0][1]["p_club_id"] == "club"
    for patch in ({"week_num": 3}, {"selected_ids": ["unbeaten:99"]}, {"selected_ids": ["unbeaten:1"] * 2}, {"mode": "automatic"}):
        with pytest.raises(ValueError):
            save_weekly_highlights(db, **{**args, **patch})
    with pytest.raises(PermissionError):
        save_weekly_highlights(db, **{**args, "actor_role": "operator"})
    assert len(calls) == 1
    class Conflict(Exception):
        code = "40001"
    def conflict(*_args):
        raise Conflict()
    db.rpc = conflict
    with pytest.raises(HighlightsConflict):
        save_weekly_highlights(db, **args)


def test_full_weekly_rankings_share_preview_evidence_and_keep_complete_field():
    from jupr_app.services.league_weekly_highlights_service import weekly_stat_rankings
    names = {pid: f"Player {pid:02d}" for pid in range(1, 25)}
    games = []
    for start in range(1, 25, 4):
        for index in range(4):
            games.append(game(len(games) + 1, t1_p1=start, t1_p2=start + 1, t2_p1=start + 2, t2_p2=start + 3))
    candidates = stat_candidates(games, names)
    # Other weeks and deleted games must not affect either value or rank.
    games += [game(40, week_num=1), game(41, deleted_at="removed")]
    rankings = weekly_stat_rankings(games, players={pid: {"name": name, "email": "private@example.com"} for pid, name in names.items()}, week_num=2)
    assert "court" not in rankings
    assert rankings["games"]["total_qualified"] == 24
    assert [row["rank"] for row in rankings["games"]["entries"]] == list(range(1, 25))
    for key, ranking in rankings.items():
        expected = [row for row in candidates if row["category"] == key]
        assert [(row["player_id"], row["metric_value"], row["metric_display"]) for row in ranking["entries"]] == [
            (row["player_ids"][0], row["value"], row["metric"]) for row in expected
        ]
    assert "private@example.com" not in str(rankings)


def test_public_weekly_records_match_legacy_cards_and_qualification():
    from jupr_app.services.public_league_results_service import _highlights, _record_stat_rankings
    rows = [
        {"player_id": 1, "player_name": "Alex", "games": 2, "wins": 2, "losses": 0, "rating_delta_jupr": -.1, "win_pct": 100},
        {"player_id": 2, "player_name": "Blair", "games": 5, "wins": 4, "losses": 1, "rating_delta_jupr": .2, "win_pct": 80},
        {"player_id": 3, "player_name": "Casey", "games": 5, "wins": 3, "losses": 2, "rating_delta_jupr": .3, "win_pct": 60},
    ]
    highlights = _highlights(rows, scope="week", min_games=4, week_num=2)
    rankings = _record_stat_rankings(rows, min_games=4)
    for key in ("biggest_climbers", "best_win_pct", "most_active"):
        assert [row["player_id"] for row in rankings[key]["entries"][:3]] == [row["player_id"] for row in highlights[key]]
    assert rankings["record_gain"]["entries"][-1]["metric_value"] == -.1
    assert [row["player_id"] for row in rankings["best_win_pct"]["entries"]] == [2, 3]
    assert _record_stat_rankings(rows, min_games=1)["best_win_pct"]["entries"][0]["player_id"] == 1
