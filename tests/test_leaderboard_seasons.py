from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from jupr_app.services.leaderboard_service import (
    LeaderboardDataUnavailable,
    LeaderboardPeriodInvalid,
    _period_rows,
    build_public_leaderboard,
)
from tests.test_public_leaderboard_projection import _Query, _Response, _Supabase as _BaseSupabase, _fixture


class _SeasonQuery(_Query):
    def __init__(self, store, table_name, comparisons):
        super().__init__(store, table_name)
        self.comparisons = comparisons

    def gte(self, key, value):
        self.comparisons.append((key, "gte", value))
        return self

    def lte(self, key, value):
        self.comparisons.append((key, "lte", value))
        return self

    def lt(self, key, value):
        self.comparisons.append((key, "lt", value))
        return self

    def execute(self):
        rows = [dict(row) for row in self.store.get(self.table_name, [])]
        for key, value in self.filters.items():
            rows = [row for row in rows if str(row.get(key)) == str(value)]
        for key, operator, value in self.comparisons:
            cutoff = datetime.fromisoformat(value.replace("Z", "+00:00"))
            selected = []
            for row in rows:
                try:
                    actual = datetime.fromisoformat(str(row.get(key)).replace("Z", "+00:00"))
                except ValueError:
                    continue
                if {"gte": actual >= cutoff, "lte": actual <= cutoff, "lt": actual < cutoff}[operator]:
                    selected.append(row)
            rows = selected
        if hasattr(self, "page_bounds"):
            rows = rows[self.page_bounds[0]:self.page_bounds[1] + 1]
        return _Response(rows)


class _Supabase(_BaseSupabase):
    def __init__(self, store):
        super().__init__(store)
        self.match_queries = []

    def table(self, name):
        comparisons = []
        if name == "matches":
            self.match_queries.append(comparisons)
        return _SeasonQuery(self.store, name, comparisons)


PERIOD = {"id": "season-2024", "name": "2024 season", "start_date": "2024-09-15",
          "end_date": "2024-09-16", "timezone": "America/Mazatlan"}


def match(mid, when, *, before=1550, after=1600, **extra):
    return {"id": mid, "club_id": "club-1", "date": when, "match_format": "doubles",
            "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4,
            "score_t1": 11, "score_t2": 7, "t1_p1_r": before, "t1_p1_r_end": after, **extra}


def fixture(*, ongoing=False):
    sb = _Supabase(_fixture().store)
    sb.store["players"][0].update(rating=1800, starting_rating=1000, wins=131, losses=19, matches_played=150)
    season = {**PERIOD, "end_date": None} if ongoing else dict(PERIOD)
    sb.store["club_leaderboard_settings"] = [{"club_id": "club-1", "published": {
        "cards": ["most_matches", "most_improved"], "show_summary": False,
        "seasons": [season], "default_season_id": season["id"], "min_games": 0,
    }, "draft": {"cards": []}}]
    sb.store["matches"] = [
        match(1, "2024-09-15T06:59:59Z", before=1500, after=1550),
        match(2, "2024-09-15T07:00:00Z", before=1550, after=1600),
        # Losing while beating expectation can increase JUPR. Use slot snapshots.
        match(3, "2024-09-17T06:59:59Z", before=1600, after=1610, score_t1=9, score_t2=11),
        match(4, "2024-09-17T07:00:00Z", before=1610, after=1800),
    ]
    return sb


def first(payload):
    return next(row for row in payload["leaderboard"] if row["player_id"] == 1)


def test_season_uses_local_boundaries_and_freezes_past_gain_while_retaining_current_rating():
    sb = fixture()
    original = deepcopy(sb.store)
    payload = build_public_leaderboard(sb, club_id="club-1")
    row = first(payload)
    assert row["rating"] == 1800
    assert row["starting_rating"] == 1550
    assert row["rating_gain_jupr"] == pytest.approx(60 / 400)
    assert (row["wins"], row["losses"], row["matches_played"], row["win_pct"]) == (1, 1, 2, 50)
    assert payload["period"] == PERIOD
    assert payload["leaderboard_settings"]["cards"] == ["most_matches", "most_improved"]
    assert payload["leaderboard_settings"]["show_summary"] is False
    assert "draft" not in payload
    assert sb.store == original
    assert sb.match_queries
    for query in sb.match_queries:
        assert ("date", "gte", "2024-09-15T07:00:00+00:00") in query
        assert ("date", "lt", "2024-09-17T07:00:00+00:00") in query
        assert any(key == "date" and operator == "lte" for key, operator, _ in query)


def test_ongoing_season_uses_current_rating_from_season_start_and_all_time_is_available():
    sb = fixture(ongoing=True)
    row = first(build_public_leaderboard(sb, club_id="club-1"))
    assert row["rating_gain_jupr"] == pytest.approx(250 / 400)
    assert (row["wins"], row["losses"], row["matches_played"]) == (2, 1, 3)
    lifetime = build_public_leaderboard(sb, club_id="club-1", season="all")
    row = first(lifetime)
    assert (row["wins"], row["matches_played"], row["starting_rating"]) == (131, 150, 1000)
    assert row["rating_gain_jupr"] == 2
    assert lifetime["period"]["id"] is None


def test_off_season_admin_rating_correction_is_not_reported_as_seasonal_improvement():
    sb = fixture()
    # The prior season finished at 1400; staff corrected the player's seed before
    # the new season. New season play starts at 1550 and finishes at 1610.
    sb.store["matches"][0]["t1_p1_r_end"] = 1400
    row = first(build_public_leaderboard(sb, club_id="club-1"))
    assert row["starting_rating"] == 1550
    assert row["rating_gain_jupr"] == pytest.approx(60 / 400)


def test_missing_first_season_snapshot_remains_unknown_despite_prior_history():
    sb = fixture()
    sb.store["matches"][1]["t1_p1_r"] = None
    row = first(build_public_leaderboard(sb, club_id="club-1"))
    assert row["starting_rating"] is None
    assert row["rating_gain_jupr"] is None


def test_new_season_resets_stus_period_stats_without_resetting_any_rating():
    sb = fixture(ongoing=True)
    config = sb.store["club_leaderboard_settings"][0]["published"]
    config["seasons"].append({"id": "next-season", "name": "Next season", "start_date": "2099-09-15",
                              "end_date": None, "timezone": "America/Mazatlan"})
    config["default_season_id"] = "next-season"
    payload = build_public_leaderboard(sb, club_id="club-1")
    row = first(payload)
    assert row["rating"] == 1800
    assert row["rating_gain_jupr"] == 0
    assert row["matches_played"] == row["wins"] == row["losses"] == 0
    assert row["win_pct"] is None
    assert payload["highlights"]["highest_rating"]
    assert all(not payload["highlights"][key] for key in ("most_improved", "most_wins", "most_matches", "best_win_pct"))


def test_only_eligible_overall_matches_count_and_history_is_not_limited_to_first_page():
    sb = fixture()
    sb.store["matches"] = [match(10, "2024-09-15T08:00:00Z", match_type="PopUp"),
                           match(11, "2024-09-15T08:01:00Z", match_type="Tournament", tournament_id=4)]
    excluded = [
        {"deleted_at": "2024-09-15T12:00:00Z"}, {"rating_scope": "unrated"}, {"match_format": "singles"},
        {"t1_p2": None, "t2_p2": None}, {"t2_p2": 1}, {"club_id": "other-club"},
        {"score_t1": 0, "score_t2": 0}, {"score_t1": -1}, {"score_t1": None},
        {"date": "invalid"}, {"date": "2099-01-01T00:00:00Z"},
    ]
    sb.store["matches"] += [match(100 + i, "2024-09-15T12:00:00Z", **item) if "date" not in item
                             else match(100 + i, item["date"]) for i, item in enumerate(excluded)]
    # A second page must contribute; duplicate match IDs cannot double count.
    sb.store["matches"] += [match(1000 + i, "2024-09-15T12:00:00Z") for i in range(510)]
    sb.store["matches"].append(dict(sb.store["matches"][0]))
    row = first(build_public_leaderboard(sb, club_id="club-1"))
    assert row["matches_played"] == row["wins"] == 512


def test_missing_snapshots_never_use_lifetime_start_or_assume_loser_loses_rating():
    sb = fixture()
    sb.store["matches"] = [match(1, "2024-09-15T08:00:00Z", before=None, after=1600)]
    payload = build_public_leaderboard(sb, club_id="club-1")
    assert first(payload)["rating_gain_jupr"] is None
    assert first(payload)["starting_rating"] is None
    assert payload["highlights"]["most_improved"] == []
    sb.store["matches"][0].update(t1_p1_r=1500, t1_p1_r_end=None)
    assert first(build_public_leaderboard(sb, club_id="club-1"))["rating_gain_jupr"] is None
    sb.store["matches"][0].update(t1_p1_r_end=1550)
    assert first(build_public_leaderboard(sb, club_id="club-1"))["rating_gain_jupr"] == pytest.approx(.125)


def test_legacy_tied_result_counts_as_a_match_without_inventing_a_win_or_loss():
    sb = fixture()
    sb.store["matches"] = [match(1, "2024-09-15T08:00:00Z", score_t1=9, score_t2=9, before=1550, after=1550)]
    row = first(build_public_leaderboard(sb, club_id="club-1"))
    assert (row["wins"], row["losses"], row["matches_played"]) == (0, 0, 1)
    assert row["rating_gain_jupr"] == row["win_pct"] == 0


def test_minimum_games_filters_performance_cards_without_hiding_current_rating_or_rows():
    sb = fixture()
    sb.store["club_leaderboard_settings"][0]["published"]["min_games"] = 3
    payload = build_public_leaderboard(sb, club_id="club-1", search="Avery", limit=1)
    assert len(payload["leaderboard"]) == 1
    assert payload["highlights"]["highest_rating"]
    assert not payload["highlights"]["most_improved"]
    assert not payload["highlights"]["best_win_pct"]


def test_invalid_or_foreign_season_fails_but_league_stats_ignore_overall_season():
    sb = fixture()
    with pytest.raises(LeaderboardPeriodInvalid):
        build_public_leaderboard(sb, club_id="club-1", season="not-this-club")
    payload = build_public_leaderboard(sb, club_id="club-1", league_name="Pro", season="not-this-club")
    assert first(payload)["matches_played"] == 8
    assert payload["period"]["id"] is None


def test_defaults_without_site_are_lifetime_and_site_outage_does_not_silently_reset_period():
    payload = build_public_leaderboard(_fixture(), club_id="club-1")
    assert first(payload)["matches_played"] == 10
    assert payload["period"]["id"] is None

    class BrokenSite(_Supabase):
        def table(self, name):
            if name == "club_leaderboard_settings":
                raise RuntimeError("upstream unavailable")
            return super().table(name)

    with pytest.raises(LeaderboardDataUnavailable):
        build_public_leaderboard(BrokenSite(fixture().store), club_id="club-1")


def test_closed_end_date_is_inclusive_and_current_rating_used_until_local_midnight():
    sb = fixture()
    rows = [{"player_id": 1, "rating": 1700}]
    in_progress = _period_rows(sb, rows, club_id="club-1", period=PERIOD,
                               now=datetime(2024, 9, 17, 6, 59, 59, tzinfo=timezone.utc))
    assert in_progress[0]["rating_gain_jupr"] == pytest.approx(150 / 400)
    closed = _period_rows(sb, rows, club_id="club-1", period=PERIOD,
                         now=datetime(2024, 9, 17, 7, tzinfo=timezone.utc))
    assert closed[0]["rating_gain_jupr"] == pytest.approx(60 / 400)
