from copy import deepcopy
import json

import pytest

from jupr_app.domain.leaderboard_metrics import EXTENDED_CARD_KEYS, compute_overall_metrics, overall_highlights
from jupr_app.services.leaderboard_service import build_public_leaderboard
from tests.test_leaderboard_seasons import PERIOD, fixture, match


def game(mid, when="2024-09-15T08:00:00Z", **changes):
    return match(mid, when, **{**{f"{slot}_r": 1400 for slot in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")}, **changes})


def setup(*, all_time=False):
    sb = fixture()
    config = sb.store["pcs_club_sites"][0]["published"]["leaderboard"]
    config["cards"] = sorted(EXTENDED_CARD_KEYS)
    if all_time:
        config["default_season_id"] = None
    return sb, config


def metric(payload, card, player=1):
    return next(row for row in payload["highlights"][card] if row["player_id"] == player)


def test_extended_cards_share_season_history_and_use_chronological_results_and_local_days():
    sb, _ = setup()
    sb.store["matches"] = [
        game(4, "2024-09-16T07:00:00Z", score_t1=9, score_t2=11),
        game(2, "2024-09-15T09:00:00Z", score_t2=5),
        game(5, "2024-09-16T07:00:00Z", score_t1=9, score_t2=9),
        game(1, score_t2=9, t1_p1_r=1200, t1_p2_r=1200),
        game(3, "2024-09-16T06:59:00Z", score_t2=8),
        game(99, "2024-09-15T06:59:59Z", score_t2=0),
        game(100, "2024-09-17T07:00:00Z", score_t2=0),
    ]
    original = deepcopy(sb.store)
    payload = build_public_leaderboard(sb, club_id="club-1")
    assert len(sb.match_queries) == 2  # One history scan: data page + exhaustion page.
    assert metric(payload, "point_differential")["metric_value"] == 9
    assert metric(payload, "average_margin")["metric_value"] == pytest.approx(1.8)
    assert metric(payload, "longest_win_streak")["metric_value"] == 3
    assert payload["highlights"]["hot_hand"] == []  # final tie resets both sides
    assert metric(payload, "close_game_record")["metric_value"] == pytest.approx(100 / 3)
    assert metric(payload, "close_game_record")["metric_sample"] == 3
    assert metric(payload, "biggest_upset")["metric_value"] == .5
    assert metric(payload, "most_upsets")["metric_value"] == 1
    assert metric(payload, "opponent_strength")["metric_value"] == 3.5
    assert metric(payload, "over_performance")["metric_value"] == pytest.approx(3 - (1 / (1 + 10 ** .5) + 2))
    assert metric(payload, "playing_days")["metric_value"] == 2
    assert metric(payload, "partner_variety")["metric_value"] == 1
    pair = metric(payload, "best_partnership")
    assert pair["metric_value"] == 60
    assert pair["metric_sample"] == 5
    assert "Blake Baseline" in pair["metric_display"]
    assert sb.store == original


@pytest.mark.parametrize("missing", [None, "NaN", "Infinity"])
def test_incomplete_ratings_make_entire_rating_derived_period_unknown(missing):
    sb, _ = setup()
    sb.store["matches"] = [game(1, t1_p1_r=1200, t1_p2_r=1200), game(2, t2_p2_r=missing)]
    payload = build_public_leaderboard(sb, club_id="club-1")
    for key in ("most_upsets", "biggest_upset", "opponent_strength", "over_performance"):
        assert payload["highlights"][key] == []
    assert metric(payload, "point_differential")["metric_value"] == 8
    assert metric(payload, "longest_win_streak")["metric_value"] == 2


def test_exact_quarter_point_upset_is_inclusive_despite_float_conversion():
    sb, _ = setup()
    sb.store["matches"] = [game(1, t1_p1_r=1506, t1_p2_r=1506, t2_p1_r=1606, t2_p2_r=1606)]
    payload = build_public_leaderboard(sb, club_id="club-1")
    assert metric(payload, "most_upsets")["metric_value"] == 1
    assert metric(payload, "biggest_upset")["metric_value"] == pytest.approx(.25)


@pytest.mark.parametrize("invalid", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("snapshot", ["t1_p1_r", "t1_p1_r_end"])
def test_nonfinite_season_snapshots_remain_unknown_and_entire_payload_is_json_safe(invalid, snapshot):
    sb, _ = setup()
    sb.store["matches"] = [game(1, **{snapshot: invalid})]
    payload = build_public_leaderboard(sb, club_id="club-1")
    player = next(row for row in payload["leaderboard"] if row["player_id"] == 1)
    assert player["rating_gain_jupr"] is None
    if snapshot == "t1_p1_r":
        assert player["starting_rating"] is None
    assert all(row["player_id"] != 1 for row in payload["highlights"]["most_improved"])
    json.dumps(payload, allow_nan=False)


def test_best_partnership_selects_best_qualifying_pair_instead_of_rejecting_best_small_sample():
    sb, config = setup()
    config["card_options"] = {"best_partnership": {"minimum": 3, "depth": 5}}
    sb.store["matches"] = [game(1)] + [
        game(mid, t1_p2=3, t2_p1=2, score_t1=11 if mid != 4 else 5)
        for mid in (2, 3, 4)
    ]
    row = metric(build_public_leaderboard(sb, club_id="club-1"), "best_partnership")
    assert row["metric_value"] == pytest.approx(200 / 3)
    assert row["metric_sample"] == 3
    assert "Casey Court" in row["metric_display"]


def test_global_games_and_per_card_relevant_sample_both_apply_without_hiding_table_rows():
    sb, config = setup()
    sb.store["matches"] = [game(1, score_t2=9), game(2)]
    config["min_games"] = 2
    config["card_options"] = {"close_game_record": {"minimum": 2, "depth": 5},
                              "longest_win_streak": {"minimum": 0, "depth": 1}}
    payload = build_public_leaderboard(sb, club_id="club-1")
    assert len(payload["leaderboard"]) == 2
    assert payload["highlights"]["close_game_record"] == []
    assert len(payload["highlights"]["longest_win_streak"]) == 1
    config["min_games"] = 3
    payload = build_public_leaderboard(sb, club_id="club-1")
    assert payload["highlights"]["longest_win_streak"] == []
    assert payload["highlights"]["highest_rating"]
    config["card_options"]["highest_rating"] = {"minimum": 3, "depth": 5}
    assert build_public_leaderboard(sb, club_id="club-1")["highlights"]["highest_rating"] == []


def test_all_time_legacy_cards_do_not_read_match_history_and_extended_cards_read_once():
    sb = fixture()
    config = sb.store["pcs_club_sites"][0]["published"]["leaderboard"]
    config["default_season_id"] = None
    build_public_leaderboard(sb, club_id="club-1")
    assert sb.match_queries == []
    config["cards"] = ["playing_days"]
    config["timezone"] = "America/Mazatlan"
    sb.store["matches"] = [game(1, "2024-09-15T23:00:00Z"), game(2, "2024-09-16T01:00:00Z")]
    payload = build_public_leaderboard(sb, club_id="club-1")
    assert len(sb.match_queries) == 2  # One history scan: data page + exhaustion page.
    assert metric(payload, "playing_days")["metric_value"] == 1
    assert payload["period"]["timezone"] == "America/Mazatlan"
    config["timezone"] = "UTC"
    assert metric(build_public_leaderboard(sb, club_id="club-1"), "playing_days")["metric_value"] == 2


def test_extended_stats_use_same_club_deleted_unrated_singles_and_duplicate_filters():
    sb, _ = setup(all_time=True)
    sb.store["matches"] = [game(1), game(1), game(2, rating_scope="unrated"), game(3, rating_scope="singles"),
                           game(4, match_format="singles"), game(5, deleted_at="2024-09-15T10:00:00Z"),
                           game(6, club_id="other-club"), game(7, t1_p2=None), game(8, t2_p2=1),
                           game(9, score_t1=0, score_t2=0), game(10, score_t1=10.8),
                           game(11, score_t1=True), game(12, score_t2="NaN")]
    payload = build_public_leaderboard(sb, club_id="club-1")
    assert metric(payload, "point_differential")["metric_value"] == 4
    assert metric(payload, "longest_win_streak")["metric_sample"] == 1
    assert all(row["player_id"] != 99 for rows in payload["highlights"].values() for row in rows)


def test_new_empty_season_has_no_extended_leaders_and_does_not_erase_rating():
    sb, config = setup()
    config["seasons"] = [{**PERIOD, "start_date": "2099-09-15", "end_date": None}]
    payload = build_public_leaderboard(sb, club_id="club-1")
    assert all(payload["highlights"][key] == [] for key in EXTENDED_CARD_KEYS)
    assert payload["highlights"]["highest_rating"]


def test_pair_ties_are_stable_and_depth_can_show_ten_players():
    matches = [game(1, t1_p2=3, t2_p1=2), game(2)]
    stats = compute_overall_metrics(matches, names={"2": "Second", "3": "Third"}, timezone="UTC")
    assert stats["1"]["best_partner_name"] == "Second"
    rows = [{"player_id": i, "rank": i, "rating_jupr": 4, "matches_played": 0} for i in range(1, 13)]
    cards = overall_highlights(rows, metrics={}, min_games=10,
                               card_options={"highest_rating": {"minimum": 0, "depth": 10}})
    assert len(cards["highest_rating"]) == 10
