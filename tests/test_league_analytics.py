from __future__ import annotations

import pytest

from jupr_app.domain.league_analytics import (
    MEASURABLE_PLAYER_STATS,
    award_category_catalog,
    canonical_league_matches,
    compute_league_player_analytics,
    compute_team_league_standings,
)
from jupr_app.services.admin_league_awards_service import (
    _award_records,
    _computed_configured_awards,
)


def _match(match_id: int, **patch):
    row = {
        "id": match_id,
        "club_id": "club",
        "league": "Open",
        "date": f"2026-08-{match_id:02d}",
        "week_tag": f"Week {match_id}",
        "t1_p1": 1,
        "t1_p2": 2,
        "t2_p1": 3,
        "t2_p2": 4,
        "score_t1": 11,
        "score_t2": 9,
        "t1_p1_r": 1600,
        "t1_p2_r": 1600,
        "t2_p1_r": 1800,
        "t2_p2_r": 1800,
    }
    row.update(patch)
    return row


def test_canonical_league_match_filter_reports_every_exclusion() -> None:
    matches = [
        _match(1),
        _match(2, deleted_at="2026-08-10T00:00:00Z"),
        _match(3, excluded_from_ratings=True),
        _match(4, score_t2=11),
        _match(5, t2_p2=2),
        _match(6, league="Other"),
        _match(7, club_id="elsewhere"),
    ]

    result = canonical_league_matches(
        matches, club_id="club", league_name="Open"
    )

    assert [row["id"] for row in result.included] == [1]
    assert result.discovered_count == 5
    assert result.exclusion_counts == {
        "deleted": 1,
        "excluded_from_ratings": 1,
        "tied_score": 1,
        "invalid_player_sides": 1,
    }
    assert result.provenance()["included_match_ids"] == [1]


def test_player_analytics_cover_record_points_upsets_partners_and_attendance() -> None:
    players = [
        {"id": 1, "name": "Alex", "rating": 1600},
        {"id": 2, "name": "Blair", "rating": 1600},
        {"id": 3, "name": "Casey", "rating": 1800},
        {"id": 4, "name": "Devon", "rating": 1800},
    ]
    ratings = [
        {
            "club_id": "club",
            "league_name": "Open",
            "player_id": 1,
            "rating": 1600,
            "starting_rating": 1400,
        }
    ]

    result = compute_league_player_analytics(
        [_match(1)],
        club_id="club",
        league_name="Open",
        players=players,
        league_ratings=ratings,
        expected_weeks=2,
    )
    alex = next(row for row in result["players"] if row["player_id"] == 1)

    assert alex["rating_jupr"] == 4.0
    assert alex["rating_gain_jupr"] == 0.5
    assert (alex["games"], alex["wins"], alex["losses"]) == (1, 1, 0)
    assert (alex["points_for"], alex["points_against"]) == (11, 9)
    assert alex["point_differential"] == 2
    assert alex["close_games"] == alex["close_wins"] == 1
    assert alex["upset_wins"] == 1
    assert alex["largest_upset_jupr"] == 0.5
    assert alex["average_opponent_jupr"] == 4.5
    assert alex["expected_wins"] < 0.5
    assert alex["wins_above_expected"] > 0.5
    assert alex["best_partner_id"] == 2
    assert alex["best_partnership_win_pct"] == 1.0
    assert alex["attendance_pct"] == 0.5


def test_missing_pre_match_rating_never_becomes_fifty_fifty_expectation() -> None:
    partial = _match(
        2,
        score_t1=9,
        score_t2=11,
        t1_p1_r=None,
        t1_p2_r=1600,
        t2_p1_r=1700,
        t2_p2_r=1700,
    )

    result = compute_league_player_analytics(
        [_match(1), partial],
        club_id="club",
        league_name="Open",
    )
    alex = next(row for row in result["players"] if row["player_id"] == 1)

    assert alex["games"] == 2
    assert alex["expected_wins"] is None
    assert alex["wins_above_expected"] is None
    assert alex["expected_model"] is None


def test_singles_match_is_canonical_without_partner_slots() -> None:
    singles_match = _match(
        8,
        t1_p2=None,
        t2_p2=None,
        t1_p1_r=1600,
        t2_p1_r=1800,
    )

    result = compute_league_player_analytics(
        [singles_match],
        club_id="club",
        league_name="Open",
        match_format="singles",
        players=[
            {"id": 1, "name": "Alex", "rating": 1610},
            {"id": 3, "name": "Casey", "rating": 1790},
        ],
    )

    assert result["provenance"]["included_count"] == 1
    assert result["provenance"]["exclusion_counts"] == {}
    assert {row["player_name"] for row in result["players"]} == {"Alex", "Casey"}
    alex = next(row for row in result["players"] if row["player_id"] == 1)
    assert (alex["games"], alex["wins"], alex["losses"]) == (1, 1, 0)
    assert alex["best_partner_id"] is None
    assert alex["partner_variety"] == 0
    assert alex["expected_model"] == "canonical_elo_pre_match_singles_v1"


def test_team_standings_use_head_to_head_before_point_differential() -> None:
    teams = [
        {
            "id": team_id,
            "team_name": f"Team {team_id}",
            "status": "confirmed",
        }
        for team_id in ("A", "B", "C", "D")
    ]
    fixtures = [
        # A and B both finish 2-1; A owns their direct meeting.
        {"phase": "regular", "status": "complete", "team_a_id": "A", "team_b_id": "B", "team_a_score": 11, "team_b_score": 10},
        {"phase": "regular", "status": "complete", "team_a_id": "A", "team_b_id": "C", "team_a_score": 11, "team_b_score": 10},
        {"phase": "regular", "status": "complete", "team_a_id": "D", "team_b_id": "A", "team_a_score": 11, "team_b_score": 0},
        {"phase": "regular", "status": "complete", "team_a_id": "B", "team_b_id": "C", "team_a_score": 11, "team_b_score": 0},
        {"phase": "regular", "status": "complete", "team_a_id": "B", "team_b_id": "D", "team_a_score": 11, "team_b_score": 0},
        {"phase": "regular", "status": "complete", "team_a_id": "C", "team_b_id": "D", "team_a_score": 11, "team_b_score": 10},
    ]

    standings = compute_team_league_standings(fixtures, teams)
    a = next(row for row in standings if row["team_id"] == "A")
    b = next(row for row in standings if row["team_id"] == "B")

    assert a["wins"] == b["wins"] == 2
    assert a["losses"] == b["losses"] == 1
    assert b["point_differential"] > a["point_differential"]
    assert a["head_to_head_score"] > b["head_to_head_score"]
    assert a["rank"] < b["rank"]
    assert a["standing_score"] > b["standing_score"]

    champion_awards = _computed_configured_awards(
        {
            "catalog": award_category_catalog(),
            "player_analytics": [],
            "team_analytics": standings,
        },
        {
            "categories": {
                "team_champion": {
                    "enabled": True,
                    "depth": 1,
                    "minimum": 1,
                }
            }
        },
    )
    assert champion_awards is not None
    assert [
        row["team_id"]
        for row in champion_awards
        if row["category_key"] == "team_champion"
    ] == ["A"]


def test_forfeit_counts_in_team_record_without_rated_points() -> None:
    teams = [
        {"id": "A", "team_name": "Aces", "status": "confirmed"},
        {"id": "B", "team_name": "Dinkers", "status": "confirmed"},
    ]
    standings = compute_team_league_standings(
        [
            {
                "phase": "regular",
                "status": "forfeit",
                "team_a_id": "A",
                "team_b_id": "B",
                "winner_team_id": "B",
                "team_a_score": None,
                "team_b_score": None,
            }
        ],
        teams,
    )

    assert standings[0]["team_id"] == "B"
    assert (standings[0]["wins"], standings[0]["losses"]) == (1, 0)
    assert (standings[1]["wins"], standings[1]["losses"]) == (0, 1)
    assert standings[0]["points_for"] == standings[0]["points_against"] == 0


def test_award_catalog_exposes_player_and_team_measures() -> None:
    catalog = award_category_catalog()
    by_key = {row["key"]: row for row in catalog}

    assert len(catalog) == 20
    assert by_key["attendance"]["metric"] == "attendance_pct"
    assert by_key["over_performance"]["metric"] == "wins_above_expected"
    assert by_key["best_partnership"]["minimum_metric"] == "best_partnership_games"
    assert by_key["team_champion"]["recipient_type"] == "team"
    assert by_key["team_wins"]["minimum_metric"] == "games_played"
    assert (
        by_key["team_point_differential"]["minimum_metric"]
        == "games_played"
    )
    assert {
        "points_for",
        "longest_win_streak",
        "largest_upset_jupr",
        "partner_variety",
        "attendance_pct",
    } <= set(MEASURABLE_PLAYER_STATS)


def test_configured_awards_keep_deterministic_cutoff_ties_and_team_recipient() -> None:
    analytics = {
        "catalog": award_category_catalog(),
        "player_analytics": [
            {"player_id": 2, "player_name": "Blair", "games": 5, "wins": 5},
            {"player_id": 1, "player_name": "Alex", "games": 5, "wins": 5},
            {"player_id": 3, "player_name": "Casey", "games": 5, "wins": 4},
        ],
        "team_analytics": [
            {
                "team_id": "t1",
                "team_name": "Aces",
                "games_played": 3,
                "standing_score": 3,
            }
        ],
    }
    config = {
        "categories": {
            "most_wins": {"enabled": True, "depth": 1, "minimum": 1},
            "team_champion": {"enabled": True, "depth": 1, "minimum": 1},
        }
    }

    awards = _computed_configured_awards(analytics, config)

    assert awards is not None
    player_awards = [
        row for row in awards if row["category_key"] == "most_wins"
    ]
    assert [row["player_name"] for row in player_awards] == ["Alex", "Blair"]
    assert all(row["is_co_winner"] for row in player_awards)
    team_award = next(
        row for row in awards if row["category_key"] == "team_champion"
    )
    assert team_award["recipient_type"] == "team"
    assert team_award["team_id"] == "t1"


def test_award_cutoff_ties_have_unique_identities_and_correct_places() -> None:
    analytics = {
        "catalog": award_category_catalog(),
        "player_analytics": [
            {"player_id": 1, "player_name": "Alex", "games": 6, "wins": 6},
            {"player_id": 2, "player_name": "Blair", "games": 6, "wins": 5},
            {"player_id": 3, "player_name": "Casey", "games": 6, "wins": 5},
            {"player_id": 4, "player_name": "Devon", "games": 6, "wins": 4},
        ],
        "team_analytics": [],
    }
    awards = _computed_configured_awards(
        analytics,
        {
            "categories": {
                "most_wins": {
                    "enabled": True,
                    "depth": 2,
                    "minimum": 1,
                }
            }
        },
    )

    assert awards is not None
    assert [
        (row["player_name"], row["rank"], row["is_co_winner"])
        for row in awards
    ] == [
        ("Alex", 1, False),
        ("Blair", 2, True),
        ("Casey", 2, True),
    ]
    assert len({row["award_key"] for row in awards}) == 3


def test_durable_award_record_keeps_computed_and_override_evidence() -> None:
    records = _award_records(
        [
            {
                "award_key": "most_wins:1:player:1",
                "category_key": "most_wins",
                "category_label": "Most Wins",
                "recipient_type": "player",
                "player_id": 2,
                "player_name": "Blair",
                "recipient_name": "Blair",
                "rank": 1,
                "metric_value": 4,
                "metric_display": "4",
                "computed_player_id": 1,
                "computed_recipient_name": "Alex",
                "computed_metric_value": 6,
            }
        ],
        source_snapshot={"provenance": {"rule_version": "test"}},
        override_notes={
            "most_wins:1:player:1": "Committee correction after score review"
        },
    )

    assert records == [
        {
            "award_key": "most_wins:1:player:1",
            "category_key": "most_wins",
            "category_label": "Most Wins",
            "recipient_type": "player",
            "player_id": 2,
            "team_id": None,
            "recipient_name": "Blair",
            "placement": 1,
            "is_co_winner": False,
            "metric_value": 4,
            "computed_metric_value": 6,
            "computed_player_id": 1,
            "computed_team_id": None,
            "computed_recipient_name": "Alex",
            "metric_display": "4",
            "manual_label": None,
            "is_override": True,
            "override_reason": "Committee correction after score review",
            "public_visible": True,
            "source_snapshot": {
                "provenance": {"rule_version": "test"}
            },
        }
    ]
