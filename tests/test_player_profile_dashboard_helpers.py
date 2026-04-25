import pandas as pd

from jupr_app.ui.pages.players import (
    _aggregate_repeating_badges,
    _build_profile_snapshot,
    _month_day,
    _player_is_claimed_or_verified,
)


def test_aggregate_repeating_badges_stacks_duplicates():
    badges = [
        {"badge_id": "alpha", "name": "Alpha", "stack_count": 1, "prestige": 4, "last_earned_at": "2026-01-01"},
        {"badge_id": "alpha", "name": "Alpha", "stack_count": 2, "prestige": 4, "last_earned_at": "2026-02-01"},
        {"badge_id": "beta", "name": "Beta", "stack_count": 1, "prestige": 2, "last_earned_at": "2026-01-15"},
    ]
    out = _aggregate_repeating_badges(badges)
    by_id = {row["badge_id"]: row for row in out}
    assert by_id["alpha"]["stack_count"] == 3
    assert str(by_id["alpha"]["last_earned_at"]).startswith("2026-02-01")
    assert by_id["beta"]["stack_count"] == 1


def test_build_profile_snapshot_formats_summary():
    df = pd.DataFrame(
        [
            {"id": 1, "Date": "2026-01-01", "Result": "WIN", "Overall Δ": 0.01, "Overall After": 3.0},
            {"id": 2, "Date": "2026-01-02", "Result": "LOSS", "Overall Δ": -0.02, "Overall After": 2.98},
            {"id": 3, "Date": "2026-01-03", "Result": "WIN", "Overall Δ": 0.03, "Overall After": 3.01},
        ]
    )
    row = pd.Series({"rating": 1200.0})
    snap = _build_profile_snapshot(df, row, pid=7)

    assert snap["wins"] == 2
    assert snap["losses"] == 1
    assert snap["matches"] == 3
    assert snap["win_pct"] == (2 / 3) * 100
    labels = {item["Stat"]: item["Value"] for item in snap["stats"]}
    assert labels["Current JUPR"] == "3.010"
    assert labels["Rated matches"] == "3"


def test_build_profile_snapshot_empty_state_uses_player_rating():
    row = pd.Series({"rating": 1600.0})
    snap = _build_profile_snapshot(pd.DataFrame(), row, pid=99)
    labels = {item["Stat"]: item["Value"] for item in snap["stats"]}
    assert labels["Current JUPR"] == "4.000"
    assert labels["Rated matches"] == "0"


def test_claimed_or_verified_detection_and_month_day_formatting():
    assert _player_is_claimed_or_verified(pd.Series({"is_verified": True})) is True
    assert _player_is_claimed_or_verified(pd.Series({"verified_at": "2026-01-01"})) is True
    assert _player_is_claimed_or_verified(pd.Series({"is_verified": False, "verified_at": ""})) is False
    assert _month_day("2026-04-22") == "April 22"
