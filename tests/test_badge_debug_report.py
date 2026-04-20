import pandas as pd

from jupr_app.domain.gamification.badge_debug import (
    _build_badge_diagnostics,
    build_badge_debug_report,
)


class DummyContext:
    def __init__(self):
        self.club_id = "club-1"
        self.df_matches = pd.DataFrame()
        self.df_players_all = pd.DataFrame()


def test_build_badge_debug_report_smoke():
    ctx = DummyContext()
    report = build_badge_debug_report(
        ctx,
        club_id="club-1",
        league_id=None,
        player_id=1,
        badge_id="participant",
    )

    assert report.badge_id == "participant"
    assert report.club_id == "club-1"
    assert report.player_id == 1
    assert report.errors == []


def test_high_roller_diagnostics_below_threshold():
    facts = pd.DataFrame(
        [{"player_id": 5, "match_id": f"m{i}", "win": True} for i in range(99)]
    )
    evaluation = type("Evaluation", (), {"facts": facts, "ctx": None})()

    diagnostics = _build_badge_diagnostics(
        evaluation=evaluation,
        badge_id="high_roller",
        player_id=5,
    )

    assert diagnostics["filtered_player_distinct_win_match_ids"] == 99
    assert diagnostics["threshold_required"] == 100
    assert diagnostics["qualifies_boolean"] is False


def test_high_roller_diagnostics_at_threshold():
    facts = pd.DataFrame(
        [{"player_id": 8, "match_id": f"m{i}", "win": True} for i in range(100)]
    )
    evaluation = type("Evaluation", (), {"facts": facts, "ctx": None})()

    diagnostics = _build_badge_diagnostics(
        evaluation=evaluation,
        badge_id="high_roller",
        player_id=8,
    )

    assert diagnostics["filtered_player_distinct_win_match_ids"] == 100
    assert diagnostics["qualifies_boolean"] is True


def test_high_roller_diagnostics_counts_distinct_wins():
    facts = pd.DataFrame(
        [
            {"player_id": 11, "match_id": "m1", "win": True},
            {"player_id": 11, "match_id": "m1", "win": True},
            {"player_id": 11, "match_id": "m2", "win": True},
            {"player_id": 11, "match_id": "m3", "win": False},
        ]
    )
    evaluation = type("Evaluation", (), {"facts": facts, "ctx": None})()

    diagnostics = _build_badge_diagnostics(
        evaluation=evaluation,
        badge_id="high_roller",
        player_id=11,
    )

    assert diagnostics["filtered_player_win_rows"] == 3
    assert diagnostics["filtered_player_distinct_win_match_ids"] == 2
