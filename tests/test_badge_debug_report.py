import pandas as pd

from jupr_app.domain.gamification.badge_debug import build_badge_debug_report


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
