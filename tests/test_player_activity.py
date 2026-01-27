from datetime import datetime, timedelta, timezone

import pandas as pd

from jupr_app.domain.player_activity import (
    add_activity_columns,
    build_player_activity_update,
    should_mark_inactive,
)
from jupr_app.ui.pages.leaderboards import select_leaderboard_players


def test_should_mark_inactive_threshold_boundary():
    now = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
    last_game = now - timedelta(days=14)
    assert should_mark_inactive(last_game, None, now_utc=now) is True

    just_under = now - timedelta(days=14) + timedelta(seconds=1)
    assert should_mark_inactive(just_under, None, now_utc=now) is False


def test_should_mark_inactive_uses_created_at_when_no_games():
    now = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
    created = now - timedelta(days=15)
    assert should_mark_inactive(None, created, now_utc=now) is True


def test_add_activity_columns_sets_active_flag():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "inactive_at": [None, "2024-01-01T00:00:00Z"],
        }
    )
    out = add_activity_columns(df)
    assert bool(out.loc[0, "active"]) is True
    assert bool(out.loc[1, "active"]) is False


def test_build_player_activity_update_reactivates_and_sets_last_game_at():
    existing = "2024-01-01T00:00:00+00:00"
    match_time = datetime(2024, 1, 10, 12, 0, tzinfo=timezone.utc)
    payload = build_player_activity_update(existing, match_time)
    assert payload["inactive_at"] is None
    assert payload["last_game_at"] == match_time.isoformat()


def test_build_player_activity_update_uses_latest_match_time():
    existing = "2024-01-12T00:00:00+00:00"
    match_time = datetime(2024, 1, 10, 12, 0, tzinfo=timezone.utc)
    payload = build_player_activity_update(existing, match_time)
    assert payload["last_game_at"] == "2024-01-12T00:00:00+00:00"


def test_select_leaderboard_players_respects_toggle():
    df_all = pd.DataFrame({"id": [1, 2, 3]})
    df_active = df_all[df_all["id"] != 3].copy()

    assert select_leaderboard_players(df_active, df_all, "Active").equals(df_active)
    assert select_leaderboard_players(df_active, df_all, "See all").equals(df_all)
