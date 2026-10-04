from datetime import datetime, timedelta, timezone

import pandas as pd

from jupr_app.domain.player_activity import (
    add_activity_columns,
    build_player_activity_update,
    is_player_leaderboard_active,
    should_mark_inactive,
)
from jupr_app.ui.pages.leaderboards import select_leaderboard_players


def test_should_mark_inactive_threshold_boundary():
    now = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
    last_game = now - timedelta(days=14)
    assert should_mark_inactive(last_game, None, now_utc=now) is True

    just_under = now - timedelta(days=14) + timedelta(seconds=1)
    assert should_mark_inactive(just_under, None, now_utc=now) is False


def test_never_played_profiles_are_inactive_regardless_of_creation_date():
    now = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
    for created in (None, now, now - timedelta(days=15)):
        assert should_mark_inactive(None, created, now_utc=now) is True
    assert should_mark_inactive(now + timedelta(days=1), now_utc=now) is True


def test_add_activity_columns_sets_active_flag():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "inactive_at": [None, "2024-01-01T00:00:00Z", None],
            "last_game_at": [datetime.now(timezone.utc).isoformat(), "2024-01-01T00:00:00Z", None],
        }
    )
    out = add_activity_columns(df)
    assert bool(out.loc[0, "active"]) is True
    assert bool(out.loc[1, "active"]) is False
    assert bool(out.loc[2, "active"]) is False


def test_build_player_activity_update_reactivates_and_sets_last_game_at():
    existing = "2024-01-01T00:00:00+00:00"
    match_time = datetime(2024, 1, 10, 12, 0, tzinfo=timezone.utc)
    payload = build_player_activity_update(existing, match_time, now_utc=match_time)
    assert payload["active"] is True
    assert payload["inactive_at"] is None
    assert payload["last_game_at"] == match_time.isoformat()


def test_build_player_activity_update_uses_latest_match_time():
    existing = "2024-01-12T00:00:00+00:00"
    match_time = datetime(2024, 1, 10, 12, 0, tzinfo=timezone.utc)
    payload = build_player_activity_update(existing, match_time)
    assert payload["last_game_at"] == "2024-01-12T00:00:00+00:00"


def test_entering_historical_game_does_not_make_player_currently_active():
    now = datetime(2026, 10, 4, tzinfo=timezone.utc)
    old_game = now - timedelta(days=30)
    payload = build_player_activity_update(None, old_game, now_utc=now)
    assert payload == {"last_game_at": old_game.isoformat(), "active": False, "inactive_at": now.isoformat()}
    # An older import must not replace a more recent recorded game.
    recent = now - timedelta(days=1)
    payload = build_player_activity_update(recent, old_game, now_utc=now)
    assert payload == {"last_game_at": recent.isoformat(), "active": True, "inactive_at": None}


def test_game_recency_never_overrides_retired_or_explicitly_hidden_profiles():
    now = datetime(2026, 10, 4, tzinfo=timezone.utc)
    recent = {"name": "Player", "active": True, "inactive_at": None, "last_game_at": now.isoformat()}
    assert is_player_leaderboard_active(recent, now_utc=now)
    for exclusion in ({"active": False}, {"inactive_at": now.isoformat()}, {"name": "Old (MERGED into Player #1)"}):
        assert not is_player_leaderboard_active({**recent, **exclusion}, now_utc=now)


def test_select_leaderboard_players_respects_toggle():
    df_all = pd.DataFrame({"id": [1, 2, 3]})
    df_active = df_all[df_all["id"] != 3].copy()

    assert select_leaderboard_players(df_active, df_all, "Active").equals(df_active)
    assert select_leaderboard_players(df_active, df_all, "See all").equals(df_all)
