import pandas as pd
import pytest

from jupr_app.domain.league_night_roster import (
    RosterChangeError,
    apply_roster_change,
    roster_change_availability,
)


def _make_roster(player_ids, ratings=None):
    ratings = ratings or [1200.0] * len(player_ids)
    rows = []
    court = 1
    slot = 1
    for idx, pid in enumerate(player_ids):
        rows.append(
            {
                "player_id": int(pid),
                "name": f"Player {pid}",
                "rating": float(ratings[idx]),
                "court": int(court),
                "slot": int(slot),
            }
        )
        slot += 1
        if slot > 4:
            court += 1
            slot = 1
    return pd.DataFrame(rows)


def test_roster_change_availability_between_rounds():
    ok, msg = roster_change_availability("CONFIRM_MOVEMENT", current_round=2, total_rounds=5, is_admin=True)
    assert ok is True
    assert msg == ""

    ok, msg = roster_change_availability("PLAY_ROUND", current_round=2, total_rounds=5, is_admin=True)
    assert ok is False
    assert "between rounds" in msg

    ok, msg = roster_change_availability("CONFIRM_MOVEMENT", current_round=5, total_rounds=5, is_admin=True)
    assert ok is False
    assert "complete" in msg


def test_apply_roster_change_rejects_locked_round():
    roster = _make_roster([1, 2, 3, 4])
    with pytest.raises(RosterChangeError, match="locked"):
        apply_roster_change(
            roster_df=roster,
            change_type="add",
            new_player={"id": 5, "name": "Player 5", "rating": 1200.0},
            roster_locked=True,
        )


def test_apply_roster_change_validation():
    roster = _make_roster([1, 2, 3, 4])
    with pytest.raises(RosterChangeError, match="already active"):
        apply_roster_change(
            roster_df=roster,
            change_type="add",
            new_player={"id": 2, "name": "Player 2", "rating": 1200.0},
        )

    with pytest.raises(RosterChangeError, match="same player"):
        apply_roster_change(
            roster_df=roster,
            change_type="substitute",
            replaced_player_id=2,
            new_player={"id": 2, "name": "Player 2", "rating": 1200.0},
        )


def test_substitute_updates_future_roster():
    roster = _make_roster([1, 2, 3, 4, 5, 6, 7, 8])
    result = apply_roster_change(
        roster_df=roster,
        change_type="substitute",
        replaced_player_id=3,
        new_player={"id": 99, "name": "Sub", "rating": 1300.0},
        court_sizes=[4, 4],
    )

    updated_ids = result.roster_df["player_id"].astype(int).tolist()
    assert 3 not in updated_ids
    assert 99 in updated_ids
    assert result.court_sizes == [4, 4]


def test_add_player_rebalances_courts():
    roster = _make_roster([1, 2, 3, 4, 5, 6, 7, 8])
    result = apply_roster_change(
        roster_df=roster,
        change_type="add",
        new_player={"id": 99, "name": "Late", "rating": 1150.0},
        court_sizes=[4, 4],
    )

    updated_ids = result.roster_df["player_id"].astype(int).tolist()
    assert 99 in updated_ids
    assert sum(result.court_sizes) == len(result.roster_df)
    assert len(result.roster_df) == 9


def test_add_player_benches_lowest_rating_first():
    roster = _make_roster([1, 2, 3, 4, 5, 6])
    result = apply_roster_change(
        roster_df=roster,
        change_type="add",
        new_player={"id": 99, "name": "Late", "rating": 1300.0},
    )

    assert 99 in result.roster_df["player_id"].astype(int).tolist()
    assert len(result.bench_ids) == 2
    assert 99 not in result.bench_ids


def test_multiple_substitutions_can_be_applied_sequentially():
    roster = _make_roster([1, 2, 3, 4, 5, 6, 7, 8])

    first = apply_roster_change(
        roster_df=roster,
        change_type="substitute",
        replaced_player_id=2,
        new_player={"id": 90, "name": "Sub A", "rating": 1250.0},
        court_sizes=[4, 4],
    )

    second = apply_roster_change(
        roster_df=first.roster_df,
        change_type="substitute",
        replaced_player_id=5,
        new_player={"id": 91, "name": "Sub B", "rating": 1260.0},
        court_sizes=first.court_sizes,
    )

    updated_ids = second.roster_df["player_id"].astype(int).tolist()
    assert 2 not in updated_ids
    assert 5 not in updated_ids
    assert 90 in updated_ids
    assert 91 in updated_ids
