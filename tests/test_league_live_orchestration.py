from __future__ import annotations

import pytest

from jupr_app.domain.league_live_orchestration import (
    LeagueLiveDomainError,
    build_league_live_roster_suggestion,
    build_league_live_round_plan,
)


def _roster(count: int = 8) -> list[dict]:
    return [
        {
            "player_id": player_id,
            "player_name": f"Player {player_id}",
            "rating": 1500 - (player_id * 10),
        }
        for player_id in range(1, count + 1)
    ]


def _matches() -> list[dict]:
    return [
        {
            "court": 1,
            "t1_p1": 1,
            "t1_p2": 2,
            "t2_p1": 3,
            "t2_p2": 4,
            "score_t1": 11,
            "score_t2": 7,
        },
        {
            "court": 2,
            "t1_p1": 5,
            "t1_p2": 6,
            "t2_p1": 7,
            "t2_p2": 8,
            "score_t1": 6,
            "score_t2": 11,
        },
    ]


def test_roster_suggestion_returns_deterministic_courts_and_bench() -> None:
    suggestion = build_league_live_roster_suggestion(_roster(6), round_number=2)

    assert suggestion["court_sizes"] == [5]
    assert suggestion["bench_count"] == 1
    assert suggestion["bench_player_ids"] == [6]
    assert suggestion["courts"][0]["format_type"] == "5-Player"
    assert suggestion["courts"][0]["round_number"] == 2
    assert len(suggestion["fingerprint"]) == 64


def test_bench_override_requires_reason_and_exact_capacity() -> None:
    with pytest.raises(LeagueLiveDomainError, match="Explain the bench override"):
        build_league_live_roster_suggestion(
            _roster(6),
            bench_player_ids=[1],
        )

    overridden = build_league_live_roster_suggestion(
        _roster(6),
        bench_player_ids=[1],
        bench_override_reason="Players requested the first sit-out rotation.",
    )

    assert overridden["bench_override_applied"] is True
    assert overridden["bench_player_ids"] == [1]
    assert {row["bench_reason"] for row in overridden["bench"]} == {"operator_override"}


def test_round_plan_is_python_authoritative_and_deterministic() -> None:
    suggestion = build_league_live_roster_suggestion(_roster())

    first = build_league_live_round_plan(
        session_id="session-1",
        round_number=1,
        total_rounds=3,
        session_updated_at="2026-07-19T12:00:00+00:00",
        roster=suggestion["roster"],
        courts=suggestion["courts"],
        matches=_matches(),
    )
    second = build_league_live_round_plan(
        session_id="session-1",
        round_number=1,
        total_rounds=3,
        session_updated_at="2026-07-19T12:00:00+00:00",
        roster=suggestion["roster"],
        courts=suggestion["courts"],
        matches=_matches(),
    )

    assert first["operation_key"] == second["operation_key"]
    assert first["movement"]["authority"] == "python_fastapi"
    assert first["movement"]["next_round"] == 2
    assert sum(len(court["players_json"]) for court in first["next_courts"]) == 8
    assert all(len(court["players_json"]) == 4 for court in first["next_courts"])


def test_manual_movement_override_requires_reason_and_balanced_courts() -> None:
    suggestion = build_league_live_roster_suggestion(_roster())
    base = build_league_live_round_plan(
        session_id="session-1",
        round_number=1,
        total_rounds=3,
        session_updated_at="v1",
        roster=suggestion["roster"],
        courts=suggestion["courts"],
        matches=_matches(),
    )
    moved = [row for row in base["movement"]["rows"] if row["direction"] != "stay"]
    overrides = [{"player_id": row["player_id"], "to_court": row["from_court"]} for row in moved]

    with pytest.raises(LeagueLiveDomainError, match="Explain a manual movement override"):
        build_league_live_round_plan(
            session_id="session-1",
            round_number=1,
            total_rounds=3,
            session_updated_at="v1",
            roster=suggestion["roster"],
            courts=suggestion["courts"],
            matches=_matches(),
            movement_overrides=overrides,
        )

    overridden = build_league_live_round_plan(
        session_id="session-1",
        round_number=1,
        total_rounds=3,
        session_updated_at="v1",
        roster=suggestion["roster"],
        courts=suggestion["courts"],
        matches=_matches(),
        movement_overrides=overrides,
        override_reason="Keep the same courts for an accessibility accommodation.",
    )

    assert overridden["movement"]["override_applied"] is True
    assert all(row["from_court"] == row["to_court"] for row in overridden["movement"]["rows"])

    with pytest.raises(LeagueLiveDomainError, match="requires four or five"):
        build_league_live_round_plan(
            session_id="session-1",
            round_number=1,
            total_rounds=3,
            session_updated_at="v1",
            roster=suggestion["roster"],
            courts=suggestion["courts"],
            matches=_matches(),
            movement_overrides=overrides[:1],
            override_reason="Keep one player down for a reviewed accommodation.",
        )


@pytest.mark.parametrize("action", ["add", "substitute"])
def test_round_plan_validates_add_and_substitute_roster_contracts(action: str) -> None:
    suggestion = build_league_live_roster_suggestion(_roster())
    roster_change = {
        "action": action,
        "player": {"player_id": 9, "player_name": "Player 9", "rating": 1415},
    }
    if action == "substitute":
        roster_change["replaced_player_id"] = 4

    planned = build_league_live_round_plan(
        session_id="session-1",
        round_number=1,
        total_rounds=3,
        session_updated_at="v1",
        roster=suggestion["roster"],
        courts=suggestion["courts"],
        matches=_matches(),
        roster_change=roster_change,
    )

    next_ids = {row["player_id"] for row in planned["next_roster"]}
    assert 9 in next_ids
    assert (4 in next_ids) is (action == "add")
    assert all(len(court["players_json"]) in {4, 5} for court in planned["next_courts"])
    assert sum(len(court["players_json"]) for court in planned["next_courts"]) == (9 if action == "add" else 8)


def test_round_plan_rejects_cross_court_match() -> None:
    suggestion = build_league_live_roster_suggestion(_roster())
    invalid = _matches()
    invalid[0]["t2_p2"] = 8

    with pytest.raises(LeagueLiveDomainError, match="same court"):
        build_league_live_round_plan(
            session_id="session-1",
            round_number=1,
            total_rounds=3,
            session_updated_at="v1",
            roster=suggestion["roster"],
            courts=suggestion["courts"],
            matches=invalid,
        )
