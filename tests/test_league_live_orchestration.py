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
    assert [
        [int(player["player_id"]) for player in court["players_json"]]
        for court in first["next_courts"]
    ] == [[1, 2, 3, 7], [4, 8, 5, 6]]


def test_manual_movement_override_needs_no_reason_and_keeps_balanced_courts() -> None:
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

    overridden = build_league_live_round_plan(
        session_id="session-1",
        round_number=1,
        total_rounds=3,
        session_updated_at="v1",
        roster=suggestion["roster"],
        courts=suggestion["courts"],
        matches=_matches(),
        movement_overrides=overrides,
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
        )


def test_ordered_court_board_override_preserves_exact_cards_and_slots() -> None:
    suggestion = build_league_live_roster_suggestion(_roster())
    common = {
        "session_id": "session-board",
        "round_number": 1,
        "total_rounds": 3,
        "session_updated_at": "v1",
        "roster": suggestion["roster"],
        "courts": suggestion["courts"],
        "matches": _matches(),
    }
    base = build_league_live_round_plan(**common)
    ordered = {
        int(court["court_number"]): [int(player["player_id"]) for player in court["players_json"]]
        for court in base["next_courts"]
    }
    ordered[1][0], ordered[2][0] = ordered[2][0], ordered[1][0]
    overrides = [
        {"player_id": player_id, "to_court": court_number, "to_slot": slot}
        for court_number, player_ids in ordered.items()
        for slot, player_id in enumerate(player_ids, start=1)
    ]

    planned = build_league_live_round_plan(
        **common,
        movement_overrides=overrides,
    )

    assert planned["movement"]["override_applied"] is True
    assert [
        [int(player["player_id"]) for player in court["players_json"]]
        for court in planned["next_courts"]
    ] == [ordered[1], ordered[2]]
    assert all(
        [int(player["slot"]) for player in court["players_json"]] == list(range(1, len(court["players_json"]) + 1))
        for court in planned["next_courts"]
    )


def test_ordered_court_board_rejects_missing_player_and_unbalanced_court() -> None:
    suggestion = build_league_live_roster_suggestion(_roster())
    common = {
        "session_id": "session-board",
        "round_number": 1,
        "total_rounds": 3,
        "session_updated_at": "v1",
        "roster": suggestion["roster"],
        "courts": suggestion["courts"],
        "matches": _matches(),
        "override_reason": "Operator reviewed this intentional court arrangement.",
    }
    base = build_league_live_round_plan(**common)
    overrides = [
        {"player_id": int(player["player_id"]), "to_court": int(court["court_number"]), "to_slot": slot}
        for court in base["next_courts"]
        for slot, player in enumerate(court["players_json"], start=1)
    ]

    with pytest.raises(LeagueLiveDomainError, match="assign every next-round player"):
        build_league_live_round_plan(**common, movement_overrides=overrides[:-1])

    first_court = int(base["next_courts"][0]["court_number"])
    last = dict(overrides[-1])
    last.update({"to_court": first_court, "to_slot": 5})
    with pytest.raises(LeagueLiveDomainError, match="requires four or five"):
        build_league_live_round_plan(**common, movement_overrides=[*overrides[:-1], last])


def test_ordered_court_board_can_exchange_a_reviewed_bench_player() -> None:
    suggestion = build_league_live_roster_suggestion(_roster(11))
    matches = [
        _matches()[0],
        {**_matches()[1], "t1_p1": 6, "t1_p2": 7, "t2_p1": 8, "t2_p2": 9},
    ]
    common = {
        "session_id": "session-bench-board",
        "round_number": 1,
        "total_rounds": 3,
        "session_updated_at": "v1",
        "roster": suggestion["roster"],
        "courts": suggestion["courts"],
        "matches": matches,
    }
    base = build_league_live_round_plan(
        **common,
        bench_player_ids=suggestion["bench_player_ids"],
    )
    ordered = {
        int(court["court_number"]): [int(player["player_id"]) for player in court["players_json"]]
        for court in base["next_courts"]
    }
    outgoing_player_id = ordered[1][0]
    incoming_player_id = int(base["bench_player_ids"][0])
    ordered[1][0] = incoming_player_id
    overrides = [
        {"player_id": player_id, "to_court": court_number, "to_slot": slot}
        for court_number, player_ids in ordered.items()
        for slot, player_id in enumerate(player_ids, start=1)
    ]
    overrides.append({"player_id": outgoing_player_id, "to_court": 0})

    planned = build_league_live_round_plan(
        **common,
        movement_overrides=overrides,
        bench_player_ids=[outgoing_player_id],
    )

    assert planned["bench_player_ids"] == [outgoing_player_id]
    assert int(planned["next_courts"][0]["players_json"][0]["player_id"]) == incoming_player_id
    assert planned["movement"]["override_applied"] is True


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


def test_attendance_preserves_people_who_sit_out_and_accepts_their_return() -> None:
    suggestion = build_league_live_roster_suggestion(_roster())
    first = build_league_live_round_plan(
        session_id="attendance", round_number=1, total_rounds=3, session_updated_at="v1",
        roster=suggestion["roster"], courts=suggestion["courts"], matches=_matches(),
        roster_change={"action": "attendance", "active_player_ids": [1, 2, 3, 4], "court_sizes": [4]},
    )
    assert len(first["next_courts"]) == 1
    assert {row["player_id"] for row in first["next_roster"]} == set(range(1, 9))
    assert first["bench_player_ids"] == [5, 6, 7, 8]
    assert all(row["bench_reason"] == "not_playing" for row in first["bench"])
    assert {row["last_court_number"] for row in first["bench"]} == {2}

    second = build_league_live_round_plan(
        session_id="attendance", round_number=2, total_rounds=3, session_updated_at="v2",
        roster=first["next_roster"], courts=first["next_courts"], matches=[_matches()[0]],
        roster_change={"action": "attendance", "active_player_ids": [1, 2, 5, 6]},
    )
    assert {row["player_id"] for row in second["next_courts"][0]["players_json"]} == {1, 2, 5, 6}
    assert {row["player_id"] for row in second["next_roster"]} == set(range(1, 9))
    assert set(second["bench_player_ids"]) == {3, 4, 7, 8}
    assert [row["player_id"] for row in suggestion["roster"]] == list(range(1, 9))


def test_attendance_can_add_multiple_new_players_and_preserve_exact_board_order() -> None:
    suggestion = build_league_live_roster_suggestion(_roster())
    common = dict(
        session_id="attendance", round_number=1, total_rounds=3, session_updated_at="v1",
        roster=suggestion["roster"], courts=suggestion["courts"], matches=_matches(),
        roster_change={
            "action": "attendance", "active_player_ids": [1, 2, 9, 10],
            "players": _roster(10)[-2:], "court_sizes": [4],
        },
    )
    overrides = [
        {"player_id": player_id, "to_court": 1, "to_slot": slot}
        for slot, player_id in enumerate([10, 9, 2, 1], start=1)
    ] + [{"player_id": player_id, "to_court": 0} for player_id in range(3, 9)]
    planned = build_league_live_round_plan(**common, movement_overrides=overrides)
    assert [row["player_id"] for row in planned["next_courts"][0]["players_json"]] == [10, 9, 2, 1]
    assert len(planned["next_roster"]) == 10
    assert planned["roster_change"]["added_player_ids"] == [9, 10]
    assert planned["operation_key"] != build_league_live_round_plan(**common)["operation_key"]


@pytest.mark.parametrize("change,message", [
    ({"active_player_ids": [1, 1, 2, 3]}, "must be unique"),
    ({"active_player_ids": [1, 2, 3, 9]}, "outside the verified session roster"),
    ({"active_player_ids": [1, 2, 3, 4], "players": [_roster()[0]]}, "already belong"),
    ({"active_player_ids": list(range(1, 7))}, "fit every selected player"),
    ({"active_player_ids": list(range(1, 9)), "court_sizes": [4]}, "fit every selected player"),
])
def test_attendance_rejects_duplicate_unknown_and_unassigned_players(change: dict, message: str) -> None:
    suggestion = build_league_live_roster_suggestion(_roster())
    with pytest.raises(LeagueLiveDomainError, match=message):
        build_league_live_round_plan(
            session_id="attendance", round_number=1, total_rounds=3, session_updated_at="v1",
            roster=suggestion["roster"], courts=suggestion["courts"], matches=_matches(),
            roster_change={"action": "attendance", **change},
        )


def test_game_substitute_receives_own_result_without_borrowing_seat_owners_wins() -> None:
    suggestion = build_league_live_roster_suggestion(_roster(4))
    roster = suggestion["roster"] + [{**_roster(5)[-1], "status": "bench", "bench_reason": "not_playing"}]
    matches = [
        {**_matches()[0], "score_t2": 6},
        {**_matches()[0], "t1_p1": 5, "scheduled_player_ids": [1, 2, 3, 4], "substitution_reason": "Staff covered the second game."},
        {**_matches()[0], "score_t2": 8},
    ]
    common = dict(
        session_id="game-sub", round_number=1, total_rounds=3, session_updated_at="v1",
        roster=roster, courts=suggestion["courts"], matches=matches,
    )
    planned = build_league_live_round_plan(**common)
    results = {row["player_id"]: row for row in planned["movement"]["rows"]}
    assert results[1]["wins"] == 2
    assert results[2]["wins"] == 3
    assert planned["bench_player_ids"] == [5]
    assert planned["next_courts"][0]["players_json"][0]["player_id"] == 2
    changed_reason = [dict(row) for row in matches]
    changed_reason[1]["substitution_reason"] = "Player requested a break."
    assert planned["operation_key"] != build_league_live_round_plan(**{**common, "matches": changed_reason})["operation_key"]


@pytest.mark.parametrize("match,message", [
    ({"t1_p1": 9}, "outside the active court roster"),
    ({"t1_p1": 9, "scheduled_player_ids": [1, 2, 3, 4]}, "requires a reason"),
    ({"t1_p1": 10, "scheduled_player_ids": [1, 2, 3, 4], "substitution_reason": "Break"}, "outside the authorized"),
    ({"t1_p1": 5, "scheduled_player_ids": [1, 2, 3, 4], "substitution_reason": "Break"}, "another court"),
    ({"t1_p1": 9, "scheduled_player_ids": [1, 2, 3, 3], "substitution_reason": "Break"}, "distinct scheduled"),
])
def test_game_substitution_requires_authorized_identity_and_original_court(match: dict, message: str) -> None:
    suggestion = build_league_live_roster_suggestion(_roster())
    roster = suggestion["roster"] + [{**_roster(9)[-1], "status": "bench"}]
    with pytest.raises(LeagueLiveDomainError, match=message):
        build_league_live_round_plan(
            session_id="game-sub", round_number=1, total_rounds=3, session_updated_at="v1",
            roster=roster, courts=suggestion["courts"], matches=[{**_matches()[0], **match}, _matches()[1]],
        )


def test_unplayed_game_record_changes_operation_key_and_incomplete_courts_warn() -> None:
    suggestion = build_league_live_roster_suggestion(_roster())
    common = dict(
        session_id="partial", round_number=1, total_rounds=3, session_updated_at="v1",
        roster=suggestion["roster"], courts=suggestion["courts"], matches=[_matches()[0]],
    )
    first = build_league_live_round_plan(**common)
    second = build_league_live_round_plan(**common, unplayed_games=[{"row_id": "rr-2-1", "reason": "Court stopped for heat."}])
    assert first["operation_key"] != second["operation_key"]
    assert any("Court 2 has 0 completed" in warning for warning in second["warnings"])
    assert len(second["next_courts"]) == 2
