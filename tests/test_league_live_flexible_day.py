"""Rehearse a hot ladder day through the real scheduling and publish domains.

The historical session pool includes a staff member who comes off court at
check-in, covers one game, and remains selectable while courts shrink 9 -> 8 -> 7.
No database, production state, or fake rating calculation is involved: official
participation is counted directly from the publisher's accepted match records.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

import pytest

from jupr_app.domain.league_live_orchestration import (
    LeagueLiveDomainError,
    build_league_live_roster_suggestion,
    build_league_live_round_plan,
)
from jupr_app.domain.league_live_publish import (
    LeagueLivePublishError,
    build_league_live_publish_request,
)
from jupr_app.domain.schedule import get_match_schedule


PLAYER_KEYS = ("t1_p1", "t1_p2", "t2_p1", "t2_p2")
SESSION_ID = "flexible-hot-weather-day"


def _check_in() -> dict[str, Any]:
    # IDs 1-35 and staff 37 were originally on court. Missed check-in 36
    # replaces staff before the first game, preserving staff in the session pool.
    roster = [
        {
            "player_id": player_id,
            "player_name": f"Player {player_id}" if player_id != 37 else "Staff cover",
            "rating": 1800 - player_id * 10,
        }
        for player_id in range(1, 38)
    ]
    before = build_league_live_roster_suggestion(
        roster,
        court_sizes=[4] * 9,
        bench_player_ids=[36],
        bench_override_reason="Missed player has not checked in yet.",
    )
    assert {row["player_id"] for row in before["active_roster"]} == {*range(1, 36), 37}
    after = build_league_live_roster_suggestion(
        before["roster"], court_sizes=[4] * 9, bench_player_ids=[37],
        bench_override_reason="Checked-in player takes staff member's place.",
    )
    assert {row["player_id"] for row in after["active_roster"]} == set(range(1, 37))
    assert len(after["roster"]) == 37
    return after


def _games(courts: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    preview_courts = []
    scored = []
    for court in courts:
        number = court["court_number"]
        player_ids = [player["player_id"] for player in court["players_json"]]
        rows = []
        for index, game in enumerate(get_match_schedule("4-Player", player_ids), start=1):
            ids = [*game["t1"], *game["t2"]]
            row = {
                "row_id": f"rr-{number}-{index}",
                "court": number,
                "match_index": index,
                **dict(zip(PLAYER_KEYS, ids)),
            }
            rows.append(row)
            scored.append({
                **row,
                "series_key": row["row_id"],
                "series_kind": "fixed_games",
                "series_games": 1,
                "game_number": 1,
                "score_t1": 11,
                "score_t2": 5 + index,
            })
        preview_courts.append({"court": number, "matches": rows})
    return {"courts": preview_courts, "match_count": len(scored)}, scored


def _plan(
    roster: list[dict[str, Any]],
    courts: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    *,
    round_number: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    return build_league_live_round_plan(
        session_id=SESSION_ID,
        round_number=round_number,
        total_rounds=3,
        session_updated_at=f"round-{round_number}-saved",
        roster=roster,
        courts=courts,
        matches=matches,
        **kwargs,
    )


def _publish(
    preview: dict[str, Any],
    matches: list[dict[str, Any]],
    *,
    round_number: int = 1,
    unplayed_games: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return build_league_live_publish_request(
        session_id=SESSION_ID,
        round_number=round_number,
        league_name="Fall Ladder",
        week_tag="2026-10-18",
        match_date="2026-10-18",
        preview=preview,
        matches=matches,
        unplayed_games=unplayed_games or [],
        expected_match_count=preview["match_count"],
        expected_updated_at=f"round-{round_number}-saved",
        expected_operation_key="a" * 64,
        **kwargs,
    )


def _attendance(player_ids: list[int], court_count: int) -> dict[str, Any]:
    return {"action": "attendance", "active_player_ids": player_ids, "court_sizes": [4] * court_count}


def _assert_pool(plan: dict[str, Any], active_ids: set[int], court_count: int) -> None:
    assert len(plan["next_courts"]) == court_count
    assert all(len(court["players_json"]) == 4 for court in plan["next_courts"])
    actual = [player["player_id"] for court in plan["next_courts"] for player in court["players_json"]]
    assert len(actual) == len(set(actual)) == len(active_ids)
    assert set(actual) == active_ids
    assert {row["player_id"] for row in plan["next_roster"]} == set(range(1, 38))
    assert set(plan["bench_player_ids"]) == set(range(1, 38)) - active_ids


def test_nine_to_eight_to_seven_courts_preserves_actual_games_and_returning_players() -> None:
    setup = _check_in()
    initial_snapshot = deepcopy(setup)
    preview1, generated1 = _games(setup["courts"])
    matches1 = deepcopy(generated1)
    substituted = matches1[1]
    substituted["scheduled_player_ids"] = [substituted[key] for key in PLAYER_KEYS]
    original_slot = next(key for key in PLAYER_KEYS if substituted[key] == 1)
    substituted[original_slot] = 37
    substituted["substitution_reason"] = "Staff covers only this game while Player 1 rests."
    incomplete = matches1.pop()
    unplayed1 = [{
        "series_key": incomplete["series_key"],
        "game_number": 1,
        "court": 9,
        "reason": "Court 9 did not finish its final game before the round ended.",
    }]
    round1_snapshot = deepcopy(matches1)
    official1 = _publish(preview1, matches1, unplayed_games=unplayed1)
    assert len(official1["matches"]) == 26
    credited1 = Counter(row[key] for row in official1["matches"] for key in PLAYER_KEYS)
    assert credited1[1] == 2
    assert credited1[37] == 1
    assert all(credited1[player_id] == 2 for player_id in range(33, 37))

    # Four players rest; keep 32 players on eight courts. Review the normal
    # results-based placement, then intentionally exchange two court positions.
    attendance2 = _attendance(list(range(1, 33)), 8)
    suggested2 = _plan(
        setup["roster"], setup["courts"], official1["matches"],
        roster_change=attendance2, unplayed_games=unplayed1,
    )
    exact_board = [
        [player["player_id"] for player in court["players_json"]]
        for court in suggested2["next_courts"]
    ]
    exact_board[0][0], exact_board[1][0] = exact_board[1][0], exact_board[0][0]
    overrides = [
        {"player_id": player_id, "to_court": number, "to_slot": slot}
        for number, players in enumerate(exact_board, start=1)
        for slot, player_id in enumerate(players, start=1)
    ] + [{"player_id": player_id, "to_court": 0} for player_id in suggested2["bench_player_ids"]]
    round2 = _plan(
        setup["roster"], setup["courts"], official1["matches"],
        roster_change=attendance2, movement_overrides=overrides, unplayed_games=unplayed1,
    )
    _assert_pool(round2, set(range(1, 33)), 8)
    assert round2["movement"]["override_applied"] is True
    assert round2["operation_key"] != suggested2["operation_key"]
    assert [[p["player_id"] for p in c["players_json"]] for c in round2["next_courts"]] == exact_board
    assert any("Court 9" in warning for warning in round2["warnings"])
    round2_snapshot = deepcopy(round2)
    preview2, matches2 = _games(round2["next_courts"])
    official2 = _publish(preview2, matches2, round_number=2)
    assert len(official2["matches"]) == 24
    assert not ({33, 34, 35, 36, 37} & {row[key] for row in official2["matches"] for key in PLAYER_KEYS})

    # Six more players stop, two rested players rejoin: 32 - 6 + 2 = 28.
    active3 = {*range(1, 27), 33, 34}
    round3 = _plan(
        round2["next_roster"], round2["next_courts"], official2["matches"],
        round_number=2, roster_change=_attendance(sorted(active3), 7),
    )
    _assert_pool(round3, active3, 7)
    preview3, matches3 = _games(round3["next_courts"])
    official3 = _publish(preview3, matches3, round_number=3)
    assert len(official3["matches"]) == 21

    official = [*official1["matches"], *official2["matches"], *official3["matches"]]
    credited = Counter(row[key] for row in official for key in PLAYER_KEYS)
    assert len(official) == len({row["context_id"] for row in official}) == 71
    assert sum(credited.values()) == 284
    assert credited[37] == 1
    assert credited[1] == 8
    assert all(credited[player_id] == 9 for player_id in range(2, 27))
    assert all(credited[player_id] == 6 for player_id in range(27, 33))
    assert credited[33] == credited[34] == 5
    assert credited[35] == credited[36] == 2
    assert setup == initial_snapshot
    assert matches1 == round1_snapshot
    assert round2 == round2_snapshot


def test_flexible_day_rejects_unreviewed_game_omission_and_incomplete_court_override() -> None:
    setup = _check_in()
    preview, matches = _games(setup["courts"])
    with pytest.raises(LeagueLivePublishError):
        _publish(preview, matches[:-1])
    skipped = {"series_key": matches[-1]["series_key"], "game_number": 1, "court": 9, "reason": ""}
    with pytest.raises(LeagueLivePublishError):
        _publish(preview, matches[:-1], unplayed_games=[skipped])

    attendance = _attendance(list(range(1, 33)), 8)
    plan = _plan(setup["roster"], setup["courts"], matches, roster_change=attendance)
    # Omitting a playing participant from a manually reordered board must not
    # silently lose them, even when all sit-outs were already reviewed.
    overrides = [
        {"player_id": row["player_id"], "to_court": row["court_number"] or 0, "to_slot": row["slot"] or 0}
        for row in plan["next_roster"]
    ]
    with pytest.raises(LeagueLiveDomainError, match="assign every next-round player"):
        _plan(setup["roster"], setup["courts"], matches, roster_change=attendance, movement_overrides=overrides[1:])


@pytest.mark.parametrize("duplicate", ["id", "name", "attendance"])
def test_flexible_day_rejects_duplicate_player_identity(duplicate: str) -> None:
    setup = _check_in()
    _, matches = _games(setup["courts"])
    roster = deepcopy(setup["roster"])
    active_ids = list(range(1, 33))
    if duplicate == "id":
        roster.append({**roster[0], "player_name": "Different spelling"})
    elif duplicate == "name":
        roster.append({**roster[0], "player_id": 38, "player_name": "  PLAYER   1  "})
    else:
        active_ids[-1] = active_ids[0]
    with pytest.raises(LeagueLiveDomainError):
        _plan(roster, setup["courts"], matches, roster_change=_attendance(active_ids, 8))


def test_flexible_day_fingerprints_bind_substitutions_unplayed_review_and_attendance() -> None:
    setup = _check_in()
    preview, matches = _games(setup["courts"])
    substituted = deepcopy(matches)
    substituted[1]["scheduled_player_ids"] = [substituted[1][key] for key in PLAYER_KEYS]
    key = next(key for key in PLAYER_KEYS if substituted[1][key] == 1)
    substituted[1][key] = 37
    substituted[1]["substitution_reason"] = "Staff covered this one game."
    baseline_plan = _plan(setup["roster"], setup["courts"], matches)
    substitute_plan = _plan(setup["roster"], setup["courts"], substituted)
    attendance_plan = _plan(
        setup["roster"], setup["courts"], substituted,
        roster_change=_attendance(list(range(1, 33)), 8),
    )
    assert len({baseline_plan["operation_key"], substitute_plan["operation_key"], attendance_plan["operation_key"]}) == 3

    full = _publish(preview, matches)
    actual = _publish(preview, substituted)
    unplayed = [{
        "series_key": matches[-1]["series_key"], "game_number": 1, "court": 9,
        "reason": "Game unfinished when players stopped for heat.",
    }]
    unfinished_plan = _plan(setup["roster"], setup["courts"], substituted[:-1], unplayed_games=unplayed)
    reviewed_plan = _plan(
        setup["roster"], setup["courts"], substituted[:-1],
        unplayed_games=[{**unplayed[0], "reason": "Confirmed abandoned game after injury."}],
    )
    assert len({substitute_plan["operation_key"], unfinished_plan["operation_key"], reviewed_plan["operation_key"]}) == 3
    incomplete = _publish(preview, substituted[:-1], unplayed_games=unplayed)
    reviewed = _publish(preview, substituted[:-1], unplayed_games=[{**unplayed[0], "reason": "Confirmed abandoned game after injury."}])
    attendance = _publish(preview, substituted[:-1], unplayed_games=unplayed, roster_change=_attendance(list(range(1, 33)), 8))
    assert len({request["request_fingerprint"] for request in (full, actual, incomplete, reviewed, attendance)}) == 5
    assert incomplete["unplayed_games"][0]["reason"] == unplayed[0]["reason"]
