from __future__ import annotations

import re
from collections import Counter, defaultdict

import pytest

from jupr_app.domain.schedule import (
    EXPECTED_DOUBLES_GAMES_BY_FORMAT,
    ORGANIZED_RR_MAX_ROUNDS,
    SCHEDULE_MODE_FULL,
    SCHEDULE_MODE_NAIVE_FIRST_EIGHT,
    SCHEDULE_MODE_ORGANIZED,
    SUPPORTED_DOUBLES_PLAYER_COUNTS,
    calculate_schedule_metrics,
    get_match_schedule,
)


ALL_DOUBLE_COUNTS = [count for count in SUPPORTED_DOUBLES_PLAYER_COUNTS if 4 <= count <= 20]
OPTIMIZATION_SPOT_CHECK_COUNTS = [6, 10, 14, 18, 20]


def _round_number(desc: str) -> int:
    match = re.search(r"Rnd\s*(\d+)", str(desc or ""), flags=re.IGNORECASE)
    assert match is not None, f"Missing round number in description: {desc!r}"
    return int(match.group(1))


def _group_rounds(schedule: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for match in schedule:
        grouped[_round_number(str(match["desc"]))].append(match)
    return grouped


def _assert_schedule_shape(schedule: list[dict], players: list[int]) -> None:
    assert schedule
    for match in schedule:
        assert sorted(match.keys()) == ["desc", "t1", "t2"]
        assert len(match["t1"]) == 2
        assert len(match["t2"]) == 2
        participants = list(match["t1"]) + list(match["t2"])
        assert len(set(participants)) == 4
        assert all(player in players for player in participants)

    for round_matches in _group_rounds(schedule).values():
        round_players: list[int] = []
        for match in round_matches:
            round_players.extend(match["t1"])
            round_players.extend(match["t2"])
        assert len(round_players) == len(set(round_players))


@pytest.mark.parametrize("count", ALL_DOUBLE_COUNTS)
def test_supported_round_robin_formats_resolve_and_full_schedule_remains_available(count: int):
    players = list(range(1, count + 1))

    full_schedule = get_match_schedule(f"{count}-Player", players, schedule_mode=SCHEDULE_MODE_FULL)
    organized_schedule = get_match_schedule(f"{count}-Player", players, schedule_mode=SCHEDULE_MODE_ORGANIZED)

    assert full_schedule
    assert len(full_schedule) == EXPECTED_DOUBLES_GAMES_BY_FORMAT[f"{count}-Player"]
    assert organized_schedule
    _assert_schedule_shape(full_schedule, players)
    _assert_schedule_shape(organized_schedule, players)


@pytest.mark.parametrize("count", ALL_DOUBLE_COUNTS)
def test_organized_round_robin_caps_rounds_and_beats_or_matches_baseline_score(count: int):
    players = list(range(1, count + 1))
    organized_schedule = get_match_schedule(f"{count}-Player", players, schedule_mode=SCHEDULE_MODE_ORGANIZED)
    baseline_schedule = get_match_schedule(f"{count}-Player", players, schedule_mode=SCHEDULE_MODE_NAIVE_FIRST_EIGHT)

    organized_metrics = calculate_schedule_metrics(organized_schedule, players)
    baseline_metrics = calculate_schedule_metrics(baseline_schedule, players)

    print(
        f"count={count} rounds={organized_metrics['rounds_used']} "
        f"partner_range={organized_metrics['partner_range']} exposure_range={organized_metrics['exposure_range']} "
        f"adjacent={organized_metrics['adjacent_interaction_penalty']} "
        f"flips={organized_metrics['partner_to_opponent_flip_penalty']} bye_range={organized_metrics['bye_range']}"
    )

    assert organized_metrics["rounds_used"] <= ORGANIZED_RR_MAX_ROUNDS
    assert organized_metrics["unique_exposure_score"] >= baseline_metrics["unique_exposure_score"]
    assert organized_metrics["weighted_score"] >= baseline_metrics["weighted_score"]


@pytest.mark.parametrize("count", OPTIMIZATION_SPOT_CHECK_COUNTS)
def test_organized_round_robin_reduces_adjacent_repeats_and_flips_for_key_counts(count: int):
    players = list(range(1, count + 1))
    organized_metrics = calculate_schedule_metrics(
        get_match_schedule(f"{count}-Player", players, schedule_mode=SCHEDULE_MODE_ORGANIZED),
        players,
    )
    baseline_metrics = calculate_schedule_metrics(
        get_match_schedule(f"{count}-Player", players, schedule_mode=SCHEDULE_MODE_NAIVE_FIRST_EIGHT),
        players,
    )

    assert organized_metrics["adjacent_interaction_penalty"] <= baseline_metrics["adjacent_interaction_penalty"]
    assert organized_metrics["partner_to_opponent_flip_penalty"] <= baseline_metrics["partner_to_opponent_flip_penalty"]


@pytest.mark.parametrize("count", ALL_DOUBLE_COUNTS)
def test_full_round_robin_still_balances_play_and_byes(count: int):
    players = list(range(1, count + 1))
    schedule = get_match_schedule(f"{count}-Player", players, schedule_mode=SCHEDULE_MODE_FULL)

    play_counts = Counter({player: 0 for player in players})
    bye_counts = Counter({player: 0 for player in players})

    for match in schedule:
        for player in list(match["t1"]) + list(match["t2"]):
            play_counts[player] += 1

    for round_matches in _group_rounds(schedule).values():
        active_players = {player for match in round_matches for player in [*match["t1"], *match["t2"]]}
        for player in players:
            if player not in active_players:
                bye_counts[player] += 1

    assert max(play_counts.values()) - min(play_counts.values()) <= 2
    assert max(bye_counts.values()) - min(bye_counts.values()) <= 2
