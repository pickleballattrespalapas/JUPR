
from __future__ import annotations

from collections import Counter

import pytest

from jupr_app.domain.adaptive_play_engine import (
    active_participant_ids,
    advance_generator_event,
    create_generator_preview,
    mutate_generator_roster,
    save_generator_round,
    schedule_export_rows,
    skip_generator_round,
    start_generator_event,
)


def _matches(round_row):
    if round_row.get("matches"):
        return list(round_row.get("matches") or [])
    return [
        match
        for court in round_row.get("courts") or []
        for match in court.get("matches") or []
    ]


def _pair_key(a, b):
    return "|".join(sorted((str(a), str(b))))


def _exact_key(team_a, team_b):
    left = ",".join(sorted(team_a))
    right = ",".join(sorted(team_b))
    return "|".join(sorted((left, right)))


def _score_round(event, round_number, score_a=11, score_b=7):
    row = next(item for item in event["rounds"] if item["number"] == round_number)
    return save_generator_round(
        event,
        round_number=round_number,
        scores=[
            {"match_id": match["id"], "score_a": score_a, "score_b": score_b}
            for match in _matches(row)
        ],
    )


@pytest.mark.parametrize("player_count", [3, 5, 7, 9])
def test_singles_preview_balances_byes_and_avoids_repeats(player_count):
    event = create_generator_preview(
        generator_kind="round_robin",
        play_format="singles",
        title="Singles Preview",
        participant_names=[f"Player {idx}" for idx in range(1, player_count + 1)],
        total_rounds=player_count,
        court_count=0,
    )

    pairings = []
    byes = Counter()
    for round_row in event["rounds"]:
        seen = set()
        for match in _matches(round_row):
            assert len(match["sideA"]) == 1
            assert len(match["sideB"]) == 1
            assert not seen.intersection(match["sideA"] + match["sideB"])
            seen.update(match["sideA"] + match["sideB"])
            pairings.append(_pair_key(match["sideA"][0], match["sideB"][0]))
        byes.update(round_row["byeParticipantIds"])

    assert len(pairings) == len(set(pairings))
    assert max(byes.values(), default=0) - min(byes.values(), default=0) <= 1


@pytest.mark.parametrize("player_count", [4, 5, 6, 7, 8, 9, 12])
def test_doubles_preview_handles_arbitrary_player_counts(player_count):
    event = create_generator_preview(
        generator_kind="round_robin",
        play_format="doubles",
        title="Doubles Preview",
        participant_names=[f"Player {idx}" for idx in range(1, player_count + 1)],
        total_rounds=min(6, player_count),
        court_count=0,
    )

    exact = Counter()
    for round_row in event["rounds"]:
        active = set(active_participant_ids(event, round_row["number"]))
        seen = set()
        for match in _matches(round_row):
            assert len(match["sideA"]) == 2
            assert len(match["sideB"]) == 2
            assert not set(match["sideA"]).intersection(match["sideB"])
            assert not seen.intersection(match["sideA"] + match["sideB"])
            seen.update(match["sideA"] + match["sideB"])
            exact[_exact_key(match["sideA"], match["sideB"])] += 1
        assert seen.union(round_row["byeParticipantIds"]) == active

    # Repeats are allowed only after the finite exact matchup pool is exhausted.
    unique_capacity = 3 if player_count == 4 else sum(1 for _ in exact)
    if len(event["rounds"]) <= unique_capacity:
        assert max(exact.values(), default=1) == 1
    else:
        assert any(round_row["warnings"] for round_row in event["rounds"])


def test_round_robin_preview_is_all_rounds_and_ladder_preview_is_round_one_only():
    names = [f"Player {idx}" for idx in range(1, 10)]
    round_robin = create_generator_preview(
        generator_kind="round_robin",
        play_format="doubles",
        title="Round Robin",
        participant_names=names,
        total_rounds=5,
        court_count=2,
    )
    ladder = create_generator_preview(
        generator_kind="ladder",
        play_format="doubles",
        title="Ladder",
        participant_names=names,
        total_rounds=5,
        court_count=2,
    )

    assert len(round_robin["rounds"]) == 5
    assert len(ladder["rounds"]) == 1
    assert ladder["rounds"][0]["courts"]


def test_save_results_and_skip_round_then_advance():
    event = start_generator_event(
        create_generator_preview(
            generator_kind="round_robin",
            play_format="singles",
            title="Round-by-round",
            participant_names=["A", "B", "C", "D", "E"],
            total_rounds=3,
            court_count=2,
        )
    )

    event = _score_round(event, 1)
    assert event["rounds"][0]["status"] == "saved"
    event = advance_generator_event(event)
    assert event["currentRoundNumber"] == 2
    assert event["rounds"][1]["status"] == "active"

    event = skip_generator_round(event, round_number=2, reason="Weather")
    assert event["rounds"][1]["status"] == "skipped"
    event = advance_generator_event(event)
    assert event["currentRoundNumber"] == 3


def test_adaptive_add_remove_and_substitute_preserve_completed_rounds():
    event = start_generator_event(
        create_generator_preview(
            generator_kind="round_robin",
            play_format="singles",
            title="Adaptive",
            participant_names=["A", "B", "C", "D", "E"],
            total_rounds=5,
            court_count=2,
        )
    )
    event = _score_round(event, 1)
    saved_round = event["rounds"][0]
    event = advance_generator_event(event)

    event = mutate_generator_roster(event, action="add", name="New Player")
    assert event["rounds"][0] == saved_round
    assert "New Player" in [
        row["name"]
        for row in event["participants"]
        if row["id"] in active_participant_ids(event, 2)
    ]

    event = mutate_generator_roster(
        event,
        action="substitute",
        participant_id="p-1",
        name="One-round Substitute",
        substitute_scope="round",
    )
    participants = {row["name"]: row for row in event["participants"]}
    assert 2 in participants["A"]["inactive_rounds"]
    assert participants["One-round Substitute"]["inactive_from_round"] == 3

    event = mutate_generator_roster(event, action="remove", participant_id="p-2")
    assert "p-2" not in active_participant_ids(event, 2)


def test_ladder_next_round_is_generated_only_after_results_and_uses_movement():
    event = start_generator_event(
        create_generator_preview(
            generator_kind="ladder",
            play_format="doubles",
            title="Adaptive Ladder",
            participant_names=[f"Player {idx}" for idx in range(1, 10)],
            total_rounds=3,
            court_count=2,
        )
    )
    assert len(event["rounds"]) == 1

    with pytest.raises(ValueError, match="Save or skip"):
        advance_generator_event(event)

    first_groups = [list(court["participantIds"]) for court in event["rounds"][0]["courts"]]
    event = _score_round(event, 1)
    event = advance_generator_event(event)

    assert len(event["rounds"]) == 2
    assert event["currentRoundNumber"] == 2
    next_groups = [list(court["participantIds"]) for court in event["rounds"][1]["courts"]]
    assert next_groups != first_groups


def test_ladder_supports_singles_groups():
    event = create_generator_preview(
        generator_kind="ladder",
        play_format="singles",
        title="Singles Ladder",
        participant_names=[f"Player {idx}" for idx in range(1, 8)],
        total_rounds=2,
        court_count=2,
    )

    assert [court["size"] for court in event["rounds"][0]["courts"]] == [4, 3]
    assert all(
        len(match["sideA"]) == len(match["sideB"]) == 1
        for match in _matches(event["rounds"][0])
    )


def test_roster_order_can_be_changed_during_preview():
    event = create_generator_preview(
        generator_kind="round_robin",
        play_format="singles",
        title="Ordered Preview",
        participant_names=["A", "B", "C", "D"],
        total_rounds=3,
        court_count=2,
    )
    event = mutate_generator_roster(
        event,
        action="reorder",
        roster_order=["p-4", "p-3", "p-2", "p-1"],
    )

    ordered = sorted(event["participants"], key=lambda row: row["roster_order"])
    assert [row["name"] for row in ordered] == ["D", "C", "B", "A"]


def test_roster_order_rejects_duplicate_participants():
    event = create_generator_preview(
        generator_kind="round_robin",
        play_format="singles",
        title="Invalid Ordered Preview",
        participant_names=["A", "B", "C", "D"],
        total_rounds=3,
        court_count=2,
    )

    with pytest.raises(ValueError, match="Put every player in the order once"):
        mutate_generator_roster(
            event,
            action="reorder",
            roster_order=["p-1", "p-2", "p-3", "p-4", "p-4"],
        )


def test_schedule_export_has_score_columns_and_byes():
    event = create_generator_preview(
        generator_kind="round_robin",
        play_format="singles",
        title="Paper Schedule",
        participant_names=["A", "B", "C", "D", "E"],
        total_rounds=3,
        court_count=2,
    )

    rows = schedule_export_rows(event)

    assert rows
    assert {"round", "court", "side_a", "score_a", "score_b", "side_b", "byes"}.issubset(rows[0])
    assert any(row["byes"] for row in rows)


def test_ladder_prefers_four_and_five_player_courts_and_three_games_for_four():
    event = create_generator_preview(
        generator_kind="ladder",
        play_format="doubles",
        title="Twelve-player Ladder",
        participant_names=[f"Player {idx}" for idx in range(1, 13)],
        total_rounds=3,
        court_count=0,
    )

    courts = event["rounds"][0]["courts"]
    assert [court["size"] for court in courts] == [4, 4, 4]
    assert all(len(court["matches"]) == 3 for court in courts)

    nine = create_generator_preview(
        generator_kind="ladder",
        play_format="doubles",
        title="Nine-player Ladder",
        participant_names=[f"Player {idx}" for idx in range(1, 10)],
        total_rounds=2,
        court_count=0,
    )
    assert [court["size"] for court in nine["rounds"][0]["courts"]] == [5, 4]
    assert [len(court["matches"]) for court in nine["rounds"][0]["courts"]] == [5, 3]
