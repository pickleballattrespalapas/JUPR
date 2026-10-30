"""The opening doubles rotation is exact when the starting roster stays intact."""

from collections import Counter
from copy import deepcopy
from itertools import combinations, product
import json

import pytest

from jupr_app.domain import adaptive_play_engine as engine


def _preview(count, *, courts=0, mode="scored", rounds=None):
    return engine.create_generator_preview(
        generator_kind="round_robin",
        play_format="doubles",
        title="Complete partner rotation",
        participant_names=[f"Player {number}" for number in range(1, count + 1)],
        total_rounds=rounds if rounds is not None else (count - 1 if count % 4 == 0 else count),
        court_count=courts,
        scoring_mode=mode,
    )


def _persist(event):
    # Sessions cross a JSON boundary between each operator action.
    return json.loads(json.dumps(event))


def _finish(event):
    number = event["currentRoundNumber"]
    if event["scoringMode"] == "unscored":
        return engine.mark_generator_round_played(event, round_number=number)
    return engine.save_generator_round(event, round_number=number, scores=[
        {"match_id": match["id"], "score_a": 11, "score_b": number % 10}
        for match in event["rounds"][number - 1]["matches"]
    ])


def _playing(row):
    return [pid for match in row["matches"] for pid in match["sideA"] + match["sideB"]]


def _assignment(row):
    return {
        "matches": [(match["id"], match["court"], match["sideA"], match["sideB"])
                    for match in row["matches"]],
        "byes": row["byeParticipantIds"],
    }


def _assert_valid_round(event, row, *, courts=None):
    active = set(engine.active_participant_ids(event, row["number"]))
    playing = _playing(row)
    byes = row["byeParticipantIds"]
    capacity = min(courts or len(active) // 4, len(active) // 4)
    if event.get("perfectRotation") and event.get("rosterRevisions"):
        assert 1 <= len(row["matches"]) <= capacity
    else:
        assert len(row["matches"]) == capacity
    assert len(playing) == len(set(playing))
    assert len(byes) == len(set(byes))
    assert set(playing).isdisjoint(byes)
    assert set(playing) | set(byes) == active
    for match in row["matches"]:
        assert len(match["sideA"]) == len(match["sideB"]) == 2


def _assert_perfect(event, count):
    ids = {row["id"] for row in event["participants"]}
    assert len(ids) == count
    cycle = count - 1 if count % 4 == 0 else count
    rounds = event["rounds"][:cycle]
    assert len(rounds) == cycle
    partners, opponents, games, byes = Counter(), Counter(), Counter(), Counter()
    for row in rounds:
        _assert_valid_round(event, row)
        assert len(row["byeParticipantIds"]) == count % 4
        byes.update(row["byeParticipantIds"])
        for match in row["matches"]:
            a, b = match["sideA"], match["sideB"]
            games.update(a + b)
            partners.update([frozenset(a), frozenset(b)])
            opponents.update(frozenset(pair) for pair in product(a, b))
    expected_pairs = {frozenset(pair) for pair in combinations(ids, 2)}
    assert partners == Counter({pair: 1 for pair in expected_pairs})
    assert opponents == Counter({pair: 2 for pair in expected_pairs})
    assert games == Counter({pid: count - 1 for pid in ids})
    assert byes == (Counter({pid: 1 for pid in ids}) if count % 4 == 1 else Counter())


def _assert_byes_respect_bye_priority(event, row):
    history = engine.history_before_round(event, row["number"])
    for bye in row["byeParticipantIds"]:
        assert all(
            (-history["byes"][bye], history["games"][bye])
            >= (-history["byes"][pid], history["games"][pid])
            for pid in _playing(row)
        )


@pytest.mark.parametrize("count", [n for n in range(4, 41) if n % 4 in {0, 1}])
def test_every_supported_perfect_roster_has_exact_partner_opponent_and_bye_coverage(count):
    preview = _preview(count)
    assert preview["perfectRotation"] is True
    _assert_perfect(preview, count)


@pytest.mark.parametrize("count", [4, 5, 8, 9, 12, 16])
@pytest.mark.parametrize("mode", ["scored", "unscored"])
def test_choose_rounds_out_of_order_and_skip_without_changing_the_rotation(count, mode):
    event = engine.start_generator_event(_preview(count, mode=mode))
    assignments = [_assignment(row) for row in event["rounds"]]
    # Begin on the last round, then play the full original schedule in reverse.
    order = list(range(event["totalRounds"], 0, -1))
    for number in order:
        event = engine.select_generator_round(_persist(event), round_number=number)
        assert event["currentRoundNumber"] == number
        assert sum(row["status"] == "active" for row in event["rounds"]) == 1
        assert [_assignment(row) for row in event["rounds"]] == assignments
        if number == order[0]:
            event = engine.skip_generator_round(event, round_number=number)
            event = engine.select_generator_round(_persist(event), round_number=number)
        event = _finish(event)
    _assert_perfect(event, count)
    event = engine.advance_generator_event(_persist(event))
    assert event["currentRoundNumber"] == len(assignments) + 1
    assert event["adaptiveFromRound"] == len(assignments) + 1
    assert [_assignment(row) for row in event["rounds"][:len(assignments)]] == assignments


def test_next_round_wraps_to_remaining_scheduled_games_after_starting_on_round_seven():
    event = engine.start_generator_event(_preview(8))
    event = engine.select_generator_round(event, round_number=7)
    event = engine.advance_generator_event(_finish(event))
    assert event["currentRoundNumber"] == 1
    assert len(event["rounds"]) == 7
    assert event["adaptiveFromRound"] is None


def test_choosing_an_extra_round_switches_to_adaptive_without_rewriting_the_normal_cycle():
    event = engine.start_generator_event(_preview(8, rounds=9))
    original = [_assignment(row) for row in event["rounds"][:7]]
    event = engine.select_generator_round(_persist(event), round_number=8)
    assert event["adaptiveFromRound"] == 8
    assert [_assignment(row) for row in event["rounds"][:7]] == original
    _assert_valid_round(event, event["rounds"][7])


@pytest.mark.parametrize("change", [
    {"action": "add", "name": "Late arrival"},
    {"action": "remove", "participant_id": "p-3"},
    {"action": "substitute", "participant_id": "p-3", "name": "Substitute", "substitute_scope": "rest"},
])
def test_roster_changes_after_starting_late_apply_to_earlier_unplayed_rounds(change):
    event = engine.start_generator_event(_preview(8))
    event = engine.select_generator_round(event, round_number=7)
    event = _finish(event)
    completed = deepcopy(event["rounds"][6])
    event = engine.mutate_generator_roster(_persist(event), **change)
    assert event["rounds"][6] == completed
    assert set(engine.active_participant_ids(event, 7)) == {f"p-{i}" for i in range(1, 9)}
    event = engine.select_generator_round(_persist(event), round_number=1)
    _assert_valid_round(event, event["rounds"][0])
    active = set(engine.active_participant_ids(event, 1))
    assert ("p-new-1" in active) == (change["action"] != "remove")
    assert ("p-3" in active) == (change["action"] == "add")
    assert event["rounds"][6] == completed
    assert (event["adaptiveFromRound"] is None) == (change["action"] == "add")


@pytest.mark.parametrize("count", [8, 9])
@pytest.mark.parametrize("courts", [0, 2])
@pytest.mark.parametrize("mode", ["scored", "unscored"])
def test_perfect_rotation_survives_start_save_advance_and_reload(count, courts, mode):
    preview = _preview(count, courts=courts, mode=mode)
    _assert_perfect(preview, count)
    expected_assignments = [_assignment(row) for row in preview["rounds"]]
    event = engine.start_generator_event(_persist(preview))
    for number in range(1, preview["totalRounds"] + 1):
        assert event["currentRoundNumber"] == number
        assert _assignment(event["rounds"][number - 1]) == expected_assignments[number - 1]
        event = _persist(_finish(event))
        saved = deepcopy(event["rounds"][:number])
        if number < preview["totalRounds"]:
            event = engine.advance_generator_event(event)
            assert event["rounds"][:number] == saved
            event = _persist(event)
    _assert_perfect(event, count)

    # Finishing the exact rotation must not cap the open-ended session.
    completed_cycle = deepcopy(event["rounds"])
    for number in range(preview["totalRounds"] + 1, preview["totalRounds"] + 3):
        event = engine.advance_generator_event(_persist(event))
        assert event["status"] == "active"
        assert event["currentRoundNumber"] == number
        assert event["totalRounds"] == number + 1  # One ready preview beyond the active round.
        assert event["rounds"][:len(completed_cycle)] == completed_cycle
        _assert_valid_round(event, event["rounds"][-1])
        _assert_byes_respect_bye_priority(event, event["rounds"][-1])
        event = _finish(event)


@pytest.mark.parametrize("count", [8, 9])
def test_roster_reordered_before_start_still_has_complete_pair_coverage(count):
    preview = _preview(count)
    reordered = engine.mutate_generator_roster(
        preview, action="reorder",
        roster_order=[row["id"] for row in reversed(preview["participants"])],
    )
    event = engine.start_generator_event(_persist(reordered))
    for _ in range(event["totalRounds"] - 1):
        event = engine.advance_generator_event(_persist(_finish(event)))
    event = _finish(event)
    _assert_perfect(event, count)


@pytest.mark.parametrize("count,change", [
    (8, {"action": "add", "name": "Late arrival"}),
    (9, {"action": "remove", "participant_id": "p-3"}),
    (9, {"action": "substitute", "participant_id": "p-3", "name": "Substitute", "substitute_scope": "round"}),
])
def test_changed_roster_keeps_completed_and_current_games_then_adapts(count, change):
    event = engine.start_generator_event(_preview(count))
    for _ in range(2):
        event = engine.advance_generator_event(_finish(event))
    original = deepcopy(event)

    event = engine.mutate_generator_roster(event, **change)

    assert event["rounds"][:3] == original["rounds"][:3]
    assert engine.history_before_round(event, 3)["partners"] == engine.history_before_round(original, 3)["partners"]
    for _ in range(3):
        event = engine.advance_generator_event(_persist(_finish(event)))
        row = event["rounds"][event["currentRoundNumber"] - 1]
        _assert_valid_round(event, row)
        if not event.get("perfectRotation"):
            _assert_byes_respect_bye_priority(event, row)
        active = set(engine.active_participant_ids(event, row["number"]))
        if change["action"] == "add":
            assert "p-new-1" in active
        elif change["action"] == "remove":
            assert "p-3" not in active
        else:
            assert ("p-3" in active) == (row["number"] > 4)
            assert ("p-new-1" in active) == (row["number"] == 4)
    assert event["rounds"][:2] == original["rounds"][:2]


def test_legacy_opening_game_does_not_force_the_template_bye_twice():
    event = engine.start_generator_event(_preview(9))
    event.pop("fixedScheduleRoundCount")
    first = event["rounds"][0]
    old_bye = first["byeParticipantIds"][0]
    next_planned_bye = event["rounds"][1]["byeParticipantIds"][0]
    assert old_bye != next_planned_bye
    # A persisted session may have started under the old scheduler. Keep a
    # valid first round but change its bye to the new template's second bye.
    for match in first["matches"]:
        for field in ("sideA", "sideB", "teamA", "teamB"):
            if field in match:
                match[field] = [old_bye if pid == next_planned_bye else pid for pid in match[field]]
    first["byeParticipantIds"] = [next_planned_bye]
    event = _persist(_finish(event))
    completed = deepcopy(event["rounds"][0])

    event = engine.advance_generator_event(event)

    assert event["rounds"][0] == completed
    _assert_valid_round(event, event["rounds"][1])
    _assert_byes_respect_bye_priority(event, event["rounds"][1])
    assert next_planned_bye in _playing(event["rounds"][1])


def test_skipping_a_round_uses_only_played_games_and_preserves_the_skipped_games():
    event = engine.start_generator_event(_preview(9))
    assignments = [_assignment(row) for row in event["rounds"]]
    event = engine.advance_generator_event(_finish(event))
    first = deepcopy(event["rounds"][0])
    event = engine.skip_generator_round(event, round_number=2, reason="Skipped before play")
    skipped = deepcopy(event["rounds"][1])
    assert sum(engine.history_before_round(event, 3)["games"].values()) == 8
    for number in range(3, 11):
        event = engine.advance_generator_event(_persist(event))
        row = event["rounds"][number - 1]
        _assert_valid_round(event, row)
        if number <= 9:
            assert _assignment(row) == assignments[number - 1]
        else:
            _assert_byes_respect_bye_priority(event, row)
        assert event["rounds"][0] == first
        assert event["rounds"][1] == skipped
        event = _finish(event)


@pytest.mark.parametrize("count", [8, 9])
def test_single_court_schedule_respects_capacity_instead_of_forcing_two_games(count):
    event = engine.start_generator_event(_preview(count, courts=1))
    for _ in range(3):
        row = event["rounds"][event["currentRoundNumber"] - 1]
        _assert_valid_round(event, row, courts=1)
        _assert_byes_respect_bye_priority(event, row)
        event = engine.advance_generator_event(_finish(event))
