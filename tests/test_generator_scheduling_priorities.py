"""Observable perfect-first, bye-first scheduling policy across JSON actions."""
from collections import Counter
from copy import deepcopy
from itertools import combinations, product
import json

import pytest

from jupr_app.domain import adaptive_play_engine as engine
from jupr_app.domain.perfect_rotation_completion import find_perfect_completion


def assert_exact(event, ids):
    partners, opponents = Counter(), Counter()
    for row in event["rounds"]:
        playing = []
        for game in row["matches"]:
            a, b = game["sideA"], game["sideB"]
            playing.extend(a + b)
            partners.update(tuple(sorted(side)) for side in (a, b))
            opponents.update(tuple(sorted(pair)) for pair in product(a, b))
        assert len(playing) == len(set(playing))
        assert not set(playing) & set(row["byeParticipantIds"])
    pairs = {tuple(sorted(pair)) for pair in combinations(ids, 2)}
    assert partners == Counter({pair: 1 for pair in pairs})
    assert opponents == Counter({pair: 2 for pair in pairs})


@pytest.mark.parametrize("count", [n for n in range(8, 41) if n % 4 in {0, 1}])
def test_arrivals_completing_the_opening_round_recover_the_full_normal_rotation(count):
    event = engine.start_generator_event(engine.create_generator_preview(
        generator_kind="round_robin", play_format="doubles", title="Opening arrivals",
        participant_names=[str(i) for i in range(1, count - 1)], total_rounds=1,
    ))
    event["rounds"][0]["matches"][0]["scoreA"] = 5
    original = deepcopy(event["rounds"][0]["matches"])
    for name in (str(count - 1), str(count)):
        event = engine.mutate_generator_roster(event, action="add", name=name)
    byes = event["rounds"][0]["byeParticipantIds"][:2]
    event = engine.mutate_generator_roster(json.loads(json.dumps(event)), action="seat_arrivals",
        participant_ids=["p-new-1", "p-new-2", *byes], court_number=count // 4)
    assert event["rounds"][0]["matches"][:len(original)] == original
    assert event["perfectRotation"] is True
    assert event["adaptiveFromRound"] is None
    assert event["totalRounds"] == (count - 1 if count % 4 == 0 else count)
    assignments = deepcopy(event["rounds"])
    ids = [p["id"] for p in event["participants"]]
    assert_exact(event, ids)
    # Saving and selecting a different round must keep the recovered rotation.
    event = engine.save_generator_round(event, round_number=1, scores=[
        {"match_id":m["id"], "score_a":11, "score_b":7} for m in event["rounds"][0]["matches"]])
    event = engine.select_generator_round(json.loads(json.dumps(event)), round_number=event["totalRounds"])
    assert [r["matches"] for r in event["rounds"]][1:] == [r["matches"] for r in assignments][1:]
    assert_exact(event, ids)


def test_later_arrival_can_recover_compatible_history_without_rewriting_games():
    event = engine.start_generator_event(engine.create_generator_preview(
        generator_kind="round_robin", play_format="doubles", title="Later arrival",
        participant_names=[str(i) for i in range(1, 9)], total_rounds=7))
    for _ in range(2):
        row=event["rounds"][event["currentRoundNumber"]-1]
        event=engine.save_generator_round(event, round_number=row["number"], scores=[
            {"match_id":m["id"],"score_a":11,"score_b":7} for m in row["matches"]])
        event=engine.advance_generator_event(event)
    original=deepcopy(event["rounds"][:3])
    event=engine.mutate_generator_roster(event, action="add", name="Nine")
    assert event["rounds"][:3] == original
    assert event["perfectRotation"] is True
    assert_exact(event, [p["id"] for p in event["participants"]])


@pytest.mark.parametrize("count", [8, 9, 12, 16])
def test_preview_roster_growth_uses_the_normal_perfect_cycle(count):
    event = engine.create_generator_preview(
        generator_kind="round_robin", play_format="doubles", title="Preview changes",
        participant_names=[str(i) for i in range(count - 1)], total_rounds=3)
    event = engine.mutate_generator_roster(event, action="add", name="Final player")
    assert event["totalRounds"] == (count - 1 if count % 4 == 0 else count)
    assert event["perfectRotation"] is True
    assert_exact(event, [p["id"] for p in event["participants"]])


def test_repeated_partnership_or_excess_opponents_cannot_be_called_perfect():
    ids=[str(i) for i in range(8)]
    game=(["0","1"],["2","3"])
    assert find_perfect_completion(ids, [game, game]) is None
    assert find_perfect_completion(ids, [game, (["0","1"],["4","5"])]) is None
    event=engine.start_generator_event(engine.create_generator_preview(
        generator_kind="round_robin", play_format="doubles", title="Unavoidable repeats",
        participant_names=ids, total_rounds=3))
    event["rounds"][1]=deepcopy(event["rounds"][0])
    event["rounds"][1]["number"]=2
    event["rounds"][1]["status"]="saved"
    original=deepcopy(event["rounds"][:2])
    event=engine.mutate_generator_roster(event, action="reorder",roster_order=[p["id"] for p in event["participants"]])
    assert event["perfectRotation"] is False
    assert event["adaptiveFromRound"] == 1
    assert event["rounds"][:2] == original


@pytest.mark.parametrize("count,courts,rounds", [(8, 1, 14), (9, 1, 18), (12, 2, 17)])
def test_a_full_rotation_can_use_fewer_courts(count, courts, rounds):
    event = engine.create_generator_preview(
        generator_kind="round_robin", play_format="doubles", title="Fewer courts",
        participant_names=[str(i) for i in range(count)], total_rounds=rounds, court_count=courts)
    assert event["perfectRotation"] is True
    assert all(1 <= len(row["matches"]) <= courts for row in event["rounds"])
    assert_exact(event, [p["id"] for p in event["participants"]])


def test_requested_extra_rounds_remain_after_a_perfect_cycle_on_one_court():
    event = engine.create_generator_preview(
        generator_kind="round_robin", play_format="doubles", title="Extra rounds",
        participant_names=[str(i) for i in range(8)], total_rounds=16, court_count=1)
    assert event["totalRounds"] == 16
    assert len(event["rounds"]) == 16
    assert event["fixedScheduleRoundCount"] == 14
    assert_exact({"rounds": event["rounds"][:14]}, [p["id"] for p in event["participants"]])


def test_a_round_six_arrival_has_no_prearrival_byes_and_gets_a_bye_before_second_byes():
    event = engine.start_generator_event(engine.create_generator_preview(
        generator_kind="round_robin", play_format="doubles", title="Round six arrival",
        participant_names=[str(i) for i in range(6)], total_rounds=6, court_count=1))
    for number in range(1, 6):
        row = event["rounds"][number - 1]
        event = engine.save_generator_round(event, round_number=number, scores=[
            {"match_id": m["id"], "score_a": 11, "score_b": 7} for m in row["matches"]])
        if number < 5:
            event = engine.advance_generator_event(event)
    original = deepcopy(event["rounds"][:5])
    event = engine.mutate_generator_roster(json.loads(json.dumps(event)), action="add", name="Late")
    event = engine.advance_generator_event(event)
    history = engine.history_before_round(event, 6)
    assert event["rounds"][:5] == original
    assert history["games"]["p-new-1"] == 0
    assert history["byes"]["p-new-1"] == 0
    row = event["rounds"][5]
    assert "p-new-1" in row["byeParticipantIds"]
    playing = [p for m in row["matches"] for side in (m["sideA"], m["sideB"]) for p in side]
    assert max(history["byes"][p] for p in row["byeParticipantIds"]) <= min(history["byes"][p] for p in playing)
