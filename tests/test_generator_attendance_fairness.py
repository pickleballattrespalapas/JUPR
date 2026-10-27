"""Long-running sessions must not charge arrivals for other players' past byes."""
from collections import Counter
from copy import deepcopy
import json

import pytest

from jupr_app.domain import adaptive_play_engine as engine


def save_and_advance(event):
    number = event["currentRoundNumber"]
    row = next(row for row in event["rounds"] if row["number"] == number)
    event = engine.save_generator_round(event, round_number=number, scores=[
        {"match_id": match["id"], "score_a": 11, "score_b": 7} for match in row["matches"]
    ])
    return engine.advance_generator_event(json.loads(json.dumps(event)))


@pytest.mark.parametrize("play_format,count,courts", [("doubles", 5, 1), ("doubles", 9, 2), ("singles", 3, 1)])
def test_late_arrival_after_fifteen_rounds_gets_a_fair_rotation(play_format, count, courts):
    event = engine.start_generator_event(engine.create_generator_preview(
        generator_kind="round_robin", play_format=play_format, title="Long evening",
        participant_names=[str(i) for i in range(count)], court_count=courts, total_rounds=1,
    ))
    event["roundRobinHubId"] = "attendance-test"
    for _ in range(15):
        event = save_and_advance(event)
    protected = deepcopy(event["rounds"])
    event = engine.mutate_generator_roster(event, action="add", name="Late arrival")
    newcomer = event["participants"][-1]["id"]
    assert event["rounds"][:15] == protected[:15]
    assert event["rounds"][15]["matches"][:len(protected[15]["matches"])] == protected[15]["matches"]
    assert engine.history_before_round(event, 16)["byes"][newcomer] == 0
    previous = set()
    byes = Counter({p["id"]: 0 for p in event["participants"]})
    for _ in range(12):
        row = next(r for r in event["rounds"] if r["number"] == event["currentRoundNumber"])
        current = set(row["byeParticipantIds"])
        assert not previous & current, "a waiting player was assigned consecutive byes"
        previous = current
        byes.update(current)
        event = save_and_advance(event)
    assert max(byes.values()) - min(byes.values()) <= 2
    assert abs(byes[newcomer] - sum(v for p, v in byes.items() if p != newcomer) / count) <= 1


@pytest.mark.parametrize("absence", [3, 15, 30])
def test_return_after_absence_does_not_owe_catchup_byes(absence):
    event = {"participants": [{"id": f"p-{i}", "roster_order": i} for i in range(6)],
             "generatorKind": "round_robin", "playFormat": "doubles", "courtCount": 1,
             "adaptiveFromRound": 1, "rounds": []}
    for number in range(1, absence + 19):
        event["participants"][-1]["inactive_rounds"] = list(range(4, absence + 4))
        history = engine.history_before_round(json.loads(json.dumps(event)), number)
        row = engine._generate_round_robin_round(event, number, history)
        row["status"] = "played"
        event["rounds"].append(row)
    resumed = event["rounds"][absence + 3:]
    byes = Counter({p["id"]: 0 for p in event["participants"]})
    previous = set()
    for row in resumed:
        current = set(row["byeParticipantIds"])
        assert not current & previous
        byes.update(current)
        previous = current
    # Continuing players can still be owed one bye from before the return.
    assert max(byes.values()) - min(byes.values()) <= 2
    assert abs(byes["p-5"] - sum(v for p, v in byes.items() if p != "p-5") / 5) <= 1

