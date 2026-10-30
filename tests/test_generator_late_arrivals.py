from copy import deepcopy

import pytest

from jupr_app.domain.adaptive_play_engine import (
    active_participant_ids,
    advance_generator_event,
    create_generator_preview,
    generator_event_standings,
    history_before_round,
    mark_generator_round_played,
    mutate_generator_roster,
    save_generator_round,
    schedule_export_rows,
    skip_generator_round,
    start_generator_event,
)


def session(*, kind="round_robin", mode="scored", count=8, courts=2, rounds=3, fmt="doubles"):
    return start_generator_event(create_generator_preview(
        generator_kind=kind, play_format=fmt, title="Late arrivals",
        participant_names=[f"Original {i}" for i in range(count)],
        court_count=courts, total_rounds=rounds, scoring_mode=mode,
    ))


def add_arrivals(event, count=4):
    for i in range(count):
        event = mutate_generator_roster(event, action="add", name=f"Arrival {i}")
    return event


def score(event):
    number = event["currentRoundNumber"]
    return save_generator_round(event, round_number=number, scores=[
        {"match_id": match["id"], "score_a": 11, "score_b": 7}
        for match in event["rounds"][number - 1]["matches"]
    ])


@pytest.mark.parametrize("kind,mode", [("round_robin", "scored"), ("round_robin", "unscored"), ("ladder", "scored")])
@pytest.mark.parametrize("change", [
    {"action": "add", "name": "New"},
    {"action": "remove", "participant_id": "p-1"},
    {"action": "substitute", "participant_id": "p-1", "name": "Sub", "substitute_scope": "round"},
    {"action": "reorder", "roster_order": [f"p-{i}" for i in range(9, 0, -1)]},
])
def test_roster_changes_preserve_active_round_including_byes(kind, mode, change):
    original = session(kind=kind, mode=mode, count=9)
    snapshot = deepcopy(original)
    updated = mutate_generator_roster(original, **change)
    assert original == snapshot
    assert updated["rounds"][0] == snapshot["rounds"][0]
    assert updated["rosterRevisions"][-1]["effectiveRound"] == 2
    if change["action"] == "add":
        assert "p-new-1" not in active_participant_ids(updated, 1)
        assert "p-new-1" in active_participant_ids(updated, 2)
    if change["action"] == "substitute":
        assert "p-1" in active_participant_ids(updated, 1)
        assert "p-1" not in active_participant_ids(updated, 2)
        assert "p-1" in active_participant_ids(updated, 3)


@pytest.mark.parametrize("mode", ["scored", "unscored"])
@pytest.mark.parametrize("courts", [0, 2, 3])
def test_eight_plus_four_append_spare_court_then_mix_and_keep_scores(mode, courts):
    event = session(mode=mode, courts=courts)
    original_round = deepcopy(event["rounds"][0])
    event = add_arrivals(event)
    assert event["rounds"][0] == original_round
    arrivals = [p["id"] for p in event["participants"] if p["active_from_round"] == 2]
    event = mutate_generator_roster(event, action="seat_arrivals", participant_ids=arrivals, court_number=3)
    first = event["rounds"][0]
    assert first["matches"][:2] == original_round["matches"]
    assert first["byeParticipantIds"] == original_round["byeParticipantIds"]
    assert len(first["matches"]) == 3
    assert first["formatCounts"] == {"doubles": 3, "singles": 0}
    late_game = first["matches"][2]
    assert set(late_game["sideA"] + late_game["sideB"]) == set(arrivals)
    assert late_game["court"] == 3
    assert len({m["id"] for m in first["matches"]}) == 3
    assert event["courtCount"] == (0 if courts == 0 else 3)
    if mode == "scored":
        event = score(event)
        standings = generator_event_standings(event)
        assert len(standings) == 12
        assert all(row["matches"] == 1 for row in standings)
    else:
        event = mark_generator_round_played(event, round_number=1)
    completed = deepcopy(event["rounds"][0])
    assert len([row for row in schedule_export_rows(event) if row["round"] == 1]) == 3
    assert all(games == 1 for games in history_before_round(event, 2)["games"].values())
    event = advance_generator_event(event)
    assert event["rounds"][0] == completed
    assert event["currentRoundNumber"] == 2
    second = event["rounds"][1]
    assert not second["byeParticipantIds"]
    assert len(second["matches"]) == 3
    assert any(set(m["sideA"] + m["sideB"]) & set(arrivals) and
               set(m["sideA"] + m["sideB"]) - set(arrivals) for m in second["matches"])


def test_without_spare_court_arrivals_join_next_round_with_configured_capacity():
    event = add_arrivals(session())
    event = advance_generator_event(score(event))
    assert len(event["rounds"][1]["matches"]) == 2
    assert len(event["rounds"][1]["byeParticipantIds"]) == 4
    playing = {pid for m in event["rounds"][1]["matches"] for pid in m["sideA"] + m["sideB"]}
    assert {f"p-new-{i}" for i in range(1, 5)} <= playing


@pytest.mark.parametrize("mode", ["scored", "unscored"])
@pytest.mark.parametrize("number", [1, 2])
def test_five_plus_three_can_seat_the_bye_with_arrivals_and_count_both_games(mode, number):
    event = session(count=5, courts=1, rounds=5, mode=mode)
    if number == 2:
        event = advance_generator_event(score(event) if mode == "scored" else mark_generator_round_played(event, round_number=1))
    row = event["rounds"][number - 1]
    bye = row["byeParticipantIds"][0]
    row["matches"][0]["scoreA"] = 5 if mode == "scored" else None
    original = deepcopy(row["matches"])
    earlier = deepcopy(event["rounds"][:number - 1])
    history = history_before_round(event, number)
    event = add_arrivals(event, 3)
    ids = [f"p-new-{i}" for i in range(1, 4)] + [bye]
    event = mutate_generator_roster(event, action="seat_arrivals", participant_ids=ids, court_number=2)
    row = event["rounds"][number - 1]
    assert row["matches"][:1] == original
    assert not row["byeParticipantIds"]
    assert set(row["matches"][1]["sideA"] + row["matches"][1]["sideB"]) == set(ids)
    assert len({pid for match in row["matches"] for pid in match["sideA"] + match["sideB"]}) == 8
    assert next(p for p in event["participants"] if p["id"] == bye)["active_from_round"] == 1
    event = score(event) if mode == "scored" else mark_generator_round_played(event, round_number=number)
    next_history = history_before_round(event, number + 1)
    assert next_history["byes"][bye] == history["byes"][bye]
    assert next_history["games"][bye] == history["games"][bye] + 1
    assert all(next_history["games"][pid] == 1 for pid in ids if pid != bye)
    event = advance_generator_event(event)
    assert event["rounds"][:number - 1] == earlier
    assert len(event["rounds"][number]["matches"]) == 2
    assert not event["rounds"][number]["byeParticipantIds"]
    with pytest.raises(ValueError, match="waiting"):
        mutate_generator_roster(event, action="seat_arrivals", participant_ids=ids, court_number=3)


def test_arrival_game_removes_only_selected_byes_and_requires_a_new_arrival():
    event = add_arrivals(session(count=7, courts=1), 1)
    byes = event["rounds"][0]["byeParticipantIds"]
    assert len(byes) == 3
    event = mutate_generator_roster(event, action="seat_arrivals", participant_ids=["p-new-1", *byes], court_number=2)
    assert not event["rounds"][0]["byeParticipantIds"]
    event = add_arrivals(session(count=9, courts=1), 1)
    byes = event["rounds"][0]["byeParticipantIds"]
    snapshot = deepcopy(event)
    with pytest.raises(ValueError, match="at least one"):
        mutate_generator_roster(event, action="seat_arrivals", participant_ids=byes[:4], court_number=2)
    assert event == snapshot
    event = mutate_generator_roster(event, action="seat_arrivals", participant_ids=["p-new-1", *byes[:3]], court_number=2)
    assert event["rounds"][0]["byeParticipantIds"] == byes[3:]


@pytest.mark.parametrize("ids,court,error", [
    (["p-new-1", "p-new-2", "p-new-3", "p-new-4"], 1, "already has a game"),
    (["p-new-1", "p-new-2", "p-new-3"], 3, "exactly 4"),
    (["p-new-1"] * 4, 3, "different"),
    (["p-1", "p-new-2", "p-new-3", "p-new-4"], 3, "waiting"),
    (["missing", "p-new-2", "p-new-3", "p-new-4"], 3, "waiting"),
    (["p-new-1", "p-new-2", "p-new-3", "p-new-4"], 0, "between 1 and 20"),
    (["p-new-1", "p-new-2", "p-new-3", "p-new-4"], 21, "between 1 and 20"),
])
def test_invalid_arrival_game_does_not_mutate_session(ids, court, error):
    event = add_arrivals(session())
    snapshot = deepcopy(event)
    with pytest.raises(ValueError, match=error):
        mutate_generator_roster(event, action="seat_arrivals", participant_ids=ids, court_number=court)
    assert event == snapshot


@pytest.mark.parametrize("finish", [score, lambda e: skip_generator_round(e, round_number=1)])
def test_cannot_append_game_to_finished_round(finish):
    event = finish(add_arrivals(session()))
    with pytest.raises(ValueError, match="before it is finished"):
        mutate_generator_roster(event, action="seat_arrivals", participant_ids=[f"p-new-{i}" for i in range(1, 5)], court_number=3)


def test_multiple_arrival_groups_preserve_earlier_games_and_partial_scores():
    event = add_arrivals(session(), 8)
    event["rounds"][0]["matches"][0]["scoreA"] = 5
    first_two = deepcopy(event["rounds"][0]["matches"])
    for court, start in [(3, 1), (4, 5)]:
        event = mutate_generator_roster(event, action="seat_arrivals", participant_ids=[f"p-new-{i}" for i in range(start, start + 4)], court_number=court)
    assert event["rounds"][0]["matches"][:2] == first_two
    assert len(event["rounds"][0]["matches"]) == 4
    with pytest.raises(ValueError, match="waiting"):
        mutate_generator_roster(event, action="seat_arrivals", participant_ids=[f"p-new-{i}" for i in range(1, 5)], court_number=5)


def test_singles_arrivals_and_last_planned_round_can_use_spare_court():
    event = add_arrivals(session(fmt="singles", count=4, rounds=1), 2)
    event = mutate_generator_roster(event, action="seat_arrivals", participant_ids=["p-new-1", "p-new-2"], court_number=3)
    assert len(event["rounds"][0]["matches"]) == 3
    event = advance_generator_event(score(event))
    assert event["status"] == "active"
    assert event["currentRoundNumber"] == 2
    assert len(event["rounds"]) == event["totalRounds"] == 3  # Current + one prepared preview.
    event = mutate_generator_roster(event, action="add", name="Next arrival")
    assert event["participants"][-1]["active_from_round"] == 3
