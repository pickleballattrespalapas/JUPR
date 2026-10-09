"""Playing time has priority over variety when choosing round-robin byes."""

from copy import deepcopy

import pytest

from jupr_app.domain import adaptive_play_engine as engine


def _event(play_format, count, *, courts=2):
    return {
        "generatorKind": "round_robin",
        "playFormat": play_format,
        "courtCount": courts,
        "doublesCourtCount": 1,
        "singlesCourtCount": 1,
        "participants": [
            {"id": f"p-{i}", "name": f"Player {i}", "roster_order": i}
            for i in range(1, count + 1)
        ],
    }


def _playing(round_row):
    return {
        pid
        for match in round_row["matches"]
        for pid in match["sideA"] + match["sideB"]
    }


def _pressure_to_bench(history, player_id, all_ids):
    # Variety optimization would prefer to sit this player out, even though
    # their playing-time priority requires that they stay on court.
    for other in all_ids:
        if other != player_id:
            key = engine._pair_key(player_id, other)
            for field in ("partners", "opponents", "singles_opponents", "doubles_opponents"):
                history[field][key] = 100


@pytest.mark.parametrize("play_format,count,courts", [
    ("singles", 5, 2),
    ("doubles", 9, 2),
    ("doubles", 13, 3),
    ("doubles_singles", 7, 2),
])
def test_most_games_receive_byes_before_buddy_despite_pairing_pressure(play_format, count, courts):
    event = _event(play_format, count, courts=courts)
    history = engine._blank_history(event)
    history["games"] = {pid: 3 for pid in history["games"]}
    history["games"].update({"p-1": 5, "p-2": 5, "p-3": 4})
    history["byes"]["p-3"] = 1
    _pressure_to_bench(history, "p-3", list(history["games"]))

    round_row = engine._generate_round_robin_round(event, 6, history)

    assert len(round_row["byeParticipantIds"]) == 1
    assert set(round_row["byeParticipantIds"]) <= {"p-1", "p-2"}
    assert "p-3" in _playing(round_row)


@pytest.mark.parametrize("play_format,count,courts", [
    ("singles", 7, 2),
    ("doubles", 13, 2),
    ("doubles_singles", 11, 2),
])
def test_capacity_limited_round_sits_highest_game_tiers_first(play_format, count, courts):
    event = _event(play_format, count, courts=courts)
    history = engine._blank_history(event)
    history["games"] = {f"p-{i}": i // 3 for i in range(1, count + 1)}
    _pressure_to_bench(history, "p-1", list(history["games"]))

    round_row = engine._generate_round_robin_round(event, 6, history)
    playing = _playing(round_row)
    byes = set(round_row["byeParticipantIds"])

    assert len(playing) == (4 if play_format == "singles" else 8 if play_format == "doubles" else 6)
    assert playing.isdisjoint(byes)
    assert playing | byes == set(history["games"])
    assert min(history["games"][pid] for pid in byes) >= max(history["games"][pid] for pid in playing)


@pytest.mark.parametrize("play_format,count", [("singles", 5), ("doubles", 9), ("doubles_singles", 7)])
def test_equal_games_protect_player_with_more_prior_byes(play_format, count):
    event = _event(play_format, count)
    history = engine._blank_history(event)
    history["games"] = {pid: 4 for pid in history["games"]}
    history["byes"]["p-3"] = 2
    _pressure_to_bench(history, "p-3", list(history["games"]))

    round_row = engine._generate_round_robin_round(event, 6, history)

    assert "p-3" in _playing(round_row)
    assert len(round_row["byeParticipantIds"]) == 1


def _singles_session(*, mode="scored"):
    return engine.start_generator_event(engine.create_generator_preview(
        generator_kind="round_robin", play_format="singles", title="Bye fairness",
        participant_names=["A", "B", "C", "D", "E"],
        total_rounds=4, court_count=2, scoring_mode=mode,
    ))


def _save(event, round_number):
    row = event["rounds"][round_number - 1]
    return engine.save_generator_round(event, round_number=round_number, scores=[
        {"match_id": match["id"], "score_a": 11, "score_b": 7}
        for match in row["matches"]
    ])


def _repeat_previous_schedule(event, previous_number, future_number):
    """Model an already-persisted unfair preview from the older scheduler."""
    old = event["rounds"][previous_number - 1]
    future = event["rounds"][future_number - 1]
    future["byeParticipantIds"] = list(old["byeParticipantIds"])
    for before, after in zip(old["matches"], future["matches"]):
        for field in ("sideA", "sideB", "teamA", "teamB"):
            after[field] = list(before[field])


def test_advance_replaces_existing_unfair_preview_and_preserves_saved_round():
    event = _save(_singles_session(), 1)
    waiting_player = event["rounds"][0]["byeParticipantIds"][0]
    _repeat_previous_schedule(event, 1, 2)
    original = deepcopy(event)

    updated = engine.advance_generator_event(event)

    assert event == original
    assert updated["rounds"][0] == original["rounds"][0]
    assert waiting_player in _playing(updated["rounds"][1])
    assert waiting_player not in updated["rounds"][1]["byeParticipantIds"]


def test_skipped_round_does_not_supply_games_or_byes_to_next_round():
    event = engine.advance_generator_event(_save(_singles_session(), 1))
    waiting_player = event["rounds"][0]["byeParticipantIds"][0]
    event = engine.skip_generator_round(event, round_number=2, reason="Unfair bye")
    _repeat_previous_schedule(event, 1, 3)
    previous = deepcopy(event["rounds"][:2])

    updated = engine.advance_generator_event(event)
    history = engine.history_before_round(updated, 3)

    assert updated["rounds"][:2] == previous
    assert history["games"][waiting_player] == 0
    assert history["byes"][waiting_player] == 1
    assert sum(history["games"].values()) == 4
    assert sum(history["byes"].values()) == 1
    assert waiting_player in _playing(updated["rounds"][2])


@pytest.mark.parametrize("play_format", ["singles", "doubles", "doubles_singles"])
def test_preview_and_normal_advance_are_deterministic(play_format):
    args = dict(
        generator_kind="round_robin", play_format=play_format, title="Stable preview",
        participant_names=[f"Player {i}" for i in range(9)], total_rounds=3,
        court_count=2, doubles_court_count=1, singles_court_count=2,
    )
    preview = engine.create_generator_preview(**args)
    assert preview["rounds"] == engine.create_generator_preview(**args)["rounds"]
    event = _save(engine.start_generator_event(preview), 1)
    expected = deepcopy(event["rounds"][1:])
    expected[0]["status"] = "active"

    updated = engine.advance_generator_event(event)

    assert updated["rounds"][1:] == expected


@pytest.mark.parametrize("state", ["saved", "played", "skipped", "active", "partial"])
@pytest.mark.parametrize("round_number", [2, 3])
def test_advance_preserves_future_started_rounds(state, round_number):
    if state == "played":
        event = engine.mark_generator_round_played(_singles_session(mode="unscored"), round_number=1)
        event = engine.mark_generator_round_played(event, round_number=round_number)
    else:
        event = _save(_singles_session(), 1)
        if state == "saved":
            event = _save(event, round_number)
        elif state == "skipped":
            event = engine.skip_generator_round(event, round_number=round_number)
        elif state == "active":
            event["rounds"][round_number - 1]["status"] = "active"
        else:
            event["rounds"][round_number - 1]["matches"][0]["scoreA"] = 5
    expected = deepcopy(event["rounds"][round_number - 1])
    if round_number == 2 and state == "partial":
        expected["status"] = "active"

    updated = engine.advance_generator_event(event)

    assert updated["rounds"][round_number - 1] == expected
