"""Skipped rounds remain playable, and round robins end only by choice."""

from copy import deepcopy

import pytest

from jupr_app.domain import adaptive_play_engine as engine


def _session(*, kind="round_robin", play_format="doubles", mode="scored", rounds=8):
    count = 6 if play_format == "doubles_singles" else 5
    return engine.start_generator_event(engine.create_generator_preview(
        generator_kind=kind,
        play_format=play_format,
        title="Evening round robin",
        participant_names=[f"Player {number}" for number in range(1, count + 1)],
        total_rounds=rounds,
        court_count=2 if play_format == "singles" else 1,
        doubles_court_count=1,
        singles_court_count=1,
        scoring_mode=mode,
    ))


def _finish_round(event, number):
    if event["scoringMode"] == "unscored":
        return engine.mark_generator_round_played(event, round_number=number)
    return engine.save_generator_round(event, round_number=number, scores=[
        {"match_id": match["id"], "score_a": 11, "score_b": 7}
        for match in engine._round_matches(event["rounds"][number - 1])
    ])


@pytest.mark.parametrize("event_status", ["active", "completed"])
@pytest.mark.parametrize("mode", ["scored", "unscored"])
def test_reopen_round_seven_preserves_newer_round_and_all_original_games(event_status, mode):
    event = _session(mode=mode)
    for number in range(1, 8):
        event = engine.skip_generator_round(event, round_number=number, reason="Looking for a matchup")
        event = engine.advance_generator_event(event)
    if event_status == "completed":
        # Older versions automatically closed the session after round eight.
        event = _finish_round(event, 8)
        event["status"] = "completed"
        event["completedAt"] = "2026-10-09T02:20:00+00:00"
    original = deepcopy(event)
    before_history = engine.history_before_round(event, 9)

    reopened = engine.reopen_generator_round(event, round_number=7)

    expected = deepcopy(original)
    expected["status"] = "active"
    expected["completedAt"] = None
    expected["rounds"][6].update(status="active", skippedAt=None, skipReason="")
    assert reopened == expected
    assert event == original
    assert reopened["currentRoundNumber"] == 8
    assert engine.history_before_round(reopened, 9) == before_history

    scored = _finish_round(reopened, 7)

    assert scored["rounds"][6]["status"] == ("saved" if mode == "scored" else "played")
    assert scored["currentRoundNumber"] == 8
    assert scored["rounds"][:6] == original["rounds"][:6]
    assert scored["rounds"][7] == original["rounds"][7]
    after_history = engine.history_before_round(scored, 9)
    on_court = {
        pid for match in original["rounds"][6]["matches"]
        for pid in match["sideA"] + match["sideB"]
    }
    for pid in before_history["games"]:
        assert after_history["games"][pid] == before_history["games"][pid] + int(pid in on_court)
        assert after_history["byes"][pid] == before_history["byes"][pid] + int(
            pid in original["rounds"][6]["byeParticipantIds"]
        )


def test_reopening_current_skipped_round_requires_finishing_it_before_advancing():
    event = engine.skip_generator_round(_session(), round_number=1)
    event = engine.reopen_generator_round(event, round_number=1)
    with pytest.raises(ValueError, match="Save or skip"):
        engine.advance_generator_event(event)
    event = engine.advance_generator_event(_finish_round(event, 1))
    assert event["currentRoundNumber"] == 2
    assert event["rounds"][0]["status"] == "saved"


@pytest.mark.parametrize("play_format", ["singles", "doubles", "doubles_singles"])
@pytest.mark.parametrize("mode", ["scored", "unscored"])
def test_round_robins_continue_incrementally_beyond_eight_and_fifty(play_format, mode):
    event = _session(play_format=play_format, mode=mode)
    for number in range(1, 53):
        event = _finish_round(event, number)
        previous_rounds = deepcopy(event["rounds"][:number])
        old_total = event["totalRounds"]

        updated = engine.advance_generator_event(event)

        assert updated["status"] == "active"
        assert updated.get("completedAt") is None
        assert updated["currentRoundNumber"] == number + 1
        assert updated["totalRounds"] == max(old_total, number + 1)
        assert len(updated["rounds"]) == updated["totalRounds"]
        assert updated["rounds"][:number] == previous_rounds
        next_round = updated["rounds"][number]
        assert next_round["status"] == "active"
        assert next_round["matches"]
        # The extension must retain the games-first bye rule, even once the
        # originally generated schedule is exhausted.
        history = engine.history_before_round(updated, number + 1)
        playing = {
            pid for match in next_round["matches"]
            for pid in match["sideA"] + match["sideB"]
        }
        for bye in next_round["byeParticipantIds"]:
            assert all(
                (history["games"][bye], -history["byes"][bye])
                >= (history["games"][pid], -history["byes"][pid])
                for pid in playing
            )
        event = updated
    assert event["currentRoundNumber"] == 53


@pytest.mark.parametrize("last_status", ["saved", "skipped"])
def test_legacy_completed_round_robin_can_continue(last_status):
    event = _session(rounds=1)
    event = (
        _finish_round(event, 1) if last_status == "saved"
        else engine.skip_generator_round(event, round_number=1)
    )
    event["status"] = "completed"
    event["completedAt"] = "2026-10-09T02:20:00+00:00"
    original_round = deepcopy(event["rounds"][0])

    updated = engine.advance_generator_event(event)

    assert updated["status"] == "active"
    assert updated["completedAt"] is None
    assert updated["currentRoundNumber"] == updated["totalRounds"] == 2
    assert len(updated["rounds"]) == 2
    assert updated["rounds"][0] == original_round
    assert updated["rounds"][1]["status"] == "active"


def test_ladder_still_finishes_at_configured_round_count_and_cannot_reopen():
    event = _session(kind="ladder", rounds=1)
    event = engine.skip_generator_round(event, round_number=1)
    with pytest.raises(ValueError, match="only for Round-Robin"):
        engine.reopen_generator_round(event, round_number=1)

    completed = engine.advance_generator_event(event)

    assert completed["status"] == "completed"
    assert completed["completedAt"]
    assert completed["currentRoundNumber"] == completed["totalRounds"] == 1
    assert len(completed["rounds"]) == 1


@pytest.mark.parametrize("round_status", ["active", "preview", "saved", "played"])
def test_reopen_rejects_non_skipped_round_without_mutating_it(round_status):
    event = _session()
    event["rounds"][0]["status"] = round_status
    original = deepcopy(event)
    with pytest.raises(ValueError, match="Only a skipped round"):
        engine.reopen_generator_round(event, round_number=1)
    assert event == original


def test_reopen_rejects_unstarted_event_and_missing_round():
    event = engine.skip_generator_round(_session(), round_number=1)
    event["status"] = "preview"
    with pytest.raises(ValueError, match="Only a started"):
        engine.reopen_generator_round(event, round_number=1)
    event["status"] = "active"
    with pytest.raises(ValueError, match="Round 99 was not found"):
        engine.reopen_generator_round(event, round_number=99)


@pytest.mark.parametrize("mode", ["scored", "unscored"])
def test_explicit_finish_skips_untouched_current_round_and_preserves_future_previews(mode):
    event = _session(mode=mode)
    event = engine.advance_generator_event(_finish_round(event, 1))
    original = deepcopy(event)

    completed = engine.complete_generator_event(event)

    assert event == original
    assert completed["status"] == "completed"
    assert completed["completedAt"]
    assert completed["currentRoundNumber"] == 2
    assert completed["rounds"][0] == original["rounds"][0]
    assert completed["rounds"][1]["status"] == "skipped"
    assert completed["rounds"][1]["skipReason"] == "Session finished before this round was played."
    assert completed["rounds"][1]["matches"] == original["rounds"][1]["matches"]
    assert completed["rounds"][2:] == original["rounds"][2:]
    assert engine.history_before_round(completed, 3) == engine.history_before_round(original, 3)
    assert engine.complete_generator_event(completed) == completed
    reopened = engine.reopen_generator_round(completed, round_number=2)
    assert reopened["status"] == reopened["rounds"][1]["status"] == "active"


@pytest.mark.parametrize("score_field", ["scoreA", "scoreB"])
def test_explicit_finish_rejects_current_partial_scores(score_field):
    event = _session()
    event["rounds"][0]["matches"][0][score_field] = 0
    original = deepcopy(event)
    with pytest.raises(ValueError, match="Save or clear entered scores"):
        engine.complete_generator_event(event)
    assert event == original


def test_explicit_finish_rejects_other_open_round_even_when_current_is_saved():
    event = engine.skip_generator_round(_session(), round_number=1)
    event = engine.advance_generator_event(event)
    event = _finish_round(event, 2)
    event = engine.reopen_generator_round(event, round_number=1)
    original = deepcopy(event)
    with pytest.raises(ValueError, match="every open round"):
        engine.complete_generator_event(event)
    assert event == original
    event = _finish_round(event, 1)
    assert engine.complete_generator_event(event)["status"] == "completed"


def test_ladder_explicit_finish_still_requires_current_round_terminal():
    event = _session(kind="ladder")
    with pytest.raises(ValueError, match="skip the current round"):
        engine.complete_generator_event(event)
    event = engine.skip_generator_round(event, round_number=1)
    completed = engine.complete_generator_event(event)
    assert completed["status"] == "completed"
    assert completed["rounds"] == event["rounds"]
