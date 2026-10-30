"""Operational regressions for bounded standalone RR planning."""
from copy import deepcopy
from collections import Counter

import pytest

from jupr_app.domain import adaptive_play_engine as engine


def session(count=18, total=6, play_format="doubles", scoring="unscored"):
    return engine.start_generator_event(engine.create_generator_preview(
        generator_kind="round_robin", play_format=play_format, title="Adaptive planning",
        participant_names=[f"Player {i}" for i in range(count)], total_rounds=total,
        scoring_mode=scoring,
    ))


def row(event, number):
    return next(item for item in event["rounds"] if item["number"] == number)


def players(round_row):
    return [pid for match in round_row["matches"] for pid in match["sideA"] + match["sideB"]]


def test_roster_change_generates_only_next_round_and_preserves_all_committed_games(monkeypatch):
    event = session()
    event = engine.mark_generator_round_played(event, round_number=1)
    event = engine.advance_generator_event(event)
    # A separately recorded future round must also survive regeneration.
    row(event, 6)["status"] = "played"
    before = deepcopy(event)
    generated = []
    original = engine._generate_round_robin_round

    def track(value, number, history):
        generated.append(number)
        return original(value, number, history)

    monkeypatch.setattr(engine, "_generate_round_robin_round", track)
    result = engine.mutate_generator_roster(event, action="remove", participant_id="p-18")
    assert generated == [3]
    assert row(result, 1) == row(before, 1)
    assert row(result, 2) == row(before, 2)
    assert row(result, 6) == row(before, 6)
    assert "p-18" not in players(row(result, 3))
    assert row(result, 4)["generationPending"]
    assert row(result, 4)["matches"] == row(result, 4)["byeParticipantIds"] == []
    assert event == before


def test_batch_departures_change_roster_once_and_replan_once(monkeypatch):
    event = session(count=20)
    generated = []
    original = engine._generate_round_robin_round

    def track(value, number, history):
        generated.append(number)
        return original(value, number, history)

    monkeypatch.setattr(engine, "_generate_round_robin_round", track)
    removed = ["p-18", "p-19", "p-20"]
    result = engine.mutate_generator_roster(event, action="remove", participant_ids=removed)
    assert generated == [2]
    assert len(result["rosterRevisions"]) == 1
    assert result["rosterRevisions"][0]["participantIds"] == removed
    assert row(result, 1) == row(event, 1)
    assert set(players(row(result, 2))).isdisjoint(removed)
    assert set(engine.active_participant_ids(result, 6)).isdisjoint(removed)
    for invalid in (["p-1", "missing"], ["p-1", "p-1"]):
        with pytest.raises(ValueError):
            engine.mutate_generator_roster(event, action="remove", participant_ids=invalid)
    assert all(p["inactive_from_round"] is None for p in event["participants"])


def test_select_distant_round_generates_only_requested_window_with_actual_history(monkeypatch):
    event = session(count=7, total=12, play_format="singles")
    event = engine.mark_generator_round_played(event, round_number=1)
    event = engine.advance_generator_event(event)
    event = engine.mutate_generator_roster(event, action="remove", participant_id="p-7")
    saved = deepcopy(row(event, 1))
    captures = []
    original = engine._generate_round_robin_round

    def capture(value, number, history):
        captures.append((number, deepcopy(history)))
        return original(value, number, history)

    monkeypatch.setattr(engine, "_generate_round_robin_round", capture)
    result = engine.select_generator_round(event, round_number=10)
    assert [number for number, _ in captures] == [10, 11]
    assert captures[0][1] == engine.history_before_round(event, 2)
    assert result["currentRoundNumber"] == 10
    assert row(result, 10)["status"] == "active"
    assert row(result, 10)["matches"] and row(result, 11)["matches"]
    assert row(result, 1) == saved
    assert row(result, 9)["generationPending"]
    assert row(result, 12)["generationPending"]


def test_incremental_play_continues_past_original_preview_without_duplicate_players_or_extra_byes():
    event = session(count=6, total=3, play_format="singles")
    event = engine.mutate_generator_roster(event, action="remove", participant_id="p-6")
    byes = Counter()
    previous_byes = set()
    # Each five-player cycle must give everyone exactly one bye.
    for number in range(1, 13):
        current = row(event, number)
        assert current["status"] == "active" and not current.get("generationPending")
        assigned = players(current)
        active = set(engine.active_participant_ids(event, number))
        assert len(assigned) == len(set(assigned))
        assert set(assigned) | set(current["byeParticipantIds"]) == active
        if number > 1:
            assert not previous_byes & set(current["byeParticipantIds"])
            byes.update(current["byeParticipantIds"])
            previous_byes = set(current["byeParticipantIds"])
            values = [byes[pid] for pid in active]
            assert max(values) - min(values) <= 1
        event = engine.mark_generator_round_played(event, round_number=number)
        event = engine.advance_generator_event(event)
        assert row(event, number + 2)["matches"]
    assert event["currentRoundNumber"] == 13
    assert event["totalRounds"] >= 14


def test_deferred_round_cannot_be_silently_saved_or_marked_played_and_can_be_reopened():
    event = engine.mutate_generator_roster(session(count=6, play_format="singles"),
        action="remove", participant_id="p-6")
    with pytest.raises(ValueError, match="generate its matchups"):
        engine.mark_generator_round_played(event, round_number=5)
    scored = deepcopy(event)
    scored["scoringMode"] = "scored"
    with pytest.raises(ValueError, match="generate its matchups"):
        engine.save_generator_round(scored, round_number=5, scores=[])
    skipped = engine.skip_generator_round(event, round_number=5)
    reopened = engine.reopen_generator_round(skipped, round_number=5)
    assert row(reopened, 5)["matches"]
    assert not row(reopened, 5).get("generationPending")
    assert row(reopened, 1) == row(event, 1)


@pytest.mark.parametrize("count, rounds", [(4, 3), (5, 5), (8, 7), (9, 9), (12, 11), (16, 15)])
def test_unchanged_perfect_rotation_retains_full_schedule(count, rounds):
    event = session(count=count, total=rounds)
    fixed = deepcopy(event["rounds"])
    event = engine.mark_generator_round_played(event, round_number=1)
    event = engine.advance_generator_event(event)
    assert len(event["rounds"]) == rounds
    assert all(not item.get("generationPending") for item in event["rounds"])
    assert [r["matches"] for r in event["rounds"][1:]] == [r["matches"] for r in fixed[1:]]


def test_backwards_selection_leaves_exactly_one_active_materialized_round():
    event = engine.mutate_generator_roster(session(count=6, play_format="singles"),
        action="remove", participant_id="p-6")
    event = engine.select_generator_round(event, round_number=5)
    event = engine.select_generator_round(event, round_number=3)
    assert [(r["number"], bool(r["matches"])) for r in event["rounds"] if r["status"] == "active"] == [(3, True)]
    assert row(event, 5)["generationPending"]
    assert row(event, 4)["matches"]


def test_prepared_next_round_is_reused_unless_actual_play_changes(monkeypatch):
    event = engine.mutate_generator_roster(session(count=6, play_format="singles", scoring="scored"),
        action="remove", participant_id="p-6")
    planned = deepcopy(row(event, 2))
    original = engine._generate_round_robin_round
    generated = []

    def track(value, number, history):
        generated.append(number)
        return original(value, number, history)

    monkeypatch.setattr(engine, "_generate_round_robin_round", track)
    scores = [{"match_id": m["id"], "score_a": 11, "score_b": 7} for m in row(event, 1)["matches"]]
    result = engine.advance_generator_event(engine.save_generator_round(event, round_number=1, scores=scores))
    assert generated == [3]
    assert row(result, 2)["matches"] == planned["matches"]
    generated.clear()
    # A game that was not played changes fairness history, so the preview is
    # safely replaced instead of carrying forward fictitious participation.
    scores[0].update(score_a=None, score_b=None)
    engine.advance_generator_event(engine.save_generator_round(event, round_number=1, scores=scores))
    assert generated == [2, 3]


@pytest.mark.parametrize("audience", ["admin", "public"])
def test_batch_departure_persists_and_deferred_round_can_be_selected_after_reload(audience):
    from test_generator_round_controls_service import GeneratorSession
    from jupr_app.services import admin_play_generator_service as admin
    from jupr_app.services import public_play_generator_service as public

    game = GeneratorSession(audience, rounds=8, player_count=10)
    before = deepcopy(game.session["event"]["rounds"][0])
    args = dict(club_id="club", session_key=game.session["session_key"], action="remove",
        participant_id=None, participant_ids=["p-8", "p-9", "p-10"], name=None,
        player_id=None, substitute_scope="rest", roster_order=None, expected_version=game.session["version"])
    if audience == "public":
        args.update(edit_token=game.edit_token, requester_hash="a" * 64, idempotency_key="batch-departures-one-save")
        mutation = public.mutate_public_play_generator_roster
    else:
        args.update(game.actor)
        mutation = admin.mutate_play_generator_roster
    result = mutation(game.db, **args)
    game.session = result["session"]
    event = game.reload()["event"]
    assert row(event, 1) == before
    assert len(event["rosterRevisions"]) == 1
    assert row(event, 7)["generationPending"]
    game.call("select", round_number=7)
    event = game.reload()["event"]
    assert row(event, 7)["matches"] and row(event, 8)["matches"]
    assert not set(players(row(event, 7))) & {"p-8", "p-9", "p-10"}
    assert event["currentRoundNumber"] == 7
