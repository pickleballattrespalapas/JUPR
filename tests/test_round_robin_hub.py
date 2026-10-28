from copy import deepcopy

import pytest

from jupr_app.domain.adaptive_play_engine import (
    create_generator_preview, start_generator_event, generator_event_standings,
    mutate_generator_roster, select_generator_round, save_generator_round,
)
from jupr_app.domain.round_robin_hub import (
    apply_hub_command, build_hub_snapshot, reconcile_hub_events, player_identity,
)

START = "2026-10-20T03:00:00+00:00"
LATER = "2026-10-20T03:10:00+00:00"


def limited_rolling_event():
    value = start_generator_event(create_generator_preview(generator_kind="round_robin", play_format="doubles",
        title="Rolling", participant_names=[f"Player {i}" for i in range(12)], court_count=1, total_rounds=3))
    return value


def test_later_round_bye_players_can_finish_before_slow_first_round_and_standings_wait():
    state, events = setup({"a": limited_rolling_event()}, courts=2)
    original = deepcopy(events["a"]["rounds"])
    ready = build_hub_snapshot(state, events, now=START)["eligible_queue"]
    assert {m["round_number"] for m in ready} == {1, 2}
    assert len(original) == 2
    first = next(m for m in ready if m["round_number"] == 1)
    later = next(m for m in ready if m["round_number"] == 2)
    state, events = command(state, events, "assign_game", court_id="c0", **{k: first[k] for k in ("session_key", "match_id")})
    state, events = command(state, events, "assign_game", court_id="c1", **{k: later[k] for k in ("session_key", "match_id")})
    state, events = score(state, events, "c1")
    assert events["a"]["currentRoundNumber"] == 1
    assert events["a"]["rounds"][1]["status"] == "preview"
    assert all(row["matches"] == 0 for row in generator_event_standings(events["a"]))
    assert [m["sideA"] + m["sideB"] for m in events["a"]["rounds"][1]["matches"]] == [
        m["sideA"] + m["sideB"] for m in original[1]["matches"]]
    state, events = score(state, events, "c0")
    assert events["a"]["currentRoundNumber"] == 2
    assert [r["status"] for r in events["a"]["rounds"][:2]] == ["saved", "active"]
    assert sum(row["matches"] for row in generator_event_standings(events["a"])) == 4


def test_later_matches_cannot_leapfrog_an_earlier_queued_game():
    state, events = setup()
    future = events["a"]["rounds"][1]["matches"][0]
    snap = build_hub_snapshot(state, events, now=START)
    blocked = next(m for m in snap["blocked_queue"] if m["match_id"] == future["id"])
    assert "A player has an unfinished match in an earlier round." in blocked["blockers"]
    with pytest.raises(ValueError, match="not ready"):
        command(state, events, "assign_game", session_key="a", match_id=future["id"], court_id="c0")


@pytest.mark.parametrize("action", ["remove", "substitute"])
def test_roster_edit_preserves_later_live_match_and_defers_target_departure(action):
    state, events = setup({"a": limited_rolling_event()}, courts=2)
    later = events["a"]["rounds"][1]["matches"][0]
    state, events = command(state, events, "assign_game", session_key="a", match_id=later["id"], court_id="c1")
    protected = deepcopy(events["a"]["rounds"][1]["matches"][0])
    target = protected["sideA"][0]
    before = deepcopy(events)
    events["a"] = mutate_generator_roster(events["a"], action=action, participant_id=target,
        name="Replacement" if action == "substitute" else None, substitute_scope="rest")
    state, events = reconcile_hub_events(state, before, events, now=START)
    assert events["a"]["rounds"][1]["matches"][0] == protected
    assert state["courts"][1]["current"]["match_id"] == protected["id"]
    participant = next(p for p in events["a"]["participants"] if p["id"] == target)
    assert participant["inactive_from_round"] == 3
    from jupr_app.domain.adaptive_play_engine import active_participant_ids
    assert target not in active_participant_ids(events["a"], 3)
    assert target in events["a"]["rounds"][0]["byeParticipantIds"]


def test_finish_drains_latest_started_round_and_keeps_released_start_history():
    state, events = setup({"a": limited_rolling_event()}, courts=2)
    later = events["a"]["rounds"][1]["matches"][0]
    state, events = command(state, events, "assign_game", session_key="a", match_id=later["id"], court_id="c1")
    state, events = command(state, events, "release_game", court_id="c1")
    details = build_hub_snapshot(state, events, now=START)["groups"][0]
    assert details["finish_boundary_round"] == 2 and details["finish_pending_matches"] == 4
    state, events = command(state, events, "finish_group", session_key="a")
    assert events["a"]["hubFinishAfterRound"] == 2
    for _ in range(4):
        state, events = command(state, events, "auto_fill_courts")
        occupied = next(c["id"] for c in state["courts"] if c["current"])
        state, events = score(state, events, occupied)
    assert events["a"]["status"] == "completed"
    assert [r["status"] for r in events["a"]["rounds"]] == ["saved", "saved"]
    assert sum(row["matches"] for row in generator_event_standings(events["a"])) == 16


def test_finish_untouched_group_and_adopt_unlinked_single_round_with_byes():
    unlinked = start_generator_event(create_generator_preview(generator_kind="round_robin", play_format="doubles",
        title="Five", participant_names=[f"Player {i}" for i in range(5)], total_rounds=1, court_count=0))
    state, events = setup({"a": unlinked})
    assert events["a"]["hubPreparedThroughRound"] >= 2
    assert events["a"]["roundRobinHubId"]
    state, events = command(state, events, "finish_group", session_key="a")
    assert events["a"]["status"] == "completed"
    assert events["a"]["currentRoundNumber"] == 1 and events["a"]["rounds"]


def test_detach_rejects_future_results_pending_earlier_round():
    state, events = setup({"a": limited_rolling_event()})
    later = events["a"]["rounds"][1]["matches"][0]
    state, events = command(state, events, "assign_game", session_key="a", match_id=later["id"], court_id="c0")
    state, events = score(state, events, "c0")
    with pytest.raises(ValueError, match="partially scored"):
        command(state, events, "remove_group", session_key="a")


def test_profile_identity_blocks_earlier_obligation_under_another_roster_name():
    value = limited_rolling_event()
    first_id = value["rounds"][0]["matches"][0]["sideA"][0]
    later_id = value["rounds"][1]["matches"][0]["sideA"][0]
    for participant in value["participants"]:
        if participant["id"] in {first_id, later_id}:
            participant["player_id"] = 123
    state, events = setup({"a": value})
    later = events["a"]["rounds"][1]["matches"][0]
    snapshot = build_hub_snapshot(state, events, now=START)
    entry = next(m for m in snapshot["blocked_queue"] if m["match_id"] == later["id"])
    assert "A player has an unfinished match in an earlier round." in entry["blockers"]


def test_finish_counts_future_reservation_and_promotes_it_before_completion():
    state, events = setup({"a": limited_rolling_event()}, courts=1)
    first = events["a"]["rounds"][0]["matches"][0]
    later = events["a"]["rounds"][1]["matches"][0]
    state, events = command(state, events, "assign_game", session_key="a", match_id=first["id"], court_id="c0")
    state, events = command(state, events, "reserve_game_for_court", session_key="a", match_id=later["id"], court_id="c0")
    state, events = command(state, events, "finish_group", session_key="a")
    assert events["a"]["hubFinishAfterRound"] == 2
    state, events = score(state, events, "c0")
    assert state["courts"][0]["current"]["match_id"] == later["id"]
    state, events = score(state, events, "c0")
    for _ in range(2):
        state, events = command(state, events, "auto_fill_courts")
        state, events = score(state, events, "c0")
    assert events["a"]["status"] == "completed" and state["courts"][0]["current"] is None


def test_new_plan_does_not_depend_on_which_court_finishes_first():
    value = start_generator_event(create_generator_preview(generator_kind="round_robin", play_format="doubles",
        title="Deterministic", participant_names=[f"Player {i}" for i in range(12)], total_rounds=1, court_count=0))
    state, events = setup({"a": value}, courts=3)
    state, events = command(state, events, "auto_fill_courts")
    _, a = score(deepcopy(state), deepcopy(events), "c0")
    _, b = score(deepcopy(state), deepcopy(events), "c1")
    assert a["a"]["rounds"][1] == b["a"]["rounds"][1]


def test_released_started_game_retains_lineup_and_delays_departure():
    state, events = setup({"a": limited_rolling_event()}, courts=1)
    later = events["a"]["rounds"][1]["matches"][0]
    target = later["sideA"][0]
    state, events = command(state, events, "assign_game", session_key="a", match_id=later["id"], court_id="c0")
    state, events = command(state, events, "release_game", court_id="c0")
    state, events = command(state, events, "remove_player", session_key="a", participant_id=target)
    preserved = next(m for m in events["a"]["rounds"][1]["matches"] if m["id"] == later["id"])
    assert preserved["sideA"] == later["sideA"] and preserved["sideB"] == later["sideB"]
    assert next(p for p in events["a"]["participants"] if p["id"] == target)["inactive_from_round"] == 3
    assert build_hub_snapshot(state, events, now=START)["groups"][0]["finish_boundary_round"] == 2


def event(prefix="A", count=8, scoring="scored", player_ids=None):
    result = start_generator_event(create_generator_preview(generator_kind="round_robin", play_format="doubles",
        title=prefix, participant_names=[f"{prefix} {n}" for n in range(count)], player_ids=player_ids or [],
        court_count=0, total_rounds=3, scoring_mode=scoring))
    result["roundRobinHubId"] = "hub"
    return result


def setup(events=None, courts=2):
    events = events or {"a": event()}
    state = {"courts": [{"id": f"c{n}", "label": f"Court {n}", "open": True, "current": None, "next": None,
                         "dedicated_session_key": None} for n in range(courts)], "groups": [], "players": {}}
    for key in events:
        state, events = apply_hub_command(state, events, action="add_group", payload={"session_key": key}, now=START)
    return state, events


def command(state, events, action, now=START, **payload):
    return apply_hub_command(state, events, action=action, payload=payload, now=now)


def current_matches(events, key="a"):
    e = events[key]
    return next(r for r in e["rounds"] if r["number"] == e["currentRoundNumber"])["matches"]


def score(state, events, court_id, now=LATER):
    court = next(c for c in state["courts"] if c["id"] == court_id)
    a = court["current"]
    return command(state, events, "score_and_release", now=now, court_id=court_id,
        session_key=a["session_key"], match_id=a["match_id"], score_a=11, score_b=6)


def test_partial_score_releases_court_without_counting_standings_or_advancing():
    state, events = setup(courts=1)
    first, second = current_matches(events)
    state, events = command(state, events, "assign_next_court", session_key="a", match_id=first["id"])
    state, events = score(state, events, "c0")
    assert state["courts"][0]["current"] is None
    assert events["a"]["rounds"][0]["status"] == "active"
    assert all(p["matches"] == 0 for p in generator_event_standings(events["a"]))
    assert events["a"]["currentRoundNumber"] == 1
    assert len(build_hub_snapshot(state, events, now=LATER)["eligible_queue"]) == 1
    with pytest.raises(ValueError, match="Finish the current round"):
        command(state, events, "advance_round", session_key="a")
    state, events = command(state, events, "assign_next_court", session_key="a", match_id=second["id"])
    state, events = score(state, events, "c0")
    assert events["a"]["rounds"][0]["status"] == "saved"
    assert all(p["matches"] == 1 for p in generator_event_standings(events["a"]))
    assert events["a"]["currentRoundNumber"] == 2
    assert events["a"]["hubAutoAdvancedFromRound"] == 1
    assert {m["round_number"] for m in build_hub_snapshot(state, events, now=LATER)["eligible_queue"]} == {2}
    with pytest.raises(ValueError, match="Finish the current round"):
        command(state, events, "advance_round", session_key="a")


def test_bye_wait_survives_round_and_is_first_priority():
    state, events = setup({"a": event(count=9)}, courts=2)
    bye = events["a"]["rounds"][0]["byeParticipantIds"][0]
    bye_person = next(p for p in events["a"]["participants"] if p["id"] == bye)
    state, events = command(state, events, "auto_fill_courts")
    state, events = score(state, events, "c0")
    state, events = score(state, events, "c1")
    assert events["a"]["currentRoundNumber"] == 2
    snapshot = build_hub_snapshot(state, events, now=LATER)
    top = snapshot["eligible_queue"][0]
    assert bye in [p["id"] for p in top["side_a"] + top["side_b"]]
    assert top["wait_seconds"] == 600
    assert state["players"][player_identity(bye_person, "a", state)]["wait_since"] == START


def test_pause_freezes_wait_cancels_reservation_and_keeps_existing_wait_on_resume():
    state, events = setup(courts=1)
    first, second = current_matches(events)
    state, events = command(state, events, "assign_game", session_key="a", match_id=first["id"], court_id="c0")
    state, events = command(state, events, "reserve_game_for_court", session_key="a", match_id=second["id"], court_id="c0")
    participant = next(p for p in events["a"]["participants"] if p["id"] == second["sideA"][0])
    identity = player_identity(participant, "a", state)
    state, events = command(state, events, "pause_player", now="2026-10-20T03:05:00Z", identity=identity)
    assert state["courts"][0]["next"] is None
    snap = build_hub_snapshot(state, events, now=LATER)
    assert next(p for p in snap["players"] if p["identity"] == identity)["wait_seconds"] == 300
    assert any("paused" in " ".join(m["blockers"]) for m in snap["blocked_queue"])
    state, events = command(state, events, "resume_player", now=LATER, identity=identity)
    snap = build_hub_snapshot(state, events, now="2026-10-20T03:12:00Z")
    assert next(p for p in snap["players"] if p["identity"] == identity)["wait_seconds"] == 420


def test_dedicated_courts_exclude_other_groups_but_allow_owner_to_use_shared_courts():
    state, events = setup({"a": event("A", 4), "b": event("B", 4)}, courts=2)
    config = deepcopy(state["courts"])
    config[0]["dedicated_session_key"] = "a"
    state, events = command(state, events, "configure_courts", courts=config)
    a, b = current_matches(events, "a")[0], current_matches(events, "b")[0]
    assigned, _ = command(state, events, "assign_game", session_key="a", match_id=a["id"], court_id="c1")
    assert assigned["courts"][1]["current"]["session_key"] == "a"
    with pytest.raises(ValueError, match="not available"):
        command(state, events, "assign_game", session_key="b", match_id=b["id"], court_id="c0")
    state, events = command(state, events, "auto_fill_courts")
    assert [c["current"]["session_key"] for c in state["courts"]] == ["a", "b"]


@pytest.mark.parametrize("action", ["assign_next_court", "auto_fill_courts"])
def test_dedicated_rr_overflows_onto_shared_court_when_its_two_courts_are_busy(action):
    state, events = setup({"a": event("A", 12), "b": event("B", 4)}, courts=4)
    config = deepcopy(state["courts"])
    config[0]["dedicated_session_key"] = config[1]["dedicated_session_key"] = "a"
    config[3]["dedicated_session_key"] = "b"
    state, events = command(state, events, "configure_courts", courts=config)
    first, second, third = current_matches(events)
    for index, match in enumerate((first, second)):
        state, events = command(state, events, "assign_game", session_key="a", match_id=match["id"], court_id=f"c{index}")
    ready = next(m for m in build_hub_snapshot(state, events, now=START)["eligible_queue"]
                 if m["session_key"] == "a")
    assert ready["allowed_court_ids"] == ["c0", "c1", "c2"]
    state, events = command(state, events, action, session_key="a", match_id=third["id"])
    assert state["courts"][2]["current"]["match_id"] == third["id"]
    assert state["courts"][2]["current"]["session_key"] == "a"
    assert all(c["current"]["session_key"] == "a" for c in state["courts"][:3])


@pytest.mark.parametrize("action", ["assign_next_court", "auto_fill_courts"])
def test_automatic_assignment_prefers_dedicated_court_even_when_shared_court_is_listed_first(action):
    state, events = setup({"a": event("A", 4), "b": event("B", 4)}, courts=2)
    config = deepcopy(state["courts"])
    config[1]["dedicated_session_key"] = "a"
    state, events = command(state, events, "configure_courts", courts=config)
    state, events = command(state, events, action, session_key="a", match_id=current_matches(events)[0]["id"])
    assert state["courts"][1]["current"]["session_key"] == "a"
    if action == "auto_fill_courts":
        assert state["courts"][0]["current"]["session_key"] == "b"
    else:
        assert state["courts"][0]["current"] is None


def test_dedicated_rr_can_reserve_shared_court_and_promote_after_other_group_finishes():
    state, events = setup({"a": event("A", 8), "b": event("B", 8)}, courts=2)
    config = deepcopy(state["courts"])
    config[0]["dedicated_session_key"] = "a"
    state, events = command(state, events, "configure_courts", courts=config)
    a1, a2 = current_matches(events, "a")
    b1, b2 = current_matches(events, "b")
    state, events = command(state, events, "assign_game", session_key="a", match_id=a1["id"], court_id="c0")
    state, events = command(state, events, "assign_game", session_key="b", match_id=b1["id"], court_id="c1")
    with pytest.raises(ValueError, match="not available"):
        command(state, events, "reserve_game_for_court", session_key="b", match_id=b2["id"], court_id="c0")
    state, events = command(state, events, "reserve_game_for_court", session_key="a", match_id=a2["id"], court_id="c1")
    assert state["courts"][1]["next"]["session_key"] == "a"
    state, events = score(state, events, "c1")
    assert state["courts"][1]["current"]["match_id"] == a2["id"]
    assert state["courts"][1]["next"] is None


def test_move_to_shared_court_preserves_lineup_and_cannot_use_another_groups_dedicated_court():
    state, events = setup({"a": event("A", 4), "b": event("B", 4)}, courts=3)
    config = deepcopy(state["courts"])
    config[0]["dedicated_session_key"] = "a"
    config[2]["dedicated_session_key"] = "b"
    state, events = command(state, events, "configure_courts", courts=config)
    match = current_matches(events)[0]
    state, events = command(state, events, "assign_game", session_key="a", match_id=match["id"], court_id="c0")
    original = deepcopy(state["courts"][0]["current"])
    with pytest.raises(ValueError, match="empty court available"):
        command(state, events, "move_game", court_id="c0", target_court_id="c2")
    state, events = command(state, events, "move_game", court_id="c0", target_court_id="c1")
    assert state["courts"][0]["current"] is None
    assert state["courts"][1]["current"] == original
    moved = current_matches(events)[0]
    assert (moved["sideA"], moved["sideB"]) == (match["sideA"], match["sideB"])


def test_closed_dedicated_court_does_not_block_owner_from_shared_courts():
    state, events = setup({"a": event("A", 4)}, courts=2)
    config = deepcopy(state["courts"])
    config[0].update(dedicated_session_key="a", open=False)
    state, events = command(state, events, "configure_courts", courts=config)
    assert build_hub_snapshot(state, events, now=START)["eligible_queue"][0]["allowed_court_ids"] == ["c1"]
    state, events = command(state, events, "auto_fill_courts")
    assert state["courts"][0]["current"] is None
    assert state["courts"][1]["current"]["session_key"] == "a"


def test_shared_overflow_still_respects_longest_wait_across_groups():
    state, events = setup({"a": event("A", 12), "b": event("B", 4)}, courts=3)
    config = deepcopy(state["courts"])
    config[0]["dedicated_session_key"] = config[1]["dedicated_session_key"] = "a"
    state, events = command(state, events, "configure_courts", courts=config)
    for index, match in enumerate(current_matches(events)[:2]):
        state, events = command(state, events, "assign_game", session_key="a", match_id=match["id"], court_id=f"c{index}")
    for participant in events["b"]["participants"]:
        state["players"][player_identity(participant, "b", state)]["wait_since"] = "2026-10-20T02:55:00+00:00"
    state, events = command(state, events, "auto_fill_courts", now=LATER)
    assert state["courts"][2]["current"]["session_key"] == "b"
    assert any(m["session_key"] == "a" for m in build_hub_snapshot(state, events, now=LATER)["eligible_queue"])


def test_player_profile_cannot_be_assigned_across_two_groups():
    state, events = setup({"a": event("A", 4, player_ids=[1, 2, 3, 4]), "b": event("B", 4, player_ids=[1, 5, 6, 7])})
    state, events = command(state, events, "assign_next_court", session_key="a", match_id=current_matches(events)[0]["id"])
    assert not build_hub_snapshot(state, events, now=LATER)["eligible_queue"]
    with pytest.raises(ValueError, match="not ready"):
        command(state, events, "assign_next_court", session_key="b", match_id=current_matches(events, "b")[0]["id"])


def test_reserved_game_promotes_atomically_after_score_and_labels_follow_assignment():
    state, events = setup(courts=1)
    first, second = current_matches(events)
    state, events = command(state, events, "assign_game", session_key="a", match_id=first["id"], court_id="c0")
    state, events = command(state, events, "reserve_game_for_court", session_key="a", match_id=second["id"], court_id="c0")
    assert current_matches(events)[1]["reservedCourtLabel"] == "Court 0"
    state, events = score(state, events, "c0")
    assert state["courts"][0]["current"]["match_id"] == second["id"]
    assert state["courts"][0]["next"] is None
    assert current_matches(events)[0]["courtLabel"] == "Court 0"
    assert current_matches(events)[1]["courtLabel"] == "Court 0"


def test_unscored_match_completion_advances_automatically_only_after_whole_round():
    state, events = setup({"a": event(scoring="unscored")})
    state, events = command(state, events, "auto_fill_courts")
    for index in range(2):
        assignment = state["courts"][index]["current"]
        state, events = command(state, events, "mark_played", court_id=f"c{index}", **assignment)
        assert events["a"]["currentRoundNumber"] == (1 if index == 0 else 2)
        assert events["a"]["rounds"][0]["status"] == ("active" if index == 0 else "played")


def test_old_generator_score_page_reconciles_court_release_and_wait_times():
    state, events = setup()
    state, events = command(state, events, "auto_fill_courts")
    after = deepcopy(events)
    after["a"] = save_generator_round(events["a"], round_number=1, scores=[
        {"match_id": m["id"], "score_a": 11, "score_b": 5} for m in current_matches(events)])
    state, after = reconcile_hub_events(state, events, after, now=LATER)
    assert all(c["current"] is None for c in state["courts"])
    assert all(p["wait_since"] == LATER for p in state["players"].values())


def test_roster_remove_rebuilds_only_waiting_games_preserves_assigned_and_past_results():
    state, events = setup(courts=1)
    first, second = deepcopy(current_matches(events))
    state, events = command(state, events, "assign_game", session_key="a", match_id=first["id"], court_id="c0")
    before = deepcopy(events)
    events["a"] = mutate_generator_roster(events["a"], action="remove", participant_id=second["sideA"][0])
    state, events = reconcile_hub_events(state, before, events, now=LATER)
    assert len(current_matches(events)) == 1
    assert current_matches(events)[0]["id"] == first["id"]
    assert current_matches(events)[0]["sideA"] == first["sideA"]
    assert len(events["a"]["rounds"][0]["byeParticipantIds"]) == 3
    state, events = score(state, events, "c0")
    assert events["a"]["rounds"][0]["status"] == "saved"


def test_add_four_arrivals_to_active_round_creates_queued_game_without_changing_assigned():
    state, events = setup({"a": event(count=4)}, courts=1)
    state, events = command(state, events, "auto_fill_courts")
    first = deepcopy(current_matches(events)[0])
    for n in range(4):
        before = deepcopy(events)
        events["a"] = mutate_generator_roster(events["a"], action="add", name=f"Arrival {n}")
        state, events = reconcile_hub_events(state, before, events, now=LATER)
    assert len(current_matches(events)) == 2
    assert events["a"]["hubPreparedThroughRound"] >= 2
    assert current_matches(events)[0] == first
    assert len(build_hub_snapshot(state, events, now=LATER)["eligible_queue"]) == 1


def test_pending_substitution_is_immediate_but_assigned_match_is_preserved():
    state, events = setup()
    first, second = current_matches(events)
    state, events = command(state, events, "assign_game", session_key="a", match_id=first["id"], court_id="c0")
    before = deepcopy(events)
    old = second["sideA"][0]
    events["a"] = mutate_generator_roster(events["a"], action="substitute", participant_id=old, name="Replacement")
    state, events = reconcile_hub_events(state, before, events, now=LATER)
    pending = build_hub_snapshot(state, events, now=LATER)["eligible_queue"]
    assert "Replacement" in pending[0]["label"]
    assert old not in [p["id"] for p in pending[0]["side_a"] + pending[0]["side_b"]]


def test_out_of_order_selection_and_scoring_are_rejected_only_for_attached_sessions():
    state, events = setup()
    with pytest.raises(ValueError, match="rounds in order"):
        select_generator_round(events["a"], round_number=2)
    with pytest.raises(ValueError, match="current round"):
        save_generator_round(events["a"], round_number=2, scores=[])
    standalone = deepcopy(events["a"])
    standalone.pop("roundRobinHubId")
    assert select_generator_round(standalone, round_number=2)["currentRoundNumber"] == 2


def test_mutations_are_pure_and_reject_closing_occupied_court():
    state, events = setup(courts=1)
    before = deepcopy((state, events))
    changed, after = command(state, events, "auto_fill_courts")
    assert (state, events) == before
    config = deepcopy(changed["courts"])
    config[0]["open"] = False
    with pytest.raises(ValueError, match="not available"):
        command(changed, after, "configure_courts", courts=config)


def test_mixed_guest_and_linked_same_name_requires_review_without_changing_official_links():
    a = event("A", 4, player_ids=[1, 2, 3, 4])
    b = event("B", 4)
    b["participants"][0]["name"] = "A 0"
    state, events = setup({"a": a, "b": b})
    snap = build_hub_snapshot(state, events, now=LATER)
    assert snap["identity_pending_count"] == 1
    assert not snap["eligible_queue"]
    refs = [ref("a", a["participants"][0]), ref("b", b["participants"][0])]
    original = deepcopy(events)
    state, events = command(state, events, "resolve_identities", resolution="same", participants=refs)
    assert events == original
    assert state["participant_identities"]["b"][b["participants"][0]["id"]] == "player:1"
    assert events["b"]["participants"][0].get("player_id") is None
    state, events = command(state, events, "auto_fill_courts")
    assert sum(bool(c["current"]) for c in state["courts"]) == 1
    with pytest.raises(ValueError, match="shared guest identities"):
        command(state, events, "remove_group", session_key="a")


def test_moneyball_partial_scores_preserve_frozen_expectations_and_full_roster_rules():
    from jupr_app.domain.moneyball_expectations import freeze_moneyball
    preview = create_generator_preview(generator_kind="round_robin", play_format="doubles", title="Moneyball",
        participant_names=[f"Money {n}" for n in range(8)], player_ids=list(range(1, 9)),
        total_rounds=7, event_format="moneyball_expectations")
    frozen = freeze_moneyball(preview, {p["id"]: 3 + index * .15 for index, p in enumerate(preview["participants"])})
    money = start_generator_event(frozen)
    money["roundRobinHubId"] = "hub"
    state, events = setup({"a": money}, courts=1)
    lock = deepcopy(events["a"]["moneyballPrivate"])
    first, second = current_matches(events)
    state, events = command(state, events, "assign_next_court", session_key="a", match_id=first["id"])
    with pytest.raises(ValueError):
        command(state, events, "score_and_release", court_id="c0", session_key="a", match_id=first["id"], score_a=10, score_b=6)
    state, events = score(state, events, "c0")
    assert all(p["matches"] == 0 for p in generator_event_standings(events["a"]))
    with pytest.raises(ValueError, match="starting lineup is fixed"):
        mutate_generator_roster(events["a"], action="add", name="Late")
    state, events = command(state, events, "assign_next_court", session_key="a", match_id=second["id"])
    state, events = score(state, events, "c0")
    assert all(p["matches"] == 1 for p in generator_event_standings(events["a"]))
    assert events["a"]["moneyballPrivate"] == lock
    assert {m["round_number"] for m in build_hub_snapshot(state, events, now=LATER)["eligible_queue"]} == {2}
    while events["a"]["status"] == "active":
        state, events = command(state, events, "auto_fill_courts")
        state, events = score(state, events, "c0")
    assert events["a"]["currentRoundNumber"] == 7
    assert len(events["a"]["rounds"]) == 7
    assert all(r["status"] == "saved" for r in events["a"]["rounds"])
    assert events["a"]["moneyballPrivate"] == lock
    assert not build_hub_snapshot(state, events, now=LATER)["eligible_queue"]


def test_attaching_out_of_order_results_is_rejected_and_score_correction_preserves_wait():
    bad = event()
    bad["rounds"][1]["status"] = "saved"
    with pytest.raises(ValueError, match="out-of-order"):
        setup({"a": bad})
    state, events = setup({"a": event(count=4)}, courts=1)
    state, events = command(state, events, "auto_fill_courts")
    state, events = score(state, events, "c0")
    before_wait = deepcopy(state["players"])
    before = deepcopy(events)
    events["a"] = save_generator_round(events["a"], round_number=1, scores=[
        {"match_id": events["a"]["rounds"][0]["matches"][0]["id"], "score_a": 11, "score_b": 8}])
    state, events = reconcile_hub_events(state, before, events, now="2026-10-20T03:20:00Z")
    assert state["players"] == before_wait
    assert events["a"]["currentRoundNumber"] == 2
    assert events["a"]["rounds"][1] == before["a"]["rounds"][1]


def ref(key, player):
    return {"session_key": key, "participant_id": str(player["id"])}


def numbered_groups():
    a, b = event("Alpha", 4), event("Beta", 4)
    for e in (a, b):
        for index, p in enumerate(e["participants"], 1):
            p["name"] = str(index)
    return {"a": a, "b": b}


def test_numbered_guests_review_then_different_run_concurrently_and_rename_is_not_identity():
    state, events = setup(numbered_groups())
    snap = build_hub_snapshot(state, events, now=LATER)
    assert snap["identity_pending_count"] == 4
    assert len(snap["players"]) == 8
    assert not snap["eligible_queue"]
    assert {p["display_name"] for p in snap["identity_review"][0]["participants"]} == {"1 · Alpha", "1 · Beta"}
    state, events = command(state, events, "resolve_identities", resolution="all_different")
    assert build_hub_snapshot(state, events, now=LATER)["identity_pending_count"] == 0
    state, events = command(state, events, "auto_fill_courts")
    assert [c["current"]["session_key"] for c in state["courts"]] == ["a", "b"]
    before, before_players = deepcopy(events), deepcopy(state["players"])
    events["a"]["participants"][0]["name"] = "Alex One"
    state, events = reconcile_hub_events(state, before, events, now=LATER)
    assert state["players"] == before_players
    assert len(build_hub_snapshot(state, events, now=LATER)["identity_review"]) == 4


def test_pending_names_only_block_involved_people_and_distinct_profiles_need_no_confirmation():
    groups = numbered_groups()
    groups["c"] = event("Unrelated", 4)
    state, events = setup(groups, courts=3)
    state, events = command(state, events, "auto_fill_courts")
    assert [c["current"]["session_key"] for c in state["courts"] if c["current"]] == ["c"]
    a, b = event("Same", 4, player_ids=[1, 2, 3, 4]), event("Same", 4, player_ids=[5, 6, 7, 8])
    state, events = setup({"a": a, "b": b})
    snap = build_hub_snapshot(state, events, now=START)
    assert snap["identity_pending_count"] == 0
    assert len(snap["eligible_queue"]) == 2
    assert len({p["display_name"] for r in snap["identity_review"] for p in r["participants"]}) == 8
    with pytest.raises(ValueError, match="Different club profiles"):
        command(state, events, "resolve_identities", resolution="same", participants=[ref("a", a["participants"][0]), ref("b", b["participants"][0])])


def test_same_person_conservatively_merges_wait_pause_and_full_split_copies_both():
    state, events = setup(numbered_groups())
    a, b = events["a"]["participants"][0], events["b"]["participants"][0]
    aid, bid = player_identity(a, "a", state), player_identity(b, "b", state)
    state["players"][aid]["wait_since"] = "2026-10-20T03:03:00+00:00"
    state, events = command(state, events, "pause_player", now="2026-10-20T03:05:00+00:00", identity=bid)
    refs = [ref("a", a), ref("b", b)]
    state, events = command(state, events, "resolve_identities", now=LATER, resolution="same", participants=refs)
    identity = player_identity(b, "b", state)
    assert identity == aid
    person = next(p for p in build_hub_snapshot(state, events, now=LATER)["players"] if p["identity"] == identity)
    assert person["paused"] and person["wait_seconds"] == 300
    state, events = command(state, events, "resolve_identities", now=LATER, resolution="different", participants=refs)
    assert player_identity(b, "b", state) == bid
    assert state["players"][aid] == state["players"][bid]
    assert state["players"][bid]["paused"]


def test_partial_transitive_split_rejected_full_split_allowed_and_batch_preserves_same():
    groups = {k: event(k, 4) for k in ("a", "b", "c", "d")}
    for e in groups.values():
        e["participants"][0]["name"] = "Sam"
    state, events = setup(groups, courts=4)
    refs = [ref(k, events[k]["participants"][0]) for k in groups]
    state, events = command(state, events, "resolve_identities", resolution="same", participants=refs[:2])
    state, events = command(state, events, "resolve_identities", resolution="same", participants=refs[1:3])
    with pytest.raises(ValueError, match="every entry"):
        command(state, events, "resolve_identities", resolution="different", participants=[refs[0], refs[2]])
    state, events = command(state, events, "resolve_identities", resolution="all_different")
    ids = [state["participant_identities"][r["session_key"]][r["participant_id"]] for r in refs]
    assert ids[0] == ids[1] == ids[2] and ids[3] != ids[0]
    state, events = command(state, events, "resolve_identities", resolution="different", participants=refs[:3])
    assert len({state["participant_identities"][r["session_key"]][r["participant_id"]] for r in refs}) == 4


def test_same_profile_never_splits_even_in_all_different_batch_with_guest():
    groups = {"a": event("Same", 4, player_ids=[1, 2, 3, 4]), "b": event("Same", 4, player_ids=[1, 5, 6, 7]), "c": event("Same", 4)}
    state, events = setup(groups, courts=3)
    state, events = command(state, events, "resolve_identities", resolution="all_different")
    assert state["participant_identities"]["a"][events["a"]["participants"][0]["id"]] == "player:1"
    assert state["participant_identities"]["b"][events["b"]["participants"][0]["id"]] == "player:1"
    state, events = command(state, events, "auto_fill_courts")
    assert sum(bool(c["current"]) for c in state["courts"]) == 2


def test_legacy_upgrade_preserves_live_score_pause_and_different_can_resolve_while_playing():
    state, events = setup(numbered_groups())
    legacy = {key: deepcopy(state[key]) for key in ("courts", "groups")}
    legacy["players"] = {f"name:{n}": {"paused": n == 4, "wait_since": START, "pause_started_at": LATER if n == 4 else None} for n in range(1, 5)}
    legacy["courts"][0]["current"] = {"session_key": "a", "round_number": 1, "match_id": current_matches(events)[0]["id"], "assigned_at": START}
    before = deepcopy(legacy)
    snap = build_hub_snapshot(legacy, events, now=LATER)
    assert legacy == before and snap["identity_pending_count"] == 4
    state, events = command(legacy, events, "resolve_identities", resolution="all_different", now=LATER)
    assert state["courts"][0]["current"] == legacy["courts"][0]["current"]
    assert all(p["wait_since"] == START for p in state["players"].values())
    assert sum(p["paused"] for p in state["players"].values()) == 2
    merged, merged_events = command(state, events, "resolve_identities", resolution="same", participants=[ref("a", events["a"]["participants"][0]), ref("b", events["b"]["participants"][0])])
    assert merged["courts"][0]["current"] == state["courts"][0]["current"]
    assert merged_events["a"]["rounds"] == events["a"]["rounds"]
    scored, _ = score(legacy, events, "c0")
    assert scored["courts"][0]["current"] is None
    assert scored["identity_version"] == 2


def test_pending_identity_propagates_to_renamed_same_person_occurrences():
    groups = {k: event(k, 4) for k in ("a", "b", "c")}
    groups["a"]["participants"][0]["name"] = "Sam"
    groups["b"]["participants"][0]["name"] = "Other spelling"
    groups["c"]["participants"][0]["name"] = "Sam"
    state, events = setup(groups, courts=3)
    state, events = command(state, events, "resolve_identities", resolution="same", participants=[ref("a", groups["a"]["participants"][0]), ref("b", groups["b"]["participants"][0])])
    snap = build_hub_snapshot(state, events, now=START)
    assert not snap["eligible_queue"]
    assert len([m for m in snap["blocked_queue"] if m["round_number"] == 1]) == 3


def test_same_profile_different_spelling_is_visible_in_complete_correction_component():
    groups = {"a": event("A", 4, player_ids=[1, 2, 3, 4]), "b": event("B", 4, player_ids=[1, 5, 6, 7]), "c": event("C", 4)}
    for key, name in {"a": "Sam", "b": "Samuel", "c": "Sam"}.items():
        groups[key]["participants"][0]["name"] = name
    state, events = setup(groups)
    state, events = command(state, events, "resolve_identities", resolution="same", participants=[ref("a", groups["a"]["participants"][0]), ref("c", groups["c"]["participants"][0])])
    review = build_hub_snapshot(state, events, now=START)["identity_review"]
    assert len(review) == 1 and len(review[0]["participants"]) == 3
    state, events = command(state, events, "resolve_identities", resolution="different", participants=review[0]["participants"])
    assert state["participant_identities"]["a"][groups["a"]["participants"][0]["id"]] == "player:1"
    assert state["participant_identities"]["b"][groups["b"]["participants"][0]["id"]] == "player:1"
    assert state["participant_identities"]["c"][groups["c"]["participants"][0]["id"]].startswith("guest:")


def test_identity_name_review_normalization_matches_sql_latin_and_whitespace_rules():
    groups = {"a": event("A", 4), "b": event("B", 4)}
    groups["a"]["participants"][0]["name"] = " José\u00a0One "
    groups["b"]["participants"][0]["name"] = "JOSÉ\tONE"
    state, events = setup(groups)
    assert build_hub_snapshot(state, events, now=START)["identity_pending_count"] == 1


def test_automatic_rounds_continue_beyond_schedule_without_advancing_other_group():
    a = start_generator_event(create_generator_preview(generator_kind="round_robin", play_format="doubles",
        title="A", participant_names=[f"A {n}" for n in range(4)], total_rounds=1))
    a["roundRobinHubId"] = "hub"
    state, events = setup({"a": a, "b": event("B", 4)}, courts=1)
    original_b = deepcopy(events["b"])
    for expected in range(2, 5):
        match = current_matches(events)[0]
        state, events = command(state, events, "assign_next_court", session_key="a", match_id=match["id"])
        assert "hubAutoAdvancedFromRound" not in events["a"]
        state, events = score(state, events, "c0")
        assert events["a"]["currentRoundNumber"] == expected
        assert events["a"]["hubAutoAdvancedFromRound"] == expected - 1
        assert events["a"]["rounds"][-2]["status"] == "active"
        assert events["a"]["rounds"][-1]["number"] == expected + 1
        assert state["courts"][0]["current"] is None
        assert events["b"] == original_b
    assert all(p["matches"] == 3 for p in generator_event_standings(events["a"]))


def test_automatic_playoff_progression_waits_for_both_semifinals_and_finishes_final():
    from jupr_app.domain.generator_playoffs import start_generator_playoff
    a = event("Playoff", 8)
    a = save_generator_round(a, round_number=1, scores=[{"match_id": m["id"], "score_a": 11, "score_b": 7} for m in a["rounds"][0]["matches"]])
    a = start_generator_playoff(a, playoff_format="top_eight")
    state, events = setup({"a": a})
    state, events = command(state, events, "auto_fill_courts")
    state, events = score(state, events, "c0")
    assert events["a"]["currentRoundNumber"] == 2
    state, events = score(state, events, "c1")
    assert events["a"]["currentRoundNumber"] == 3
    final = current_matches(events)[0]
    semifinals = events["a"]["rounds"][1]["matches"]
    assert final["sideA"] == semifinals[0]["sideA"]
    assert final["sideB"] == semifinals[1]["sideA"]
    state, events = command(state, events, "auto_fill_courts")
    state, events = score(state, events, "c0")
    assert events["a"]["status"] == "completed"
    assert events["a"]["currentRoundNumber"] == 3
    assert len(events["a"]["rounds"]) == 3
    assert not build_hub_snapshot(state, events, now=LATER)["eligible_queue"]


def test_next_round_retains_pause_and_applies_scheduled_roster_departure():
    state, events = setup()
    state, events = command(state, events, "auto_fill_courts")
    departed = current_matches(events)[0]["sideA"][0]
    paused = next(p for p in events["a"]["participants"] if p["id"] == current_matches(events)[1]["sideA"][0])
    identity = player_identity(paused, "a", state)
    state, events = command(state, events, "pause_player", identity=identity)
    before = deepcopy(events)
    events["a"] = mutate_generator_roster(events["a"], action="remove", participant_id=departed)
    state, events = reconcile_hub_events(state, before, events, now=START)
    state, events = score(state, events, "c0")
    state, events = score(state, events, "c1")
    assert events["a"]["currentRoundNumber"] == 2
    assert departed not in events["a"]["rosterByRound"]["2"]
    assert state["players"][identity]["paused"]
    assert state["players"][identity]["pause_started_at"] == LATER
    assert departed in before["a"]["rounds"][0]["matches"][0]["sideA"]
    assert events["a"]["rounds"][0]["matches"][0]["sideA"] == before["a"]["rounds"][0]["matches"][0]["sideA"]


def test_manager_already_advanced_unscored_round_does_not_advance_twice():
    from jupr_app.domain.adaptive_play_engine import advance_generator_event, mark_generator_round_played
    state, events = setup({"a": event(scoring="unscored")})
    state, events = command(state, events, "auto_fill_courts")
    after = deepcopy(events)
    after["a"] = advance_generator_event(mark_generator_round_played(after["a"], round_number=1))
    state, after = reconcile_hub_events(state, events, after, now=LATER)
    assert after["a"]["currentRoundNumber"] == 2
    assert after["a"]["hubAutoAdvancedFromRound"] == 1
    assert after["a"]["rounds"][1]["status"] == "active"
    assert all(c["current"] is None for c in state["courts"])


def test_fill_resumes_legacy_completed_round_once_and_dispatch_clears_end_boundary():
    state, events = setup({"a": event(count=4)})
    events["a"] = save_generator_round(events["a"], round_number=1, scores=[{"match_id": current_matches(events)[0]["id"], "score_a": 11, "score_b": 7}])
    before = deepcopy(events)
    assert build_hub_snapshot(state, events, now=LATER)["groups"][0]["can_advance"]
    assert events == before  # Reading the hub cannot progress it.
    state, events = command(state, events, "auto_fill_courts")
    assert events["a"]["currentRoundNumber"] == 2
    assert state["courts"][0]["current"]["round_number"] == 2
    assert "hubAutoAdvancedFromRound" not in events["a"]
    state, events = command(state, events, "auto_fill_courts")
    assert events["a"]["currentRoundNumber"] == 2


def test_hub_add_players_preserves_scored_playing_reserved_matches_and_waits():
    state, events = setup({"a": event(count=12)}, courts=1)
    first, second, third = current_matches(events)
    state, events = command(state, events, "assign_game", session_key="a", match_id=first["id"], court_id="c0")
    state, events = score(state, events, "c0")
    state, events = command(state, events, "assign_game", session_key="a", match_id=second["id"], court_id="c0")
    state, events = command(state, events, "reserve_game_for_court", session_key="a", match_id=third["id"], court_id="c0")
    protected = deepcopy(current_matches(events))
    old_waits, assignments = deepcopy(state["players"]), deepcopy(state["courts"])
    for number in range(4):
        state, events = command(state, events, "add_player", now=LATER, session_key="a", name=f"Arrival {number}")
        assert current_matches(events)[:3] == protected
        assert state["courts"] == assignments
        assert {key: state["players"][key] for key in old_waits} == old_waits
        if number < 3:
            assert len(events["a"]["rounds"][0]["byeParticipantIds"]) == number + 1
    assert len(current_matches(events)) == 4
    queued = build_hub_snapshot(state, events, now=LATER)["eligible_queue"]
    assert len(queued) == 1
    assert {p["name"] for p in queued[0]["side_a"] + queued[0]["side_b"]} == {f"Arrival {n}" for n in range(4)}
    assert events["a"]["currentRoundNumber"] == 1
    assert all(p["matches"] == 0 for p in generator_event_standings(events["a"]))


def test_group_rosters_preserve_local_profile_names_and_share_pause_wait():
    state, events = setup({"a": event("A", 4, player_ids=[1, 2, 3, 4]), "b": event("B", 4, player_ids=[1, 5, 6, 7])})
    state, events = command(state, events, "pause_player", now=LATER, identity="player:1")
    snapshot = build_hub_snapshot(state, events, now=LATER)
    rosters = {g["session_key"]: g["players"] for g in snapshot["groups"]}
    assert len(snapshot["players"]) == 7
    shared_a, shared_b = rosters["a"][0], rosters["b"][0]
    assert shared_a["name"] == "A 0" and shared_b["name"] == "B 0"
    assert shared_a["identity"] == shared_b["identity"] == "player:1"
    assert shared_a["paused"] and shared_b["paused"]
    assert shared_a["wait_seconds"] == shared_b["wait_seconds"] == 600
    assert shared_a["active"] and shared_b["active"]
    assert all(g["can_add_player"] and g["add_player_unavailable_reason"] is None for g in snapshot["groups"])


def test_hub_arrival_discards_unlocked_distant_previews_without_full_schedule_rebuild(monkeypatch):
    from jupr_app.domain import adaptive_play_engine as engine

    value = event(count=8)
    # Legacy distant previews are replaceable, while current games remain
    # stable. Only the single full-group successor needs to be repaired.
    template = deepcopy(value["rounds"][-1])
    for number in range(4, 21):
        row = deepcopy(template)
        row["number"] = number
        for index, match in enumerate(row["matches"], 1):
            match.update(id=f"r{number}-c{index}", round=number)
        value["rounds"].append(row)
    value.update(totalRounds=20, hubRollingEnabled=True)
    original = deepcopy(value)

    def unexpected_full_replan(*args, **kwargs):
        pytest.fail("Hub arrivals must not replan every existing preview round")

    monkeypatch.setattr(engine, "_regenerate_round_robin_from", unexpected_full_replan)
    result = engine.mutate_generator_roster(value, action="add", name="Late arrival")

    assert value == original
    assert len(result["participants"]) == 9
    newcomer = result["participants"][-1]
    assert newcomer["name"] == "Late arrival" and newcomer["active_from_round"] == 1
    assert len(result["rounds"]) == 2
    assert result["rounds"][0]["matches"] == original["rounds"][0]["matches"]
    assert newcomer["id"] in result["rounds"][0]["byeParticipantIds"]
    assert newcomer["id"] not in result["rounds"][1]["byeParticipantIds"]
    for after in result["rounds"]:
        assert newcomer["id"] in result["rosterByRound"][str(after["number"])]


def test_adding_shared_profile_reuses_pause_wait_and_departed_profile_can_rejoin():
    state, events = setup({"a": event("A", 4, player_ids=[1, 2, 3, 4]), "b": event("B", 4, player_ids=[5, 6, 7, 8])})
    state, events = command(state, events, "pause_player", now=LATER, identity="player:1")
    previous = deepcopy(state["players"]["player:1"])
    state, events = command(state, events, "add_player", now="2026-10-20T03:20:00Z", session_key="b", name="A 0", player_id=1)
    assert state["players"]["player:1"] == previous
    added = events["b"]["participants"][-1]
    assert player_identity(added, "b", state) == "player:1"
    before = deepcopy(events)
    events["b"] = mutate_generator_roster(events["b"], action="remove", participant_id=added["id"])
    state, events = reconcile_hub_events(state, before, events, now=LATER)
    entry = next(g for g in build_hub_snapshot(state, events, now=LATER)["groups"] if g["session_key"] == "b")["players"][-1]
    assert not entry["active"] and entry["inactive_from_round"] == 1
    assert entry["can_rejoin"]
    assert all(added["id"] not in ids for number, ids in events["b"].get("rosterByRound", {}).items() if int(number) >= 1)
    state, events = command(state, events, "add_player", now=LATER, session_key="b", name="A 0", player_id=1)
    assert events["b"]["participants"][-1]["id"] != added["id"]
    assert state["players"]["player:1"] == previous
    old_entry = next(p for p in events["b"]["participants"] if p["id"] == added["id"])
    assert old_entry["inactive_from_round"] == 1
    with pytest.raises(ValueError, match="already in"):
        command(state, events, "add_player", session_key="b", name="Different label", player_id=1)


@pytest.mark.parametrize("change, message", [
    ({"status": "completed"}, "active round robin"),
    ({"eventFormat": "moneyball_expectations"}, "starting lineup"),
    ({"playoff": {"format": "top_eight"}}, "Playoff teams"),
])
def test_hub_add_player_respects_existing_generator_restrictions(change, message):
    state, events = setup({"a": event(count=4)})
    events["a"].update(change)
    group = build_hub_snapshot(state, events, now=START)["groups"][0]
    assert not group["can_add_player"]
    assert message in group["add_player_unavailable_reason"]
    with pytest.raises(ValueError, match=message):
        command(state, events, "add_player", session_key="a", name="Arrival")


def test_temporarily_absent_profile_cannot_rejoin_as_duplicate_roster_entry():
    state, events = setup({"a": event("A", 4, player_ids=[1, 2, 3, 4])})
    person = events["a"]["participants"][0]
    person["inactive_rounds"] = [1]
    events["a"].setdefault("rosterByRound", {})["1"] = [p["id"] for p in events["a"]["participants"][1:]]
    with pytest.raises(ValueError, match="active or upcoming roster"):
        command(state, events, "add_player", session_key="a", name="A 0", player_id=1)


def test_rejoin_guard_includes_ungenerated_future_roster_overrides():
    state, events = setup({"a": event("A", 4, player_ids=[1, 2, 3, 4])})
    person = events["a"]["participants"][0]
    before = deepcopy(events)
    events["a"] = mutate_generator_roster(events["a"], action="remove", participant_id=person["id"])
    state, events = reconcile_hub_events(state, before, events, now=START)
    events["a"].setdefault("rosterByRound", {})["9"] = [p["id"] for p in events["a"]["participants"]]
    member = build_hub_snapshot(state, events, now=START)["groups"][0]["players"][0]
    assert not member["can_rejoin"]
    with pytest.raises(ValueError, match="active or upcoming roster"):
        command(state, events, "add_player", session_key="a", name="A 0", player_id=1)


def test_departed_profile_can_rejoin_while_playing_in_another_group_but_cannot_double_book():
    state, events = setup({"a": event("A", 4, player_ids=[1, 2, 3, 4]), "b": event("B", 4, player_ids=[1, 5, 6, 7])})
    before = deepcopy(events)
    events["b"] = mutate_generator_roster(events["b"], action="remove", participant_id=events["b"]["participants"][0]["id"])
    state, events = reconcile_hub_events(state, before, events, now=START)
    state, events = command(state, events, "assign_game", session_key="a", match_id=current_matches(events)[0]["id"], court_id="c0")
    member = build_hub_snapshot(state, events, now=START)["groups"][1]["players"][0]
    assert member["can_rejoin"] and not member["on_court"] and member["playing_elsewhere"]
    state, events = command(state, events, "add_player", session_key="b", name="B 0", player_id=1)
    snapshot = build_hub_snapshot(state, events, now=START)
    assert not any(m["session_key"] == "b" for m in snapshot["eligible_queue"])
    assert any(m["session_key"] == "b" and "A player is already assigned to a court." in m["blockers"] for m in snapshot["blocked_queue"])


def test_late_arrival_reaches_next_round_after_prior_edits_trimmed_future_rounds():
    from jupr_app.domain.adaptive_play_engine import active_participant_ids
    state, events = setup(courts=1)
    state, events = command(state, events, "add_player", session_key="a", name="Temporary")
    temporary = events["a"]["participants"][-1]["id"]
    state, events = command(state, events, "remove_player", session_key="a", participant_id=temporary)
    assert "2" in events["a"]["rosterByRound"]
    first, second = current_matches(events)
    state, events = command(state, events, "assign_game", session_key="a", match_id=first["id"], court_id="c0")
    state, events = command(state, events, "reserve_game_for_court", session_key="a", match_id=second["id"], court_id="c0")
    protected = deepcopy(current_matches(events))
    state, events = command(state, events, "add_player", session_key="a", name="9")
    arrival = events["a"]["participants"][-1]["id"]
    assert current_matches(events) == protected
    assert build_hub_snapshot(state, events, now=START)["groups"][0]["players"][-1]["on_bye"]
    state, events = score(state, events, "c0")
    state, events = score(state, events, "c0")
    assert arrival in active_participant_ids(events["a"], 2)
    assert events["a"]["hubPreparedThroughRound"] >= 2 and not events["a"]["perfectRotation"]
    assert all(arrival in active_participant_ids(events["a"], r["number"]) for r in events["a"]["rounds"] if r["number"] >= 2)
    assert len(current_matches(events)) == 2
    # The arrival sat out the current round and must play in the full-group successor.
    assert arrival not in events["a"]["rounds"][1]["byeParticipantIds"]
    assert temporary not in active_participant_ids(events["a"], 2)
    assert len(events["a"]["rounds"][-1]["byeParticipantIds"]) == 1


@pytest.mark.parametrize("scope", ["round", "rest"])
def test_future_override_substitution_respects_scope_and_preserves_dispatched_round(scope):
    from jupr_app.domain.adaptive_play_engine import active_participant_ids
    state, events = setup()
    state, events = command(state, events, "add_player", session_key="a", name="Temporary")
    state, events = command(state, events, "remove_player", session_key="a", participant_id=events["a"]["participants"][-1]["id"])
    state, events = command(state, events, "auto_fill_courts")
    original = deepcopy(current_matches(events))
    source = original[0]["sideA"][0]
    before = deepcopy(events)
    events["a"] = mutate_generator_roster(events["a"], action="substitute", participant_id=source, name="Substitute", substitute_scope=scope)
    state, events = reconcile_hub_events(state, before, events, now=START)
    replacement = events["a"]["participants"][-1]["id"]
    assert current_matches(events) == original
    state, events = score(state, events, "c0")
    state, events = score(state, events, "c1")
    assert replacement in active_participant_ids(events["a"], 2)
    assert source not in active_participant_ids(events["a"], 2)
    state, events = command(state, events, "auto_fill_courts")
    state, events = score(state, events, "c0")
    state, events = score(state, events, "c1")
    assert (replacement in active_participant_ids(events["a"], 3)) == (scope == "rest")
    assert (source in active_participant_ids(events["a"], 3)) == (scope == "round")
    assert events["a"]["rounds"][0]["matches"][0]["sideA"] == original[0]["sideA"]


def recovered_rotation_with_late_arrival():
    """Synthetic version of the reported 8-to-9-player recovered rotation."""
    result = event("Recovered", 8)
    result["participants"].append({"id": "p-new-1", "name": "Late arrival", "seed": 9,
        "roster_order": 9, "active_from_round": 3, "inactive_from_round": None, "inactive_rounds": []})
    templates = [
        [(["p-1", "p-6"], ["p-2", "p-5"]), (["p-3", "p-8"], ["p-4", "p-7"])],
        [(["p-2", "p-3"], ["p-5", "p-8"]), (["p-1", "p-4"], ["p-6", "p-7"])],
        [(["p-1", "p-8"], ["p-3", "p-6"]), (["p-2", "p-7"], ["p-4", "p-5"])],
    ]
    rows = []
    for number in range(1, 14):
        matches = [{"id": f"old-r{number}-c{index}", "round": number, "court": index,
            "sideA": a, "sideB": b, "playFormat": "doubles", "scoreA": 11 if number <= 3 else None,
            "scoreB": 5 if number <= 3 else None, "status": "scored" if number <= 3 else "scheduled"}
            for index, (a, b) in enumerate(templates[(number - 1) % 3], 1)]
        rows.append({"number": number, "status": "saved" if number <= 3 else "active" if number == 4 else "preview",
            "matches": matches, "byeParticipantIds": ["p-new-1"] if number >= 3 else [], "warnings": []})
    result.update(rounds=rows, currentRoundNumber=4, totalRounds=13, perfectRotation=True,
        adaptiveFromRound=None, hubAutoAdvancedFromRound=3, hubAssignedMatchIds=[],
        rosterRevisions=[{"action": "add", "effectiveRound": 4, "participantId": None, "name": "Late arrival"}],
        rosterByRound={str(n): [p["id"] for p in result["participants"] if n >= 3 or p["id"] != "p-new-1"] for n in range(1, 14)})
    return result


def test_repair_untouched_recovered_rotation_keeps_saved_history_and_seats_late_arrival():
    from jupr_app.domain.round_robin_hub import repair_unstarted_hub_round
    event_before = recovered_rotation_with_late_arrival()
    original = deepcopy(event_before)
    repaired = repair_unstarted_hub_round(event_before)
    assert event_before == original
    assert repaired["rounds"][:3] == original["rounds"][:3]
    assert repaired["participants"] == original["participants"]
    assert generator_event_standings(repaired) == generator_event_standings(original)
    assert repaired["currentRoundNumber"] == 4 and len(repaired["rounds"]) == 5
    assert repaired["rounds"][-1]["number"] == 5
    row = repaired["rounds"][-2]
    assert row["status"] == "active" and len(row["matches"]) == 2
    assert any("p-new-1" in m["sideA"] + m["sideB"] for m in row["matches"])
    assert len(row["byeParticipantIds"]) == 1 and "p-new-1" not in row["byeParticipantIds"]
    state, events = setup({"a": repaired})
    state, events = command(state, events, "auto_fill_courts")
    state, events = score(state, events, "c0")
    state, events = score(state, events, "c1")
    assert events["a"]["currentRoundNumber"] == 5 and events["a"]["hubPreparedThroughRound"] >= 5
    assert len(current_matches(events)) == 2


@pytest.mark.parametrize("change", ["assigned", "scored", "moneyball", "playoff", "no_boundary"])
def test_unstarted_round_repair_rejects_touched_or_fixed_events(change):
    from jupr_app.domain.round_robin_hub import repair_unstarted_hub_round
    candidate = recovered_rotation_with_late_arrival()
    if change == "assigned":
        candidate["hubAssignedMatchIds"] = [candidate["rounds"][3]["matches"][0]["id"]]
    elif change == "scored":
        candidate["rounds"][3]["matches"][0]["scoreA"] = 11
    elif change == "moneyball":
        candidate["eventFormat"] = "moneyball_expectations"
    elif change == "playoff":
        candidate["playoff"] = {"format": "top_eight"}
    else:
        candidate.pop("hubAutoAdvancedFromRound")
    with pytest.raises(ValueError, match="untouched"):
        repair_unstarted_hub_round(candidate)


def test_full_roster_can_remove_players_even_when_add_limit_is_reached():
    state, events = setup({"a": event(count=40)})
    group = build_hub_snapshot(state, events, now=START)["groups"][0]
    assert not group["can_add_player"] and group["can_remove_player"]
    state, events = command(state, events, "remove_player", session_key="a", participant_id=events["a"]["participants"][0]["id"])
    assert not build_hub_snapshot(state, events, now=START)["groups"][0]["players"][0]["active"]
