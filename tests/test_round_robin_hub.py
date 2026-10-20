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
    assert events["a"]["currentRoundNumber"] == 1
    state, events = command(state, events, "advance_round", session_key="a")
    assert events["a"]["currentRoundNumber"] == 2


def test_bye_wait_survives_round_and_is_first_priority():
    state, events = setup({"a": event(count=9)}, courts=2)
    bye = events["a"]["rounds"][0]["byeParticipantIds"][0]
    bye_person = next(p for p in events["a"]["participants"] if p["id"] == bye)
    state, events = command(state, events, "auto_fill_courts")
    state, events = score(state, events, "c0")
    state, events = score(state, events, "c1")
    state, events = command(state, events, "advance_round", now=LATER, session_key="a")
    snapshot = build_hub_snapshot(state, events, now=LATER)
    top = snapshot["eligible_queue"][0]
    assert bye in [p["id"] for p in top["side_a"] + top["side_b"]]
    assert top["wait_seconds"] == 600
    assert state["players"][player_identity(bye_person)]["wait_since"] == START


def test_pause_freezes_wait_cancels_reservation_and_keeps_existing_wait_on_resume():
    state, events = setup(courts=1)
    first, second = current_matches(events)
    state, events = command(state, events, "assign_game", session_key="a", match_id=first["id"], court_id="c0")
    state, events = command(state, events, "reserve_game_for_court", session_key="a", match_id=second["id"], court_id="c0")
    participant = next(p for p in events["a"]["participants"] if p["id"] == second["sideA"][0])
    identity = player_identity(participant)
    state, events = command(state, events, "pause_player", now="2026-10-20T03:05:00Z", identity=identity)
    assert state["courts"][0]["next"] is None
    snap = build_hub_snapshot(state, events, now=LATER)
    assert next(p for p in snap["players"] if p["identity"] == identity)["wait_seconds"] == 300
    assert any("paused" in " ".join(m["blockers"]) for m in snap["blocked_queue"])
    state, events = command(state, events, "resume_player", now=LATER, identity=identity)
    snap = build_hub_snapshot(state, events, now="2026-10-20T03:12:00Z")
    assert next(p for p in snap["players"] if p["identity"] == identity)["wait_seconds"] == 420


def test_dedicated_courts_are_exclusive_in_both_directions_and_autofill_respects_them():
    state, events = setup({"a": event("A", 4), "b": event("B", 4)}, courts=2)
    config = deepcopy(state["courts"])
    config[0]["dedicated_session_key"] = "a"
    state, events = command(state, events, "configure_courts", courts=config)
    a, b = current_matches(events, "a")[0], current_matches(events, "b")[0]
    with pytest.raises(ValueError, match="not available"):
        command(state, events, "assign_game", session_key="a", match_id=a["id"], court_id="c1")
    with pytest.raises(ValueError, match="not available"):
        command(state, events, "assign_game", session_key="b", match_id=b["id"], court_id="c0")
    state, events = command(state, events, "auto_fill_courts")
    assert [c["current"]["session_key"] for c in state["courts"]] == ["a", "b"]


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


def test_unscored_match_completion_requires_whole_round_then_explicit_advance():
    state, events = setup({"a": event(scoring="unscored")})
    state, events = command(state, events, "auto_fill_courts")
    for index in range(2):
        assignment = state["courts"][index]["current"]
        state, events = command(state, events, "mark_played", court_id=f"c{index}", **assignment)
        assert events["a"]["currentRoundNumber"] == 1
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
    assert len(events["a"]["rounds"]) == 1  # adaptive next round waits for completion
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


def test_mixed_guest_and_linked_same_name_is_rejected():
    a = event("A", 4, player_ids=[1, 2, 3, 4])
    b = event("B", 4)
    b["participants"][0]["name"] = "A 0"
    with pytest.raises(ValueError, match="both with and without"):
        setup({"a": a, "b": b})


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
        {"match_id": current_matches(events)[0]["id"], "score_a": 11, "score_b": 8}])
    state, events = reconcile_hub_events(state, before, events, now="2026-10-20T03:20:00Z")
    assert state["players"] == before_wait
