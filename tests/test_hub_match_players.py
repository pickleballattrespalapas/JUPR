"""An explicit lineup edit changes the promised game without losing its court."""
from copy import deepcopy

import pytest

from jupr_app.domain.adaptive_play_engine import generator_event_standings, create_generator_preview, start_generator_event
from jupr_app.domain.round_robin_hub import build_hub_snapshot
from test_round_robin_hub import command, setup, score, START
from test_round_robin_hub_pause_substitution import planned, people, round_row, identity, pause


def edit(state, events, match, target, replacement=None, court="c0", **extra):
    return command(state, events, "edit_match_players", session_key="a", match_id=match["id"],
        round_number=match.get("round", 1), court_id=court, participant_id=target,
        **({"replacement_participant_id": replacement} if replacement else {}), **extra)


def assigned(count=5):
    state, events = setup({"a": planned(count)}, courts=2)
    match = round_row(events["a"])["matches"][0]
    state, events = command(state, events, "assign_game", session_key="a", match_id=match["id"], court_id="c0")
    return state, events, deepcopy(match)


def test_bye_replaces_live_player_and_score_belongs_to_actual_lineup():
    state, events, match = assigned()
    target, donor = people(match)[0], round_row(events["a"])["byeParticipantIds"][0]
    court = deepcopy(state["courts"][0])
    future = deepcopy(events["a"]["rounds"][1:])
    state, events = edit(state, events, match, target, donor)
    changed = round_row(events["a"])["matches"][0]
    assert people(changed) == [donor, *people(match)[1:]]
    assert state["courts"][0] == court
    assert round_row(events["a"])["byeParticipantIds"] == [target]
    assert events["a"]["rounds"][1:] == future
    assert all(p["matches"] == 0 for p in generator_event_standings(events["a"]))
    state, events = score(state, events, "c0")
    stats = {p["participantId"]: p for p in generator_event_standings(events["a"])}
    assert stats[target]["matches"] == 0 and stats[donor]["matches"] == 1
    assert round_row(events["a"])["matches"][0]["sideA"][0] == donor


def test_swap_two_positions_within_live_game():
    state, events, match = assigned()
    a, b = match["sideA"][0], match["sideB"][0]
    state, events = edit(state, events, match, a, b)
    game = round_row(events["a"])["matches"][0]
    assert game["sideA"] == [b, match["sideA"][1]]
    assert game["sideB"] == [a, match["sideB"][1]]


def test_swap_with_waiting_game_preserves_every_other_slot():
    state, events, match = assigned(8)
    other = deepcopy(round_row(events["a"])["matches"][1])
    target, donor = people(match)[0], people(other)[2]
    state, events = edit(state, events, match, target, donor)
    first, second = round_row(events["a"])["matches"]
    assert people(first) == [donor, *people(match)[1:]]
    assert people(second) == [*people(other)[:2], target, people(other)[3]]
    assert not round_row(events["a"])["byeParticipantIds"]


@pytest.mark.parametrize("format,mode", [("doubles", "scored"), ("singles", "scored"), ("doubles", "unscored")])
def test_new_guest_joins_current_match_and_future_rotation(format, mode):
    count = 2 if format == "singles" else 4
    value = start_generator_event(create_generator_preview(generator_kind="round_robin", play_format=format, title="Substitution",
        participant_names=[f"Original {i}" for i in range(count)], total_rounds=3, scoring_mode=mode))
    state, events = setup({"a": value})
    match = deepcopy(round_row(events["a"])["matches"][0])
    state, events = command(state, events, "assign_game", session_key="a", match_id=match["id"], court_id="c0")
    state, events = edit(state, events, match, people(match)[0], name=" New Guest ")
    added = events["a"]["participants"][-1]
    assert added["name"] == "New Guest" and added["active_from_round"] == 1
    assert added["id"] in people(round_row(events["a"])["matches"][0])
    assert people(match)[0] in round_row(events["a"])["byeParticipantIds"]
    assert len(people(round_row(events["a"])["matches"][0])) == count
    # A guest who just played may fairly sit out the next round. Planning one
    # round ahead must still put them back into play within the following round.
    if mode == "unscored":
        state, events = command(state, events, "mark_played", court_id="c0", **state["courts"][0]["current"])
    else:
        state, events = score(state, events, "c0")
    assert any(added["id"] in people(m) for r in events["a"]["rounds"]
        if r["number"] in {2, 3} for m in r["matches"])


@pytest.mark.parametrize("status", ["on_court", "reserved", "played", "paused"])
def test_cannot_take_player_from_another_committed_game(status):
    state, events, match = assigned(8)
    other = round_row(events["a"])["matches"][1]
    donor = people(other)[0]
    if status in {"on_court", "played"}:
        state, events = command(state, events, "assign_game", session_key="a", match_id=other["id"], court_id="c1")
        if status == "played":
            state, events = score(state, events, "c1")
    elif status == "reserved":
        state, events = command(state, events, "reserve_game_for_court", session_key="a", match_id=other["id"], court_id="c0")
    else:
        state, events = pause(state, events, donor)
    before = deepcopy((state, events))
    with pytest.raises(ValueError, match="another|Resume"):
        edit(state, events, match, people(match)[0], donor)
    assert (state, events) == before


def test_queue_edit_and_stale_court_target_are_distinct():
    state, events, match = assigned()
    donor = round_row(events["a"])["byeParticipantIds"][0]
    with pytest.raises(ValueError, match="court assignment"):
        edit(state, events, match, people(match)[0], donor, court=None)
    state, events = command(state, events, "release_game", court_id="c0")
    with pytest.raises(ValueError, match="assignment changed"):
        edit(state, events, match, people(match)[0], donor)
    state, events = edit(state, events, match, people(match)[0], donor, court=None)
    ready = build_hub_snapshot(state, events, now=START)["eligible_queue"]
    assert any(m["match_id"] == match["id"] and donor in [p["id"] for p in m["side_a"] + m["side_b"]] for m in ready)


def test_new_guest_preserves_other_live_game_and_previously_saved_round():
    state, events, match = assigned(8)
    state, events = command(state, events, "auto_fill_courts")
    state, events = score(state, events, "c0")
    state, events = score(state, events, "c1")
    previous = deepcopy(round_row(events["a"]))
    state, events = command(state, events, "auto_fill_courts")
    assignment = state["courts"][0]["current"]
    row = round_row(events["a"], assignment["round_number"])
    match = next(m for m in row["matches"] if m["id"] == assignment["match_id"])
    other_assignment = deepcopy(state["courts"][1])
    other = deepcopy(next(m for m in row["matches"] if m["id"] == other_assignment["current"]["match_id"]))
    state, events = edit(state, events, match, people(match)[0], name="Arriving guest")
    assert round_row(events["a"]) == previous
    assert state["courts"][1] == other_assignment
    assert other in round_row(events["a"], assignment["round_number"])["matches"]


def service_fixture():
    from test_admin_round_robin_hub_service import TransactionalHubDB, session, create, command as run
    db = TransactionalHubDB()
    session(db, count=5)
    hub = run(db, create(db), "add_group", {"session_key": "rr-a"})
    match = hub["workspace"]["eligible_queue"][0]
    hub = run(db, hub, "assign_game", {"session_key": "rr-a", "match_id": match["match_id"], "court_id": "c1"})
    return db, hub, {"session_key": "rr-a", "match_id": match["match_id"], "round_number": match["round_number"],
        "court_id": "c1", "participant_id": match["side_a"][0]["id"], "name": "Guest"}


def test_joint_commit_replay_and_profile_scoping():
    from test_admin_round_robin_hub_service import command as run, raw_session
    db, hub, payload = service_fixture()
    db.storage["players"] = [{"id": 20, "club_id": "club-a", "name": "Club Player"},
        {"id": 21, "club_id": "club-b", "name": "Other Club"}]
    before = deepcopy(db.storage)
    with pytest.raises(ValueError, match="not found in this club"):
        run(db, hub, "edit_match_players", {**payload, "player_id": 21})
    assert db.storage == before
    updated = run(db, hub, "edit_match_players", {**payload, "player_id": 20}, "edit-players-once")
    persisted = deepcopy(db.storage)
    replay = run(db, hub, "edit_match_players", {**payload, "player_id": 20}, "edit-players-once")
    assert replay["replayed"] and db.storage == persisted
    assert updated["workspace"]["courts"][0]["current"]["side_a"][0]["name"] == "Club Player"
    assert raw_session(db)["state"]["page_state"]["event"]["participants"][-1]["player_id"] == 20


def test_stale_request_and_session_race_cannot_partially_edit_roster_or_court():
    from test_admin_round_robin_hub_service import command as run, raw_session
    from jupr_app.services.public_live_operation_service import PublicLiveConflictError
    from jupr_app.services import admin_round_robin_hub_service as service
    db, hub, payload = service_fixture()
    newer = run(db, hub, "rename_hub", {"title": "Changed elsewhere"})
    before = deepcopy(db.storage)
    with pytest.raises(PublicLiveConflictError):
        run(db, hub, "edit_match_players", payload)
    assert db.storage == before
    original = deepcopy(raw_session(db))
    db.before_commit = lambda current: raw_session(current).update(version=99)
    with pytest.raises(PublicLiveConflictError):
        run(db, newer, "edit_match_players", payload)
    assert raw_session(db)["state"] == original["state"]
    assert db.storage[service.HUB_TABLE] == before[service.HUB_TABLE]


def test_locked_result_or_fixed_format_rejects_edit():
    from test_admin_round_robin_hub_service import command as run, raw_session
    db, hub, payload = service_fixture()
    raw_session(db)["state"]["generator_submission"] = {"status": "pending"}
    before = deepcopy(db.storage)
    with pytest.raises(ValueError, match="locked"):
        run(db, hub, "edit_match_players", payload)
    assert db.storage == before
    for format in ("moneyball", "playoff"):
        state, events, match = assigned()
        if format == "moneyball":
            events["a"]["eventFormat"] = "moneyball_expectations"
        else:
            events["a"]["playoff"] = {"format": "top_eight"}
        with pytest.raises(ValueError, match="fixed"):
            edit(state, events, match, people(match)[0], name="Guest")


def test_later_live_match_adds_guest_from_that_round_and_cannot_steal_earlier_player():
    from test_round_robin_hub import limited_rolling_event
    state, events = setup({"a": limited_rolling_event()})
    first = deepcopy(round_row(events["a"]))
    match = deepcopy(round_row(events["a"], 2)["matches"][0])
    state, events = command(state, events, "assign_game", session_key="a", match_id=match["id"], court_id="c0")
    with pytest.raises(ValueError, match="earlier round"):
        edit(state, events, match, people(match)[0], people(first["matches"][0])[0])
    state, events = edit(state, events, match, people(match)[0], name="Later guest")
    assert round_row(events["a"]) == first
    added = events["a"]["participants"][-1]
    assert added["active_from_round"] == 2
    assert added["id"] in people(next(m for m in round_row(events["a"], 2)["matches"] if m["id"] == match["id"]))


def test_new_profile_cannot_bypass_another_groups_court_claim():
    from test_admin_round_robin_hub_service import command as run, session
    db, hub, payload = service_fixture()
    session(db, "rr-b", count=4, offset=19)
    db.storage["players"] = [{"id": 20, "club_id": "club-a", "name": "Player 20"}]
    hub = run(db, hub, "add_group", {"session_key": "rr-b"})
    hub = run(db, hub, "configure_courts", {"courts": [{"id": f"c{i}", "label": f"Court {i}", "open": True} for i in (1, 2)]})
    game = next(m for m in hub["workspace"]["eligible_queue"] if m["session_key"] == "rr-b")
    hub = run(db, hub, "assign_game", {"session_key": "rr-b", "match_id": game["match_id"], "court_id": "c2"})
    before = deepcopy(db.storage)
    with pytest.raises(ValueError, match="another court"):
        run(db, hub, "edit_match_players", {**payload, "player_id": 20})
    assert db.storage == before


def test_partial_score_cannot_be_reattributed_to_a_substitute():
    state, events, match = assigned()
    round_row(events["a"])["matches"][0]["scoreA"] = 7
    with pytest.raises(ValueError, match="Recorded games"):
        edit(state, events, match, people(match)[0], name="Guest")
