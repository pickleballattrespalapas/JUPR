"""A break must not strand available players behind an unstarted matchup.

Fixtures use deliberate lineups so these tests verify which existing promises
survive a repair, independently of the generator's matchup-selection details.
"""
from copy import deepcopy

import pytest

from jupr_app.domain.adaptive_play_engine import (
    create_generator_preview,
    generator_event_standings,
    save_generator_round,
    start_generator_event,
)
from jupr_app.domain.round_robin_hub import build_hub_snapshot, player_identity
from test_round_robin_hub import START, LATER, command, event, score, setup


def people(match):
    return list(match.get("sideA", [])) + list(match.get("sideB", []))


def round_row(value, number=1):
    return next(row for row in value["rounds"] if row["number"] == number)


def planned(count=9):
    value = event("Pause fixture", count)
    ids = [p["id"] for p in value["participants"]]
    first = round_row(value)
    first["matches"] = [
        {"id": f"fixture-r1-m{index // 4 + 1}", "round": 1,
         "court": index // 4 + 1, "playFormat": "doubles",
         "sideA": ids[index:index + 2], "sideB": ids[index + 2:index + 4],
         "scoreA": None, "scoreB": None, "status": "scheduled"}
        for index in range(0, count - count % 4, 4)
    ]
    first["byeParticipantIds"] = ids[count - count % 4:]
    return value


def identity(state, events, pid, key="a"):
    player = next(p for p in events[key]["participants"] if p["id"] == pid)
    return player_identity(player, key, state)


def pause(state, events, pid, **kwargs):
    return command(state, events, "pause_player", identity=identity(state, events, pid), **kwargs)


def ready(state, events):
    return build_hub_snapshot(state, events, now=LATER)["eligible_queue"]


def test_current_bye_can_replace_paused_slot_despite_its_future_preview():
    state, events = setup({"a": planned()})
    original = deepcopy(events["a"])
    affected, unaffected = round_row(original)["matches"]
    paused = people(affected)[0]
    donor = round_row(original)["byeParticipantIds"][0]
    assert any(donor in people(match) for row in original["rounds"][1:] for match in row["matches"])

    state, events = pause(state, events, paused)

    repaired = round_row(events["a"])
    assert unaffected in repaired["matches"]
    replacement = next(match for match in repaired["matches"] if donor in people(match))
    assert set(people(replacement)) == (set(people(affected)) - {paused}) | {donor}
    assert replacement["sideA"] == [donor, affected["sideA"][1]]
    assert replacement["sideB"] == affected["sideB"]
    assert donor not in repaired["byeParticipantIds"]
    assert paused not in repaired["byeParticipantIds"]
    assert any(entry["match_id"] == replacement["id"] for entry in ready(state, events))
    assert all(row["matches"] == 0 for row in generator_event_standings(events["a"]))
    before_read = deepcopy((state, events))
    build_hub_snapshot(state, events, now=LATER)
    assert (state, events) == before_read


@pytest.mark.parametrize("healthy_state", ["waiting", "on_court", "scored"])
def test_no_bye_carries_three_players_forward_without_changing_healthy_game_or_inventing_result(healthy_state):
    state, events = setup({"a": planned(8)}, courts=1)
    affected, healthy = deepcopy(round_row(events["a"])["matches"])
    paused = people(affected)[0]
    survivors = set(people(affected)) - {paused}
    if healthy_state != "waiting":
        state, events = command(state, events, "assign_game", session_key="a", match_id=healthy["id"], court_id="c0")
        if healthy_state == "scored":
            state, events = score(state, events, "c0")
        healthy = deepcopy(next(m for m in round_row(events["a"])["matches"] if m["id"] == healthy["id"]))

    state, events = pause(state, events, paused)

    first = round_row(events["a"])
    assert first["matches"] == [healthy]
    assert set(first["byeParticipantIds"]) == survivors
    assert paused not in first["byeParticipantIds"]
    if healthy_state == "waiting":
        state, events = command(state, events, "assign_game", session_key="a", match_id=healthy["id"], court_id="c0")
    if healthy_state != "scored":
        state, events = score(state, events, "c0")
    assert events["a"]["currentRoundNumber"] >= 2
    assert round_row(events["a"])["status"] == "saved"
    assert sum(player["matches"] for player in generator_event_standings(events["a"])) == 4
    next_ready = ready(state, events)
    assert next_ready
    assert not any(paused in [p["id"] for p in m["side_a"] + m["side_b"]] for m in next_ready)
    # Existing healthy future matchups remain intact, so all three need not
    # share the very next game. Their earned bye and original wait survive.
    assert all(state["players"][identity(state, events, pid)]["wait_since"] == START for pid in survivors)


def test_multiple_affected_games_regroup_their_available_players_only():
    state, events = setup({"a": planned(12)}, courts=3)
    affected_a, affected_b, healthy = deepcopy(round_row(events["a"])["matches"])
    paused = {people(affected_a)[0], people(affected_b)[0]}
    available = (set(people(affected_a)) | set(people(affected_b))) - paused
    for pid in sorted(paused):
        state, events = pause(state, events, pid)

    row = round_row(events["a"])
    assert healthy in row["matches"]
    rebuilt = [match for match in row["matches"] if match["id"] != healthy["id"]]
    assert len(rebuilt) == 1
    assert len(set(people(rebuilt[0]))) == 4
    assert set(people(rebuilt[0])) <= available
    assert set(row["byeParticipantIds"]) == available - set(people(rebuilt[0]))
    assert not paused & set(row["byeParticipantIds"])


def test_one_bye_preserves_one_affected_game_when_two_players_are_already_paused():
    state, events = setup({"a": planned(13)})
    first, second, healthy = deepcopy(round_row(events["a"])["matches"])
    first_paused, second_paused = people(first)[0], people(second)[0]
    donor = round_row(events["a"])["byeParticipantIds"][0]
    # Existing hubs can already contain pauses created before repair existed.
    state["players"][identity(state, events, first_paused)]["paused"] = True

    state, events = pause(state, events, second_paused)

    row = round_row(events["a"])
    replacement = next(match for match in row["matches"] if match["id"] == first["id"])
    assert replacement["sideA"] == [donor, first["sideA"][1]]
    assert replacement["sideB"] == first["sideB"]
    assert healthy in row["matches"]
    assert len(row["matches"]) == 2
    assert set(row["byeParticipantIds"]) == set(people(second)) - {second_paused}
    assert not {first_paused, second_paused} & set(row["byeParticipantIds"])


def test_underfilled_future_game_is_not_erased_by_repeated_fill_while_other_players_finish():
    value = planned(8)
    affected, healthy = deepcopy(round_row(value)["matches"])
    round_row(value)["matches"] = [healthy]
    round_row(value)["byeParticipantIds"] = people(affected)
    later = round_row(value, 2)
    later["matches"] = [
        {**affected, "id": "fixture-r2-affected", "round": 2},
        {**healthy, "id": "fixture-r2-healthy", "round": 2},
    ]
    later["byeParticipantIds"] = []
    state, events = setup({"a": value}, courts=1)
    future = deepcopy(events["a"]["rounds"][1:])
    count = len(events["a"]["rounds"])

    state, events = pause(state, events, people(affected)[0])
    for _ in range(3):
        state, events = command(state, events, "auto_fill_courts")
        assert events["a"]["rounds"][1:] == future
        assert len(events["a"]["rounds"]) == count
    assert state["courts"][0]["current"]["match_id"] == healthy["id"]


def test_current_game_scored_future_game_and_reserved_future_game_are_immutable():
    from test_round_robin_hub import limited_rolling_event

    state, events = setup({"a": limited_rolling_event()}, courts=2)
    first, second, third = [row["matches"][0] for row in events["a"]["rounds"][:3]]
    state, events = command(state, events, "assign_game", session_key="a", match_id=first["id"], court_id="c0")
    state, events = command(state, events, "assign_game", session_key="a", match_id=second["id"], court_id="c1")
    state, events = score(state, events, "c1")
    state, events = command(state, events, "reserve_game_for_court", session_key="a", match_id=third["id"], court_id="c0")
    protected = deepcopy([row["matches"] for row in events["a"]["rounds"][:3]])
    assignments = deepcopy(state["courts"])
    paused = people(first)[0]

    state, events = pause(state, events, paused)

    assert [row["matches"] for row in events["a"]["rounds"][:3]] == protected
    assert state["courts"] == assignments
    assert all(player["matches"] == 0 for player in generator_event_standings(events["a"]))
    state, events = score(state, events, "c0")
    assert state["courts"][0]["current"]["match_id"] == third["id"]
    assert sum(player["matches"] for player in generator_event_standings(events["a"])) == 8


def test_returned_started_game_keeps_its_lineup_when_a_player_pauses():
    state, events = setup({"a": planned()})
    affected = round_row(events["a"])["matches"][0]
    state, events = command(state, events, "assign_game", session_key="a", match_id=affected["id"], court_id="c0")
    state, events = command(state, events, "release_game", court_id="c0")
    started = deepcopy(round_row(events["a"])["matches"][0])
    assert started["hubStartedAt"]

    state, events = pause(state, events, people(started)[0])

    assert started in round_row(events["a"])["matches"]
    blocked = build_hub_snapshot(state, events, now=LATER)["blocked_queue"]
    assert any(m["match_id"] == started["id"] and "A player is paused." in m["blockers"] for m in blocked)


@pytest.mark.parametrize("commitment", ["assigned", "reserved", "scored"])
def test_current_bye_cannot_gain_earlier_obligation_after_committing_to_a_later_round(commitment):
    from test_round_robin_hub import limited_rolling_event

    state, events = setup({"a": limited_rolling_event()}, courts=2)
    current = deepcopy(round_row(events["a"])["matches"][0])
    later = round_row(events["a"], 2)["matches"][0]
    last = round_row(events["a"], 3)["matches"][0]
    # Occupy the other later-round players too: the only tempting donors are
    # current byes whose later obligations are already committed.
    state, events = command(state, events, "assign_game", session_key="a", match_id=last["id"], court_id="c0")
    if commitment == "reserved":
        state, events = command(state, events, "reserve_game_for_court", session_key="a", match_id=later["id"], court_id="c0")
    else:
        state, events = command(state, events, "assign_game", session_key="a", match_id=later["id"], court_id="c1")
        if commitment == "scored":
            state, events = score(state, events, "c1")
    protected = deepcopy([row["matches"] for row in events["a"]["rounds"][1:3]])
    busy = set(people(later)) | set(people(last))

    state, events = pause(state, events, people(current)[0])

    assert [row["matches"] for row in events["a"]["rounds"][1:3]] == protected
    assert all(not busy.intersection(people(match)) for match in round_row(events["a"])["matches"])


@pytest.mark.parametrize("donor_status", ["paused", "shared_profile_on_court", "unresolved_name"])
def test_paused_unresolved_and_globally_claimed_people_are_not_replacements(donor_status):
    value = planned()
    donor = round_row(value)["byeParticipantIds"][0]
    donor_row = next(p for p in value["participants"] if p["id"] == donor)
    values = {"a": value}
    if donor_status == "shared_profile_on_court":
        donor_row["player_id"] = 901
        values["b"] = event("Other group", 4, player_ids=[901, 902, 903, 904])
    elif donor_status == "unresolved_name":
        donor_row["name"] = "Repeated Guest"
        values["b"] = event("Other group", 4)
        values["b"]["participants"][0]["name"] = "Repeated Guest"
    state, events = setup(values, courts=2)
    if donor_status == "shared_profile_on_court":
        other = round_row(events["b"])["matches"][0]
        state, events = command(state, events, "assign_game", session_key="b", match_id=other["id"], court_id="c1")
    elif donor_status == "paused":
        state, events = pause(state, events, donor)
    paused = people(round_row(events["a"])["matches"][0])[0]
    healthy = deepcopy(round_row(events["a"])["matches"][1])
    other_before = deepcopy(events.get("b"))

    state, events = pause(state, events, paused)

    assert healthy in round_row(events["a"])["matches"]
    assert all(donor not in people(match) for match in round_row(events["a"])["matches"])
    assert events.get("b") == other_before


def test_too_few_available_players_do_not_generate_rounds_forever_and_resume_recovers():
    state, events = setup({"a": planned(8)})
    ids = [p["id"] for p in events["a"]["participants"]]
    for pid in ids:
        state, events = pause(state, events, pid)
    assert not ready(state, events)
    before = deepcopy(events)
    for _ in range(3):
        state, events = command(state, events, "auto_fill_courts")
    assert events == before
    assert not any(court["current"] or court["next"] for court in state["courts"])
    for pid in ids[:4]:
        state, events = command(state, events, "resume_player", identity=identity(state, events, pid))
    assert ready(state, events)
    assert all({p["id"] for p in m["side_a"] + m["side_b"]} <= set(ids[:4]) for m in ready(state, events))


def test_resume_keeps_existing_replacement_and_preserves_only_pre_pause_wait():
    state, events = setup({"a": planned()})
    pid = people(round_row(events["a"])["matches"][0])[0]
    person = identity(state, events, pid)
    state, events = pause(state, events, pid, now="2026-10-20T03:05:00+00:00")
    repaired = deepcopy(round_row(events["a"])["matches"])
    assert all(pid not in people(match) for match in repaired)
    state, events = command(state, events, "resume_player", now=LATER, identity=person)
    assert all(match in round_row(events["a"])["matches"] for match in repaired)
    snapshot = build_hub_snapshot(state, events, now="2026-10-20T03:12:00+00:00")
    assert next(p for p in snapshot["players"] if p["identity"] == person)["wait_seconds"] == 420
    assert not state["players"][person]["paused"]


@pytest.mark.parametrize("advance_first", [False, True])
def test_resume_forms_game_with_three_byes_in_the_current_round(advance_first):
    value = planned(8)
    first, healthy = deepcopy(round_row(value)["matches"])
    # Keep the same explicit lineups in round two to exercise a break that
    # continues past automatic advancement, with another court already live.
    round_row(value, 2)["matches"] = [dict(m, id=f"r2-{m['id']}", round=2) for m in (first, healthy)]
    state, events = setup({"a": value}, courts=2)
    pid = people(first)[0]
    state, events = pause(state, events, pid)
    state, events = command(state, events, "assign_game", session_key="a", match_id=healthy["id"], court_id="c0")
    if advance_first:
        state, events = score(state, events, "c0")
        assert events["a"]["currentRoundNumber"] == 2
        state, events = command(state, events, "assign_game", session_key="a", match_id=f"r2-{healthy['id']}", court_id="c0")
    number = events["a"]["currentRoundNumber"]
    live = deepcopy(round_row(events["a"], number)["matches"])
    assert len(round_row(events["a"], number)["byeParticipantIds"]) == 3

    state, events = command(state, events, "resume_player", identity=identity(state, events, pid))

    row = round_row(events["a"], number)
    assert all(match in row["matches"] for match in live)
    formed = [m for m in ready(state, events) if m["round_number"] == number]
    assert len(formed) == 1
    assert {p["id"] for p in formed[0]["side_a"] + formed[0]["side_b"]} == set(people(first))
    assert not row["byeParticipantIds"]
    before = deepcopy(events)
    state, events = command(state, events, "resume_player", identity=identity(state, events, pid))
    assert events == before
    state, events = command(state, events, "assign_next_court", session_key="a", match_id=formed[0]["match_id"])
    assert state["courts"][1]["current"]["match_id"] == formed[0]["match_id"]


@pytest.mark.parametrize("already_on_bye", [False, True])
def test_fill_recovers_four_byes_left_by_a_pre_fix_resume_without_toggling_player(already_on_bye):
    value = planned(8)
    affected, healthy = deepcopy(round_row(value)["matches"])
    if already_on_bye:
        round_row(value)["matches"] = [healthy]
        round_row(value)["byeParticipantIds"] = people(affected)
    state, events = setup({"a": value}, courts=2)
    pid = people(affected)[0]
    state, events = pause(state, events, pid)
    state, events = command(state, events, "assign_game", session_key="a", match_id=healthy["id"], court_id="c0")
    healthy = deepcopy(round_row(events["a"])["matches"][0])
    # This is the persisted state produced by the former resume behavior.
    state["players"][identity(state, events, pid)].update(paused=False, pause_started_at=None)
    row = round_row(events["a"])
    row["byeParticipantIds"].append(pid)
    row["hubPausedParticipantIds"] = []
    assert len(row["byeParticipantIds"]) == 4

    state, events = command(state, events, "auto_fill_courts")

    assert healthy in round_row(events["a"])["matches"]
    assigned = state["courts"][1]["current"]
    assert assigned and assigned["round_number"] == 1
    formed = next(m for m in round_row(events["a"])["matches"] if m["id"] == assigned["match_id"])
    assert set(people(formed)) == set(people(affected))
    assert not round_row(events["a"])["byeParticipantIds"]
    before = deepcopy(events)
    state, events = command(state, events, "auto_fill_courts")
    assert events == before


def test_pausing_an_existing_bye_records_absence_without_changing_lineups_and_resume_restores_bye():
    state, events = setup({"a": planned()})
    pid = round_row(events["a"])["byeParticipantIds"][0]
    person = identity(state, events, pid)
    lineups = deepcopy([row["matches"] for row in events["a"]["rounds"]])
    future = deepcopy(events["a"]["rounds"][1:])

    state, events = pause(state, events, pid, now="2026-10-20T03:05:00+00:00")

    assert pid not in round_row(events["a"])["byeParticipantIds"]
    assert pid in round_row(events["a"])["hubPausedParticipantIds"]
    assert [row["matches"] for row in events["a"]["rounds"]] == lineups
    assert events["a"]["rounds"][1:] == future
    state, events = command(state, events, "resume_player", now=LATER, identity=person)
    assert pid in round_row(events["a"])["byeParticipantIds"]
    assert pid not in round_row(events["a"]).get("hubPausedParticipantIds", [])
    assert [row["matches"] for row in events["a"]["rounds"]] == lineups
    snapshot = build_hub_snapshot(state, events, now="2026-10-20T03:12:00+00:00")
    assert next(p for p in snapshot["players"] if p["identity"] == person)["wait_seconds"] == 420


@pytest.mark.parametrize("kind", ["moneyball", "playoff"])
def test_fixed_moneyball_or_playoff_lineups_are_not_automatically_substituted(kind):
    if kind == "moneyball":
        from jupr_app.domain.moneyball_expectations import freeze_moneyball

        preview = create_generator_preview(generator_kind="round_robin", play_format="doubles",
            title="Fixed Moneyball", participant_names=[f"Money {i}" for i in range(8)],
            player_ids=list(range(1, 9)),
            total_rounds=7, event_format="moneyball_expectations")
        value = start_generator_event(freeze_moneyball(preview,
            {p["id"]: 3 + i * .15 for i, p in enumerate(preview["participants"])}))
        value["roundRobinHubId"] = "hub"
    else:
        from jupr_app.domain.generator_playoffs import start_generator_playoff

        value = planned(8)
        value = save_generator_round(value, round_number=1, scores=[
            {"match_id": m["id"], "score_a": 11, "score_b": 7} for m in round_row(value)["matches"]])
        value = start_generator_playoff(value, playoff_format="top_eight")
    state, events = setup({"a": value})
    before = deepcopy(events["a"])
    pid = people(round_row(before, before["currentRoundNumber"])["matches"][0])[0]

    state, events = pause(state, events, pid)

    assert events["a"] == before
    assert any("A player is paused." in match["blockers"] for match in build_hub_snapshot(state, events, now=LATER)["blocked_queue"])


def test_shared_pause_does_not_repair_a_submitted_sibling_and_replay_does_not_reapply_repair():
    from test_admin_round_robin_hub_service import (
        TransactionalHubDB, command as service_command, create, raw_session, session,
    )
    from jupr_app.services import admin_round_robin_hub_service as service

    db = TransactionalHubDB()
    session(db, count=9)
    session(db, "rr-b", count=9, offset=20)
    a = raw_session(db)["state"]["page_state"]["event"]
    pid = people(round_row(a)["matches"][0])[0]
    shared = next(p for p in a["participants"] if p["id"] == pid)
    b = raw_session(db, "rr-b")["state"]["page_state"]["event"]
    b["participants"][0]["player_id"] = shared["player_id"]
    hub = service_command(db, create(db), "add_group", {"session_key": "rr-a"})
    hub = service_command(db, hub, "add_group", {"session_key": "rr-b"})
    raw_session(db, "rr-b")["state"]["generator_submission"] = {"status": "pending"}
    locked = deepcopy(raw_session(db, "rr-b"))
    request = dict(club_id="club-a", hub_id=hub["hub"]["id"], action="pause_player",
        payload={"identity": f"player:{shared['player_id']}", "excluded_session_keys": []},
        expected_version=hub["hub"]["version"], idempotency_key="pause-repair-retry-once",
        actor_email="operator@example.test")

    service.command_round_robin_hub(db, **request)

    assert raw_session(db, "rr-b") == locked
    assert next(row for row in db.calls[-1]["p_sessions"] if row["session_key"] == "rr-b")["patch"] == {}
    persisted = deepcopy(db.storage)
    calls = len(db.calls)
    result = service.command_round_robin_hub(db, **request)
    assert result["replayed"] is True
    assert db.storage == persisted
    assert len(db.calls) == calls
