from copy import deepcopy

import pytest

from jupr_app.domain.adaptive_play_engine import create_generator_preview, start_generator_event
from jupr_app.domain.hub_lineup_controls import remove_hub_players, set_hub_match_lineup
from jupr_app.domain.round_robin_hub import apply_hub_command, build_hub_snapshot


NOW = "2026-10-28T03:00:00+00:00"


def fixture(count=9):
    event = start_generator_event(create_generator_preview(generator_kind="round_robin", play_format="doubles",
        title="Operator controls", participant_names=[f"Player {n}" for n in range(1, count + 1)],
        court_count=0, total_rounds=3))
    event["roundRobinHubId"] = "hub"
    ids = [p["id"] for p in event["participants"]]
    first = event["rounds"][0]
    first["matches"] = [{"id": f"game-{i}", "round": 1, "court": i, "playFormat": "doubles",
        "sideA": ids[(i - 1) * 4:(i - 1) * 4 + 2], "sideB": ids[(i - 1) * 4 + 2:i * 4],
        "scoreA": None, "scoreB": None, "status": "scheduled"} for i in range(1, count // 4 + 1)]
    first["byeParticipantIds"] = ids[count // 4 * 4:]
    hub = {"id": "hub", "groups": [{"session_key": "a"}], "players": {}, "courts": [
        {"id": "court", "label": "Court", "open": True, "current": None, "next": None,
         "dedicated_session_key": None}]}
    return hub, {"a": event}, ids


def edit(hub, events, a, b, **extra):
    set_hub_match_lineup(hub, events, {"session_key": "a", "match_id": "game-1",
        "side_a": a, "side_b": b, **extra}, now=NOW)


def test_exact_teams_swap_waiting_players_and_byes_without_losing_anyone():
    hub, events, p = fixture()
    later = deepcopy(events["a"]["rounds"][1:])
    edit(hub, events, [p[0], p[4]], [p[8], p[2]])
    row = events["a"]["rounds"][0]
    target, donor = row["matches"]
    assert target["sideA"] == [p[0], p[4]] and target["sideB"] == [p[8], p[2]]
    assert donor["sideA"] == [p[1], p[5]]
    assert row["byeParticipantIds"] == [p[3]]
    seated = [pid for m in row["matches"] for side in ("sideA", "sideB") for pid in m[side]]
    assert len(seated) == len(set(seated)) == 8
    assert set(seated + row["byeParticipantIds"]) == set(p)
    assert target["hubPlanLocked"] and donor["hubPlanLocked"]
    assert events["a"]["rounds"][1:] == later
    assert len(events["a"]["hubLineupEdits"]) == 1


def test_exact_team_reordering_is_preserved_by_fill_and_membership_repair():
    hub, events, p = fixture()
    edit(hub, events, [p[2], p[0]], [p[3], p[1]])
    selected = deepcopy(events["a"]["rounds"][0]["matches"][0])
    hub, events = apply_hub_command(hub, events, action="auto_fill_courts", payload={}, now=NOW)
    target = next(m for m in events["a"]["rounds"][0]["matches"] if m["id"] == selected["id"])
    assert target["sideA"] == selected["sideA"] and target["sideB"] == selected["sideB"]
    repaired = remove_hub_players(events["a"], [p[8]], now=NOW)
    target = next(m for m in repaired["rounds"][0]["matches"] if m["id"] == selected["id"])
    assert target["sideA"] == selected["sideA"] and target["sideB"] == selected["sideB"]
    assert target["hubPlanLocked"]


def test_two_replacements_can_come_from_the_same_unlocked_waiting_game():
    hub, events, p = fixture()
    edit(hub, events, [p[0], p[4]], [p[5], p[2]])
    first, second = events["a"]["rounds"][0]["matches"]
    assert first["sideA"] == [p[0], p[4]] and first["sideB"] == [p[5], p[2]]
    assert second["sideA"] == [p[1], p[3]]
    assert first["hubPlanLocked"] and second["hubPlanLocked"]


@pytest.mark.parametrize("replacement_source", ["bye", "two_donors"])
def test_next_round_exact_lineup_uses_its_round_for_every_swap(replacement_source):
    hub, events, p = fixture(count=13)
    event = events["a"]
    first = event["rounds"][0]
    second = deepcopy(first)
    second.update(number=2, status="preview")
    for match in second["matches"]:
        match.update(id=f"next-{match['id']}", round=2)
    event["rounds"][1] = second
    event["hubPlanVersion"] = 1
    # The first game still runs; players supplying replacements have finished
    # their own earlier games (or had a bye) and may join a next-round lineup.
    for match in first["matches"][1:]:
        match.update(scoreA=11, scoreB=5, status="scored")
    before_first = deepcopy(first)
    target_a = [p[0], p[12] if replacement_source == "bye" else p[4]]
    target_b = [p[2], p[3] if replacement_source == "bye" else p[8]]
    hub, events = apply_hub_command(hub, events, action="set_match_lineup", payload={
        "session_key": "a", "match_id": "next-game-1", "round_number": 2,
        "side_a": target_a, "side_b": target_b,
    }, now=NOW)
    event = events["a"]
    assert event["currentRoundNumber"] == 1
    assert event["rounds"][0] == before_first
    second = next(row for row in event["rounds"] if row["number"] == 2)
    matches = {match["id"]: match for match in second["matches"]}
    assert matches["next-game-1"]["sideA"] == target_a
    assert matches["next-game-1"]["sideB"] == target_b
    assert matches["next-game-1"]["hubPlanLocked"]
    if replacement_source == "bye":
        assert second["byeParticipantIds"] == [p[1]]
    else:
        assert matches["next-game-2"]["sideA"] == [p[1], p[5]]
        assert matches["next-game-3"]["sideA"] == [p[3], p[9]]
        assert matches["next-game-2"]["hubPlanLocked"] and matches["next-game-3"]["hubPlanLocked"]
    seated = [pid for match in second["matches"] for side in ("sideA", "sideB") for pid in match[side]]
    assert len(seated) == len(set(seated)) == 12
    assert set(seated + second["byeParticipantIds"]) == set(p)


@pytest.mark.parametrize("earlier_on_court", [False, True])
def test_next_plan_can_swap_players_whose_earlier_games_are_unfinished(earlier_on_court):
    hub, events, _ = fixture(count=8)
    event = events["a"]
    event["hubPlanVersion"] = 1
    # Use the generated next-round plan, with every Round 1 result pending.
    if earlier_on_court:
        first_match = event["rounds"][0]["matches"][0]
        hub, events = apply_hub_command(hub, events, action="assign_game", payload={
            "session_key": "a", "match_id": first_match["id"], "round_number": 1, "court_id": "court",
        }, now=NOW)

        event = events["a"]
    next_round = next(row for row in event["rounds"] if row["number"] == 2)
    target, donor = next_round["matches"]
    current_players = set(event["rounds"][0]["matches"][0]["sideA"] + event["rounds"][0]["matches"][0]["sideB"])
    if earlier_on_court and not current_players.intersection(donor["sideA"] + donor["sideB"]):
        target, donor = donor, target
    replacement = next((pid for pid in donor["sideA"] + donor["sideB"] if pid in current_players), donor["sideA"][0])
    desired_a, desired_b = [replacement, target["sideA"][1]], list(target["sideB"])
    original_round = deepcopy(event["rounds"][0])
    original_courts = deepcopy(hub["courts"])
    hub, events = apply_hub_command(hub, events, action="set_match_lineup", payload={
        "session_key": "a", "match_id": target["id"], "round_number": 2,
        "side_a": desired_a, "side_b": desired_b,
    }, now=NOW)
    event = events["a"]
    edited = next(match for row in event["rounds"] if row["number"] == 2 for match in row["matches"] if match["id"] == target["id"])
    assert edited["sideA"] == desired_a and edited["sideB"] == desired_b and edited["hubPlanLocked"]
    assert event["currentRoundNumber"] == 1 and event["rounds"][0] == original_round
    assert hub["courts"] == original_courts
    snapshot = build_hub_snapshot(hub, events, now=NOW)
    blocked = next(match for match in snapshot["blocked_queue"] if match["match_id"] == target["id"])
    assert "A player has an unfinished match in an earlier round." in blocked["blockers"]
    with pytest.raises(ValueError, match="not ready"):
        apply_hub_command(hub, events, action="assign_game", payload={
            "session_key": "a", "match_id": target["id"], "round_number": 2, "court_id": "court",
        }, now=NOW)


def test_next_plan_can_include_a_person_playing_in_another_group_without_changing_claims():
    hub, events, _ = fixture(count=8)
    _, other_events, _ = fixture(count=4)
    events["b"] = other_events["a"]
    events["a"]["hubPlanVersion"] = events["b"]["hubPlanVersion"] = 1
    hub["groups"].append({"session_key": "b"})
    target, donor = events["a"]["rounds"][1]["matches"]
    incoming = donor["sideA"][0]
    next(p for p in events["a"]["participants"] if p["id"] == incoming)["player_id"] = 321
    for player in events["b"]["participants"]:
        player["name"] = f"Other {player['name']}"
    events["b"]["participants"][0]["player_id"] = 321
    hub, events = apply_hub_command(hub, events, action="assign_game", payload={
        "session_key": "b", "match_id": "game-1", "round_number": 1, "court_id": "court",
    }, now=NOW)
    claimed_court = deepcopy(hub["courts"])
    other_round = deepcopy(events["b"]["rounds"][0])
    desired_a = [incoming, target["sideA"][1]]
    hub, events = apply_hub_command(hub, events, action="set_match_lineup", payload={
        "session_key": "a", "match_id": target["id"], "round_number": 2,
        "side_a": desired_a, "side_b": target["sideB"],
    }, now=NOW)
    edited = next(match for row in events["a"]["rounds"] if row["number"] == 2 for match in row["matches"] if match["id"] == target["id"])
    assert edited["sideA"] == desired_a and edited["hubPlanLocked"]
    assert hub["courts"] == claimed_court and events["b"]["rounds"][0] == other_round
    snapshot = build_hub_snapshot(hub, events, now=NOW)
    blocked = next(match for match in snapshot["blocked_queue"] if match["session_key"] == "a" and match["match_id"] == target["id"])
    assert "A player is already assigned to a court." in blocked["blockers"]


def test_released_started_future_game_cannot_use_plan_only_readiness_bypass():
    hub, events, p = fixture(count=13)
    event = events["a"]
    event["hubPlanVersion"] = 1
    for match in event["rounds"][0]["matches"][1:]:
        match.update(scoreA=11, scoreB=5, status="scored")
    second = deepcopy(event["rounds"][0])
    second.update(number=2, status="preview")
    for match, people in zip(second["matches"], ([p[4], p[5], p[8], p[9]], p[:4], [p[6], p[7], p[10], p[11]])):
        match.update(id=f"next-{match['id']}", round=2, sideA=list(people[:2]), sideB=list(people[2:]),
            scoreA=None, scoreB=None, status="scheduled")
    second["matches"][0]["hubStartedAt"] = NOW
    event["rounds"][1] = second
    before = deepcopy((hub, events))
    with pytest.raises(ValueError, match="unfinished game in an earlier round"):
        edit(hub, events, [p[0], p[5]], [p[8], p[9]], match_id="next-game-1", round_number=2)
    assert (hub, events) == before

@pytest.mark.parametrize("kind", ["duplicate_id", "duplicate_identity", "merged_guests", "locked_donor", "recorded_donor", "wrong_court"])
def test_invalid_exact_lineup_does_not_partially_apply_an_earlier_swap(kind):
    hub, events, p = fixture()
    a, b, extra = [p[0], p[8]], [p[4], p[2]], {}
    if kind == "duplicate_id":
        b = [p[0], p[2]]
    elif kind == "duplicate_identity":
        for participant in events["a"]["participants"]:
            if participant["id"] in (p[0], p[4]):
                participant["player_id"] = 123
    elif kind == "merged_guests":
        hub["identity_decisions"] = [{"resolution": "same", "left": {"session_key": "a", "participant_id": p[0]},
            "right": {"session_key": "a", "participant_id": p[4]}}]
    elif kind == "locked_donor":
        events["a"]["rounds"][0]["matches"][1]["hubPlanLocked"] = True
    elif kind == "recorded_donor":
        events["a"]["rounds"][0]["matches"][1].update(scoreA=11, scoreB=4, status="scored")
    else:
        extra["court_id"] = "moved-court"
    before = deepcopy((hub, events))
    with pytest.raises(ValueError):
        edit(hub, events, a, b, **extra)
    assert (hub, events) == before


def test_invalid_batch_is_atomic_before_any_membership_changes():
    _, events, p = fixture()
    before = deepcopy(events["a"])
    for ids in ([p[0], "missing"], [p[0], p[0]], []):
        with pytest.raises(ValueError):
            remove_hub_players(events["a"], ids, now=NOW)
        assert events["a"] == before


def test_batch_departures_repair_once_and_keep_history_and_committed_players(monkeypatch):
    from jupr_app.domain import round_robin_hub as core

    _, events, p = fixture()
    event = events["a"]
    first = event["rounds"][0]
    first["status"] = "saved"
    for match in first["matches"]:
        match.update(scoreA=11, scoreB=5, status="scored")
    event["currentRoundNumber"] = 2
    second = event["rounds"][1]
    second["status"] = "active"
    live = second["matches"][0]
    live["hubStartedAt"] = NOW
    committed = live["sideA"][0]
    live_people = set(live["sideA"] + live["sideB"])
    free = next(pid for pid in p if pid not in live_people)
    original = deepcopy(event)
    calls = []
    repair = core.repair_hub_roster_plan

    def counted(value):
        calls.append(deepcopy(value))
        return repair(value)

    monkeypatch.setattr(core, "repair_hub_roster_plan", counted)
    result = remove_hub_players(event, [committed, free], now=NOW)
    assert len(calls) == 1
    assert event == original
    assert result["rounds"][0] == original["rounds"][0]
    assert next(m for m in result["rounds"][1]["matches"] if m["id"] == live["id"]) == live
    participants = {p["id"]: p for p in result["participants"]}
    assert participants[committed]["inactive_from_round"] == 3
    assert participants[free]["inactive_from_round"] == 2
    for row in result["rounds"]:
        people = {pid for match in row["matches"] for side in ("sideA", "sideB") for pid in match[side]}
        for pid in (committed, free):
            if row["number"] >= participants[pid]["inactive_from_round"]:
                assert pid not in people | set(row["byeParticipantIds"])


def test_departure_repairs_future_manual_lock_without_deferring_exit():
    _, events, p = fixture()
    event = events["a"]
    future = event["rounds"][1]["matches"][0]
    future["hubPlanLocked"] = True
    departing = future["sideA"][0]
    result = remove_hub_players(event, [departing], now=NOW)
    assert next(p for p in result["participants"] if p["id"] == departing)["inactive_from_round"] == 1
    assert all(departing not in match["sideA"] + match["sideB"] for row in result["rounds"] for match in row["matches"])
    assert any(future["id"] in row["unlockedMatchIds"] for row in result["hubPlanAdjustments"])
