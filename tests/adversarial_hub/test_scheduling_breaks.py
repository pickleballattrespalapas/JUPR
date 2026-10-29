"""Fresh bounded scheduling regression sequences; no external services or data.

HUB_AUDIT_SEEDS (1..48) and HUB_AUDIT_STEPS (1..100) reduce the default 48x100
run. A failure includes its seed and recent commands for deterministic replay.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import os
import random

import pytest

from jupr_app.domain.adaptive_play_engine import active_participant_ids, create_generator_preview, start_generator_event
from jupr_app.domain.round_robin_hub import apply_hub_command, build_hub_snapshot

SEEDS = min(48, max(1, int(os.environ.get("HUB_AUDIT_SEEDS", "48"))))
STEPS = min(100, max(1, int(os.environ.get("HUB_AUDIT_STEPS", "100"))))
START = datetime(2026, 10, 23, tzinfo=timezone.utc)


def all_matches(events):
    return {(key, str(match["id"])): match for key, event in events.items()
            for row in event["rounds"] for match in row.get("matches", [])}


def lineup(match):
    return tuple(match.get("sideA") or match.get("teamA") or []), tuple(match.get("sideB") or match.get("teamB") or [])


def people(match):
    return set(sum((list(side) for side in lineup(match)), []))


def terminal(match):
    return match.get("status") in {"played", "not_played"} or (match.get("scoreA") is not None and match.get("scoreB") is not None)


def create_world(seed):
    events = {}
    for index, key in enumerate(("a", "b")):
        count = 6 + ((seed + index) % 4)
        profiles = [1, *range(100 + index * 20, 100 + index * 20 + count - 1)]
        events[key] = start_generator_event(create_generator_preview(
            generator_kind="round_robin", play_format="singles" if seed % 6 == 0 else "doubles",
            title=f"Group {key}", participant_names=[f"{key} Player {n}" for n in range(count)],
            player_ids=profiles, court_count=0, total_rounds=3))
    hub = {"id": "offline-audit", "groups": [], "players": {}, "courts": [
        {"id": f"c{i}", "label": f"Court {i}", "open": True, "current": None, "next": None,
         "dedicated_session_key": None} for i in range(3)]}
    for key in events:
        hub, events = apply_hub_command(hub, events, action="add_group", payload={"session_key": key}, now=START.isoformat())
    return hub, events


def assert_invariants(hub, events, now):
    original = deepcopy((hub, events))
    view = build_hub_snapshot(hub, events, now=now)
    assert (hub, events) == original, "snapshot mutated input"
    identities, assigned = [], []
    for court in view["courts"]:
        for slot in ("current", "next"):
            match = court.get(slot)
            if match:
                assigned.append((match["session_key"], match["match_id"]))
                identities.extend(p["identity"] for p in match["side_a"] + match["side_b"])
    assert len(identities) == len(set(identities)), "one person holds two court commitments"
    assert len(assigned) == len(set(assigned)), "one game holds two court commitments"
    for key, event in events.items():
        known = {str(p["id"]) for p in event["participants"]}
        match_ids = []
        for row in event["rounds"]:
            seated = [pid for m in row.get("matches", []) for side in lineup(m) for pid in side]
            assert not set(seated) - known, f"unknown player in group {key}"
            assert len(seated) == len(set(seated)), f"duplicate participant seat in {key} round {row['number']}"
            assert not set(seated) & set(row.get("byeParticipantIds", [])), "player is seated and on bye"
            match_ids.extend(str(m["id"]) for m in row.get("matches", []))
        assert len(match_ids) == len(set(match_ids)), "match ID repeated across rounds"
    for group in view["groups"]:
        for member in group["players"]:
            if member["on_court"]:
                assert member["current_court"]["session_key"] == group["session_key"]
            if member["playing_elsewhere"]:
                assert member["current_court"]["session_key"] != group["session_key"]
                assert not member["on_court"]
    return view


def choose_command(rng, step, seed, hub, events, view):
    key = rng.choice(("a", "b"))
    current = [c for c in hub["courts"] if c.get("current")]
    reserved = [c for c in hub["courts"] if c.get("next")]
    ready = view["eligible_queue"]
    operation = step % 20
    if operation in {3, 4, 5} and current:
        court = rng.choice(current)
        return "score_and_release", {**court["current"], "court_id": court["id"], "score_a": 11, "score_b": rng.randrange(0, 11)}
    if operation == 6:
        candidates = ready + view["blocked_queue"]
        if current and rng.random() < 0.5:
            court = rng.choice(current)
            return "resolve_game", {**court["current"], "court_id": court["id"], "resolution": "not_played"}
        if candidates:
            match = rng.choice(candidates)
            return "resolve_game", {"session_key": match["session_key"], "match_id": match["match_id"], "resolution": "not_played"}
    if operation in {7, 8, 19}:
        paused = [identity for identity, player in hub["players"].items() if player.get("paused")]
        identity = rng.choice(paused if operation in {8, 19} and paused else list(hub["players"]))
        return "resume_player" if operation in {8, 19} else "pause_player", {"identity": identity}
    if operation == 9:
        return "add_player", {"session_key": key, "name": f"Arrival {seed} {step}", "player_id": 10000 + seed * 100 + step}
    group = next(g for g in view["groups"] if g["session_key"] == key)
    removable = [p for p in group["players"] if p["can_remove"]]
    if operation == 10 and removable:
        return "remove_player", {"session_key": key, "participant_id": rng.choice(removable)["participant_id"]}
    if operation == 11 and ready:
        possible = [(c, m) for c in current if not c.get("next") for m in ready if c["id"] in m["allowed_court_ids"]]
        if possible:
            court, match = rng.choice(possible)
            return "reserve_game_for_court", {"court_id": court["id"], "session_key": match["session_key"], "match_id": match["match_id"]}
    if operation == 12 and reserved:
        return "cancel_reservation", {"court_id": rng.choice(reserved)["id"]}
    if operation == 13 and current:
        return "release_game", {"court_id": rng.choice(current)["id"]}
    if operation == 14 and removable:
        return "transfer_player", {"source_session_key": key, "destination_session_key": "b" if key == "a" else "a", "participant_id": rng.choice(removable)["participant_id"]}
    if operation == 15:
        departed = [p for p in group["players"] if p["can_rejoin"]]
        if departed:
            return "rejoin_player", {"session_key": key, "participant_id": rng.choice(departed)["participant_id"]}
    if operation == 16 and removable:
        selected = rng.choice(removable)
        return "add_player", {"session_key": key, "name": "Duplicate profile attempt", "player_id": selected["profile_id"]}
    if operation == 18 and group["can_advance"]:
        return "advance_round", {"session_key": key}
    return "auto_fill_courts", {}


@pytest.mark.parametrize("seed", range(SEEDS))
def test_seeded_two_group_scheduling_contracts(seed):
    rng = random.Random(seed)
    hub, events = create_world(seed)
    trace, successful = [], 0
    initial_rounds = {key: e["currentRoundNumber"] for key, e in events.items()}
    for step in range(STEPS):
        now = (START + timedelta(minutes=step + 1)).isoformat()
        view = assert_invariants(hub, events, now)
        action, payload = choose_command(rng, step, seed, hub, events, view)
        before = deepcopy((hub, events))
        prior_matches = deepcopy(all_matches(events))
        prior_rounds = {(key, str(match["id"])): int(row["number"])
                        for key, event in events.items() for row in event["rounds"]
                        for match in row.get("matches", [])}
        prior_assignments = {(court[slot]["session_key"], court[slot]["match_id"])
                             for court in hub["courts"] for slot in ("current", "next") if court.get(slot)}
        paused_refs = {(g["session_key"], p["participant_id"]) for g in view["groups"] for p in g["players"] if p["paused"]}
        trace.append((action, payload))
        try:
            try:
                updated_hub, updated_events = apply_hub_command(hub, events, action=action, payload=payload, now=now)
            except ValueError:
                assert (hub, events) == before, "rejected command changed input"
                continue
            assert (hub, events) == before, "successful pure command changed input"
            hub, events = updated_hub, updated_events
            successful += 1
            after_matches = all_matches(events)
            for match_key, old in prior_matches.items():
                if terminal(old):
                    assert match_key in after_matches, "recorded history disappeared"
                    new = after_matches[match_key]
                    assert (lineup(old), old.get("scoreA"), old.get("scoreB"), old.get("status")) == (lineup(new), new.get("scoreA"), new.get("scoreB"), new.get("status")), "recorded history changed"
                key, match_id = match_key
                number = prior_rounds[match_key]
                committed = (old.get("hubStartedAt") or match_key in prior_assignments
                             or match_id in before[1][key].get("hubAssignedMatchIds", [])
                             or old.get("scoreA") is not None or old.get("scoreB") is not None)
                if committed:
                    assert match_key in after_matches and lineup(after_matches[match_key]) == lineup(old), "started, scored, assigned, or reserved lineup changed"
                # An explicit staff lineup survives routine repairs. A real
                # departure may unlock it when its selected players leave.
                if old.get("hubPlanLocked") and people(old) <= set(active_participant_ids(events[key], number)):
                    assert match_key in after_matches and lineup(after_matches[match_key]) == lineup(old), "active manual lineup changed"
                if (action == "add_player" and number <= before[1][key]["currentRoundNumber"]
                        and not any((key, pid) in paused_refs for pid in people(old))):
                    assert match_key in after_matches and lineup(after_matches[match_key]) == lineup(old), "arrival changed an unaffected current lineup"
            assert_invariants(hub, events, now)
        except Exception as exc:
            pytest.fail(f"seed={seed} step={step} action={action}: {exc}\nRecent trace: {trace[-20:]!r}", pytrace=True)
    assert successful >= STEPS // 2
    if STEPS == 100:
        assert any(e["currentRoundNumber"] > initial_rounds[key] for key, e in events.items()), f"seed {seed}: no round made progress"


@pytest.mark.parametrize("count", [15, 29])
def test_arrivals_mix_the_whole_next_roster_without_replacing_current_games(count):
    """Growing arrival cohorts must not become permanent four-player pools."""
    event = start_generator_event(create_generator_preview(
        generator_kind="round_robin", play_format="doubles", title="Arrivals",
        participant_names=[f"Original {n}" for n in range(4)], court_count=0, total_rounds=20))
    hub = {"id": "arrival-audit", "groups": [], "players": {}, "courts": [
        {"id": "c1", "label": "Court 1", "open": True, "current": None, "next": None,
         "dedicated_session_key": None}]}
    hub, events = apply_hub_command(hub, {"a": event}, action="add_group",
        payload={"session_key": "a"}, now=START.isoformat())
    hub, events = apply_hub_command(hub, events, action="auto_fill_courts", payload={}, now=START.isoformat())
    for index in range(4, count):
        current_games = deepcopy(events["a"]["rounds"][0]["matches"])
        hub, events = apply_hub_command(hub, events, action="add_player",
            payload={"session_key": "a", "name": f"Arrival {index}"}, now=START.isoformat())
        value = events["a"]
        assert [row["number"] for row in value["rounds"]] == [1, 2]
        assert all(match in value["rounds"][0]["matches"] for match in current_games)
        planned = value["rounds"][1]
        seats = [pid for match in planned["matches"] for pid in people(match)]
        byes = planned["byeParticipantIds"]
        assert len(seats) == len(set(seats)) and not set(seats) & set(byes)
        assert set(seats + byes) == set(active_participant_ids(value, 2))
        assert len(planned["matches"]) == (index + 1) // 4
        assert_invariants(hub, events, START.isoformat())
    current = events["a"]["rounds"][0]
    planned = events["a"]["rounds"][1]
    current_foursomes = {frozenset(people(match)) for match in current["matches"]}
    assert all(frozenset(people(match)) not in current_foursomes for match in planned["matches"])
    assert not set(current["byeParticipantIds"]) & set(planned["byeParticipantIds"])
