"""Regression contracts for explicit identity, membership and no-result operations."""
from copy import deepcopy

import pytest

from jupr_app.domain.adaptive_play_engine import active_participant_ids
from jupr_app.domain.round_robin_hub import player_identity
from test_round_robin_hub import setup, event, command, START, build_hub_snapshot


def person(events, key="a", index=0):
    return events[key]["participants"][index]


def ref(key, participant):
    return {"session_key": key, "participant_id": participant["id"]}


def current(events, key="a"):
    e = events[key]
    return next(r for r in e["rounds"] if r["number"] == e["currentRoundNumber"])


@pytest.mark.parametrize("profile", [False, True])
def test_explicit_rejoin_preserves_identity_availability_and_history(profile):
    hub, events = setup({"a": event(count=4, player_ids=[1, 2, 3, 4] if profile else None)})
    old = deepcopy(person(events))
    identity = player_identity(old, "a", hub)
    hub, events = command(hub, events, "pause_player", identity=identity)
    hub, events = command(hub, events, "remove_player", **ref("a", old))
    hub, events = command(hub, events, "rejoin_player", **ref("a", old))
    new = events["a"]["participants"][-1]
    assert new["id"] != old["id"]
    assert player_identity(new, "a", hub) == player_identity(person(events), "a", hub)
    view = build_hub_snapshot(hub, events, now=START)
    assert view["identity_pending_count"] == 0
    assert len(events["a"]["participants"]) == 5
    assert hub["players"][player_identity(new, "a", hub)]["paused"]
    with pytest.raises(ValueError, match="active or upcoming"):
        command(hub, events, "rejoin_player", **ref("a", old))


@pytest.mark.parametrize("profile", [False, True])
def test_transfer_moves_one_membership_atomically(profile):
    hub, events = setup({"a": event(count=4, player_ids=[1, 2, 3, 4] if profile else None), "b": event("B", count=4)})
    old = deepcopy(person(events))
    identity = player_identity(old, "a", hub)
    hub, events = command(hub, events, "pause_player", identity=identity)
    hub, events = command(hub, events, "transfer_player", source_session_key="a", destination_session_key="b", participant_id=old["id"])
    assert old["id"] not in active_participant_ids(events["a"], 1)
    new = events["b"]["participants"][-1]
    assert new["id"] in active_participant_ids(events["b"], 1)
    assert player_identity(new, "b", hub) == player_identity(old, "a", hub)
    assert hub["players"][player_identity(new, "b", hub)]["paused"]
    assert build_hub_snapshot(hub, events, now=START)["identity_pending_count"] == 0


@pytest.mark.parametrize("released", [False, True])
def test_transfer_rejects_started_commitment_without_partial_mutation(released):
    hub, events = setup({"a": event(count=4), "b": event("B", count=4)})
    match = current(events)["matches"][0]
    hub, events = command(hub, events, "assign_game", session_key="a", match_id=match["id"], court_id="c0")
    if released:
        hub, events = command(hub, events, "release_game", court_id="c0")
    before = deepcopy((hub, events))
    with pytest.raises(ValueError, match="started games"):
        command(hub, events, "transfer_player", source_session_key="a", destination_session_key="b", participant_id=person(events)["id"])
    assert (hub, events) == before


def test_transfer_to_existing_profile_removes_source_without_new_duplicate():
    hub, events = setup({"a": event(count=4, player_ids=[1, 2, 3, 4]), "b": event("B", count=4, player_ids=[1, 5, 6, 7])})
    hub, events = command(hub, events, "transfer_player", source_session_key="a", destination_session_key="b", participant_id=person(events)["id"])
    assert len(events["b"]["participants"]) == 4
    assert len(active_participant_ids(events["a"], 1)) == 3


def test_explicit_variant_merge_preserves_started_game_and_one_future_seat():
    hub, events = setup({"a": event(count=4)})
    original = deepcopy(person(events))
    match = deepcopy(current(events)["matches"][0])
    hub, events = command(hub, events, "assign_game", session_key="a", match_id=match["id"], court_id="c0")
    hub, events = command(hub, events, "add_player", session_key="a", name="A different spelling")
    duplicate = deepcopy(events["a"]["participants"][-1])
    hub, events = command(hub, events, "resolve_identities", resolution="same", participants=[ref("a", original), ref("a", duplicate)])
    preserved = current(events)["matches"][0]
    assert preserved["id"] == match["id"] and preserved["sideA"] == match["sideA"] and preserved["sideB"] == match["sideB"]
    assert duplicate["id"] not in active_participant_ids(events["a"], 1)
    assert len(active_participant_ids(events["a"], 2)) == 4
    assert len(events["a"]["participants"]) == 5


def test_arrivals_preserve_waiting_games_and_form_new_game_from_byes():
    hub, events = setup({"a": event(count=7)})
    old = deepcopy(current(events)["matches"])
    hub, events = command(hub, events, "add_player", session_key="a", name="New arrival")
    assert current(events)["matches"][:len(old)] == old
    assert len(current(events)["matches"]) == 2
    assert events["a"]["participants"][-1]["id"] in sum([m["sideA"] + m["sideB"] for m in current(events)["matches"]], [])


@pytest.mark.parametrize("assigned", [False, True])
def test_not_played_resolves_game_releases_court_and_preserves_wait(assigned):
    hub, events = setup({"a": event(count=4)})
    match = deepcopy(current(events)["matches"][0])
    if assigned:
        hub, events = command(hub, events, "assign_game", session_key="a", match_id=match["id"], court_id="c0")
    waits = {key: value["wait_since"] for key, value in hub["players"].items()}
    hub, events = command(hub, events, "resolve_game", session_key="a", match_id=match["id"], resolution="not_played", reason="Duplicate game")
    completed = events["a"]["rounds"][0]["matches"][0]
    assert completed["status"] == "not_played"
    assert completed["scoreA"] is None and completed["scoreB"] is None
    assert events["a"]["currentRoundNumber"] == 2
    assert not hub["courts"][0]["current"]
    assert {key: value["wait_since"] for key, value in hub["players"].items()} == waits


def test_failed_transfer_keeps_both_groups_and_historical_lineups_unchanged():
    hub, events = setup({"a": event(count=4), "b": event("B", count=40)})
    before = deepcopy((hub, events))
    with pytest.raises(ValueError, match="maximum of 40"):
        command(hub, events, "transfer_player", source_session_key="a", destination_session_key="b", participant_id=person(events)["id"])
    assert (hub, events) == before


def test_transfer_keeps_previous_scores_and_original_roster_history():
    hub, events = setup({"a": event(count=4), "b": event("B", count=4)})
    match = current(events)["matches"][0]
    hub, events = command(hub, events, "assign_game", session_key="a", match_id=match["id"], court_id="c0")
    hub, events = command(hub, events, "score_and_release", session_key="a", match_id=match["id"], court_id="c0", score_a=11, score_b=9)
    saved = deepcopy(events["a"]["rounds"][0])
    original = deepcopy(person(events))
    hub, events = command(hub, events, "transfer_player", source_session_key="a", destination_session_key="b", participant_id=original["id"])
    assert events["a"]["rounds"][0] == saved
    assert original["id"] in active_participant_ids(events["a"], 1)
    assert original["id"] not in active_participant_ids(events["a"], 2)
    assert hub["membership_changes"][-1]["source_effective_round"] == 2
    assert hub["membership_changes"][-1]["destination_effective_round"] == 1


def test_merge_finite_started_membership_retains_ongoing_duplicate_membership():
    hub, events = setup({"a": event(count=4)})
    original = deepcopy(person(events))
    match = current(events)["matches"][0]
    hub, events = command(hub, events, "assign_game", session_key="a", match_id=match["id"], court_id="c0")
    hub, events = command(hub, events, "remove_player", **ref("a", original))
    hub, events = command(hub, events, "add_player", session_key="a", name="Other spelling")
    duplicate = events["a"]["participants"][-1]
    hub, events = command(hub, events, "resolve_identities", resolution="same", participants=[ref("a", original), ref("a", duplicate)])
    assert original["id"] in active_participant_ids(events["a"], 10)
    assert duplicate["id"] not in active_participant_ids(events["a"], 10)
    assert len(active_participant_ids(events["a"], 2)) == 4


def test_blank_scores_resolve_current_game_without_fabricated_result():
    hub, events = setup({"a": event(count=4)})
    match = current(events)["matches"][0]
    hub, events = command(hub, events, "assign_game", session_key="a", match_id=match["id"], court_id="c0")
    hub, events = command(hub, events, "score_and_release", session_key="a", match_id=match["id"], court_id="c0", score_a=None, score_b=None)
    assert events["a"]["rounds"][0]["matches"][0]["status"] == "not_played"
    assert events["a"]["rounds"][0]["matches"][0]["scoreA"] is None


def test_resolve_future_game_finds_its_own_round_and_waits_for_earlier_results():
    hub, events = setup({"a": event(count=4)})
    future = events["a"]["rounds"][1]["matches"][0]
    hub, events = command(hub, events, "resolve_game", session_key="a", match_id=future["id"], resolution="not_played")
    assert events["a"]["rounds"][1]["matches"][0]["status"] == "not_played"
    assert events["a"]["currentRoundNumber"] == 1


def test_not_played_clears_reservation_without_releasing_current_game():
    hub, events = setup({"a": event(count=8)}, courts=1)
    first, reserved = current(events)["matches"]
    hub, events = command(hub, events, "assign_game", session_key="a", match_id=first["id"], court_id="c0")
    hub, events = command(hub, events, "reserve_game_for_court", session_key="a", match_id=reserved["id"], court_id="c0")
    hub, events = command(hub, events, "resolve_game", session_key="a", match_id=reserved["id"], court_id="c0", resolution="not_played")
    assert hub["courts"][0]["next"] is None
    assert hub["courts"][0]["current"]["match_id"] == first["id"]
    view = build_hub_snapshot(hub, events, now=START)["groups"][0]
    assert (view["played_matches"], view["not_played_matches"], view["pending_matches"]) == (0, 1, 1)
    hub, events = command(hub, events, "score_and_release", session_key="a", match_id=first["id"], court_id="c0", score_a=11, score_b=7)
    assert events["a"]["currentRoundNumber"] == 2


def test_not_played_remains_not_played_in_unscored_round():
    hub, events = setup({"a": event(count=4, scoring="unscored")})
    match = current(events)["matches"][0]
    hub, events = command(hub, events, "resolve_game", session_key="a", match_id=match["id"], resolution="not_played")
    assert events["a"]["rounds"][0]["matches"][0]["status"] == "not_played"
    assert events["a"]["currentRoundNumber"] == 2


def test_one_blank_score_keeps_court_wait_and_results_unchanged():
    hub, events = setup({"a": event(count=4)})
    match = current(events)["matches"][0]
    hub, events = command(hub, events, "assign_game", session_key="a", match_id=match["id"], court_id="c0")
    before = deepcopy((hub, events))
    with pytest.raises(ValueError, match="complete, non-tied"):
        command(hub, events, "score_and_release", session_key="a", match_id=match["id"], court_id="c0", score_a=11, score_b=None)
    assert (hub, events) == before


@pytest.mark.parametrize("action", ["add", "substitute"])
@pytest.mark.parametrize("attached", [False, True])
def test_direct_generator_roster_cannot_duplicate_active_profile(action, attached):
    from jupr_app.domain.adaptive_play_engine import mutate_generator_roster
    value = event(count=5, player_ids=[1, 2, 3, 4, 5])
    if not attached:
        value.pop("roundRobinHubId", None)
    before = deepcopy(value)
    with pytest.raises(ValueError, match="active or upcoming"):
        mutate_generator_roster(value, action=action, name="Duplicate profile", player_id=1,
            participant_id=value["participants"][1]["id"] if action == "substitute" else None)
    assert value == before


def test_departed_member_court_status_identifies_destination_group():
    hub, events = setup({"a": event(count=4, player_ids=[1, 2, 3, 4]), "b": event("B", count=4, player_ids=[1, 5, 6, 7])})
    source = deepcopy(person(events))
    hub, events = command(hub, events, "remove_player", **ref("a", source))
    match = current(events, "b")["matches"][0]
    hub, events = command(hub, events, "assign_game", session_key="b", match_id=match["id"], court_id="c0")
    group = next(g for g in build_hub_snapshot(hub, events, now=START)["groups"] if g["session_key"] == "a")
    displayed = next(p for p in group["players"] if p["participant_id"] == source["id"])
    assert displayed["membership_status"] == "departed"
    assert not displayed["on_court"] and displayed["playing_elsewhere"]
    assert displayed["current_court"]["session_key"] == "b"
    assert displayed["current_court"]["group_title"] == "B"
    assert displayed["current_court"]["court_id"] == "c0"
    assert group["departed_player_count"] == 1 and group["active_player_count"] == 3


def test_arrival_preserves_released_started_game_id_and_lineup():
    hub, events = setup({"a": event(count=8)})
    match = deepcopy(current(events)["matches"][0])
    hub, events = command(hub, events, "assign_game", session_key="a", match_id=match["id"], court_id="c0")
    started = deepcopy(current(events)["matches"][0])
    hub, events = command(hub, events, "release_game", court_id="c0")
    hub, events = command(hub, events, "add_player", session_key="a", name="Arrival")
    preserved = next(m for m in current(events)["matches"] if m["id"] == match["id"])
    assert preserved["sideA"] == started["sideA"] and preserved["sideB"] == started["sideB"]
    assert preserved["hubStartedAt"] == started["hubStartedAt"]
