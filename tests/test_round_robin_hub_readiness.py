"""Player-level readiness stays accurate as courts and earlier games clear."""
from copy import deepcopy

from jupr_app.domain.round_robin_hub import build_hub_snapshot
from test_round_robin_hub import START, LATER, command, current_matches, event, score, setup


def planned_players(snapshot, key="a"):
    group = next(group for group in snapshot["groups"] if group["session_key"] == key)
    return [player for match in group["next_round_matches"] for player in match["side_a"] + match["side_b"]]


def test_next_plan_names_ready_players_and_the_court_of_each_busy_player_after_score():
    state, events = setup()
    state, events = command(state, events, "auto_fill_courts")
    playing = build_hub_snapshot(state, events, now=START)["courts"][1]["current"]
    busy = {p["id"] for p in playing["side_a"] + playing["side_b"]}
    original_plan = deepcopy(events["a"]["rounds"][1])
    state, events = score(state, events, "c0")
    before = deepcopy((state, events))
    players = planned_players(build_hub_snapshot(state, events, now=LATER))
    assert len(players) == 8
    assert sum(p["readiness"]["ready"] for p in players) == 4
    for player in players:
        status = player["readiness"]
        if player["id"] in busy:
            assert status["current_court"] == {"session_key": "a", "group_title": "A", "court_id": "c1", "court_label": "Court 1"}
            assert status["waiting_round_number"] == 1
            assert not status["ready"]
        else:
            assert status["ready"]
            assert status["current_court"] is None
            assert status["waiting_round_number"] is None
    assert events["a"]["rounds"][1] == original_plan
    assert (state, events) == before, "Reading readiness must not mutate games or court state"
    state, events = score(state, events, "c1")
    snapshot = build_hub_snapshot(state, events, now=LATER)
    assert all(p["readiness"]["ready"] for m in snapshot["eligible_queue"] for p in m["side_a"] + m["side_b"])


def test_waiting_for_an_unplayed_earlier_game_is_distinct_from_playing():
    state, events = setup()
    snapshot = build_hub_snapshot(state, events, now=START)
    for player in planned_players(snapshot):
        assert not player["readiness"]["ready"]
        assert player["readiness"]["waiting_round_number"] == 1
        assert player["readiness"]["current_court"] is None
    # Player readiness is independent of whether a court is currently open.
    for court in state["courts"]:
        court["open"] = False
    snapshot = build_hub_snapshot(state, events, now=START)
    current = [m for m in snapshot["blocked_queue"] if m["round_number"] == 1]
    assert current
    assert all(p["readiness"]["ready"] for m in current for p in m["side_a"] + m["side_b"])


def test_readiness_follows_shared_profile_to_another_round_robin_court():
    state, events = setup({"a": event("A", 4, player_ids=[1, 2, 3, 4]), "b": event("B", 4, player_ids=[1, 5, 6, 7])})
    state, events = command(state, events, "assign_next_court", session_key="a", match_id=current_matches(events)[0]["id"])
    snapshot = build_hub_snapshot(state, events, now=START)
    other = next(m for m in snapshot["blocked_queue"] if m["session_key"] == "b" and m["round_number"] == 1)
    for player in other["side_a"] + other["side_b"]:
        status = player["readiness"]
        if player["identity"] == "player:1":
            assert not status["ready"]
            assert status["current_court"]["group_title"] == "A"
            assert status["current_court"]["court_label"] == "Court 0"
        else:
            assert status["ready"]


def test_reservation_becomes_playing_in_readiness_when_the_court_releases():
    state, events = setup(courts=1)
    first, second = current_matches(events)
    state, events = command(state, events, "assign_game", session_key="a", match_id=first["id"], court_id="c0")
    state, events = command(state, events, "reserve_game_for_court", session_key="a", match_id=second["id"], court_id="c0")
    reserved = set(second["sideA"] + second["sideB"])
    for player in planned_players(build_hub_snapshot(state, events, now=START)):
        if player["id"] in reserved:
            assert not player["readiness"]["ready"]
            assert player["readiness"]["reserved_court"]["court_id"] == "c0"
            assert player["readiness"]["current_court"] is None
    state, events = score(state, events, "c0")
    for player in planned_players(build_hub_snapshot(state, events, now=LATER)):
        if player["id"] in reserved:
            assert player["readiness"]["current_court"]["court_id"] == "c0"
            assert player["readiness"]["reserved_court"] is None


def test_pauses_and_unresolved_identities_never_appear_ready():
    state, events = setup({"a": event("Same", 4), "b": event("Same", 4)})
    snapshot = build_hub_snapshot(state, events, now=START)
    assert all(p["readiness"]["identity_review"] and not p["readiness"]["ready"]
        for m in snapshot["blocked_queue"] for p in m["side_a"] + m["side_b"])
    state, events = command(state, events, "resolve_identities", resolution="all_different")
    ready = build_hub_snapshot(state, events, now=START)["eligible_queue"][0]
    paused = ready["side_a"][0]["identity"]
    state, events = command(state, events, "pause_player", identity=paused)
    snapshot = build_hub_snapshot(state, events, now=START)
    player = next(p for m in snapshot["blocked_queue"] for p in m["side_a"] + m["side_b"] if p["identity"] == paused)
    assert player["readiness"]["paused"] and not player["readiness"]["ready"]
    assert not player["readiness"]["identity_review"]
