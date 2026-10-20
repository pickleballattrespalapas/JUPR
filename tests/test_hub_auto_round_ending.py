from copy import deepcopy

import pytest

from jupr_app.domain.adaptive_play_engine import (
    advance_generator_event, complete_generator_event, create_generator_preview,
    generator_event_standings, hub_auto_advance_boundary, mark_generator_round_played,
    mutate_generator_roster, save_generator_round, start_generator_event,
)
from jupr_app.domain.generator_playoffs import generator_playoff_options, start_generator_playoff
from jupr_app.domain.round_robin_hub import reconcile_hub_events, validate_hub_session_change


def waiting_event(scoring="scored"):
    event = start_generator_event(create_generator_preview(
        generator_kind="round_robin", play_format="doubles", title="Rolling group",
        participant_names=[f"Player {i}" for i in range(12)], total_rounds=5,
        scoring_mode=scoring,
    ))
    event["roundRobinHubId"] = "hub"
    if scoring == "unscored":
        event = mark_generator_round_played(event, round_number=1)
    else:
        event = save_generator_round(event, round_number=1, scores=[
            {"match_id": match["id"], "score_a": 11, "score_b": 5}
            for match in event["rounds"][0]["matches"]
        ])
    event = advance_generator_event(event)
    event["hubAutoAdvancedFromRound"] = 1
    return event


def hub():
    return {"courts": [{"id": "c1", "label": "Main", "open": True, "current": None, "next": None}],
            "groups": [{"session_key": "rr"}], "players": {}}


@pytest.mark.parametrize("scoring", ["scored", "unscored"])
def test_finish_at_untouched_boundary_preserves_roster_and_all_round_history(scoring):
    event = waiting_event(scoring)
    event = mutate_generator_roster(event, action="remove", participant_id=event["participants"][0]["id"])
    event = mutate_generator_roster(event, action="add", name="Late arrival")
    before = deepcopy(event)
    assert hub_auto_advance_boundary(event) == 1
    completed = complete_generator_event(event)
    assert event == before
    assert completed["status"] == "completed"
    assert completed["currentRoundNumber"] == 2
    assert completed["rounds"][0] == event["rounds"][0]
    assert completed["rounds"][1]["status"] == "skipped"
    assert completed["rounds"][1]["matches"] == event["rounds"][1]["matches"]
    assert completed["rounds"][2:] == event["rounds"][2:]
    for key in ("participants", "rosterByRound", "rosterRevisions"):
        assert completed.get(key) == event.get(key)
    assert "hubAutoAdvancedFromRound" not in completed
    def recorded_stats(value):
        return {row["participantId"]: {key: row[key] for key in ("matches", "wins", "losses", "pointsFor", "pointsAgainst")}
                for row in generator_event_standings(value) if row["matches"]}
    assert recorded_stats(completed) == recorded_stats(event)
    _, reconciled = reconcile_hub_events(hub(), {"rr": event}, {"rr": completed}, now="2026-10-20T04:00:00Z")
    assert reconciled["rr"]["status"] == "completed"


def test_playoffs_use_completed_boundary_and_current_roster_without_mutating_options():
    event = waiting_event()
    departed = generator_event_standings(event)[0]["participantId"]
    event = mutate_generator_roster(event, action="remove", participant_id=departed)
    event = mutate_generator_roster(event, action="add", name="Late arrival")
    before = deepcopy(event)
    options = generator_playoff_options(event)
    assert event == before
    assert options["sourceRound"] == 1
    assert not any(row["reason"] for row in options["formats"])
    assert departed not in {row["participantId"] for row in options["seeds"]}
    assert "Late arrival" not in {row["name"] for row in options["seeds"]}

    playoff = start_generator_playoff(event, playoff_format="top_eight")
    assert event == before
    assert playoff["currentRoundNumber"] == 2
    assert playoff["playoff"]["sourceRound"] == 1
    assert playoff["rounds"][0] == event["rounds"][0]
    assert all(row["stage"] == "playoff" for row in playoff["rounds"][1:])
    assert "hubAutoAdvancedFromRound" not in playoff
    for key in ("participants", "rosterByRound", "rosterRevisions"):
        assert playoff.get(key) == event.get(key)
    assert generator_event_standings(playoff) == generator_event_standings(event)
    _, reconciled = reconcile_hub_events(hub(), {"rr": event}, {"rr": playoff}, now="2026-10-20T04:00:00Z")
    assert reconciled["rr"]["playoff"] == playoff["playoff"]


@pytest.mark.parametrize("change", ["assigned", "reserved", "court_label", "score_a", "score_b", "played",
                                    "completed_at", "stale_marker", "earlier_open", "future_score", "future_played"])
def test_stale_marker_never_discards_started_or_unresolved_play(change):
    event = waiting_event()
    current = event["rounds"][1]["matches"][0]
    if change == "assigned":
        event["hubAssignedMatchIds"] = [current["id"]]
    elif change == "reserved":
        current["reservedCourtLabel"] = "Main"
    elif change == "court_label":
        current["courtLabel"] = "Main"
    elif change == "score_a":
        current["scoreA"] = 0
    elif change == "score_b":
        current["scoreB"] = 0
    elif change == "played":
        current["status"] = "played"
    elif change == "completed_at":
        current["hubCompletedAt"] = "2026-10-20T03:30:00Z"
    elif change == "stale_marker":
        event["hubAutoAdvancedFromRound"] = 2
    elif change == "earlier_open":
        event["rounds"][0]["status"] = "active"
    elif change == "future_score":
        event["rounds"][2]["matches"][0]["scoreA"] = 11
    elif change == "future_played":
        event["rounds"][2]["status"] = "played"
    before = deepcopy(event)
    assert hub_auto_advance_boundary(event) is None
    with pytest.raises(ValueError):
        complete_generator_event(event)
    with pytest.raises(ValueError):
        start_generator_playoff(event, playoff_format="top_eight")
    assert event == before


@pytest.mark.parametrize("slot", ["current", "next"])
@pytest.mark.parametrize("action", ["finish", "playoff"])
def test_actual_court_claim_still_blocks_transition_if_projection_were_stale(slot, action):
    event = waiting_event()
    state = hub()
    state["courts"][0][slot] = {"session_key": "rr", "round_number": 2,
                                  "match_id": event["rounds"][1]["matches"][0]["id"]}
    changed = complete_generator_event(event) if action == "finish" else start_generator_playoff(event, playoff_format="top_eight")
    with pytest.raises(ValueError):
        validate_hub_session_change(state, {"rr": event}, "rr", changed)


def test_standalone_ending_and_finite_format_restrictions_are_unchanged():
    event = waiting_event()
    event.pop("roundRobinHubId")
    assert hub_auto_advance_boundary(event) is None
    assert complete_generator_event(event)["status"] == "completed"
    with pytest.raises(ValueError, match="Save or skip"):
        start_generator_playoff(event, playoff_format="top_eight")
    unscored = waiting_event("unscored")
    assert all(row["reason"] == "Playoffs require scored standings." for row in generator_playoff_options(unscored)["formats"])
    from jupr_app.domain.moneyball_expectations import freeze_moneyball
    from test_moneyball_expectations import preview as moneyball_preview, save as save_moneyball
    moneyball = moneyball_preview()
    moneyball = start_generator_event(freeze_moneyball(moneyball, {p["id"]: 3 + i * .15 for i, p in enumerate(moneyball["participants"])}))
    moneyball = advance_generator_event(save_moneyball(moneyball, 1))
    moneyball.update(roundRobinHubId="hub", hubAutoAdvancedFromRound=1)
    assert hub_auto_advance_boundary(moneyball) is None
    with pytest.raises(ValueError):
        complete_generator_event(moneyball)
    assert all("Moneyball ends" in row["reason"] for row in generator_playoff_options(moneyball)["formats"])


def test_validator_rejects_replacing_started_round_with_unrelated_playoff():
    event = waiting_event()
    playoff = start_generator_playoff(event, playoff_format="top_eight")
    event["rounds"][1]["matches"][0]["scoreA"] = 7
    with pytest.raises(ValueError, match="Finish the current round"):
        validate_hub_session_change(hub(), {"rr": event}, "rr", playoff)
