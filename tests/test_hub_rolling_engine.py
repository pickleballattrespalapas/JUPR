from copy import deepcopy
import json

import pytest

from jupr_app.domain import adaptive_play_engine as engine
from jupr_app.domain.generator_playoffs import start_generator_playoff


def event(play_format="doubles", **kwargs):
    result = engine.start_generator_event(engine.create_generator_preview(
        generator_kind="round_robin", play_format=play_format, title="Rolling group",
        participant_names=[f"Player {number}" for number in range(9)], total_rounds=5, **kwargs,
    ))
    result["roundRobinHubId"] = "hub"
    return result


def save_current(value):
    number = value["currentRoundNumber"]
    row = next(row for row in value["rounds"] if row["number"] == number)
    return engine.save_generator_round(value, round_number=number, scores=[
        {"match_id": match["id"], "score_a": 11, "score_b": 7} for match in row["matches"]
    ])


@pytest.mark.parametrize("play_format,options", [("doubles", {}), ("singles", {}),
    ("doubles_singles", {"doubles_court_count": 1, "singles_court_count": 1})])
def test_preparation_uses_every_preceding_planned_match_once_and_preserves_future(play_format, options):
    value = event(play_format, **options)
    expected = deepcopy(value["rounds"][3])
    value["rounds"] = [row for row in value["rounds"] if row["number"] != 4]
    original = deepcopy(value)
    prepared = engine.prepare_hub_round(value, 4)
    assert value == original
    assert prepared["hubRollingEnabled"] is True
    assert prepared["currentRoundNumber"] == 1
    assert next(row for row in prepared["rounds"] if row["number"] == 4) == expected
    assert [row for row in prepared["rounds"] if row["number"] != 4] == original["rounds"]
    assert engine.prepare_hub_round(json.loads(json.dumps(value)), 4) == prepared
    assert engine.prepare_hub_round(prepared, 4) == prepared


def test_rolling_advance_retains_partial_and_released_future_games_and_all_templates():
    value = engine.prepare_hub_round(event(), 2)
    value["rounds"][1]["status"] = "active"
    value["rounds"][1]["matches"][0].update(scoreA=11, scoreB=4, status="scored", hubCompletedAt="saved")
    value["rounds"][2]["matches"][0]["hubStartedAt"] = "already dispatched"
    value = save_current(value)
    original = deepcopy(value)
    advanced = engine.advance_generator_event(value)
    assert value == original
    assert advanced["currentRoundNumber"] == 2
    assert advanced["rounds"] == original["rounds"]
    advanced = save_current(advanced)
    next_value = engine.advance_generator_event(advanced)
    assert next_value["currentRoundNumber"] == 3
    assert next_value["rounds"][2]["status"] == "active"
    assert next_value["rounds"][2]["matches"] == original["rounds"][2]["matches"]
    assert next_value["rounds"][3:] == original["rounds"][3:]


def test_rolling_advance_appends_one_missing_successor_without_replanning_past():
    value = event()
    value["rounds"] = value["rounds"][:1]
    value["totalRounds"] = 1
    value = engine.prepare_hub_round(value, 1)
    value = save_current(value)
    advanced = engine.advance_generator_event(value)
    assert len(advanced["rounds"]) == 2
    assert advanced["currentRoundNumber"] == 2
    assert advanced["rounds"][0] == value["rounds"][0]
    assert advanced["rounds"][1]["status"] == "active"


@pytest.mark.parametrize("released_marker", ["match", "round"])
def test_regeneration_never_truncates_or_replaces_protected_future_rounds(released_marker):
    value = engine.prepare_hub_round(event(), 2)
    released = value["rounds"][1]["matches"][0] if released_marker == "match" else value["rounds"][1]
    released["hubStartedAt"] = "released from court"
    value["rounds"][2]["matches"][0].update(scoreA=0)
    value["hubAssignedMatchIds"] = [value["rounds"][3]["matches"][0]["id"]]
    protected = deepcopy(value["rounds"][1:4])
    value["totalRounds"] = 1  # A legacy count must not truncate the actual rows.
    engine._regenerate_round_robin_from(value, 2)
    assert value["rounds"][1:4] == protected
    assert len(value["rounds"]) == 5


@pytest.mark.parametrize("action", ["save", "played", "skip", "reopen"])
def test_manager_cannot_finalize_future_round_before_oldest_unresolved(action):
    value = engine.prepare_hub_round(event(scoring_mode="unscored" if action == "played" else "scored"), 2)
    if action == "reopen":
        value["rounds"][1]["status"] = "skipped"
    before = deepcopy(value)
    with pytest.raises(ValueError):
        if action == "save":
            engine.save_generator_round(value, round_number=2, scores=[])
        elif action == "played":
            engine.mark_generator_round_played(value, round_number=2)
        elif action == "skip":
            engine.skip_generator_round(value, round_number=2)
        else:
            engine.reopen_generator_round(value, round_number=2)
    assert value == before


def test_finish_boundary_prevents_more_preparation_and_completes_without_advancing():
    value = engine.prepare_hub_round(event(), 1)
    value = save_current(value)
    value["hubFinishAfterRound"] = 1
    with pytest.raises(ValueError, match="finishing"):
        engine.prepare_hub_round(value, 2)
    completed = engine.advance_generator_event(value)
    assert completed["status"] == "completed"
    assert completed["currentRoundNumber"] == 1
    assert completed["rounds"] == value["rounds"]
    blocked = deepcopy(value)
    blocked["rounds"][1]["status"] = "active"
    blocked["rounds"][1]["matches"][0].update(scoreA=11, scoreB=4)
    with pytest.raises(ValueError, match="every open round"):
        engine.advance_generator_event(blocked)


def test_moneyball_and_playoff_preparation_respect_finite_schedule_and_dependencies():
    from test_moneyball_expectations import preview as moneyball_preview
    moneyball = engine.start_generator_event(moneyball_preview())
    moneyball["roundRobinHubId"] = "hub"
    assert engine.prepare_hub_round(moneyball, 2)["rounds"] == moneyball["rounds"]
    with pytest.raises(ValueError, match="fixed schedule"):
        engine.prepare_hub_round(moneyball, moneyball["totalRounds"] + 1)

    value = save_current(event())
    playoff = start_generator_playoff(value, playoff_format="top_eight")
    last = playoff["rounds"][-1]["number"]
    before = deepcopy(playoff)
    with pytest.raises(ValueError, match="semifinal scores"):
        engine.prepare_hub_round(playoff, last)
    assert playoff == before
    saved = save_current(playoff)
    prepared = engine.prepare_hub_round(saved, last)
    final = prepared["rounds"][-1]["matches"][0]
    semis = saved["rounds"][-2]["matches"]
    assert final["sideA"] == semis[0]["sideA"]
    assert final["sideB"] == semis[1]["sideA"]
    assert prepared["rounds"][:-1] == saved["rounds"][:-1]
    assert saved["rounds"][-1]["matches"][0]["sideA"] == []
    with pytest.raises(ValueError, match="fixed schedule"):
        engine.prepare_hub_round(prepared, last + 1)
