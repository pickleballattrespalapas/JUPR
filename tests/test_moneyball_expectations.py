from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from jupr_app.domain import moneyball_probability
from jupr_app.domain.adaptive_play_engine import (
    create_generator_preview, start_generator_event, save_generator_round,
    advance_generator_event, complete_generator_event, generator_event_standings,
    mutate_generator_roster, skip_generator_round, select_generator_round,
)
from jupr_app.domain.moneyball_expectations import freeze_moneyball, public_event, settlement, convert_existing_moneyball, validate_score
import numpy as np
from jupr_app.domain.generator_playoffs import start_generator_playoff
from jupr_app.services.public_play_generator_service import (
    preview_public_play_generator, create_public_play_generator_session,
    public_play_generator_session_payload, build_public_play_generator_export,
)
from jupr_app.services.admin_play_generator_service import preview_play_generator
from test_public_play_generator_service import FakeSupabase


def preview(**overrides):
    options = dict(generator_kind="round_robin", play_format="doubles", title="Moneyball test",
                   participant_names=[f"Player {i}" for i in range(8)], player_ids=list(range(1, 9)),
                   total_rounds=7, event_format="moneyball_expectations")
    return create_generator_preview(**{**options, **overrides})


@pytest.fixture(scope="module")
def frozen():
    event = preview()
    return freeze_moneyball(event, {p["id"]: 3+i*.15 for i, p in enumerate(event["participants"])})


def save(event, number, a=11, b=7):
    row = next(r for r in event["rounds"] if r["number"] == number)
    return save_generator_round(event, round_number=number, scores=[
        {"match_id": m["id"], "score_a": a, "score_b": b} for m in row["matches"]])


@pytest.mark.parametrize("changes", [dict(participant_names=[f"P{i}" for i in range(9)], player_ids=list(range(9))),
    dict(play_format="singles"), dict(generator_kind="ladder"), dict(scoring_mode="unscored"),
    dict(total_rounds=8), dict(court_count=1), dict(player_ids=[]), dict(player_ids=[1]*8)])
def test_invalid_moneyball_setups_are_rejected(changes):
    with pytest.raises(ValueError):
        preview(**changes)


def test_fixed_setup_and_interim_expectations(frozen):
    event = start_generator_event(frozen)
    before = deepcopy(event["moneyballPrivate"])
    rows = generator_event_standings(event)
    assert all(r["moneyballScore"] is None and r["expectedWins"] == 0 for r in rows)
    event = save(event, 1)
    rows = generator_event_standings(event)
    assert len(rows) == 8 and sum(r["wins"] for r in rows) == 4
    assert sum(r["expectedWins"] for r in rows) == pytest.approx(4)
    assert sum(r["expectedDifferential"] for r in rows) == pytest.approx(0)
    assert all(r["matches"] == 1 and not r["moneyballFinal"] for r in rows)
    assert not settlement(event, rows)["winners"]
    assert event["moneyballPrivate"] == before
    for fn in [lambda: mutate_generator_roster(event, action="add", name="Late arrival"),
               lambda: skip_generator_round(event, round_number=2),
               lambda: complete_generator_event(event),
               lambda: start_generator_playoff(event, playoff_format="top_eight")]:
        with pytest.raises(ValueError): fn()


def test_moneyball_cannot_submit_blank_games_as_not_played(frozen):
    event = start_generator_event(frozen)
    before = deepcopy(event)
    scores = [{"match_id": m["id"], "score_a": 11, "score_b": 7} for m in event["rounds"][0]["matches"]]
    scores[-1].update(score_a=None, score_b=None)
    with pytest.raises(ValueError, match="complete fixed rotation"):
        save_generator_round(event, round_number=1, scores=scores)
    assert event == before


@pytest.mark.parametrize("fixture_name,games", [("frozen", 7), ("six_frozen", 4), ("six_nine_frozen", 6)])
def test_preview_expectations_cover_full_schedule_and_match_final_standings(fixture_name, games, request):
    frozen = request.getfixturevalue(fixture_name)
    before = deepcopy(frozen)
    public = public_event(frozen)
    expectations = public["moneyballExpectations"]
    total_matches = len(frozen["moneyballPrivate"]["config"]["matches"])
    assert frozen == before
    assert [r["participantId"] for r in expectations] == [p["id"] for p in frozen["participants"]]
    assert sum(r["expectedWins"] for r in expectations) == pytest.approx(2 * total_matches)
    assert sum(r["expectedLosses"] for r in expectations) == pytest.approx(2 * total_matches)
    assert sum(r["expectedDifferential"] for r in expectations) == pytest.approx(0)
    for row in expectations:
        assert row["scheduledGames"] == games
        assert row["expectedWins"] + row["expectedLosses"] == pytest.approx(games)
        assert 0 < row["expectedWins"] < games
    serialized = json.dumps(public)
    for private in ["moneyballPrivate", '"rating":', "margin_probabilities", "balance_logit_offsets"]:
        assert private not in serialized
    event = start_generator_event(frozen)
    assert all(r["expectedWins"] == 0 for r in generator_event_standings(event))
    for round_number in range(1, len(event["rounds"]) + 1):
        event = save(event, round_number)
        assert public_event(event)["moneyballExpectations"] == expectations
    final = {r["participantId"]: r for r in generator_event_standings(event)}
    for expected in expectations:
        actual = final[expected["participantId"]]
        assert expected["scheduledGames"] == actual["matches"]
        assert expected["expectedWins"] == pytest.approx(actual["expectedWins"])
        assert expected["expectedDifferential"] == pytest.approx(actual["expectedDifferential"])


def test_standard_preview_has_no_moneyball_expectations():
    event = preview(event_format="standard")
    assert public_event(event) == event
    assert "moneyballExpectations" not in public_event(event)


def test_fifteen_point_distribution_is_symmetric_normalized_and_target_aware():
    from pathlib import Path
    base = json.loads(Path(moneyball_probability.__file__).with_name("moneyball_model_v1.json").read_text())
    assert moneyball_probability.model_for_target(base, 11) is base
    model = moneyball_probability.model_for_target(base, 15)
    differences = [-.8, 0, .8]
    probabilities = moneyball_probability.match_probabilities(model, differences)
    support = np.array(model["margin_support"])
    wins = probabilities[:, support > 0].sum(axis=1)
    original = moneyball_probability.match_probabilities(base, differences)
    original_wins = original[:, np.array(base["margin_support"]) > 0].sum(axis=1)
    assert probabilities.sum(axis=1) == pytest.approx([1, 1, 1])
    assert probabilities[0] == pytest.approx(probabilities[2][::-1])
    assert wins[1] == pytest.approx(.5)
    assert wins[0] < original_wins[0] < .5 < original_wins[2] < wins[2]
    assert np.all(probabilities > 0) and max(support) == 15 and min(support) == -15
    # Independently count fair scoring-point paths to 15-0 and to deuce.
    assert probabilities[1, -1] == pytest.approx(.5**15)
    assert probabilities @ support == pytest.approx([-(probabilities[2] @ support), 0, probabilities[2] @ support])
    assert model["validation"]["historical_15_point_validation"] is False


@pytest.fixture(scope="module")
def converted_fifteen():
    event = start_generator_event(preview(event_format="standard", participant_names=[f"Player {i}" for i in range(5)],
                                          player_ids=list(range(1, 6)), total_rounds=5))
    event = advance_generator_event(save(event, 1, 11, 15))
    original = deepcopy(event)
    ratings = {p["id"]: 3+i*.15 for i, p in enumerate(event["participants"])}
    player_ids = {p["id"]: p["player_id"] for p in event["participants"]}
    result = convert_existing_moneyball(event, ratings, player_ids=player_ids,
                                       target_score=15, converted_at="2026-10-18T01:45:00+00:00")
    assert event == original
    for key in ["rounds", "participants", "status", "createdAt", "startedAt", "currentRoundNumber", "sourceEventUid", "ratingMode"]:
        assert result[key] == original[key]
    return result


def test_live_fifteen_conversion_keeps_games_and_matches_full_reference(converted_fifteen):
    event = deepcopy(converted_fifteen)
    expected = public_event(event)["moneyballExpectations"]
    assert event["moneyball"]["targetScore"] == 15 and event["moneyball"]["convertedAfterGames"] == 1
    assert event["moneyballPrivate"]["fairness"]["simulation_fairness_check_passed"]
    assert all(r["scheduledGames"] == 4 for r in expected)
    assert sum(r["expectedWins"] for r in expected) == pytest.approx(10)
    first = event["rounds"][0]
    assert (first["matches"][0]["scoreA"], first["matches"][0]["scoreB"]) == (11, 15)
    lock = deepcopy(event["moneyballPrivate"])
    for number, a, b in [(2, 15, 0), (3, 0, 15), (4, 17, 15), (5, 15, 13)]:
        event = advance_generator_event(save(event, number, a, b))
    assert event["status"] == "completed"
    assert public_event(event)["moneyballExpectations"] == expected
    assert event["moneyballPrivate"] == lock
    by_id = {m["id"]: m for r in event["rounds"] for m in r["matches"]}
    scored = deepcopy(lock["config"])
    for m in scored["matches"]:
        m.update(score1=by_id[m["id"]]["scoreA"], score2=by_id[m["id"]]["scoreB"])
    reference = moneyball_probability.score(scored, lock)["standings"]
    rows = generator_event_standings(event)
    assert [r["participantId"] for r in rows] == [r["player_id"] for r in reference]
    for row, ref in zip(rows, reference):
        assert row["moneyballScore"] == pytest.approx(ref["moneyball_score"])
        assert row["expectedDifferential"] == pytest.approx(ref["expected_point_differential"])
    assert sum(w["payoutMxn"] for w in settlement(event, rows)["winners"]) == 500
    event["moneyball"]["targetScore"] = 11
    with pytest.raises(ValueError, match="matchups"):
        generator_event_standings(event)


@pytest.mark.parametrize("a,b,valid", [(15, 0, True), (11, 15, True), (17, 15, True), (15, 13, True),
                                      (11, 7, False), (15, 14, False), (16, 12, False), (14, 12, False), (True, 15, False)])
def test_fifteen_score_rules(a, b, valid):
    if valid:
        validate_score(a, b, 15)
    else:
        with pytest.raises(ValueError): validate_score(a, b, 15)


def test_conversion_rejects_changed_profiles_incomplete_ratings_and_incompatible_scores():
    event = start_generator_event(preview(event_format="standard", participant_names=[f"P{i}" for i in range(5)],
                                          player_ids=list(range(1, 6)), total_rounds=5))
    event = save(event, 1, 15, 11)
    ratings = {p["id"]: 3.5 for p in event["participants"]}
    ids = {p["id"]: p["player_id"] for p in event["participants"]}
    for level, links, goal in [(ratings, ids, 11), ({}, ids, 15), (ratings, {**ids, "p-1": 99}, 15),
                               (ratings, {**ids, "p-1": 2}, 15), (ratings, ids, 21)]:
        with pytest.raises(ValueError):
            convert_existing_moneyball(event, level, player_ids=links, target_score=goal, converted_at="now")


def test_final_scores_match_prototype_and_score_corrections(frozen):
    event = start_generator_event(frozen)
    for number in range(1, 8):
        event = save(event, number, 11, number)
        event = advance_generator_event(event)
    assert event["status"] == "completed" and len(event["rounds"]) == 7
    rows = generator_event_standings(event)
    lock = event["moneyballPrivate"]
    played = deepcopy(lock["config"])
    by_id = {m["id"]: m for r in event["rounds"] for m in r["matches"]}
    for m in played["matches"]:
        m.update(score1=by_id[m["id"]]["scoreA"], score2=by_id[m["id"]]["scoreB"])
    reference = moneyball_probability.score(played, lock)["standings"]
    assert [r["participantId"] for r in rows] == [r["player_id"] for r in reference]
    for row, expected in zip(rows, reference):
        assert row["moneyballScore"] == pytest.approx(expected["moneyball_score"], abs=1e-10)
        assert row["expectedWins"] == pytest.approx(expected["expected_wins"])
        assert row["expectedDifferential"] == pytest.approx(expected["expected_point_differential"])
    payout = settlement(event, rows)
    assert payout["final"] and sum(w["payoutMxn"] for w in payout["winners"]) == 800
    for fn in [lambda: advance_generator_event(event), lambda: select_generator_round(event, round_number=7)]:
        with pytest.raises(ValueError): fn()
    changed = save(event, 1, 7, 11)
    assert changed["moneyballPrivate"] == lock
    assert generator_event_standings(changed) != rows


@pytest.mark.parametrize("a,b", [(11,10),(11,11),(10,8),(12,8),(True,3),(11,-1)])
def test_invalid_finished_scores(frozen,a,b):
    with pytest.raises(ValueError, match="Moneyball games"):
        save(start_generator_event(frozen), 1, a,b)


def test_deuce_and_frozen_integrity(frozen):
    event = save(start_generator_event(frozen), 1, 15,13)
    assert all(abs(r["differential"]) == 2 for r in generator_event_standings(event))
    edited = deepcopy(event)
    edited["moneyballPrivate"]["balance_logit_offsets"][0] += 1
    with pytest.raises(ValueError, match="setup changed"): generator_event_standings(edited)
    edited = deepcopy(event)
    edited["rounds"][0]["matches"][0]["sideA"].reverse()
    with pytest.raises(ValueError, match="matchups"): generator_event_standings(edited)
    event["name"] = "Renamed Moneyball"
    assert len(generator_event_standings(event)) == 8


def test_ties_split_pool_without_rounding_influencing_rank(frozen):
    event = start_generator_event(frozen)
    for number in range(1,8): event = save(event, number)
    event = complete_generator_event(event)
    rows = generator_event_standings(event)
    for row in rows: row["rank"] = 1
    payout = settlement(event,rows)
    assert len(payout["winners"]) == 8 and all(w["payoutMxn"] == 100 for w in payout["winners"])


@pytest.mark.parametrize("fixture_name", ["frozen", "six_frozen", "six_nine_frozen"])
def test_public_preview_start_export_hide_ratings_and_freeze_database_values(fixture_name, request, monkeypatch):
    frozen = request.getfixturevalue(fixture_name)
    count = len(frozen["participants"])
    # Reuse independently prepared synthetic lock while exercising real services.
    monkeypatch.setattr("jupr_app.domain.moneyball_expectations._prepare", lambda _: deepcopy(frozen["moneyballPrivate"]))
    db = FakeSupabase()
    db.db["players"] = [{"id": i+1, "name": f"Player {i}", "club_id": "club", "rating": (3+i*.15)*400} for i in range(count)]
    setup = dict(club_id="club", generator_kind="round_robin", play_format="doubles",title="Moneyball test",
                 participant_names=[p["name"] for p in db.db["players"]],
                 participant_player_ids={p["name"]:p["id"] for p in db.db["players"]},
                 total_rounds=frozen["totalRounds"],court_count=0,event_format="moneyball_expectations")
    result = preview_public_play_generator(db,**setup)["preview"]
    assert "moneyballPrivate" not in result
    assert result["moneyballExpectations"] == public_event(frozen)["moneyballExpectations"]
    staff_setup = {key: value for key, value in setup.items() if key != "participant_player_ids"}
    staff = preview_play_generator(db, **staff_setup, player_ids=list(range(1, count + 1)))["preview"]
    assert staff["moneyballExpectations"] == result["moneyballExpectations"]
    assert staff["previewFingerprint"] == result["previewFingerprint"]
    assert "moneyballPrivate" not in staff
    session = create_public_play_generator_session(db,**setup,preview_fingerprint=result["previewFingerprint"],
        idempotency_key="moneyball-test-0001",requester_hash="a"*64,token_secret="x"*48)["session"]
    stored = db.db["live_sessions"][0]
    assert "moneyballPrivate" in stored["state"]["page_state"]["event"]
    assert "moneyballPrivate" not in session["event"]
    assert session["event"]["moneyballExpectations"] == result["moneyballExpectations"]
    db.db["players"][0]["rating"] += 200
    assert public_play_generator_session_payload(stored)["standings"] == session["standings"]
    assert public_play_generator_session_payload(stored)["event"]["moneyballExpectations"] == result["moneyballExpectations"]
    replay = create_public_play_generator_session(db,**setup,preview_fingerprint=result["previewFingerprint"],
        idempotency_key="moneyball-test-0001",requester_hash="a"*64,token_secret="x"*48)
    assert replay["idempotent_replay"] and replay["session"] == session
    assert len(db.db["live_sessions"]) == 1
    export = build_public_play_generator_export(db,club_id="club",session_key=session["session_key"],export_format="json")
    assert "balance_logit_offsets" not in export["content"] and "moneyballPrivate" not in export["content"]
    assert public_event(stored["state"]["page_state"]["event"]) == session["event"]


def test_missing_or_foreign_ratings_never_get_defaulted():
    db = FakeSupabase()
    names = [f"Player {i}" for i in range(8)]
    db.db["players"] = [{"id": i+1,"name":name,"club_id":"club","rating":1200 if i else None} for i,name in enumerate(names)]
    with pytest.raises(ValueError,match="starting rating"):
        preview_public_play_generator(db,club_id="club",generator_kind="round_robin",play_format="doubles",
            title="Moneyball",participant_names=names,participant_player_ids={name:i+1 for i,name in enumerate(names)},
            total_rounds=7,court_count=0,event_format="moneyball_expectations")


@pytest.mark.parametrize("count", [n for n in range(4,41) if n % 4 in {0,1}])
def test_every_supported_perfect_size_geometry(count):
    from jupr_app.domain.perfect_round_robin import perfect_doubles_schedule
    from jupr_app.domain.moneyball_expectations import config_from_event
    schedule = perfect_doubles_schedule([f"p-{i}" for i in range(count)])
    event = preview(participant_names=[f"Player {i}" for i in range(count)],
                    player_ids=list(range(1,count+1)), total_rounds=len(schedule))
    config = config_from_event(event, {p["id"]:3.5 for p in event["participants"]})
    moneyball_probability.validate_config(config)
    a, support, probabilities, pct, pmfs, expected = moneyball_probability.distributions(config,
        __import__("json").loads(__import__("pathlib").Path(moneyball_probability.__file__).with_name("moneyball_model_v1.json").read_text()))
    assert a.shape == (count*(count-1)//4,count)
    assert pmfs.shape == (count,22*(count-1)+1)
    assert all(abs(p.sum()-1)<1e-12 for p in pmfs)
    assert len(event["rounds"]) == (count-1 if count%4==0 else count)


def test_substitution_disqualifies_both_and_preserves_everyone_elses_expectations(frozen):
    event = save(start_generator_event(frozen),1)
    event = advance_generator_event(event)
    before = deepcopy(event)
    changed = mutate_generator_roster(event, action="substitute",participant_id="p-1",name="Substitute",player_id=99,substitute_scope="rest")
    assert changed["moneyballPrivate"] == before["moneyballPrivate"]
    assert changed["rounds"][0] == before["rounds"][0]
    assert changed["moneyballDisqualified"] == ["p-1","p-sub-1"]
    for number in range(2,8):
        changed = save(changed,number,11,7)
        before = save(before,number,11,7)
    changed = complete_generator_event(changed)
    before = complete_generator_event(before)
    rows = {r["participantId"]:r for r in generator_event_standings(changed)}
    original = {r["participantId"]:r for r in generator_event_standings(before)}
    assert rows["p-1"]["matches"] == 1 and rows["p-sub-1"]["matches"] == 6
    for pid in ["p-1","p-sub-1"]:
        assert not rows[pid]["moneyballEligible"] and rows[pid]["rank"] == 0
        assert rows[pid]["moneyballScore"] is None
    for pid in original.keys()-{"p-1"}:
        for key in ["moneyballScore","expectedWins","expectedDifferential","wins","differential"]:
            assert rows[pid][key] == original[pid][key]
    winners = settlement(changed,list(rows.values()))["winners"]
    assert not {w["participantId"] for w in winners} & {"p-1","p-sub-1"}
    assert sum(w["payoutMxn"] for w in winners)==800


def test_temporary_and_chained_substitutes_cannot_win_or_change_saved_games(frozen):
    event = start_generator_event(frozen)
    changed = mutate_generator_roster(event,action="substitute",participant_id="p-1",name="Temporary",player_id=90,substitute_scope="round")
    assert "p-sub-1" in str(changed["rounds"][0]["matches"])
    assert changed["rounds"][1] == event["rounds"][1]
    with pytest.raises(ValueError,match="different player"):
        mutate_generator_roster(changed,action="substitute",participant_id="p-2",name="Duplicate",player_id=90)
    rest = mutate_generator_roster(event,action="substitute",participant_id="p-1",name="First",player_id=91)
    second = mutate_generator_roster(rest,action="substitute",participant_id="p-sub-1",name="Second",player_id=92)
    assert set(second["moneyballDisqualified"]) == {"p-1","p-sub-1","p-sub-2"}
    assert second["participants"][-1]["moneyballSlot"] == "p-1"
    for n in range(1,8):second=save(second,n)
    second=complete_generator_event(second)
    assert all(not r["moneyballEligible"] for r in generator_event_standings(second) if r["participantId"] in second["moneyballDisqualified"])


def test_pool_scales_with_starting_lineup_and_no_substitute_contributions():
    from jupr_app.domain.moneyball_expectations import _prepare
    event=preview(participant_names=[f"Player {i}" for i in range(5)],player_ids=list(range(1,6)),total_rounds=5)
    event=freeze_moneyball(event,{p["id"]:3.5 for p in event["participants"]})
    event=start_generator_event(event)
    assert event["moneyball"]["prizePoolMxn"]==500 and event["moneyball"]["requiredRounds"]==5
    event=mutate_generator_roster(event,action="substitute",participant_id=event["rounds"][0]["matches"][0]["sideA"][0],name="Guest")
    for n in range(1,6):event=save(event,n)
    event=complete_generator_event(event)
    assert settlement(event,generator_event_standings(event))["totalGames"]==5
    assert sum(w["payoutMxn"] for w in settlement(event,generator_event_standings(event))["winners"])==500


@pytest.fixture(scope="module")
def six_frozen():
    event = preview(participant_names=[f"Player {i}" for i in range(6)],
                    player_ids=list(range(1, 7)), total_rounds=6)
    return freeze_moneyball(event, {p["id"]: 3+i*.15 for i, p in enumerate(event["participants"])})


@pytest.fixture(scope="module")
def six_nine_frozen():
    event = preview(participant_names=[f"Player {i}" for i in range(6)],
                    player_ids=list(range(1, 7)), total_rounds=9)
    return freeze_moneyball(event, {p["id"]: 3+i*.15 for i, p in enumerate(event["participants"])})


@pytest.mark.parametrize("fixture_name,rounds,games_each,partners_each,opponent_counts", [
    ("six_frozen", 6, 4, 4, {1, 2}), ("six_nine_frozen", 9, 6, 5, {2, 3}),
])
def test_six_player_rotation_geometry_and_bye_spacing(fixture_name, rounds, games_each, partners_each, opponent_counts, request):
    event = request.getfixturevalue(fixture_name)
    assert len(event["rounds"]) == rounds and not event["perfectRotation"]
    assert event["fixedScheduleRoundCount"] == rounds
    ids = [p["id"] for p in event["participants"]]
    partners, opponents, games, byes = Counter(), Counter(), Counter(), {pid: [] for pid in ids}
    for row in event["rounds"]:
        assert len(row["matches"]) == 1 and len(row["byeParticipantIds"]) == 2
        for pid in row["byeParticipantIds"]:
            byes[pid].append(row["number"])
        match = row["matches"][0]
        a, b = match["sideA"], match["sideB"]
        games.update(a+b)
        partners.update(tuple(sorted(team)) for team in (a,b))
        opponents.update(tuple(sorted((x,y))) for x in a for y in b)
    assert set(games.values()) == {games_each}
    assert len(partners) == 6 * partners_each // 2
    assert len(opponents) == 15 and set(opponents.values()) == opponent_counts
    for pid in ids:
        assert sum(pid in pair for pair in partners) == partners_each
        assert sum(count == 2 for pair, count in partners.items() if pid in pair) == games_each - partners_each
        assert [(number - 1) // 3 for number in byes[pid]] == list(range(rounds // 3))
        assert all(right - left > 1 for left, right in zip(byes[pid], byes[pid][1:]))
    assert f"{rounds} rounds" in event["moneyball"]["rotationSummary"]
    assert len(event["moneyballPrivate"]["player_total_margin_pmf"][0]) == 2 * 11 * games_each + 1
    assert event["moneyballPrivate"]["fairness"]["simulation_fairness_check_passed"]


@pytest.mark.parametrize("change", ["missing", "duplicate_round", "repeated_partner", "byes"])
@pytest.mark.parametrize("fixture_name", ["six_frozen", "six_nine_frozen"])
def test_six_player_probability_validation_rejects_unbalanced_rotations(fixture_name, change, request):
    config = deepcopy(request.getfixturevalue(fixture_name)["moneyballPrivate"]["config"])
    if change == "missing":
        config["matches"].pop()
    elif change == "duplicate_round":
        config["matches"][0]["round"] = 2
    elif change == "repeated_partner":
        config["matches"][1]["team1"] = list(config["matches"][0]["team1"])
    else:
        config["matches"][2]["round"], config["matches"][3]["round"] = 4, 3
    with pytest.raises(ValueError, match="Six-player"):
        moneyball_probability.validate_config(config)


@pytest.mark.parametrize("changes", [dict(total_rounds=5), dict(total_rounds=8), dict(court_count=2)])
def test_six_player_setup_requires_the_full_one_court_rotation(changes):
    with pytest.raises(ValueError, match="full fixed rotation"):
        preview(**{**dict(participant_names=[f"P{i}" for i in range(6)],
                          player_ids=list(range(1,7)), total_rounds=6), **changes})


@pytest.mark.parametrize("fixture_name,games_each", [("six_frozen", 4), ("six_nine_frozen", 6)])
def test_six_player_scoring_completion_correction_and_substitution(fixture_name, games_each, request):
    six_frozen = json.loads(json.dumps(request.getfixturevalue(fixture_name)))
    rounds = six_frozen["totalRounds"]
    event = start_generator_event(six_frozen)
    with pytest.raises(ValueError, match="all scheduled"):
        complete_generator_event(event)
    with pytest.raises(ValueError, match="fixed rotation"):
        skip_generator_round(event, round_number=1)
    for number in range(1, rounds + 1):
        event = save(event, number, 11, 7 if number % 2 else 4)
        if number == 6 and rounds == 9:
            with pytest.raises(ValueError, match="all scheduled"):
                complete_generator_event(event)
            assert not settlement(event, generator_event_standings(event))["winners"]
    event = complete_generator_event(event)
    rows = generator_event_standings(event)
    assert all(r["matches"] == games_each and r["moneyballFinal"] for r in rows)
    lock = six_frozen["moneyballPrivate"]
    played = deepcopy(lock["config"])
    for match, row in zip(played["matches"], event["rounds"]):
        match.update(score1=row["matches"][0]["scoreA"], score2=row["matches"][0]["scoreB"])
    reference = {r["player_id"]: r for r in moneyball_probability.score(played,lock)["standings"]}
    for row in rows:
        assert row["moneyballScore"] == pytest.approx(reference[row["participantId"]]["moneyball_score"])
        assert row["losses"] == reference[row["participantId"]]["losses"]
    assert sum(w["payoutMxn"] for w in settlement(event,rows)["winners"]) == 600
    corrected = save(event,1,11,9)
    assert corrected["moneyballPrivate"] == lock
    assert generator_event_standings(corrected) != rows
    # A replacement inherits only the remaining games; both players are DQ.
    changed = mutate_generator_roster(start_generator_event(six_frozen), action="substitute",
        participant_id="p-1", name="Substitute", player_id=99)
    for number in range(1, rounds + 1):
        changed = save(changed,number)
    changed = complete_generator_event(changed)
    rows = generator_event_standings(changed)
    assert changed["moneyballPrivate"] == lock
    assert {r["participantId"] for r in rows if not r["moneyballEligible"]} == {"p-1","p-sub-1"}
    result = settlement(changed,rows)
    assert result["prizePoolMxn"] == 600 and result["totalGames"] == rounds
    assert not {w["participantId"] for w in result["winners"]} & {"p-1","p-sub-1"}
    assert sum(w["payoutMxn"] for w in result["winners"]) == 600


def test_six_player_rotation_choice_changes_the_frozen_preview(six_frozen, six_nine_frozen):
    assert six_nine_frozen["rounds"][:6] == six_frozen["rounds"]
    assert six_nine_frozen["previewFingerprint"] != six_frozen["previewFingerprint"]
    assert six_nine_frozen["moneyballPrivate"]["lock_sha256"] != six_frozen["moneyballPrivate"]["lock_sha256"]


def test_legacy_frozen_results_and_payout_are_unchanged():
    fixture = json.loads(Path(__file__).with_name("fixtures").joinpath("moneyball_v2_frozen.json").read_text())
    event = fixture["event"]
    before = deepcopy(event)
    rows = generator_event_standings(event)
    assert rows == fixture["standings"]
    assert settlement(event, rows) == fixture["settlement"]
    assert event == before
    assert "scoringSummary" not in public_event(event)["moneyball"]
    # An old, unfinished session also stays on its frozen points-only formula.
    event["status"] = "active"
    event["rounds"][-1]["status"] = "preview"
    interim = generator_event_standings(event)
    assert all(not row["moneyballFinal"] and "moneyballScoringSummary" not in row for row in interim)
    event["rounds"][-1]["status"] = "saved"
    event["status"] = "completed"
    assert generator_event_standings(event) == fixture["standings"]


def test_new_formula_prefers_winning_games_over_fewer_blowout_wins():
    event = preview(title="Wins priority")
    event = freeze_moneyball(event, {p["id"]: 3.5 for p in event["participants"]})
    assert event["moneyballPrivate"]["version"] == "fair-moneyball-event-v3"
    assert event["moneyballPrivate"]["scoring"] == {
        "method": "wins_margin_percentile_blend", "wins_weight": .7, "margin_weight": .3,
    }
    event = start_generator_event(event)
    scores = iter([(11, 0), (6, 11), (11, 3), (11, 2), (11, 3), (7, 11), (11, 0),
                   (8, 11), (11, 0), (8, 11), (11, 7), (6, 11), (11, 7), (8, 11)])
    for row in event["rounds"]:
        round_scores = []
        for match in row["matches"]:
            a, b = next(scores)
            round_scores.append({"match_id": match["id"], "score_a": a, "score_b": b})
        event = save_generator_round(event, round_number=row["number"], scores=round_scores)
    event = complete_generator_event(event)
    rows = generator_event_standings(event)
    assert (rows[0]["name"], rows[0]["wins"], rows[0]["differential"]) == ("Player 3", 6, 19)
    assert (rows[1]["name"], rows[1]["wins"], rows[1]["differential"]) == ("Player 0", 5, 42)
    assert settlement(event, rows)["winners"][0]["name"] == "Player 3"
    assert all(row["moneyballScoringSummary"] == public_event(event)["moneyball"]["scoringSummary"] for row in rows)
    assert "70% win performance" in rows[0]["moneyballScoringSummary"]
    # An extra point in a loss still improves the player's score without a win.
    first = event["rounds"][0]
    loser = first["matches"][0]["sideB"][0]
    corrected = save_generator_round(event, round_number=1, scores=[
        {"match_id": first["matches"][0]["id"], "score_a": 11, "score_b": 1},
        {"match_id": first["matches"][1]["id"], "score_a": 6, "score_b": 11},
    ])
    before = next(row for row in rows if row["participantId"] == loser)
    after = next(row for row in generator_event_standings(corrected) if row["participantId"] == loser)
    assert after["wins"] == before["wins"]
    assert after["moneyballScore"] > before["moneyballScore"]
    assert corrected["moneyballPrivate"] == event["moneyballPrivate"]


def test_exact_win_distribution_for_unequal_matchups():
    # Three independent games with win chances .2, .5, .8 have masses
    # [.08, .42, .42, .08] for 0, 1, 2, 3 wins, including for the opponents.
    a = np.array([[1, -1], [1, -1], [1, -1]])
    support = np.array([-2, 2])
    probabilities = np.array([[.8, .2], [.5, .5], [.2, .8]])
    actual = moneyball_probability.win_distributions(a, support, probabilities)
    assert actual == pytest.approx(np.array([[.04, .29, .71, .96]] * 2))


def test_simulation_and_final_scorer_use_identical_joint_outcomes(frozen):
    lock = frozen["moneyballPrivate"]
    config, model = lock["config"], lock["model"]
    a, support, probabilities, pct, _, _ = moneyball_probability.distributions(config, model)
    number, seed = 20, 9823
    totals, wins, _, z = moneyball_probability.simulate(
        a, support, probabilities, pct, number, seed,
        win_percentiles=np.array(lock["player_win_percentiles"]), scoring=lock["scoring"])
    # Reconstruct the sampled shared game outcomes independently, then feed
    # completed scores through the actual event scorer.
    rng = np.random.default_rng(seed)
    margins = []
    for probabilities_for_match in probabilities:
        cdf = np.cumsum(probabilities_for_match)
        cdf[-1] = 1
        margins.append(support[np.searchsorted(cdf, rng.random(number), side="right")])
    for sample in range(number):
        played = deepcopy(config)
        for j, match in enumerate(played["matches"]):
            margin = int(margins[j][sample])
            match.update(score1=11 if margin > 0 else 11+margin,
                         score2=11-margin if margin > 0 else 11)
        rows = {r["player_id"]: r for r in moneyball_probability.score(played, lock)["standings"]}
        for i, player in enumerate(config["players"]):
            row = rows[player["id"]]
            assert row["wins"] == wins[sample, i]
            assert row["actual_point_differential"] == totals[sample, i]
            assert row["moneyball_score"] == pytest.approx(100/(1+np.exp(-z[sample, i]-lock["balance_logit_offsets"][i])))


@pytest.mark.parametrize("version,scoring", [
    ("fair-moneyball-event-v99", None), ("fair-moneyball-event-v3", None),
    ("fair-moneyball-event-v3", {"method": "wins_margin_percentile_blend", "wins_weight": .3, "margin_weight": .7}),
    ("fair-moneyball-event-v2", {"method": "wins_margin_percentile_blend", "wins_weight": .7, "margin_weight": .3}),
])
def test_invalid_scoring_versions_fail_closed(frozen, version, scoring):
    event = deepcopy(frozen)
    lock = event["moneyballPrivate"]
    lock["version"] = version
    lock.pop("scoring", None)
    if scoring is not None:
        lock["scoring"] = scoring
    lock.pop("lock_sha256")
    lock["lock_sha256"] = moneyball_probability.digest(lock)
    with pytest.raises(ValueError, match="scoring"):
        generator_event_standings(event)


@pytest.mark.parametrize("target", [11, 15])
def test_wide_rating_four_player_groups_pass_independent_balance_audit(target):
    event = preview(participant_names=[f"Player {i}" for i in range(4)],
                    player_ids=[1, 2, 3, 4], total_rounds=3)
    frozen = freeze_moneyball(event, {p["id"]: 3+2*i/3 for i, p in enumerate(event["participants"])}, target_score=target)
    audit = frozen["moneyballPrivate"]["fairness"]
    assert audit["audit_simulations"] == 1000000
    assert audit["max_audit_deviation_percentage_points"] <= .5
    assert audit["simulation_fairness_check_passed"]
