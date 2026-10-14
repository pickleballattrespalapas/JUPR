from copy import deepcopy

import pytest

from jupr_app.domain import moneyball_probability
from jupr_app.domain.adaptive_play_engine import (
    create_generator_preview, start_generator_event, save_generator_round,
    advance_generator_event, complete_generator_event, generator_event_standings,
    mutate_generator_roster, skip_generator_round, select_generator_round,
)
from jupr_app.domain.moneyball_expectations import freeze_moneyball, public_event, settlement
from jupr_app.domain.generator_playoffs import start_generator_playoff
from jupr_app.services.public_play_generator_service import (
    preview_public_play_generator, create_public_play_generator_session,
    public_play_generator_session_payload, build_public_play_generator_export,
)
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


def test_public_preview_start_export_hide_ratings_and_freeze_database_values(frozen,monkeypatch):
    # Reuse independently prepared synthetic lock while exercising real services.
    monkeypatch.setattr("jupr_app.domain.moneyball_expectations._prepare", lambda _: deepcopy(frozen["moneyballPrivate"]))
    db = FakeSupabase()
    db.db["players"] = [{"id": i+1, "name": f"Player {i}", "club_id": "club", "rating": (3+i*.15)*400} for i in range(8)]
    setup = dict(club_id="club", generator_kind="round_robin", play_format="doubles",title="Moneyball test",
                 participant_names=[p["name"] for p in db.db["players"]],
                 participant_player_ids={p["name"]:p["id"] for p in db.db["players"]},
                 total_rounds=7,court_count=0,event_format="moneyball_expectations")
    result = preview_public_play_generator(db,**setup)["preview"]
    assert "moneyballPrivate" not in result
    session = create_public_play_generator_session(db,**setup,preview_fingerprint=result["previewFingerprint"],
        idempotency_key="moneyball-test-0001",requester_hash="a"*64,token_secret="x"*48)["session"]
    stored = db.db["live_sessions"][0]
    assert "moneyballPrivate" in stored["state"]["page_state"]["event"]
    assert "moneyballPrivate" not in session["event"]
    db.db["players"][0]["rating"] += 200
    assert public_play_generator_session_payload(stored)["standings"] == session["standings"]
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
