"""Correct saved scores without replaying games or changing later matchups."""

from copy import deepcopy

import pytest

from test_generator_round_controls_service import GeneratorSession
from test_public_play_generator_service import matches
from test_generator_playoffs import scored_event
from jupr_app.domain.adaptive_play_engine import (
    advance_generator_event, create_generator_preview, save_generator_round, start_generator_event,
)
from jupr_app.domain.generator_playoffs import start_generator_playoff
from jupr_app.services.public_live_operation_service import PublicLiveConflictError


def scores_for(round_row, a=8, b=11):
    return [{"match_id": m["id"], "score_a": a, "score_b": b} for m in matches(round_row)]


@pytest.mark.parametrize("audience", ["admin", "public"])
@pytest.mark.parametrize("completed", [False, True])
def test_edit_old_scores_updates_standings_once_and_preserves_later_games(audience, completed):
    game = GeneratorSession(audience)
    game.call("save", round_number=1, scores=scores_for(game.session["event"]["rounds"][0], 11, 8))
    game.call("advance")
    game.score(2)
    if completed:
        game.call("complete")
    before = deepcopy(game.session)
    first_round = before["event"]["rounds"][0]
    first_match = matches(first_round)[0]
    game.call("save", round_number=1, scores=scores_for(first_round))
    result = game.reload()

    assert result["status"] == before["status"]
    assert result["event"]["status"] == before["event"]["status"]
    assert result["current_round_number"] == before["current_round_number"] == 2
    assert result["event"]["rounds"][1:] == before["event"]["rounds"][1:]
    assert result["event"]["participants"] == before["event"]["participants"]
    corrected = result["event"]["rounds"][0]
    assert corrected["status"] == "saved"
    assert corrected["byeParticipantIds"] == first_round["byeParticipantIds"]
    assert matches(corrected)[0] == {**first_match, "scoreA": 8, "scoreB": 11}
    old_stats = {r["participantId"]: r for r in before["standings"]}
    new_stats = {r["participantId"]: r for r in result["standings"]}
    for pid, old in old_stats.items():
        new = new_stats[pid]
        assert new["matches"] == old["matches"]
        if pid in first_match["sideA"]:
            assert new["wins"] == old["wins"] - 1
            assert new["losses"] == old["losses"] + 1
            assert new["pointsFor"] == old["pointsFor"] - 3
            assert new["pointsAgainst"] == old["pointsAgainst"] + 3
        elif pid in first_match["sideB"]:
            assert new["wins"] == old["wins"] + 1
            assert new["losses"] == old["losses"] - 1


@pytest.mark.parametrize("audience", ["admin", "public"])
@pytest.mark.parametrize("lock", ["pending", "approved", "rejected", "published_ids", "published_at"])
def test_submitted_or_published_scores_stay_locked(audience, lock):
    game = GeneratorSession(audience)
    game.score(1)
    if lock.startswith("published_"):
        game.stored["state"]["official_publish"] = (
            {"published_match_ids": ["r1-c1"]} if lock == "published_ids" else {"published_at": "2026-10-10T00:00:00Z"}
        )
    else:
        game.stored["state"]["generator_submission"] = {"status": lock}
    before = deepcopy(game.stored)
    with pytest.raises(ValueError, match="[Pp]ublished|[Ss]ubmitted|locked"):
        game.call("save", round_number=1, scores=scores_for(game.session["event"]["rounds"][0]))
    assert game.stored == before
    if audience == "public":
        assert game.reload()["results_locked"] is True


@pytest.mark.parametrize("audience", ["admin", "public"])
def test_correction_rejects_stale_version_and_invalid_scores(audience):
    game = GeneratorSession(audience)
    game.score(1)
    before = deepcopy(game.stored)
    score_rows = scores_for(game.session["event"]["rounds"][0])
    stale = game.session["version"] - 1 if audience == "public" else "stale-version"
    with pytest.raises((ValueError, PublicLiveConflictError), match="changed"):
        game.call("save", round_number=1, scores=score_rows, expected_version=stale)
    assert game.stored == before
    for invalid in [[], [{**score_rows[0], "score_a": None}], [{**score_rows[0], "score_a": 11}], [{**score_rows[0], "score_b": 100}]]:
        with pytest.raises(ValueError):
            game.call("save", round_number=1, scores=invalid)
        assert game.stored == before


def test_public_correction_requires_organizer_and_retry_does_not_apply_twice():
    game = GeneratorSession("public")
    game.score(1)
    before = deepcopy(game.stored)
    score_rows = scores_for(game.session["event"]["rounds"][0])
    with pytest.raises(PermissionError):
        game.call("save", round_number=1, scores=score_rows, edit_token="wrong-organizer")
    assert game.stored == before
    request = dict(round_number=1, scores=score_rows, expected_version=game.session["version"], idempotency_key="score-correction-retry-0001")
    first = game.call("save", **request)
    replay = game.call("save", **request)
    assert replay["idempotent_replay"] is True
    assert replay["session"] == first["session"]
    committed = deepcopy(game.stored)
    with pytest.raises(PublicLiveConflictError):
        game.call("save", **{**request, "scores": [{**score_rows[0], "score_a": 9}]})
    assert game.stored == committed


def test_one_court_correction_preserves_other_court_and_input_event():
    event = scored_event(9)
    before = deepcopy(event)
    row = event["rounds"][0]
    scores = [{"match_id": m["id"], "score_a": m["scoreA"], "score_b": m["scoreB"]} for m in matches(row)]
    scores[0].update(score_a=8, score_b=11)
    corrected = save_generator_round(event, round_number=1, scores=scores)
    assert event == before
    assert corrected["rounds"][0]["matches"][1] == before["rounds"][0]["matches"][1]
    assert corrected["rounds"][1:] == before["rounds"][1:]


def test_playoff_scores_can_be_corrected_until_next_round_but_cannot_reseed():
    playoff = start_generator_playoff(scored_event(), playoff_format="top_eight")
    number = playoff["currentRoundNumber"]
    row = next(r for r in playoff["rounds"] if r["number"] == number)
    playoff = save_generator_round(playoff, round_number=number, scores=scores_for(row, 11, 8))
    playoff = save_generator_round(playoff, round_number=number, scores=scores_for(row))
    advanced = advance_generator_event(playoff)
    assert advanced["rounds"][-1]["matches"][0]["sideA"] == row["matches"][0]["sideB"]
    for locked in (1, number):
        with pytest.raises(ValueError, match="Only the current playoff"):
            save_generator_round(advanced, round_number=locked, scores=[])


def test_ladder_correction_is_allowed_before_dependent_round_is_generated():
    event = start_generator_event(create_generator_preview(
        generator_kind="ladder", play_format="doubles", title="Ladder",
        participant_names=[f"Player {i}" for i in range(8)], total_rounds=3, court_count=2,
    ))
    scores = scores_for(event["rounds"][0])
    event = save_generator_round(event, round_number=1, scores=scores)
    event = save_generator_round(event, round_number=1, scores=scores_for(event["rounds"][0], 11, 8))
    event = advance_generator_event(event)
    with pytest.raises(ValueError, match="after the next round"):
        save_generator_round(event, round_number=1, scores=scores)
