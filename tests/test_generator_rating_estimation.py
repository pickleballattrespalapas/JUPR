from __future__ import annotations

import copy
import math

import pytest

from jupr_app.domain.generator_rating_estimation import estimate_generator_player_ratings


def game(a, b, score_a=11, score_b=7, **extra):
    return {"sideA": a, "sideB": b, "scoreA": score_a, "scoreB": score_b, **extra}


def estimate(games, known=None, targets=None, **kwargs):
    return estimate_generator_player_ratings(
        games, known if known is not None else {"partner": 3.5, "o1": 3.5, "o2": 3.5},
        targets or ["new"], **kwargs,
    )


def test_uses_group_average_when_no_completed_games():
    result = estimate([], known={"a": 3, "b": 4.5})["new"]
    assert result["rating_jupr"] == 3.75
    assert result["method"] == "group_average"
    assert result["games_used"] == 0
    assert result["provisional"] is True


def test_scores_move_estimate_but_sparse_evidence_is_shrunk_toward_group():
    win = estimate([game(["new", "partner"], ["o1", "o2"], 11, 1)])["new"]
    loss = estimate([game(["new", "partner"], ["o1", "o2"], 1, 11)])["new"]
    assert 3.5 < win["rating_jupr"] < 3.5 + 2 * math.log10(12 / 2)
    assert loss["rating_jupr"] < 3.5
    assert win["rating_jupr"] - 3.5 == pytest.approx(3.5 - loss["rating_jupr"])
    assert win["games_used"] == win["anchored_games"] == 1


def test_same_score_accounts_for_partner_and_opponent_strength():
    match = game(["new", "partner"], ["o1", "o2"], 11, 7)
    weak_partner = estimate([match], {"partner": 3, "o1": 4, "o2": 4})["new"]["rating_jupr"]
    strong_partner = estimate([match], {"partner": 5, "o1": 3, "o2": 3})["new"]["rating_jupr"]
    # The same known-player average and score, with a different team assignment.
    assert weak_partner > strong_partner


def test_more_consistent_evidence_supports_larger_adjustment():
    match = game(["new", "partner"], ["o1", "o2"], 11, 4)
    one = estimate([match])["new"]["rating_jupr"]
    many = estimate([match] * 6)["new"]["rating_jupr"]
    assert 3.5 < one < many


def test_unknowns_are_solved_jointly_and_ignore_previous_estimates_as_anchors():
    games = [game(["a", "partner"], ["b", "o1"], 11, 2),
             game(["a", "b"], ["partner", "o1"], 11, 7)]
    known = {"partner": 3.5, "o1": 3.5}
    result = estimate(games, known, ["a", "b"])
    rerun = estimate(list(reversed(games)), {**known, "a": 6.5}, ["b", "a"])
    assert result == rerun
    assert result["a"]["rating_jupr"] > result["b"]["rating_jupr"]
    assert result["a"]["known_players"] == 2


def test_unanchored_group_does_not_invent_an_absolute_skill_level():
    games = [game(["a", "b"], ["c", "d"], 11, 0)]
    result = estimate(games, {}, ["a", "b", "c", "d"])
    assert {item["rating_jupr"] for item in result.values()} == {3.5}
    assert {item["method"] for item in result.values()} == {"default_estimate"}
    assert {item["games_used"] for item in result.values()} == {0}
    # A different group may provide a fallback mean, but not evidence of these
    # players' relative levels until they connect to a known profile.
    disconnected = estimate(games, {"unrelated": 4}, ["a", "b", "c", "d"])
    assert {item["rating_jupr"] for item in disconnected.values()} == {4}
    assert {item["method"] for item in disconnected.values()} == {"group_average"}


def test_evidence_can_connect_to_known_players_through_other_unknowns():
    games = [game(["a", "b"], ["c", "d"], 11, 7),
             game(["c", "d"], ["p", "q"], 11, 2)]
    result = estimate(games, {"p": 3.5, "q": 3.5}, ["a", "b", "c", "d"])
    assert result["a"]["games_used"] == 1
    assert result["a"]["anchored_games"] == 0
    assert result["a"]["method"] == "score_estimate"
    assert result["c"]["games_used"] == 2
    assert result["c"]["anchored_games"] == 1


@pytest.mark.parametrize("change", [
    {"scoreA": None}, {"scoreA": float("nan")}, {"scoreA": float("inf")},
    {"scoreA": True}, {"scoreA": -1}, {"scoreA": 3.5}, {"scoreA": 7},
    {"status": "non_play"}, {"match_format": "singles"},
    {"sideA": ["new", "new"]}, {"sideB": ["o1", "unresolved"]},
])
def test_invalid_or_unresolved_games_do_not_adjust_rating(change):
    match = {**game(["new", "partner"], ["o1", "o2"]), **change}
    assert estimate([match])["new"]["method"] == "group_average"


def test_shutout_finite_bounded_and_duplicate_match_not_counted_twice():
    match = game(["new", "partner"], ["o1", "o2"], 11, 0, id="match")
    result = estimate([match, copy.deepcopy(match)])["new"]
    assert 1 <= result["rating_jupr"] <= 7
    assert result["games_used"] == 1
    assert estimate([match])["new"] == result


def test_match_ids_reused_in_different_rounds_are_distinct_games():
    match = game(["new", "partner"], ["o1", "o2"], 11, 4, id="match")
    first, second = {**match, "round_number": 1}, {**match, "round_number": 2}
    result = estimate([first, second, copy.deepcopy(second)])["new"]
    assert result["games_used"] == 2
    assert result["rating_jupr"] > estimate([first])["new"]["rating_jupr"]


def test_singles_uses_separate_format_and_scale_and_accepts_legacy_team_fields():
    games = [{"teamA": ["new"], "teamB": ["p"], "scoreA": 15, "scoreB": 3},
             game(["new", "p"], ["q", "r"], 0, 11)]
    result = estimate(games, {"p": 4}, match_format="singles")["new"]
    assert result["rating_jupr"] > 4
    assert result["games_used"] == 1
    assert estimate(games, {"p": 4})["new"]["games_used"] == 0


def test_invalid_known_ratings_are_not_anchors_and_inputs_are_unchanged():
    games = [game(["new", "partner"], ["o1", "o2"])]
    known = {"partner": float("inf"), "o1": 0, "o2": 8, "valid": 4}
    before = copy.deepcopy(games)
    result = estimate(games, known)["new"]
    assert result["rating_jupr"] == 4
    assert result["known_players"] == 1
    assert games == before
    assert math.isinf(known["partner"])


def test_empty_targets_and_invalid_format():
    assert estimate_generator_player_ratings([], {}, []) == {}
    with pytest.raises(ValueError, match="doubles or singles"):
        estimate([], match_format="mixed")
