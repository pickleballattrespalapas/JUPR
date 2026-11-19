from copy import deepcopy

from jupr_app.domain.tournaments import (
    build_playoff_games,
    compute_round_robin_standings_with_tiebreaks,
)
from jupr_app.domain.tournaments.tiebreak_rulings import _rulings
from jupr_app.services.tournament_round_robin_tiebreak_service import (
    build_round_robin_tiebreak_explanations,
    round_robin_ranking_policy,
)


def _bracket():
    ruling = _rulings()[0]
    ids = {number: f"team-{number}" for number in range(1, 9)}
    ids.update(dict(zip((1, 5, 7, 8), ruling["team_ids"])))
    teams = [{"id": ids[n], "team_number": n} for n in range(1, 9)]
    # Completed eight-team RR reproducing the director's four-way 3–4 tie.
    results = [
        (2, 1, 15, 11), (3, 8, 15, 6), (4, 7, 15, 11), (5, 6, 15, 2),
        (3, 4, 7, 15), (1, 7, 15, 13), (8, 6, 15, 4), (2, 5, 15, 12),
        (6, 2, 2, 15), (7, 8, 15, 8), (4, 1, 15, 4), (5, 3, 10, 15),
        (7, 5, 6, 15), (8, 4, 16, 14), (2, 3, 8, 15), (6, 1, 7, 15),
        (1, 3, 11, 15), (4, 2, 11, 15), (5, 8, 16, 14), (6, 7, 6, 15),
        (4, 5, 15, 6), (8, 1, 15, 10), (2, 7, 15, 12), (3, 6, 15, 1),
        (7, 3, 15, 6), (8, 2, 4, 15), (1, 5, 15, 12), (6, 4, 3, 15),
    ]
    games = [
        {"team_a_id": ids[a], "team_b_id": ids[b], "score_a": sa, "score_b": sb,
         "draw_id": ruling["draw_id"], "tournament_id": ruling["tournament_id"]}
        for a, b, sa, sb in results
    ]
    return teams, games


def test_director_ruling_moves_all_four_to_differential_and_updates_playoff_seed():
    teams, games = _bracket()
    originals = deepcopy((teams, games))
    result = compute_round_robin_standings_with_tiebreaks(teams, games)
    standings = result["standings"]
    assert [s["team_number"] for s in standings] == [2, 4, 3, 7, 5, 8, 1, 6]
    assert [s["differential"] for s in standings[3:7]] == [7, 4, -11, -11]
    assert (teams, games) == originals
    audit = next(a for a in result["tiebreaks"] if a["wins"] == 3)
    assert [(s["criterion"], s["outcome"]) for s in audit["steps"]] == [
        ("HEAD_TO_HEAD", "UNRESOLVED"),
        ("POINT_DIFFERENTIAL", "PARTIALLY_RESOLVED"),
        ("HEAD_TO_HEAD_POINT_DIFFERENTIAL", "RESOLVED"),
    ]
    assert audit["steps"][0]["groups_after"] == [audit["team_ids"]]
    assert audit["steps"][1]["groups_before"] == [audit["team_ids"]]
    policy = round_robin_ranking_policy(result["tiebreaks"])
    assert "all four teams" in policy["description"]
    assert policy["criteria"][3] == "HEAD_TO_HEAD_POINT_DIFFERENTIAL"
    playoff = build_playoff_games(tournament_id="test", advance_count=4, standings=standings)
    assert playoff[0]["team_b_id"] == teams[6]["id"]
    explanations = build_round_robin_tiebreak_explanations([audit], standings, games)
    explanation = explanations[0]
    assert "Tournament director ruling applied" in explanation["summary"]
    assert "All four" in explanation["steps"][0]["detail"] or "all four" in explanation["steps"][0]["detail"]
    assert "Team 8 +5" in explanation["steps"][2]["detail"]
    assert "Team 1 -5" in explanation["steps"][2]["detail"]
    assert "15–10" in explanation["steps"][2]["detail"]
    assert "Head-to-head and point differential decided" not in explanation["summary"]


def test_ruling_does_not_change_other_draws_or_tournaments():
    for field in ("draw_id", "tournament_id"):
        teams, games = _bracket()
        for game in games:
            game[field] = "another-event"
        result = compute_round_robin_standings_with_tiebreaks(teams, games)
        assert [s["team_number"] for s in result["standings"]][3:7] == [5, 1, 7, 8]
        assert all("director_ruling_id" not in audit for audit in result["tiebreaks"])


def test_score_correction_that_removes_four_way_tie_does_not_force_ruling():
    teams, games = _bracket()
    games[-2]["score_a"], games[-2]["score_b"] = 12, 15
    result = compute_round_robin_standings_with_tiebreaks(teams, games)
    assert all("director_ruling_id" not in audit for audit in result["tiebreaks"])


def test_ruling_is_independent_of_query_row_order():
    teams, games = _bracket()
    first = compute_round_robin_standings_with_tiebreaks(teams, games)
    reversed_rows = compute_round_robin_standings_with_tiebreaks(teams[::-1], games[::-1])
    assert first["standings"] == reversed_rows["standings"]
