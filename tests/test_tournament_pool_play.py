from collections import Counter, defaultdict
from itertools import combinations

import pytest

from jupr_app.domain.tournaments import (
    build_draw_round_robin_games,
    compute_pool_standings,
    normalize_pool_config,
    recommend_pool_qualifiers,
    validate_pool_schedule,
)
from jupr_app.domain.tournaments.bracket_builder import (
    PLAYOFF_TEMPLATES,
    build_playoff_games,
    build_round_robin_games,
    compute_podium_from_playoffs,
)


def teams(count):
    return [{"id": f"team-{i}", "team_number": i} for i in range(1, count + 1)]


def config_for_sizes(*sizes):
    start = 1
    pools = []
    for index, size in enumerate(sizes):
        pool_id = chr(ord("A") + index)
        pools.append({"id": pool_id, "name": f"Pool {pool_id}",
                      "team_ids": [f"team-{i}" for i in range(start, start + size)],
                      "advance_count": min(2, size)})
        start += size
    return {"version": 1, "pools": pools}


def score_games(games):
    result = []
    for game in games:
        a, b = game["team_a_id"], game["team_b_id"]
        a_wins = int(a.split("-")[-1]) < int(b.split("-")[-1])
        result.append({**game, "score_a": 11 if a_wins else 4, "score_b": 4 if a_wins else 11,
                       "winner_team_id": a if a_wins else b, "loser_team_id": b if a_wins else a,
                       "finalized_at": "2026-10-31T20:00:00Z"})
    return result


@pytest.mark.parametrize("sizes,expected_games", [((4, 4), 12), ((3, 3), 6), ((4, 3), 9), ((2, 2), 2), ((3, 3, 3), 9), ((8, 8), 56)])
def test_pool_schedule_plays_each_within_pool_pair_once(sizes, expected_games):
    roster = teams(sum(sizes))
    config = config_for_sizes(*sizes)
    games = build_draw_round_robin_games(tournament_id="tour", teams=roster, pool_config=config)
    expected_pairs = {frozenset(pair) for pool in config["pools"] for pair in combinations(pool["team_ids"], 2)}
    assert len(games) == expected_games
    assert {frozenset((row["team_a_id"], row["team_b_id"])) for row in games} == expected_pairs
    coordinates = [(row["rr_round_number"], row["rr_slot_number"]) for row in games]
    assert len(set(coordinates)) == len(coordinates)
    rounds = defaultdict(list)
    appearances = Counter()
    for row in games:
        assert set(row) == {"tournament_id", "stage", "rr_round_number", "rr_slot_number", "team_a_id", "team_b_id"}
        rounds[row["rr_round_number"]].extend((row["team_a_id"], row["team_b_id"]))
        appearances.update((row["team_a_id"], row["team_b_id"]))
    for players in rounds.values():
        assert len(players) == len(set(players))
    for pool in config["pools"]:
        for team_id in pool["team_ids"]:
            assert appearances[team_id] == len(pool["team_ids"]) - 1
    validate_pool_schedule(roster, games, config, require_complete=True)


def test_three_team_pool_has_one_rotating_bye_each():
    config = config_for_sizes(3, 3)
    games = build_draw_round_robin_games(tournament_id="tour", teams=teams(6), pool_config=config)
    for pool in config["pools"]:
        byes = []
        for round_number in (1, 2, 3):
            active = {row[side] for row in games if row["rr_round_number"] == round_number
                      for side in ("team_a_id", "team_b_id")}
            bye = set(pool["team_ids"]) - active
            assert len(bye) == 1
            byes.extend(bye)
        assert set(byes) == set(pool["team_ids"])


def test_unsplit_draw_preserves_existing_schedule():
    roster = teams(8)
    assert build_draw_round_robin_games(tournament_id="tour", teams=roster) == build_round_robin_games(
        tournament_id="tour", team_ids_by_number={row["team_number"]: row["id"] for row in roster})


@pytest.mark.parametrize("mutation,message", [
    (lambda c: c.update(version=True), "version"),
    (lambda c: c["pools"].pop(), "two and eight"),
    (lambda c: c["pools"][1].update(id="A"), "unique short"),
    (lambda c: c["pools"][1].update(name="pool a"), "unique name"),
    (lambda c: c["pools"][0].update(team_ids=["team-1"]), "at least two"),
    (lambda c: c["pools"][1]["team_ids"].__setitem__(0, "team-1"), "exactly one"),
    (lambda c: c["pools"][0]["team_ids"].append("foreign"), "only saved"),
    (lambda c: c["pools"][0]["team_ids"].pop(), "every draw team"),
    (lambda c: c["pools"][0].update(advance_count=2.0), "whole number"),
    (lambda c: c["pools"][0].update(advance_count=True), "whole number"),
    (lambda c: c["pools"][0].update(advance_count=5), "whole number"),
    (lambda c: [p.update(advance_count=0) for p in c["pools"]], "between two"),
])
def test_rejects_invalid_pool_partition_or_advancement(mutation, message):
    config = config_for_sizes(4, 4)
    mutation(config)
    with pytest.raises(ValueError, match=message):
        normalize_pool_config(config, teams(8))


def test_zero_advancers_from_one_pool_is_explicitly_supported():
    config = config_for_sizes(3, 3)
    config["pools"][0]["advance_count"] = 0
    assert normalize_pool_config(config, teams(6)) == config


def test_config_normalization_trims_labels_but_does_not_mutate_input():
    config = config_for_sizes(3, 3)
    config["pools"][0]["name"] = "  Pool A  "
    normalized = normalize_pool_config(config, teams(6))
    assert normalized["pools"][0]["name"] == "Pool A"
    assert config["pools"][0]["name"] == "  Pool A  "


def test_standings_are_pool_specific_and_crossovers_use_place_not_raw_wins():
    roster = teams(7)
    config = config_for_sizes(4, 3)
    games = score_games(build_draw_round_robin_games(tournament_id="tour", teams=roster, pool_config=config))
    pools = compute_pool_standings(roster, games, config)
    assert [[row["team_id"] for row in pool["standings"]] for pool in pools] == [
        ["team-1", "team-2", "team-3", "team-4"], ["team-5", "team-6", "team-7"]]
    assert [[row["wins"] for row in pool["standings"]] for pool in pools] == [[3, 2, 1, 0], [2, 1, 0]]
    qualifiers = recommend_pool_qualifiers(roster, games, config)
    assert qualifiers == ["team-1", "team-5", "team-2", "team-6"]
    standings = [row for pool in pools for row in pool["standings"]]
    bracket = build_playoff_games(tournament_id="tour", advance_count=4, standings=standings, seed_team_ids=qualifiers)
    assert [(row["team_a_id"], row["team_b_id"]) for row in bracket[:2]] == [
        ("team-1", "team-6"), ("team-5", "team-2")]


def test_retirement_shortfall_does_not_silently_replace_pool_qualifier():
    roster = teams(6)
    roster[0]["competition_status"] = "RETIRED"
    roster[1]["competition_status"] = "RETIRED"
    config = config_for_sizes(3, 3)
    games = score_games(build_draw_round_robin_games(tournament_id="tour", teams=roster, pool_config=config))
    pools = compute_pool_standings(roster, games, config)
    assert pools[0]["standings"][0]["team_id"] == "team-3"
    assert pools[0]["standings"][-1]["retired"]
    assert recommend_pool_qualifiers(roster, games, config) == ["team-3", "team-4", "team-5"]


def test_schedule_guards_missing_cross_pool_repeated_and_foreign_pairings():
    roster = teams(6)
    config = config_for_sizes(3, 3)
    games = build_draw_round_robin_games(tournament_id="tour", teams=roster, pool_config=config)
    validate_pool_schedule(roster, games[:-1], config)
    with pytest.raises(ValueError, match="Every within-pool pairing"):
        validate_pool_schedule(roster, games[:-1], config, require_complete=True)
    with pytest.raises(ValueError, match="repeat"):
        validate_pool_schedule(roster, [*games, games[0]], config)
    with pytest.raises(ValueError, match="cross"):
        validate_pool_schedule(roster, [{**games[0], "team_a_id": "team-1", "team_b_id": "team-4"}], config)
    with pytest.raises(ValueError, match="assigned draw teams"):
        validate_pool_schedule(roster, [{**games[0], "team_a_id": "foreign"}], config)


def test_series_children_and_playoffs_do_not_change_pool_standings():
    roster = teams(6)
    config = config_for_sizes(3, 3)
    games = score_games(build_draw_round_robin_games(tournament_id="tour", teams=roster, pool_config=config))
    extras = [{**games[0], "series_parent_game_id": "parent", "score_a": 999},
              {**games[0], "stage": "PLAYOFF", "team_a_id": "team-1", "team_b_id": "team-4"}]
    assert compute_pool_standings(roster, games + extras, config) == compute_pool_standings(roster, games, config)


@pytest.mark.parametrize("count", range(2, 17))
def test_each_playoff_size_resolves_with_real_byes_and_one_podium(count):
    standings = [{"team_id": f"team-{i}", "seed": i} for i in range(1, count + 1)]
    games = build_playoff_games(tournament_id="tour", advance_count=count, standings=standings)
    assert len(games) == (count if count >= 4 else count - 1)
    initial_ids = [row[side] for row in games for side in ("team_a_id", "team_b_id") if row[side]]
    assert Counter(initial_ids) == Counter({row["team_id"]: 1 for row in standings})
    resolved = {}
    for game in games:
        for side in ("a", "b"):
            source = game[f"team_{side}_source"]
            if "winnerOf" in source:
                game[f"team_{side}_id"] = resolved[source["winnerOf"]]["winner_team_id"]
            if "loserOf" in source:
                game[f"team_{side}_id"] = resolved[source["loserOf"]]["loser_team_id"]
        ids = [game["team_a_id"], game["team_b_id"]]
        assert all(ids) and ids[0] != ids[1]
        ids.sort(key=lambda value: int(value.split("-")[-1]))
        game.update(winner_team_id=ids[0], loser_team_id=ids[1], finalized_at="2026-10-31T20:00:00Z")
        resolved[game["playoff_game_code"]] = game
    podium = compute_podium_from_playoffs(games)
    assert podium == [{"placement": i, "team_id": f"team-{i}"} for i in range(1, min(3, count) + 1)]


def test_three_qualifier_bracket_gives_top_seed_real_bye():
    assert [(row["round"], row["teamA"], row["teamB"]) for row in PLAYOFF_TEMPLATES[3]["games"]] == [
        ("SF", {"seed": 2}, {"seed": 3}), ("Final", {"seed": 1}, {"winnerOf": "P1"})]


def test_larger_bracket_with_missing_bronze_is_not_a_two_team_podium():
    final = {"playoff_round": "Final", "winner_team_id": "a", "loser_team_id": "b", "finalized_at": "now"}
    semifinals = [{"playoff_round": "SF", "winner_team_id": "a", "loser_team_id": "c", "finalized_at": "now"},
                  {"playoff_round": "SF", "winner_team_id": "b", "loser_team_id": "d", "finalized_at": "now"}]
    assert compute_podium_from_playoffs([*semifinals, final]) is None


def test_three_pools_two_each_avoid_first_playoff_pool_rematch():
    roster = teams(9)
    config = config_for_sizes(3, 3, 3)
    games = score_games(build_draw_round_robin_games(tournament_id="tour", teams=roster, pool_config=config))
    qualifiers = recommend_pool_qualifiers(roster, games, config)
    assert set(qualifiers) == {"team-1", "team-4", "team-7", "team-2", "team-5", "team-8"}
    mapping = {team_id: pool["id"] for pool in config["pools"] for team_id in pool["team_ids"]}
    standings = [row for pool in compute_pool_standings(roster, games, config) for row in pool["standings"]]
    bracket = build_playoff_games(tournament_id="tour", advance_count=6, standings=standings, seed_team_ids=qualifiers)
    direct_games = [row for row in bracket if row["team_a_id"] and row["team_b_id"]]
    assert direct_games
    assert all(mapping[row["team_a_id"]] != mapping[row["team_b_id"]] for row in direct_games)
    assert set(qualifiers[:3]) == {"team-1", "team-4", "team-7"}
    assert recommend_pool_qualifiers(roster, games, config) == qualifiers


def test_unequal_quotas_never_promote_runner_over_pool_winner_to_avoid_rematch():
    roster = teams(6)
    config = config_for_sizes(3, 3)
    config["pools"][0]["advance_count"] = 3
    config["pools"][1]["advance_count"] = 1
    games = score_games(build_draw_round_robin_games(tournament_id="tour", teams=roster, pool_config=config))
    qualifiers = recommend_pool_qualifiers(roster, games, config)
    # Three from one pool and one from the other must have one same-pool semi.
    # Preserve place priority and let the operator review this unavoidable case.
    assert qualifiers == ["team-1", "team-4", "team-2", "team-3"]
