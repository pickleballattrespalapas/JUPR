"""Pool assignments and competition rules within one tournament draw.

Pools share the draw's teams, game lifecycle, playoff bracket, and podium.  Pool
membership is stored once on the draw, never inferred from a display name.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import re
from typing import Any

from .bracket_builder import (
    PLAYOFF_TEMPLATES,
    build_round_robin_games,
    compute_round_robin_standings_with_tiebreaks,
)


def normalize_pool_config(
    config: Any, teams: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Validate a complete exact-roster partition; preserve explicit pool order."""
    if config is None:
        return None
    if not isinstance(config, dict) or type(config.get("version")) is not int or config["version"] != 1:
        raise ValueError("Pool configuration must use version 1.")
    raw_pools = config.get("pools")
    if not isinstance(raw_pools, list) or not 2 <= len(raw_pools) <= 8:
        raise ValueError("Split the draw into between two and eight pools.")
    if not 4 <= len(teams) <= 16:
        raise ValueError("Pool play supports draws of four through sixteen teams.")
    team_ids = [str(row.get("id") or "").strip() for row in teams]
    if any(not team_id for team_id in team_ids) or len(set(team_ids)) != len(team_ids):
        raise ValueError("Every draw team must have a unique saved ID before assigning pools.")
    expected_ids = set(team_ids)
    assigned_ids: set[str] = set()
    pool_ids: set[str] = set()
    pool_names: set[str] = set()
    pools = []
    for raw in raw_pools:
        if not isinstance(raw, dict):
            raise ValueError("Every pool must have a name, team assignments and advance count.")
        pool_id = str(raw.get("id") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", pool_id) or pool_id in pool_ids:
            raise ValueError("Every pool must have a unique short identifier.")
        name = str(raw.get("name") or "").strip()
        if not name or len(name) > 80 or name.casefold() in pool_names:
            raise ValueError("Every pool must have a unique name of at most 80 characters.")
        raw_ids = raw.get("team_ids")
        if not isinstance(raw_ids, list) or len(raw_ids) < 2:
            raise ValueError("Each pool needs at least two teams.")
        ids = [str(value or "").strip() for value in raw_ids]
        if len(set(ids)) != len(ids) or assigned_ids.intersection(ids):
            raise ValueError("Each team must belong to exactly one pool.")
        if not set(ids).issubset(expected_ids):
            raise ValueError("Pool assignments must contain only saved teams from this draw.")
        count = raw.get("advance_count")
        if type(count) is not int or not 0 <= count <= len(ids):
            raise ValueError("Each pool's advance count must be a whole number from zero through its team count.")
        pools.append({"id": pool_id, "name": name, "team_ids": ids, "advance_count": count})
        assigned_ids.update(ids)
        pool_ids.add(pool_id)
        pool_names.add(name.casefold())
    if assigned_ids != expected_ids:
        raise ValueError("Assign every draw team to exactly one pool before saving.")
    if not 2 <= sum(pool["advance_count"] for pool in pools) <= 16:
        raise ValueError("The combined pools must advance between two and sixteen teams.")
    return {"version": 1, "pools": pools}


def _pool_round_robin(tournament_id: str, ids: list[str]) -> list[dict[str, Any]]:
    # The standalone draw minimum remains four.  Small pools are valid because
    # their qualifiers progress into the parent draw's shared championship.
    if len(ids) >= 4:
        return build_round_robin_games(
            tournament_id=tournament_id,
            team_ids_by_number=dict(enumerate(ids, start=1)),
        )
    rotation: list[str | None] = list(ids)
    if len(rotation) % 2:
        rotation.append(None)
    games = []
    for round_index in range(len(rotation) - 1):
        for left, right in zip(rotation[:len(rotation) // 2], reversed(rotation[len(rotation) // 2:])):
            if left is not None and right is not None:
                games.append({
                    "tournament_id": tournament_id,
                    "stage": "ROUND_ROBIN",
                    "rr_round_number": round_index + 1,
                    "rr_slot_number": 1,
                    "team_a_id": left,
                    "team_b_id": right,
                })
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    return games


def build_draw_round_robin_games(
    *, tournament_id: str, teams: list[dict[str, Any]], pool_config: Any = None
) -> list[dict[str, Any]]:
    """Build a whole draw or one single RR per pool, sharing round/slot space."""
    config = normalize_pool_config(pool_config, teams)
    if config is None:
        return build_round_robin_games(
            tournament_id=tournament_id,
            team_ids_by_number={int(row["team_number"]): str(row["id"]) for row in teams},
        )
    slots: dict[int, int] = defaultdict(int)
    games = []
    for pool in config["pools"]:
        for game in _pool_round_robin(tournament_id, pool["team_ids"]):
            round_number = game["rr_round_number"]
            slots[round_number] += 1
            games.append({**game, "rr_slot_number": slots[round_number]})
    return sorted(games, key=lambda game: (game["rr_round_number"], game["rr_slot_number"]))


def _round_robin_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in games
        if str(row.get("stage") or "ROUND_ROBIN").upper() == "ROUND_ROBIN"
        and not row.get("series_parent_game_id")
    ]


def validate_pool_schedule(
    teams: list[dict[str, Any]], games: list[dict[str, Any]], pool_config: Any,
    *, require_complete: bool = False,
) -> None:
    """Reject duplicate/cross-pool games; optionally require every expected pair.

    ``require_complete`` checks schedule coverage, not score completion.  The
    existing finalized-result readiness check remains authoritative for scores.
    """
    config = normalize_pool_config(pool_config, teams)
    if config is None:
        return
    memberships = {team_id: pool["id"] for pool in config["pools"] for team_id in pool["team_ids"]}
    expected = {frozenset(pair) for pool in config["pools"] for pair in combinations(pool["team_ids"], 2)}
    observed: set[frozenset[str]] = set()
    for game in _round_robin_games(games):
        a, b = str(game.get("team_a_id") or ""), str(game.get("team_b_id") or "")
        if not a or a == b or a not in memberships or b not in memberships:
            raise ValueError("Every pool game must contain two different assigned draw teams.")
        if memberships[a] != memberships[b]:
            raise ValueError("Round-robin pool games cannot cross between pools.")
        pair = frozenset((a, b))
        if pair in observed:
            raise ValueError("A single pool round-robin cannot repeat a team pairing.")
        observed.add(pair)
    if require_complete and observed != expected:
        raise ValueError("Every within-pool pairing must be scheduled exactly once before activation or playoffs.")


def compute_pool_standings(
    teams: list[dict[str, Any]], games: list[dict[str, Any]], pool_config: Any
) -> list[dict[str, Any]]:
    """Apply the existing wins/tiebreak policy separately inside each pool."""
    config = normalize_pool_config(pool_config, teams)
    if config is None:
        return []
    validate_pool_schedule(teams, games, config)
    teams_by_id = {str(team["id"]): team for team in teams}
    rr_games = _round_robin_games(games)
    result = []
    for pool in config["pools"]:
        ids = set(pool["team_ids"])
        standings_result = compute_round_robin_standings_with_tiebreaks(
            [teams_by_id[team_id] for team_id in pool["team_ids"]],
            [game for game in rr_games if str(game.get("team_a_id")) in ids],
        )
        standings = []
        for rank, row in enumerate(standings_result["standings"], start=1):
            standings.append({
                **row,
                "pool_id": pool["id"], "pool_name": pool["name"], "pool_rank": rank,
                "recommended_qualifier": rank <= pool["advance_count"] and not row.get("retired"),
            })
        result.append({
            "pool_id": pool["id"], "pool_name": pool["name"],
            "advance_count": pool["advance_count"],
            "standings": standings, "tiebreaks": standings_result["tiebreaks"],
        })
    return result


def _avoid_immediate_pool_rematches(
    qualifiers: list[tuple[int, int, str]], advance_count: int,
) -> list[tuple[int, int, str]]:
    """Improve first-match crossovers without promoting a lower pool place.

    Only equal-place seeds can swap. Each accepted swap strictly reduces the
    number of direct same-pool pairings, bounding the search by bracket size.
    Ties retain the existing pool order; missing quota qualifiers stay missing.
    """
    ordered = list(qualifiers)
    if len(ordered) != advance_count:
        return ordered
    template = PLAYOFF_TEMPLATES.get(advance_count, {})
    direct_pairs = [
        (int(game["teamA"]["seed"]) - 1, int(game["teamB"]["seed"]) - 1)
        for game in template.get("games", [])
        if "seed" in game["teamA"] and "seed" in game["teamB"]
    ]

    def rematches(rows: list[tuple[int, int, str]]) -> int:
        return sum(
            rows[left][1] == rows[right][1]
            for left, right in direct_pairs if left < len(rows) and right < len(rows)
        )

    while True:
        score = rematches(ordered)
        if not score:
            return ordered
        best_score, best_swap = score, None
        # Prefer moving runners-up/lower places before changing pool winners.
        ranks = sorted({row[0] for row in ordered}, reverse=True)
        for rank in ranks:
            indices = [i for i, row in enumerate(ordered) if row[0] == rank]
            for left, right in reversed(list(combinations(indices, 2))):
                candidate = list(ordered)
                candidate[left], candidate[right] = candidate[right], candidate[left]
                candidate_score = rematches(candidate)
                if candidate_score < best_score:
                    best_score, best_swap = candidate_score, (left, right)
        if best_swap is None:
            return ordered
        left, right = best_swap
        ordered[left], ordered[right] = ordered[right], ordered[left]


def recommend_pool_qualifiers(
    teams: list[dict[str, Any]], games: list[dict[str, Any]], pool_config: Any
) -> list[str]:
    """Recommend by pool place, improving crossovers inside equal-place bands.

    Raw wins are never compared across pools. Two pools advancing two teams
    retain A1, B1, A2, B2 for crossover semifinals. For other layouts, equal-place
    seeds may swap to avoid an immediate pool rematch; manual review still owns
    the final selections, especially when uneven quotas force repeat meetings.
    """
    pools = compute_pool_standings(teams, games, pool_config)
    qualifiers = sorted(
        (row["pool_rank"], pool_index, str(row["team_id"]))
        for pool_index, pool in enumerate(pools)
        for row in pool["standings"] if row["recommended_qualifier"]
    )
    advance_count = sum(pool["advance_count"] for pool in pools)
    ordered = _avoid_immediate_pool_rematches(qualifiers, advance_count)
    return [team_id for _, _, team_id in ordered]
