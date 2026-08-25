from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Tournament setup permits divisions as large as 16 teams.  Keep one shared
# executable contract so setup, the operator readiness model, and game
# generation cannot disagree about a valid draw size.  Preserve the existing
# four-team product minimum while extending the executable upper bound.
SUPPORTED_TEAM_COUNTS = list(range(4, 17))


ROUND_ROBIN_TEMPLATES: dict[int, dict[str, Any]] = {
    4: {
        "teamCount": 4,
        "rounds": [
            {"round": 1, "games": [{"slot": 1, "teamA": 2, "teamB": 1}, {"slot": 2, "teamA": 3, "teamB": 4}], "byes": []},
            {"round": 2, "games": [{"slot": 1, "teamA": 4, "teamB": 2}, {"slot": 2, "teamA": 1, "teamB": 3}], "byes": []},
            {"round": 3, "games": [{"slot": 1, "teamA": 4, "teamB": 1}, {"slot": 2, "teamA": 2, "teamB": 3}], "byes": []},
        ],
    },
    5: {
        "teamCount": 5,
        "rounds": [
            {"round": 1, "games": [{"slot": 1, "teamA": 1, "teamB": 4}, {"slot": 2, "teamA": 2, "teamB": 3}], "byes": [5]},
            {"round": 2, "games": [{"slot": 1, "teamA": 3, "teamB": 1}, {"slot": 2, "teamA": 4, "teamB": 5}], "byes": [2]},
            {"round": 3, "games": [{"slot": 1, "teamA": 5, "teamB": 3}, {"slot": 2, "teamA": 1, "teamB": 2}], "byes": [4]},
            {"round": 4, "games": [{"slot": 1, "teamA": 2, "teamB": 5}, {"slot": 2, "teamA": 3, "teamB": 4}], "byes": [1]},
            {"round": 5, "games": [{"slot": 1, "teamA": 2, "teamB": 4}, {"slot": 2, "teamA": 5, "teamB": 1}], "byes": [3]},
        ],
    },
    7: {
        "teamCount": 7,
        "rounds": [
            {"round": 1, "games": [{"slot": 1, "teamA": 1, "teamB": 6}, {"slot": 2, "teamA": 2, "teamB": 5}, {"slot": 3, "teamA": 3, "teamB": 4}], "byes": [7]},
            {"round": 2, "games": [{"slot": 1, "teamA": 4, "teamB": 2}, {"slot": 2, "teamA": 5, "teamB": 1}, {"slot": 3, "teamA": 6, "teamB": 7}], "byes": [3]},
            {"round": 3, "games": [{"slot": 1, "teamA": 2, "teamB": 7}, {"slot": 2, "teamA": 3, "teamB": 6}, {"slot": 3, "teamA": 4, "teamB": 5}], "byes": [1]},
            {"round": 4, "games": [{"slot": 1, "teamA": 5, "teamB": 3}, {"slot": 2, "teamA": 6, "teamB": 2}, {"slot": 3, "teamA": 7, "teamB": 1}], "byes": [4]},
            {"round": 5, "games": [{"slot": 1, "teamA": 3, "teamB": 1}, {"slot": 2, "teamA": 4, "teamB": 7}, {"slot": 3, "teamA": 5, "teamB": 6}], "byes": [2]},
            {"round": 6, "games": [{"slot": 1, "teamA": 6, "teamB": 4}, {"slot": 2, "teamA": 7, "teamB": 3}, {"slot": 3, "teamA": 1, "teamB": 2}], "byes": [5]},
            {"round": 7, "games": [{"slot": 1, "teamA": 7, "teamB": 5}, {"slot": 2, "teamA": 1, "teamB": 4}, {"slot": 3, "teamA": 2, "teamB": 3}], "byes": [6]},
        ],
    },
    8: {
        "teamCount": 8,
        "rounds": [
            {"round": 1, "games": [{"slot": 1, "teamA": 2, "teamB": 1}, {"slot": 2, "teamA": 3, "teamB": 8}, {"slot": 3, "teamA": 4, "teamB": 7}, {"slot": 4, "teamA": 5, "teamB": 6}], "byes": []},
            {"round": 2, "games": [{"slot": 1, "teamA": 3, "teamB": 4}, {"slot": 2, "teamA": 1, "teamB": 7}, {"slot": 3, "teamA": 8, "teamB": 6}, {"slot": 4, "teamA": 2, "teamB": 5}], "byes": []},
            {"round": 3, "games": [{"slot": 1, "teamA": 6, "teamB": 2}, {"slot": 2, "teamA": 7, "teamB": 8}, {"slot": 3, "teamA": 4, "teamB": 1}, {"slot": 4, "teamA": 5, "teamB": 3}], "byes": []},
            {"round": 4, "games": [{"slot": 1, "teamA": 7, "teamB": 5}, {"slot": 2, "teamA": 8, "teamB": 4}, {"slot": 3, "teamA": 2, "teamB": 3}, {"slot": 4, "teamA": 6, "teamB": 1}], "byes": []},
            {"round": 5, "games": [{"slot": 1, "teamA": 1, "teamB": 3}, {"slot": 2, "teamA": 4, "teamB": 2}, {"slot": 3, "teamA": 5, "teamB": 8}, {"slot": 4, "teamA": 6, "teamB": 7}], "byes": []},
            {"round": 6, "games": [{"slot": 1, "teamA": 4, "teamB": 5}, {"slot": 2, "teamA": 8, "teamB": 1}, {"slot": 3, "teamA": 2, "teamB": 7}, {"slot": 4, "teamA": 3, "teamB": 6}], "byes": []},
            {"round": 7, "games": [{"slot": 1, "teamA": 7, "teamB": 3}, {"slot": 2, "teamA": 8, "teamB": 2}, {"slot": 3, "teamA": 1, "teamB": 5}, {"slot": 4, "teamA": 6, "teamB": 4}], "byes": []},
        ],
    },
}

PLAYOFF_TEMPLATES: dict[int, dict[str, Any]] = {
    4: {"advanceCount": 4, "games": [
        {"id": "P1", "name": "Semifinal 1", "round": "SF", "slot": 1, "teamA": {"seed": 1}, "teamB": {"seed": 4}},
        {"id": "P2", "name": "Semifinal 2", "round": "SF", "slot": 2, "teamA": {"seed": 2}, "teamB": {"seed": 3}},
        {"id": "P3", "name": "Gold Medal Match", "round": "Final", "slot": 1, "teamA": {"winnerOf": "P1"}, "teamB": {"winnerOf": "P2"}},
        {"id": "P4", "name": "Bronze Medal Match", "round": "Bronze", "slot": 1, "teamA": {"loserOf": "P1"}, "teamB": {"loserOf": "P2"}},
    ]},
    5: {"advanceCount": 5, "games": [
        {"id": "P1", "name": "Play-in", "round": "QF", "slot": 1, "teamA": {"seed": 4}, "teamB": {"seed": 5}},
        {"id": "P2", "name": "Semifinal 1", "round": "SF", "slot": 1, "teamA": {"seed": 1}, "teamB": {"winnerOf": "P1"}},
        {"id": "P3", "name": "Semifinal 2", "round": "SF", "slot": 2, "teamA": {"seed": 2}, "teamB": {"seed": 3}},
        {"id": "P4", "name": "Gold Medal Match", "round": "Final", "slot": 1, "teamA": {"winnerOf": "P2"}, "teamB": {"winnerOf": "P3"}},
        {"id": "P5", "name": "Bronze Medal Match", "round": "Bronze", "slot": 1, "teamA": {"loserOf": "P2"}, "teamB": {"loserOf": "P3"}},
    ]},
    6: {"advanceCount": 6, "games": [
        {"id": "P1", "name": "Quarterfinal 1", "round": "QF", "slot": 1, "teamA": {"seed": 4}, "teamB": {"seed": 5}},
        {"id": "P2", "name": "Quarterfinal 2", "round": "QF", "slot": 2, "teamA": {"seed": 3}, "teamB": {"seed": 6}},
        {"id": "P3", "name": "Semifinal 1", "round": "SF", "slot": 1, "teamA": {"seed": 1}, "teamB": {"winnerOf": "P1"}},
        {"id": "P4", "name": "Semifinal 2", "round": "SF", "slot": 2, "teamA": {"seed": 2}, "teamB": {"winnerOf": "P2"}},
        {"id": "P5", "name": "Gold Medal Match", "round": "Final", "slot": 1, "teamA": {"winnerOf": "P3"}, "teamB": {"winnerOf": "P4"}},
        {"id": "P6", "name": "Bronze Medal Match", "round": "Bronze", "slot": 1, "teamA": {"loserOf": "P3"}, "teamB": {"loserOf": "P4"}},
    ]},
}


@dataclass
class TeamStanding:
    team_id: str
    team_number: int
    wins: int = 0
    losses: int = 0
    points_for: int = 0
    points_against: int = 0

    @property
    def differential(self) -> int:
        return self.points_for - self.points_against


def _load_rr6_template() -> dict[str, Any]:
    asset_path = Path(__file__).resolve().parents[2] / "ui" / "assets" / "tournaments" / "rr-6.csv"
    rounds: dict[int, list[dict[str, int]]] = {}
    with asset_path.open("r", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            round_number = int(row["round"])
            rounds.setdefault(round_number, []).append(
                {
                    "slot": int(row["slot"]),
                    "teamA": int(row["teamA"]),
                    "teamB": int(row["teamB"]),
                }
            )
    return {
        "teamCount": 6,
        "rounds": [
            {"round": round_number, "games": sorted(games, key=lambda game: game["slot"]), "byes": []}
            for round_number, games in sorted(rounds.items())
        ],
    }


def _round_robin_template(team_count: int) -> dict[str, Any] | None:
    if team_count == 6:
        return _load_rr6_template()
    configured = ROUND_ROBIN_TEMPLATES.get(team_count)
    if configured is not None:
        return configured
    if team_count not in SUPPORTED_TEAM_COUNTS:
        return None

    # Circle-method schedule for the larger setup-supported divisions.  A
    # ``None`` participant is the single rotating bye in odd-sized draws.
    # Every unordered pair appears exactly once and no team appears twice in
    # a round.  Existing 4-8 team templates remain unchanged for historical
    # schedule continuity.
    participants: list[int | None] = list(range(1, team_count + 1))
    if team_count % 2:
        participants.append(None)
    round_count = len(participants) - 1
    rounds: list[dict[str, Any]] = []
    rotation = list(participants)
    for round_index in range(round_count):
        games: list[dict[str, int]] = []
        byes: list[int] = []
        for pair_index in range(len(rotation) // 2):
            team_a = rotation[pair_index]
            team_b = rotation[-(pair_index + 1)]
            if team_a is None or team_b is None:
                bye = team_b if team_a is None else team_a
                if bye is not None:
                    byes.append(int(bye))
                continue
            # Alternate the fixed participant's side to avoid giving one team
            # the same orientation for an entire large round robin.
            if pair_index == 0 and round_index % 2:
                team_a, team_b = team_b, team_a
            games.append(
                {
                    "slot": len(games) + 1,
                    "teamA": int(team_a),
                    "teamB": int(team_b),
                }
            )
        rounds.append(
            {
                "round": round_index + 1,
                "games": games,
                "byes": sorted(byes),
            }
        )
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    return {"teamCount": team_count, "rounds": rounds}


def build_round_robin_games(*, tournament_id: str, team_ids_by_number: dict[int, str]) -> list[dict[str, Any]]:
    team_count = len(team_ids_by_number)
    template = _round_robin_template(team_count)
    if not template:
        raise ValueError(f"Unsupported team count: {team_count}")

    games: list[dict[str, Any]] = []
    for round_data in template["rounds"]:
        for game in round_data["games"]:
            games.append(
                {
                    "tournament_id": tournament_id,
                    "stage": "ROUND_ROBIN",
                    "rr_round_number": int(round_data["round"]),
                    "rr_slot_number": int(game["slot"]),
                    "team_a_id": team_ids_by_number[int(game["teamA"])],
                    "team_b_id": team_ids_by_number[int(game["teamB"])],
                }
            )
    return games


def compute_round_robin_standings(teams: list[dict[str, Any]], games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    standings = {t["id"]: TeamStanding(team_id=t["id"], team_number=int(t["team_number"])) for t in teams}
    for game in games:
        score_a, score_b = game.get("score_a"), game.get("score_b")
        if score_a is None or score_b is None:
            continue
        try:
            score_a, score_b = int(score_a), int(score_b)
        except Exception:
            continue
        if score_a == 0 and score_b == 0:
            continue
        team_a, team_b = standings.get(game.get("team_a_id")), standings.get(game.get("team_b_id"))
        if not team_a or not team_b:
            continue
        team_a.points_for += score_a
        team_a.points_against += score_b
        team_b.points_for += score_b
        team_b.points_against += score_a
        if score_a > score_b:
            team_a.wins += 1
            team_b.losses += 1
        elif score_b > score_a:
            team_b.wins += 1
            team_a.losses += 1

    standings_list = sorted(list(standings.values()), key=lambda s: (-s.wins, -s.differential, -s.points_for, s.team_number))
    grouped: dict[tuple[int, int, int], list[TeamStanding]] = {}
    for standing in standings_list:
        grouped.setdefault((standing.wins, standing.differential, standing.points_for), []).append(standing)

    resolved: list[TeamStanding] = []
    for group in grouped.values():
        if len(group) == 2:
            winner = _head_to_head_winner(group[0].team_id, group[1].team_id, games)
            if winner == group[0].team_id:
                resolved.extend([group[0], group[1]])
                continue
            if winner == group[1].team_id:
                resolved.extend([group[1], group[0]])
                continue
        resolved.extend(sorted(group, key=lambda s: s.team_number))

    return [
        {
            "team_id": standing.team_id,
            "team_number": standing.team_number,
            "wins": standing.wins,
            "losses": standing.losses,
            "points_for": standing.points_for,
            "points_against": standing.points_against,
            "differential": standing.differential,
            "seed": seed,
        }
        for seed, standing in enumerate(resolved, start=1)
    ]


def compute_podium_from_rr(teams: list[dict[str, Any]], games: list[dict[str, Any]], *, max_placements: int = 3) -> list[dict[str, Any]]:
    standings = compute_round_robin_standings(teams, games)
    return [
        {"placement": idx, "team_id": row["team_id"], "seed": row.get("seed")}
        for idx, row in enumerate(standings[: max(0, max_placements)], start=1)
    ]


def compute_podium_from_playoffs(games: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    final_game = next((g for g in games if g.get("playoff_round") == "Final"), None)
    bronze_game = next((g for g in games if g.get("playoff_round") == "Bronze"), None)
    if not final_game or not bronze_game:
        return None
    if not final_game.get("finalized_at") or not bronze_game.get("finalized_at"):
        return None
    if not final_game.get("winner_team_id") or not final_game.get("loser_team_id"):
        return None
    if not bronze_game.get("winner_team_id"):
        return None
    return [
        {"placement": 1, "team_id": final_game["winner_team_id"]},
        {"placement": 2, "team_id": final_game["loser_team_id"]},
        {"placement": 3, "team_id": bronze_game["winner_team_id"]},
    ]


def build_playoff_games(*, tournament_id: str, advance_count: int, standings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    template = PLAYOFF_TEMPLATES.get(int(advance_count))
    if not template:
        raise ValueError(f"Unsupported playoff advance count: {advance_count}")
    seed_map = {int(row["seed"]): row["team_id"] for row in standings}
    return [
        {
            "tournament_id": tournament_id,
            "stage": "PLAYOFF",
            "playoff_game_code": game["id"],
            "playoff_round": game["round"],
            "team_a_id": seed_map.get(int(game["teamA"]["seed"])) if "seed" in game["teamA"] else None,
            "team_b_id": seed_map.get(int(game["teamB"]["seed"])) if "seed" in game["teamB"] else None,
            "team_a_source": game["teamA"],
            "team_b_source": game["teamB"],
        }
        for game in template["games"]
    ]


def resolve_playoff_dependencies(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updates: dict[str, dict[str, Any]] = {}

    def resolve_source(source: Any, by_code: dict[str, dict[str, Any]]) -> tuple[bool, str | None]:
        if not isinstance(source, dict):
            return False, None
        if "winnerOf" in source:
            return True, by_code.get(source["winnerOf"], {}).get("winner_team_id")
        if "loserOf" in source:
            return True, by_code.get(source["loserOf"], {}).get("loser_team_id")
        return False, None

    def set_update(game: dict[str, Any], field: str, value: Any) -> None:
        updates.setdefault(game["id"], {"id": game["id"]})[field] = value

    local_games = [dict(game) for game in games]
    changed = True
    while changed:
        changed = False
        by_code = {game.get("playoff_game_code"): game for game in local_games if game.get("playoff_game_code")}
        for game in local_games:
            if game.get("stage") != "PLAYOFF":
                continue
            dep_a, desired_a = resolve_source(game.get("team_a_source"), by_code)
            dep_b, desired_b = resolve_source(game.get("team_b_source"), by_code)
            for key, is_dep, desired in (("team_a_id", dep_a, desired_a), ("team_b_id", dep_b, desired_b)):
                if not is_dep:
                    continue
                if desired is None and game.get(key) is not None:
                    set_update(game, key, None)
                    _clear_game_results(game, updates)
                    game[key] = None
                    changed = True
                elif desired is not None and game.get(key) != desired:
                    set_update(game, key, desired)
                    _clear_game_results(game, updates)
                    game[key] = desired
                    changed = True
            for field in ["winner_team_id", "loser_team_id", "finalized_at", "score_a", "score_b"]:
                if field in updates.get(game["id"], {}):
                    game[field] = updates[game["id"]].get(field)
    return list(updates.values())


def _head_to_head_winner(team_a_id: str, team_b_id: str, games: list[dict[str, Any]]) -> str | None:
    for game in games:
        if {game.get("team_a_id"), game.get("team_b_id")} != {team_a_id, team_b_id}:
            continue
        score_a, score_b = game.get("score_a"), game.get("score_b")
        if score_a is None or score_b is None:
            continue
        try:
            score_a, score_b = int(score_a), int(score_b)
        except Exception:
            continue
        if score_a == score_b:
            return None
        return game.get("team_a_id") if score_a > score_b else game.get("team_b_id")
    return None


def _clear_game_results(game: dict[str, Any], updates: dict[str, dict[str, Any]]) -> None:
    updates.setdefault(game["id"], {"id": game["id"]}).update(
        {"score_a": None, "score_b": None, "winner_team_id": None, "loser_team_id": None, "finalized_at": None}
    )


def finalize_game(game: dict[str, Any]) -> dict[str, Any]:
    score_a = int(game.get("score_a") or 0)
    score_b = int(game.get("score_b") or 0)
    if score_a <= 0 and score_b <= 0:
        raise ValueError("Scores are required to finalize a game.")
    if score_a == score_b:
        raise ValueError("Ties are not supported for tournament games.")
    if game.get("team_a_id") is None or game.get("team_b_id") is None:
        raise ValueError("Both teams must be assigned before scoring.")
    winner_id, loser_id = (game.get("team_a_id"), game.get("team_b_id")) if score_a > score_b else (game.get("team_b_id"), game.get("team_a_id"))
    return {
        "score_a": score_a,
        "score_b": score_b,
        "winner_team_id": winner_id,
        "loser_team_id": loser_id,
        "finalized_at": datetime.now(timezone.utc).isoformat(),
    }
