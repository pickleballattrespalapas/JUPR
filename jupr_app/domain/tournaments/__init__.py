from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


ROUND_ROBIN_TEMPLATES: dict[int, dict[str, Any]] = {
    4: {
        "teamCount": 4,
        "rounds": [
            {
                "round": 1,
                "games": [
                    {"slot": 1, "teamA": 2, "teamB": 1},
                    {"slot": 2, "teamA": 3, "teamB": 4},
                ],
                "byes": [],
            },
            {
                "round": 2,
                "games": [
                    {"slot": 1, "teamA": 4, "teamB": 2},
                    {"slot": 2, "teamA": 1, "teamB": 3},
                ],
                "byes": [],
            },
            {
                "round": 3,
                "games": [
                    {"slot": 1, "teamA": 4, "teamB": 1},
                    {"slot": 2, "teamA": 2, "teamB": 3},
                ],
                "byes": [],
            },
        ],
    },
    5: {
        "teamCount": 5,
        "rounds": [
            {
                "round": 1,
                "games": [
                    {"slot": 1, "teamA": 1, "teamB": 4},
                    {"slot": 2, "teamA": 2, "teamB": 3},
                ],
                "byes": [5],
            },
            {
                "round": 2,
                "games": [
                    {"slot": 1, "teamA": 3, "teamB": 1},
                    {"slot": 2, "teamA": 4, "teamB": 5},
                ],
                "byes": [2],
            },
            {
                "round": 3,
                "games": [
                    {"slot": 1, "teamA": 5, "teamB": 3},
                    {"slot": 2, "teamA": 1, "teamB": 2},
                ],
                "byes": [4],
            },
            {
                "round": 4,
                "games": [
                    {"slot": 1, "teamA": 2, "teamB": 5},
                    {"slot": 2, "teamA": 3, "teamB": 4},
                ],
                "byes": [1],
            },
            {
                "round": 5,
                "games": [
                    {"slot": 1, "teamA": 2, "teamB": 4},
                    {"slot": 2, "teamA": 5, "teamB": 1},
                ],
                "byes": [3],
            },
        ],
    },
    6: {
        "teamCount": 6,
        "rounds": [
            {
                "round": 1,
                "games": [
                    {"slot": 1, "teamA": 2, "teamB": 1},
                    {"slot": 2, "teamA": 3, "teamB": 6},
                    {"slot": 3, "teamA": 4, "teamB": 5},
                ],
                "byes": [],
            },
            {
                "round": 2,
                "games": [
                    {"slot": 1, "teamA": 3, "teamB": 4},
                    {"slot": 2, "teamA": 6, "teamB": 1},
                    {"slot": 3, "teamA": 2, "teamB": 5},
                ],
                "byes": [],
            },
            {
                "round": 3,
                "games": [
                    {"slot": 1, "teamA": 6, "teamB": 4},
                    {"slot": 2, "teamA": 2, "teamB": 3},
                    {"slot": 3, "teamA": 1, "teamB": 5},
                ],
                "byes": [],
            },
            {
                "round": 4,
                "games": [
                    {"slot": 1, "teamA": 4, "teamB": 1},
                    {"slot": 2, "teamA": 5, "teamB": 3},
                    {"slot": 3, "teamA": 2, "teamB": 6},
                ],
                "byes": [],
            },
            {
                "round": 5,
                "games": [
                    {"slot": 1, "teamA": 5, "teamB": 6},
                    {"slot": 2, "teamA": 1, "teamB": 3},
                    {"slot": 3, "teamA": 2, "teamB": 4},
                ],
                "byes": [],
            },
        ],
    },
    7: {
        "teamCount": 7,
        "rounds": [
            {
                "round": 1,
                "games": [
                    {"slot": 1, "teamA": 1, "teamB": 6},
                    {"slot": 2, "teamA": 2, "teamB": 5},
                    {"slot": 3, "teamA": 3, "teamB": 4},
                ],
                "byes": [7],
            },
            {
                "round": 2,
                "games": [
                    {"slot": 1, "teamA": 4, "teamB": 2},
                    {"slot": 2, "teamA": 5, "teamB": 1},
                    {"slot": 3, "teamA": 6, "teamB": 7},
                ],
                "byes": [3],
            },
            {
                "round": 3,
                "games": [
                    {"slot": 1, "teamA": 2, "teamB": 7},
                    {"slot": 2, "teamA": 3, "teamB": 6},
                    {"slot": 3, "teamA": 4, "teamB": 5},
                ],
                "byes": [1],
            },
            {
                "round": 4,
                "games": [
                    {"slot": 1, "teamA": 5, "teamB": 3},
                    {"slot": 2, "teamA": 6, "teamB": 2},
                    {"slot": 3, "teamA": 7, "teamB": 1},
                ],
                "byes": [4],
            },
            {
                "round": 5,
                "games": [
                    {"slot": 1, "teamA": 3, "teamB": 1},
                    {"slot": 2, "teamA": 4, "teamB": 7},
                    {"slot": 3, "teamA": 5, "teamB": 6},
                ],
                "byes": [2],
            },
            {
                "round": 6,
                "games": [
                    {"slot": 1, "teamA": 6, "teamB": 4},
                    {"slot": 2, "teamA": 7, "teamB": 3},
                    {"slot": 3, "teamA": 1, "teamB": 2},
                ],
                "byes": [5],
            },
            {
                "round": 7,
                "games": [
                    {"slot": 1, "teamA": 7, "teamB": 5},
                    {"slot": 2, "teamA": 1, "teamB": 4},
                    {"slot": 3, "teamA": 2, "teamB": 3},
                ],
                "byes": [6],
            },
        ],
    },
    8: {
        "teamCount": 8,
        "rounds": [
            {
                "round": 1,
                "games": [
                    {"slot": 1, "teamA": 2, "teamB": 1},
                    {"slot": 2, "teamA": 3, "teamB": 8},
                    {"slot": 3, "teamA": 4, "teamB": 7},
                    {"slot": 4, "teamA": 5, "teamB": 6},
                ],
                "byes": [],
            },
            {
                "round": 2,
                "games": [
                    {"slot": 1, "teamA": 3, "teamB": 4},
                    {"slot": 2, "teamA": 1, "teamB": 7},
                    {"slot": 3, "teamA": 8, "teamB": 6},
                    {"slot": 4, "teamA": 2, "teamB": 5},
                ],
                "byes": [],
            },
            {
                "round": 3,
                "games": [
                    {"slot": 1, "teamA": 6, "teamB": 2},
                    {"slot": 2, "teamA": 7, "teamB": 8},
                    {"slot": 3, "teamA": 4, "teamB": 1},
                    {"slot": 4, "teamA": 5, "teamB": 3},
                ],
                "byes": [],
            },
            {
                "round": 4,
                "games": [
                    {"slot": 1, "teamA": 7, "teamB": 5},
                    {"slot": 2, "teamA": 8, "teamB": 4},
                    {"slot": 3, "teamA": 2, "teamB": 3},
                    {"slot": 4, "teamA": 6, "teamB": 1},
                ],
                "byes": [],
            },
            {
                "round": 5,
                "games": [
                    {"slot": 1, "teamA": 1, "teamB": 3},
                    {"slot": 2, "teamA": 4, "teamB": 2},
                    {"slot": 3, "teamA": 5, "teamB": 8},
                    {"slot": 4, "teamA": 6, "teamB": 7},
                ],
                "byes": [],
            },
            {
                "round": 6,
                "games": [
                    {"slot": 1, "teamA": 4, "teamB": 5},
                    {"slot": 2, "teamA": 8, "teamB": 1},
                    {"slot": 3, "teamA": 2, "teamB": 7},
                    {"slot": 4, "teamA": 3, "teamB": 6},
                ],
                "byes": [],
            },
            {
                "round": 7,
                "games": [
                    {"slot": 1, "teamA": 7, "teamB": 3},
                    {"slot": 2, "teamA": 8, "teamB": 2},
                    {"slot": 3, "teamA": 1, "teamB": 5},
                    {"slot": 4, "teamA": 6, "teamB": 4},
                ],
                "byes": [],
            },
        ],
    },
}

PLAYOFF_TEMPLATES: dict[int, dict[str, Any]] = {
    4: {
        "advanceCount": 4,
        "games": [
            {"id": "P1", "name": "Semifinal 1", "round": "SF", "slot": 1, "teamA": {"seed": 1}, "teamB": {"seed": 4}},
            {"id": "P2", "name": "Semifinal 2", "round": "SF", "slot": 2, "teamA": {"seed": 2}, "teamB": {"seed": 3}},
            {"id": "P3", "name": "Gold Medal Match", "round": "Final", "slot": 1, "teamA": {"winnerOf": "P1"}, "teamB": {"winnerOf": "P2"}},
            {"id": "P4", "name": "Bronze Medal Match", "round": "Bronze", "slot": 1, "teamA": {"loserOf": "P1"}, "teamB": {"loserOf": "P2"}},
        ],
    },
    5: {
        "advanceCount": 5,
        "games": [
            {"id": "P1", "name": "Play-in", "round": "QF", "slot": 1, "teamA": {"seed": 4}, "teamB": {"seed": 5}},
            {"id": "P2", "name": "Semifinal 1", "round": "SF", "slot": 1, "teamA": {"seed": 1}, "teamB": {"winnerOf": "P1"}},
            {"id": "P3", "name": "Semifinal 2", "round": "SF", "slot": 2, "teamA": {"seed": 2}, "teamB": {"seed": 3}},
            {"id": "P4", "name": "Gold Medal Match", "round": "Final", "slot": 1, "teamA": {"winnerOf": "P2"}, "teamB": {"winnerOf": "P3"}},
            {"id": "P5", "name": "Bronze Medal Match", "round": "Bronze", "slot": 1, "teamA": {"loserOf": "P2"}, "teamB": {"loserOf": "P3"}},
        ],
    },
    6: {
        "advanceCount": 6,
        "games": [
            {"id": "P1", "name": "Quarterfinal 1", "round": "QF", "slot": 1, "teamA": {"seed": 4}, "teamB": {"seed": 5}},
            {"id": "P2", "name": "Quarterfinal 2", "round": "QF", "slot": 2, "teamA": {"seed": 3}, "teamB": {"seed": 6}},
            {"id": "P3", "name": "Semifinal 1", "round": "SF", "slot": 1, "teamA": {"seed": 1}, "teamB": {"winnerOf": "P1"}},
            {"id": "P4", "name": "Semifinal 2", "round": "SF", "slot": 2, "teamA": {"seed": 2}, "teamB": {"winnerOf": "P2"}},
            {"id": "P5", "name": "Gold Medal Match", "round": "Final", "slot": 1, "teamA": {"winnerOf": "P3"}, "teamB": {"winnerOf": "P4"}},
            {"id": "P6", "name": "Bronze Medal Match", "round": "Bronze", "slot": 1, "teamA": {"loserOf": "P3"}, "teamB": {"loserOf": "P4"}},
        ],
    },
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


def build_round_robin_games(*, tournament_id: str, team_ids_by_number: dict[int, str]) -> list[dict[str, Any]]:
    team_count = len(team_ids_by_number)
    template = ROUND_ROBIN_TEMPLATES.get(team_count)
    if not template:
        raise ValueError(f"Unsupported team count: {team_count}")

    games: list[dict[str, Any]] = []
    for round_data in template["rounds"]:
        for game in round_data["games"]:
            team_a_num = int(game["teamA"])
            team_b_num = int(game["teamB"])
            games.append(
                {
                    "tournament_id": tournament_id,
                    "stage": "ROUND_ROBIN",
                    "rr_round_number": int(round_data["round"]),
                    "rr_slot_number": int(game["slot"]),
                    "team_a_id": team_ids_by_number[team_a_num],
                    "team_b_id": team_ids_by_number[team_b_num],
                }
            )
    return games


def compute_round_robin_standings(
    teams: list[dict[str, Any]],
    games: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    standings = {t["id"]: TeamStanding(team_id=t["id"], team_number=int(t["team_number"])) for t in teams}

    for game in games:
        score_a = game.get("score_a")
        score_b = game.get("score_b")
        if score_a is None or score_b is None:
            continue
        try:
            score_a = int(score_a)
            score_b = int(score_b)
        except Exception:
            continue
        if score_a == 0 and score_b == 0:
            continue

        team_a = standings.get(game.get("team_a_id"))
        team_b = standings.get(game.get("team_b_id"))
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

    standings_list = list(standings.values())

    def base_key(s: TeamStanding):
        return (-s.wins, -s.differential, -s.points_for, s.team_number)

    standings_list.sort(key=base_key)

    grouped: dict[tuple[int, int, int], list[TeamStanding]] = {}
    for s in standings_list:
        key = (s.wins, s.differential, s.points_for)
        grouped.setdefault(key, []).append(s)

    resolved: list[TeamStanding] = []
    for key, group in grouped.items():
        if len(group) == 2:
            a, b = group
            head_to_head_winner = _head_to_head_winner(a.team_id, b.team_id, games)
            if head_to_head_winner == a.team_id:
                resolved.extend([a, b])
                continue
            if head_to_head_winner == b.team_id:
                resolved.extend([b, a])
                continue
        resolved.extend(sorted(group, key=lambda s: s.team_number))

    results = []
    for idx, s in enumerate(resolved, start=1):
        results.append(
            {
                "team_id": s.team_id,
                "team_number": s.team_number,
                "wins": s.wins,
                "losses": s.losses,
                "points_for": s.points_for,
                "points_against": s.points_against,
                "differential": s.differential,
                "seed": idx,
            }
        )
    return results


def compute_podium_from_rr(
    teams: list[dict[str, Any]],
    games: list[dict[str, Any]],
    *,
    max_placements: int = 3,
) -> list[dict[str, Any]]:
    standings = compute_round_robin_standings(teams, games)
    placements: list[dict[str, Any]] = []
    for idx, row in enumerate(standings[: max(0, max_placements)], start=1):
        placements.append({"placement": idx, "team_id": row["team_id"], "seed": row.get("seed")})
    return placements


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


def validate_podium_placements(
    placements: list[dict[str, Any]],
    *,
    max_placements: int = 3,
) -> None:
    seen_team_ids: set[str] = set()
    for placement in placements:
        place = int(placement.get("placement", 0) or 0)
        if place < 1 or place > max_placements:
            raise ValueError("Podium placement must be between 1 and 3.")
        team_id = placement.get("team_id")
        if not team_id:
            raise ValueError("Podium placement requires a team.")
        if team_id in seen_team_ids:
            raise ValueError("Podium placements must use distinct teams.")
        seen_team_ids.add(team_id)


def build_podium_payload(
    tournament_id: str,
    placements: list[dict[str, Any]],
    source: str,
) -> list[dict[str, Any]]:
    ordered = sorted(placements, key=lambda row: int(row.get("placement", 0) or 0))
    validate_podium_placements(ordered, max_placements=3)
    payload = []
    for row in ordered:
        payload.append(
            {
                "tournament_id": tournament_id,
                "placement": int(row["placement"]),
                "team_id": row["team_id"],
                "source": source,
            }
        )
    return payload


def _head_to_head_winner(team_a_id: str, team_b_id: str, games: list[dict[str, Any]]) -> str | None:
    for game in games:
        if {
            game.get("team_a_id"),
            game.get("team_b_id"),
        } != {team_a_id, team_b_id}:
            continue
        score_a = game.get("score_a")
        score_b = game.get("score_b")
        if score_a is None or score_b is None:
            continue
        try:
            score_a = int(score_a)
            score_b = int(score_b)
        except Exception:
            continue
        if score_a == score_b:
            return None
        return game.get("team_a_id") if score_a > score_b else game.get("team_b_id")
    return None


def build_playoff_games(
    *,
    tournament_id: str,
    advance_count: int,
    standings: list[dict[str, Any]],
    best_of: int = 1,
) -> list[dict[str, Any]]:
    template = PLAYOFF_TEMPLATES.get(int(advance_count))
    if not template:
        raise ValueError(f"Unsupported playoff advance count: {advance_count}")

    seed_map = {int(row["seed"]): row["team_id"] for row in standings}
    games: list[dict[str, Any]] = []

    for template_game in template["games"]:
        team_a_source = template_game["teamA"]
        team_b_source = template_game["teamB"]
        team_a_id = seed_map.get(int(team_a_source["seed"])) if "seed" in team_a_source else None
        team_b_id = seed_map.get(int(team_b_source["seed"])) if "seed" in team_b_source else None
        for series_game in range(1, best_of + 1):
            games.append(
                {
                    "tournament_id": tournament_id,
                    "stage": "PLAYOFF",
                    "playoff_game_code": template_game["id"],
                    "series_game_number": series_game,
                    "playoff_round": template_game["round"],
                    "team_a_id": team_a_id,
                    "team_b_id": team_b_id,
                    "team_a_source": team_a_source,
                    "team_b_source": team_b_source,
                }
            )

    return games


def resolve_series_results(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Determine series winner for playoff games.

    For best-of-3:
        First team to 2 wins advances.
    For single game:
        Normal winner logic applies.

    Returns:
        List of updates compatible with sb_update.
    """

    updates = []
    series_groups: dict[str, list[dict[str, Any]]] = {}

    for game in games:
        if game.get("stage") != "PLAYOFF":
            continue
        code = game.get("playoff_game_code")
        if not code:
            continue
        series_groups.setdefault(code, []).append(game)

    for series_games in series_groups.values():
        ordered_games = sorted(series_games, key=_series_game_sort_key)
        required_wins = 2 if len(ordered_games) >= 3 else 1
        wins: dict[str, int] = {}

        for game in ordered_games:
            winner = game.get("winner_team_id")
            if not winner:
                continue
            wins[winner] = wins.get(winner, 0) + 1

        winning_team_id = next((team_id for team_id, win_count in wins.items() if win_count >= required_wins), None)
        if not winning_team_id:
            continue

        deciding_game = None
        win_counter = 0
        for game in ordered_games:
            if game.get("winner_team_id") != winning_team_id:
                continue
            win_counter += 1
            if win_counter == required_wins:
                deciding_game = game
                break

        if not deciding_game:
            continue

        loser_id = None
        deciding_team_a = deciding_game.get("team_a_id")
        deciding_team_b = deciding_game.get("team_b_id")
        if deciding_team_a and deciding_team_b:
            if deciding_team_a == winning_team_id:
                loser_id = deciding_team_b
            elif deciding_team_b == winning_team_id:
                loser_id = deciding_team_a

        if loser_id is None:
            for team_id in wins:
                if str(team_id) != str(winning_team_id):
                    loser_id = team_id
                    break

        if loser_id is None:
            participants_in_order: list[Any] = []
            seen_participants: set[str] = set()
            for game in ordered_games:
                for participant in (game.get("team_a_id"), game.get("team_b_id")):
                    participant_key = str(participant or "").strip()
                    if not participant_key or participant_key in seen_participants:
                        continue
                    seen_participants.add(participant_key)
                    participants_in_order.append(participant)

            winning_team_key = str(winning_team_id)
            for participant_id in participants_in_order:
                if str(participant_id) != winning_team_key:
                    loser_id = participant_id
                    break

        updates.append(
            {
                "id": deciding_game["id"],
                "winner_team_id": winning_team_id,
                "loser_team_id": loser_id,
                "finalized_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    return updates


def resolve_playoff_dependencies(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updates: dict[str, dict[str, Any]] = {}
    local_games = [dict(game) for game in games if game.get("stage") == "PLAYOFF" and game.get("playoff_game_code")]

    def set_update(game_id: str, field: str, value: Any) -> None:
        updates.setdefault(game_id, {"id": game_id})[field] = value

    changed = True
    while changed:
        changed = False
        series_groups: dict[str, list[dict[str, Any]]] = {}
        for game in local_games:
            series_groups.setdefault(game["playoff_game_code"], []).append(game)

        series_outcomes = _compute_series_outcomes(series_groups)

        def resolve_source(source: Any) -> tuple[bool, str | None]:
            parsed = _parse_source(source)
            if not parsed:
                return False, None
            if "winnerOf" in parsed:
                return True, series_outcomes.get(parsed["winnerOf"], {}).get("winner_team_id")
            if "loserOf" in parsed:
                return True, series_outcomes.get(parsed["loserOf"], {}).get("loser_team_id")
            return False, None

        for code, series_games in series_groups.items():
            anchor = _pick_series_anchor(series_games)
            dep_a, desired_team_a = resolve_source(anchor.get("team_a_source"))
            dep_b, desired_team_b = resolve_source(anchor.get("team_b_source"))

            for series_game in series_games:
                participants_changed = False

                for key, is_dependency, desired_value in (
                    ("team_a_id", dep_a, desired_team_a),
                    ("team_b_id", dep_b, desired_team_b),
                ):
                    if not is_dependency:
                        continue
                    if series_game.get(key) == desired_value:
                        continue
                    set_update(series_game["id"], key, desired_value)
                    series_game[key] = desired_value
                    participants_changed = True

                if participants_changed:
                    _clear_game_results(series_game, updates)
                    changed = True

    return list(updates.values())


def _series_game_sort_key(game: dict[str, Any]) -> tuple[int, str]:
    return int(game.get("series_game_number") or 1), str(game.get("id") or "")


def _pick_series_anchor(series_games: list[dict[str, Any]]) -> dict[str, Any]:
    ordered_games = sorted(series_games, key=_series_game_sort_key)
    return next((game for game in ordered_games if int(game.get("series_game_number") or 1) == 1), ordered_games[0])


def _compute_series_outcomes(series_groups: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    outcomes: dict[str, dict[str, Any]] = {}
    for code, series_games in series_groups.items():
        ordered_games = sorted(series_games, key=_series_game_sort_key)
        required_wins = 2 if len(ordered_games) >= 3 else 1
        wins: dict[str, int] = {}

        for game in ordered_games:
            winner = game.get("winner_team_id")
            if winner:
                wins[winner] = wins.get(winner, 0) + 1

        winner_team_id = next((team_id for team_id, win_count in wins.items() if win_count >= required_wins), None)
        loser_team_id = None
        if winner_team_id:
            participants = {
                game.get("team_a_id") for game in ordered_games if game.get("team_a_id")
            } | {
                game.get("team_b_id") for game in ordered_games if game.get("team_b_id")
            }
            participants.discard(winner_team_id)
            loser_team_id = next(iter(participants), None)

        outcomes[code] = {
            "winner_team_id": winner_team_id,
            "loser_team_id": loser_team_id,
        }

    return outcomes


def _parse_source(source: Any) -> dict[str, Any] | None:
    if source is None:
        return None
    if isinstance(source, dict):
        return source
    return None


def _clear_game_results(game: dict[str, Any], updates: dict[str, dict[str, Any]]) -> None:
    gid = game["id"]
    upd = updates.setdefault(gid, {"id": gid})
    upd.update(
        {
            "score_a": None,
            "score_b": None,
            "winner_team_id": None,
            "loser_team_id": None,
            "finalized_at": None,
        }
    )
    game["score_a"] = None
    game["score_b"] = None
    game["winner_team_id"] = None
    game["loser_team_id"] = None
    game["finalized_at"] = None


def finalize_game(game: dict[str, Any]) -> dict[str, Any]:
    score_a = int(game.get("score_a") or 0)
    score_b = int(game.get("score_b") or 0)
    if score_a <= 0 and score_b <= 0:
        raise ValueError("Scores are required to finalize a game.")
    if score_a == score_b:
        raise ValueError("Ties are not supported for tournament games.")
    if game.get("team_a_id") is None or game.get("team_b_id") is None:
        raise ValueError("Both teams must be assigned before scoring.")

    if score_a > score_b:
        winner_id = game.get("team_a_id")
        loser_id = game.get("team_b_id")
    else:
        winner_id = game.get("team_b_id")
        loser_id = game.get("team_a_id")

    return {
        "score_a": score_a,
        "score_b": score_b,
        "winner_team_id": winner_id,
        "loser_team_id": loser_id,
        "finalized_at": datetime.now(timezone.utc).isoformat(),
    }
