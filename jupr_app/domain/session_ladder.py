from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CourtGameTemplate:
    game_number: int
    sit_out_player_index: int | None
    team_a: tuple[int, int]
    team_b: tuple[int, int]


@dataclass(frozen=True)
class CourtGameResult:
    team_a: tuple[int, int]
    team_b: tuple[int, int]
    score_a: int
    score_b: int


@dataclass(frozen=True)
class PlayerStats:
    player_id: int
    wins: int
    losses: int
    points_for: int
    points_against: int

    @property
    def point_differential(self) -> int:
        return self.points_for - self.points_against


def round_robin_template(players_per_court: int) -> list[CourtGameTemplate]:
    if players_per_court == 4:
        return [
            CourtGameTemplate(game_number=1, sit_out_player_index=None, team_a=(0, 1), team_b=(2, 3)),
            CourtGameTemplate(game_number=2, sit_out_player_index=None, team_a=(0, 2), team_b=(1, 3)),
            CourtGameTemplate(game_number=3, sit_out_player_index=None, team_a=(0, 3), team_b=(1, 2)),
        ]
    if players_per_court == 5:
        return [
            CourtGameTemplate(game_number=1, sit_out_player_index=0, team_a=(1, 2), team_b=(3, 4)),
            CourtGameTemplate(game_number=2, sit_out_player_index=1, team_a=(0, 3), team_b=(2, 4)),
            CourtGameTemplate(game_number=3, sit_out_player_index=2, team_a=(0, 4), team_b=(1, 3)),
            CourtGameTemplate(game_number=4, sit_out_player_index=3, team_a=(0, 1), team_b=(2, 4)),
            CourtGameTemplate(game_number=5, sit_out_player_index=4, team_a=(0, 2), team_b=(1, 3)),
        ]
    raise ValueError("players_per_court must be 4 or 5")


def compute_court_stats(player_ids: list[int], results: list[CourtGameResult]) -> list[PlayerStats]:
    stats: dict[int, dict[str, int]] = {
        int(pid): {"wins": 0, "losses": 0, "pf": 0, "pa": 0} for pid in player_ids
    }
    for game in results:
        winners = game.team_a if game.score_a > game.score_b else game.team_b
        losers = game.team_b if game.score_a > game.score_b else game.team_a
        for pid in game.team_a:
            stats[int(pid)]["pf"] += int(game.score_a)
            stats[int(pid)]["pa"] += int(game.score_b)
        for pid in game.team_b:
            stats[int(pid)]["pf"] += int(game.score_b)
            stats[int(pid)]["pa"] += int(game.score_a)
        for pid in winners:
            stats[int(pid)]["wins"] += 1
        for pid in losers:
            stats[int(pid)]["losses"] += 1
    return [
        PlayerStats(
            player_id=pid,
            wins=vals["wins"],
            losses=vals["losses"],
            points_for=vals["pf"],
            points_against=vals["pa"],
        )
        for pid, vals in stats.items()
    ]


def rank_players_with_tiebreak(player_ids: list[int], results: list[CourtGameResult]) -> list[int]:
    stats = {row.player_id: row for row in compute_court_stats(player_ids, results)}
    h2h = _head_to_head_wins(player_ids, results)

    def _sort_key(pid: int) -> tuple[int, int, int, int, int]:
        row = stats[pid]
        return (
            row.wins,
            row.point_differential,
            row.points_for,
            h2h.get(pid, 0),
            -player_ids.index(pid),
        )

    return sorted(player_ids, key=_sort_key, reverse=True)


def _head_to_head_wins(player_ids: list[int], results: list[CourtGameResult]) -> dict[int, int]:
    tied_ids = {int(pid) for pid in player_ids}
    values = {int(pid): 0 for pid in player_ids}
    for game in results:
        a = {int(pid) for pid in game.team_a if int(pid) in tied_ids}
        b = {int(pid) for pid in game.team_b if int(pid) in tied_ids}
        if not a or not b:
            continue
        if game.score_a > game.score_b:
            for pid in a:
                values[pid] += 1
        elif game.score_b > game.score_a:
            for pid in b:
                values[pid] += 1
    return values


def apply_adjacent_court_movement(
    ranked_courts: list[list[int]],
    *,
    movers_per_boundary: int = 1,
) -> list[list[int]]:
    if movers_per_boundary <= 0:
        raise ValueError("movers_per_boundary must be positive")
    if not ranked_courts:
        return []

    moved = [list(court) for court in ranked_courts]
    for index in range(len(ranked_courts) - 1):
        upper = ranked_courts[index]
        lower = ranked_courts[index + 1]
        if movers_per_boundary >= len(upper) or movers_per_boundary >= len(lower):
            raise ValueError("movers_per_boundary must be smaller than each court size")

        upper_out = upper[-movers_per_boundary:]
        lower_out = lower[:movers_per_boundary]

        moved[index] = upper[:-movers_per_boundary] + lower_out
        moved[index + 1] = upper_out + lower[movers_per_boundary:]
        ranked_courts = [list(court) for court in moved]
    return moved


SESSION_TRANSITIONS: dict[str, dict[str, str]] = {
    "draft": {"start": "active"},
    "active": {"complete": "completed", "cancel": "cancelled"},
    "completed": {"reopen": "active", "archive": "archived"},
    "cancelled": {},
    "archived": {},
}


def transition_session_state(current_state: str, action: str) -> str:
    normalized_state = str(current_state or "").strip().lower()
    normalized_action = str(action or "").strip().lower()
    next_state = SESSION_TRANSITIONS.get(normalized_state, {}).get(normalized_action)
    if not next_state:
        raise ValueError(f"invalid transition: {current_state} -> {action}")
    return next_state


def build_session_resume_pointer(session_id: str, round_number: int, court_id: str) -> dict[str, Any]:
    sid = str(session_id or "").strip()
    cid = str(court_id or "").strip()
    if not sid or not cid:
        raise ValueError("session_id and court_id are required")
    if int(round_number) <= 0:
        raise ValueError("round_number must be positive")
    route = f"/sessions/{sid}/rounds/{int(round_number)}/courts/{cid}"
    return {"session_id": sid, "round_number": int(round_number), "court_id": cid, "route": route}
