from __future__ import annotations

from collections import defaultdict
from typing import Any


def generateRoundGames(players: list[int], templateType: str) -> list[dict[str, Any]]:
    players = [int(pid) for pid in players]
    template = str(templateType or "").strip().lower()
    if template in {"4", "4p", "four"}:
        if len(players) != 4:
            raise ValueError("4-player template requires exactly 4 players")
        return [
            {"game_number": 1, "teamA": [players[0], players[1]], "teamB": [players[2], players[3]], "sit_out": None},
            {"game_number": 2, "teamA": [players[0], players[2]], "teamB": [players[1], players[3]], "sit_out": None},
            {"game_number": 3, "teamA": [players[0], players[3]], "teamB": [players[1], players[2]], "sit_out": None},
        ]

    if template in {"5", "5p", "five"}:
        if len(players) != 5:
            raise ValueError("5-player template requires exactly 5 players")
        return [
            {"game_number": 1, "teamA": [players[1], players[2]], "teamB": [players[3], players[4]], "sit_out": players[0]},
            {"game_number": 2, "teamA": [players[0], players[3]], "teamB": [players[2], players[4]], "sit_out": players[1]},
            {"game_number": 3, "teamA": [players[0], players[4]], "teamB": [players[1], players[3]], "sit_out": players[2]},
            {"game_number": 4, "teamA": [players[0], players[1]], "teamB": [players[2], players[4]], "sit_out": players[3]},
            {"game_number": 5, "teamA": [players[0], players[2]], "teamB": [players[1], players[3]], "sit_out": players[4]},
        ]

    raise ValueError("templateType must be one of: 4p, 5p")


def computeCourtStandings(games: list[dict[str, Any]], players: list[int]) -> list[dict[str, Any]]:
    stats: dict[int, dict[str, Any]] = {
        int(pid): {
            "player_id": int(pid),
            "wins": 0,
            "losses": 0,
            "pf": 0,
            "pa": 0,
            "pd": 0,
            "h2h_opponent_wins": 0,
            "playoff_required": False,
        }
        for pid in players
    }

    for game in games:
        team_a = [int(pid) for pid in game.get("teamA", [])]
        team_b = [int(pid) for pid in game.get("teamB", [])]
        score_a = int(game.get("scoreA", 0) or 0)
        score_b = int(game.get("scoreB", 0) or 0)

        if not team_a or not team_b:
            continue

        for pid in team_a:
            if pid in stats:
                stats[pid]["pf"] += score_a
                stats[pid]["pa"] += score_b
        for pid in team_b:
            if pid in stats:
                stats[pid]["pf"] += score_b
                stats[pid]["pa"] += score_a

        if score_a == score_b:
            continue
        winners, losers = (team_a, team_b) if score_a > score_b else (team_b, team_a)
        for pid in winners:
            if pid in stats:
                stats[pid]["wins"] += 1
        for pid in losers:
            if pid in stats:
                stats[pid]["losses"] += 1

    for pid in stats:
        stats[pid]["pd"] = int(stats[pid]["pf"]) - int(stats[pid]["pa"])

    base = sorted(
        stats.values(),
        key=lambda row: (row["wins"], row["pd"], row["pf"], -row["player_id"]),
        reverse=True,
    )
    for i, row in enumerate(base, start=1):
        row["rank"] = i
    return base


def resolveTies(standings: list[dict[str, Any]], games: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if not standings:
        return []

    rows = [dict(item) for item in standings]
    for row in rows:
        row.setdefault("playoff_required", False)
        row.setdefault("h2h_opponent_wins", 0)
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["wins"]), int(row["pd"]), int(row["pf"]))].append(row)

    resolved: list[dict[str, Any]] = []
    for _, group in sorted(grouped.items(), key=lambda kv: kv[0], reverse=True):
        if len(group) == 1:
            group[0]["tie_break"] = "WinsPDPF"
            resolved.extend(group)
            continue

        tied_ids = {int(item["player_id"]) for item in group}
        h2h_scores = _head_to_head_opponent_wins(tied_ids, games or [])

        ordered = sorted(
            group,
            key=lambda row: (int(row["wins"]), int(row["pd"]), int(row["pf"]), int(h2h_scores.get(int(row["player_id"]), 0))),
            reverse=True,
        )

        subgroups: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
        for item in ordered:
            h2h = int(h2h_scores.get(int(item["player_id"]), 0))
            item["h2h_opponent_wins"] = h2h
            subgroups[(int(item["wins"]), int(item["pd"]), int(item["pf"]), h2h)].append(item)

        for subgroup in subgroups.values():
            if len(subgroup) > 1:
                for row in subgroup:
                    row["playoff_required"] = True
                    row["tie_break"] = "PlayoffRequired"
            else:
                subgroup[0]["tie_break"] = "HeadToHead"
        resolved.extend(ordered)

    resolved = sorted(
        resolved,
        key=lambda row: (
            int(row["wins"]),
            int(row["pd"]),
            int(row["pf"]),
            int(row.get("h2h_opponent_wins", 0)),
            -int(bool(row.get("playoff_required", False))),
        ),
        reverse=True,
    )
    for i, row in enumerate(resolved, start=1):
        row["rank"] = i
    return resolved


def getMovers(standings: list[dict[str, Any]], moversPerCourt: int) -> dict[str, list[int]]:
    movers = int(moversPerCourt)
    if movers not in {1, 2}:
        raise ValueError("moversPerCourt must be 1 or 2")
    if len(standings) < (movers * 2):
        raise ValueError("Not enough players in standings for requested moversPerCourt")

    ordered = sorted(standings, key=lambda item: int(item.get("rank", 10_000)))
    up = [int(item["player_id"]) for item in ordered[:movers]]
    down = [int(item["player_id"]) for item in ordered[-movers:]]
    return {"up": up, "down": down}


def applyMovement(courtPods: list[list[int]], moversPerCourt: int) -> list[list[int]]:
    movers = int(moversPerCourt)
    if movers not in {1, 2}:
        raise ValueError("moversPerCourt must be 1 or 2")
    if not courtPods:
        return []

    pods = [list(map(int, pod)) for pod in courtPods]
    total = len(pods)
    for pod in pods:
        if len(pod) <= movers:
            raise ValueError("Each court pod must contain more players than moversPerCourt")

    outgoing_up: dict[int, list[int]] = {}
    outgoing_down: dict[int, list[int]] = {}
    for idx, pod in enumerate(pods):
        outgoing_up[idx] = pod[:movers] if idx > 0 else []
        outgoing_down[idx] = pod[-movers:] if idx < (total - 1) else []

    next_pods: list[list[int]] = []
    for idx, pod in enumerate(pods):
        cut_left = movers if idx > 0 else 0
        cut_right = len(pod) - movers if idx < (total - 1) else len(pod)
        survivors = pod[cut_left:cut_right]

        incoming_from_above = outgoing_down.get(idx - 1, [])
        incoming_from_below = outgoing_up.get(idx + 1, [])
        next_pods.append(incoming_from_above + survivors + incoming_from_below)

    return next_pods


def _head_to_head_opponent_wins(tied_ids: set[int], games: list[dict[str, Any]]) -> dict[int, int]:
    values = {int(pid): 0 for pid in tied_ids}
    for game in games:
        team_a = {int(pid) for pid in game.get("teamA", [])}
        team_b = {int(pid) for pid in game.get("teamB", [])}
        score_a = int(game.get("scoreA", 0) or 0)
        score_b = int(game.get("scoreB", 0) or 0)

        tied_on_a = tied_ids.intersection(team_a)
        tied_on_b = tied_ids.intersection(team_b)
        if not tied_on_a or not tied_on_b:
            continue

        if score_a > score_b:
            for pid in tied_on_a:
                values[int(pid)] += 1
        elif score_b > score_a:
            for pid in tied_on_b:
                values[int(pid)] += 1
    return values
