"""Overall card metrics calculated from the same canonical games as seasons.

This deliberately does not change league awards: Overall includes rated popup
and tournament games and retains legacy ties without inventing a win or loss.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from math import isclose, isfinite
from typing import Any
from zoneinfo import ZoneInfo


# key -> (value field, sample field, display format)
LEADERBOARD_CARD_METRICS = {
    "highest_rating": ("rating_jupr", "matches_played", "rating"),
    "most_improved": ("rating_gain_jupr", "matches_played", "signed_rating"),
    "best_win_pct": ("win_pct", "matches_played", "percent"),
    "most_wins": ("wins", "matches_played", "wins"),
    "most_matches": ("matches_played", "matches_played", "games"),
    "hot_hand": ("current_win_streak", "games", "wins"),
    "point_differential": ("point_differential", "games", "points"),
    "average_margin": ("average_margin", "games", "margin"),
    "longest_win_streak": ("longest_win_streak", "games", "wins"),
    "close_game_record": ("close_win_pct", "close_games", "close"),
    "biggest_upset": ("largest_upset_jupr", "upset_wins", "upset"),
    "most_upsets": ("upset_wins", "upset_wins", "wins"),
    "opponent_strength": ("average_opponent_jupr", "games", "rating"),
    "over_performance": ("wins_above_expected", "games", "expected"),
    "best_partnership": ("best_partnership_win_pct", "best_partnership_games", "partner"),
    "partner_variety": ("partner_variety", "games", "partners"),
    "playing_days": ("playing_days", "playing_days", "days"),
}
LEADERBOARD_CARD_KEYS = tuple(LEADERBOARD_CARD_METRICS)
EXTENDED_CARD_KEYS = frozenset(LEADERBOARD_CARD_KEYS[5:])
_POSITIVE_ONLY = {"hot_hand", "longest_win_streak", "biggest_upset", "most_upsets"}
_SLOTS = ("t1_p1", "t1_p2", "t2_p1", "t2_p2")


def _rating(value: Any) -> float | None:
    try:
        result = float(value)
        if not isfinite(result):
            return None
        return result / 400 if abs(result) > 20 else result
    except (ValueError, TypeError):
        return None


def compute_overall_metrics(
    matches: list[dict[str, Any]], *, names: dict[str, str], timezone: str,
    partnership_minimum: int = 0,
) -> dict[str, dict[str, Any]]:
    """Consume already filtered, chronological, unique Overall doubles games.

    All rating-derived statistics require complete pre-game snapshots across a
    player's entire selected period. A partial history is not a zero or a valid
    estimate. Playing days use local calendar dates; ties break win streaks.
    """
    zone = ZoneInfo(timezone)
    totals: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "games": 0, "wins": 0, "points": 0, "current_win_streak": 0,
        "longest_win_streak": 0, "close_games": 0, "close_wins": 0,
        "rating_games": 0, "opponent_total": 0.0, "expected_wins": 0.0,
        "upset_wins": 0, "largest_upset_jupr": None, "days": set(),
        "partners": defaultdict(lambda: {"games": 0, "wins": 0}),
    })
    for match in matches:
        ratings = [_rating(match.get(f"{slot}_r")) for slot in _SLOTS]
        complete = all(rating is not None for rating in ratings)
        team_ratings = ((ratings[0] + ratings[1]) / 2, (ratings[2] + ratings[3]) / 2) if complete else None
        day = datetime.fromisoformat(str(match["date"]).replace("Z", "+00:00"))
        # The service canonicalizes naive timestamps as UTC before this point.
        day_key = day.astimezone(zone).date().isoformat()
        for side in (0, 1):
            score = int(match["score_t1" if side == 0 else "score_t2"])
            opposing_score = int(match["score_t2" if side == 0 else "score_t1"])
            margin = score - opposing_score
            won = margin > 0
            team = [str(match[slot]) for slot in _SLOTS[side * 2:side * 2 + 2]]
            for index, pid in enumerate(team):
                current = totals[pid]
                current["games"] += 1
                current["wins"] += int(won)
                current["points"] += margin
                current["days"].add(day_key)
                current["current_win_streak"] = current["current_win_streak"] + 1 if won else 0
                current["longest_win_streak"] = max(current["longest_win_streak"], current["current_win_streak"])
                if abs(margin) <= 2:
                    current["close_games"] += 1
                    current["close_wins"] += int(won)
                partner = current["partners"][team[1 - index]]
                partner["games"] += 1
                partner["wins"] += int(won)
                if team_ratings is not None:
                    own, other = team_ratings[side], team_ratings[1 - side]
                    current["rating_games"] += 1
                    current["opponent_total"] += other
                    # JUPR is Elo / 400, so this is the standard Elo /400
                    # logistic probability expressed in JUPR units.
                    expected = 1 / (1 + 10 ** max(-300, min(300, other - own)))
                    current["expected_wins"] += expected
                    gap = other - own
                    if won and (gap >= 0.25 or isclose(gap, 0.25, rel_tol=0, abs_tol=1e-12)):
                        current["upset_wins"] += 1
                        current["largest_upset_jupr"] = max(current["largest_upset_jupr"] or 0, gap)

    result: dict[str, dict[str, Any]] = {}
    for pid, current in totals.items():
        games = current["games"]
        complete = current["rating_games"] == games
        # Qualify each pair BEFORE picking the player's best partnership.
        partners = [(other, pair) for other, pair in current["partners"].items()
                    if pair["games"] >= max(1, partnership_minimum)]
        partners.sort(key=lambda item: (-(item[1]["wins"] / item[1]["games"]),
                                        -item[1]["games"], -item[1]["wins"], item[0]))
        other, pair = partners[0] if partners else (None, {"games": 0, "wins": 0})
        result[pid] = {
            "games": games, "point_differential": current["points"],
            "average_margin": current["points"] / games,
            "current_win_streak": current["current_win_streak"],
            "longest_win_streak": current["longest_win_streak"],
            "close_games": current["close_games"], "close_wins": current["close_wins"],
            "close_win_pct": 100 * current["close_wins"] / current["close_games"] if current["close_games"] else None,
            "upset_wins": current["upset_wins"] if complete else None,
            "largest_upset_jupr": current["largest_upset_jupr"] if complete else None,
            "average_opponent_jupr": current["opponent_total"] / games if complete else None,
            "wins_above_expected": current["wins"] - current["expected_wins"] if complete else None,
            "best_partnership_win_pct": 100 * pair["wins"] / pair["games"] if pair["games"] else None,
            "best_partnership_games": pair["games"], "best_partnership_wins": pair["wins"],
            "best_partner_name": names.get(other, f"Player {other}") if other else None,
            "partner_variety": len(current["partners"]), "playing_days": len(current["days"]),
        }
    return result


def _display(kind: str, value: float, stats: dict[str, Any]) -> str:
    if kind == "rating":
        return f"{value:.3f}"
    if kind == "signed_rating":
        return f"{value:+.3f}"
    if kind == "percent":
        return f"{value:.1f}%"
    if kind == "points":
        return f"{value:+.0f} pts"
    if kind == "margin":
        return f"{value:+.1f} pts/game"
    if kind == "close":
        return f"{value:.1f}% ({stats['close_wins']}/{stats['close_games']})"
    if kind == "upset":
        return f"{value:+.3f} JUPR"
    if kind == "expected":
        return f"{value:+.1f} wins"
    if kind == "partner":
        return f"{value:.1f}% with {stats['best_partner_name']} ({stats['best_partnership_wins']}/{stats['best_partnership_games']})"
    return f"{value:.0f} {kind}"


def team_upset_highlights(
    matches: list[dict[str, Any]], *, players: list[dict[str, Any]],
    status: str, search: str, min_games: int, options: dict[str, int],
) -> list[dict[str, Any]]:
    """Rank each partnership once, using its largest qualifying upset.

    Games and snapshot completeness belong to the unordered pair, so playing
    with a different partner cannot qualify or disqualify this partnership.
    ``matches`` is the same canonical period history used by the other cards.
    """
    players_by_id = {str(row["player_id"]): row for row in players}
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for match in matches:
        ratings = [_rating(match.get(f"{slot}_r")) for slot in _SLOTS]
        complete = all(rating is not None for rating in ratings)
        team_ratings = ((ratings[0] + ratings[1]) / 2, (ratings[2] + ratings[3]) / 2) if complete else None
        for side in (0, 1):
            pair = tuple(sorted(str(match[slot]) for slot in _SLOTS[side * 2:side * 2 + 2]))
            if not all(pid in players_by_id for pid in pair):
                continue
            totals = pairs.setdefault(pair, {"games": 0, "rating_games": 0, "upset_wins": 0, "largest_upset": 0.0})
            totals["games"] += 1
            if team_ratings is None:
                continue
            totals["rating_games"] += 1
            gap = team_ratings[1 - side] - team_ratings[side]
            won = match["score_t1" if side == 0 else "score_t2"] > match["score_t2" if side == 0 else "score_t1"]
            if won and (gap >= 0.25 or isclose(gap, 0.25, rel_tol=0, abs_tol=1e-12)):
                totals["upset_wins"] += 1
                totals["largest_upset"] = max(totals["largest_upset"], gap)

    minimum = max(1, int(options.get("minimum", 0)))
    depth = max(1, min(10, int(options.get("depth", 5))))
    needle = search.casefold()
    eligible = []
    for pair, totals in pairs.items():
        members = [players_by_id[pid] for pid in pair]
        active = all(member.get("is_active") is True for member in members)
        if (status == "active" and not active) or (status == "inactive" and active):
            continue
        if needle and not any(needle in str(member["player_name"]).casefold() for member in members):
            continue
        if totals["games"] < max(1, min_games) or totals["upset_wins"] < minimum:
            continue
        if totals["rating_games"] != totals["games"]:
            continue
        team_members = [{"player_id": member["player_id"], "player_name": member["player_name"]} for member in members]
        eligible.append({
            "team_key": ":".join(pair), "team_members": team_members,
            "player_id": None, "player_name": " & ".join(member["player_name"] for member in team_members),
            "club_id": members[0]["club_id"], "league_name": members[0]["league_name"],
            "is_active": active, "matches_played": totals["games"],
            "metric_value": totals["largest_upset"],
            "metric_display": _display("upset", totals["largest_upset"], {}),
            "metric_sample": totals["upset_wins"],
        })
    eligible.sort(key=lambda row: (-row["metric_value"], -row["metric_sample"], -row["matches_played"], row["team_key"]))
    return [{**row, "rank": rank, "rank_position": rank} for rank, row in enumerate(eligible[:depth], start=1)]


def overall_highlights(
    rows: list[dict[str, Any]], *, metrics: dict[str, dict[str, Any]],
    min_games: int, card_options: dict[str, dict[str, int]],
) -> dict[str, list[dict[str, Any]]]:
    """Public-safe highlight entries; calculation details never leave this layer."""
    output: dict[str, list[dict[str, Any]]] = {}
    for key, (field, sample_field, kind) in LEADERBOARD_CARD_METRICS.items():
        if key == "biggest_upset":
            # This card is populated separately from complete partnerships,
            # never by duplicating the same win into two individual entries.
            output[key] = []
            continue
        options = card_options.get(key) or {}
        minimum = int(options.get("minimum", 0))
        depth = max(1, min(10, int(options.get("depth", 5))))
        eligible = []
        for row in rows:
            stats = {**row, **metrics.get(str(row["player_id"]), {})}
            value = stats.get(field)
            sample = int(stats.get(sample_field) or 0)
            games = int(stats.get("games" if key in EXTENDED_CARD_KEYS else "matches_played") or 0)
            if value is None or not isfinite(float(value)):
                continue
            if key == "highest_rating":
                if sample < minimum:
                    continue
            elif games < max(1, min_games) or sample < max(1, minimum):
                continue
            if key in _POSITIVE_ONLY and value <= 0:
                continue
            eligible.append({**row, "metric_value": value, "metric_display": _display(kind, value, stats),
                             "metric_sample": sample})
        # More relevant observations resolve ties, then the existing rating rank.
        eligible.sort(key=lambda row: (-row["metric_value"], -row["metric_sample"], int(row.get("rank") or 0)))
        output[key] = eligible[:depth]
    return output
