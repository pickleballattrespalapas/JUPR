from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from math import pow
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


MEASURABLE_PLAYER_STATS = (
    "rating_jupr",
    "rating_gain_jupr",
    "games",
    "wins",
    "losses",
    "win_pct",
    "points_for",
    "points_against",
    "point_differential",
    "average_margin",
    "longest_win_streak",
    "current_win_streak",
    "close_games",
    "close_wins",
    "close_win_pct",
    "largest_upset_jupr",
    "upset_wins",
    "average_opponent_jupr",
    "expected_wins",
    "wins_above_expected",
    "best_partnership_win_pct",
    "best_partnership_games",
    "partner_variety",
    "weeks_played",
    "attendance_pct",
)


AWARD_CATEGORY_CATALOG = (
    {"key": "highest_rating", "label": "Highest Rating", "recipient_type": "player", "metric": "rating_jupr", "format": "rating"},
    {"key": "most_improved", "label": "Most Improved", "recipient_type": "player", "metric": "rating_gain_jupr", "format": "signed_rating"},
    {"key": "best_win_pct", "label": "Best Win Percentage", "recipient_type": "player", "metric": "win_pct", "format": "percent"},
    {"key": "most_wins", "label": "Most Wins", "recipient_type": "player", "metric": "wins", "format": "integer", "default_enabled": True},
    {"key": "iron_player", "label": "Iron Player", "recipient_type": "player", "metric": "games", "format": "integer"},
    {"key": "hot_hand", "label": "Hot Hand", "recipient_type": "player", "metric": "current_win_streak", "format": "integer"},
    {"key": "point_differential", "label": "Best Point Differential", "recipient_type": "player", "metric": "point_differential", "format": "signed_integer"},
    {"key": "average_margin", "label": "Best Average Margin", "recipient_type": "player", "metric": "average_margin", "format": "decimal"},
    {"key": "longest_win_streak", "label": "Longest Win Streak", "recipient_type": "player", "metric": "longest_win_streak", "format": "integer"},
    {"key": "close_game_record", "label": "Best Close-Game Record", "recipient_type": "player", "metric": "close_win_pct", "format": "percent", "minimum_metric": "close_games"},
    {"key": "biggest_upset", "label": "Biggest Upset", "recipient_type": "player", "metric": "largest_upset_jupr", "format": "rating", "minimum_metric": "upset_wins"},
    {"key": "most_upsets", "label": "Most Upset Wins", "recipient_type": "player", "metric": "upset_wins", "format": "integer", "minimum_metric": "upset_wins"},
    {"key": "opponent_strength", "label": "Strongest Opposition", "recipient_type": "player", "metric": "average_opponent_jupr", "format": "rating"},
    {"key": "over_performance", "label": "Most Above Expectation", "recipient_type": "player", "metric": "wins_above_expected", "format": "signed_decimal"},
    {"key": "best_partnership", "label": "Best Partnership", "recipient_type": "player", "metric": "best_partnership_win_pct", "format": "percent", "minimum_metric": "best_partnership_games"},
    {"key": "partner_variety", "label": "Most Partners", "recipient_type": "player", "metric": "partner_variety", "format": "integer"},
    {"key": "attendance", "label": "Best Attendance", "recipient_type": "player", "metric": "attendance_pct", "format": "percent", "minimum_metric": "weeks_played"},
    {"key": "team_champion", "label": "Team Champion", "recipient_type": "team", "metric": "standing_score", "format": "team_record", "minimum_metric": "games_played"},
    {"key": "team_wins", "label": "Most Team Wins", "recipient_type": "team", "metric": "wins", "format": "integer", "minimum_metric": "games_played"},
    {"key": "team_point_differential", "label": "Best Team Point Differential", "recipient_type": "team", "metric": "point_differential", "format": "signed_integer", "minimum_metric": "games_played"},
)


@dataclass(frozen=True)
class CanonicalLeagueMatches:
    included: list[dict[str, Any]]
    exclusion_counts: dict[str, int]
    discovered_count: int

    def provenance(self) -> dict[str, Any]:
        return {
            "rule_version": "canonical_league_matches_v1",
            "discovered_count": self.discovered_count,
            "included_count": len(self.included),
            "excluded_count": self.discovered_count - len(self.included),
            "exclusion_counts": dict(sorted(self.exclusion_counts.items())),
            "included_match_ids": [
                row.get("id") for row in self.included if row.get("id") is not None
            ],
            "included_match_dates": sorted(
                {
                    str(row.get("date") or row.get("played_on") or "")[:10]
                    for row in self.included
                    if row.get("date") or row.get("played_on")
                }
            ),
        }


def _rows(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        return [
            {str(key): _json_value(item) for key, item in row.items()}
            for row in value.to_dict("records")
        ]
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _json_value(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _int(value: Any) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
        result = int(number)
        return result if number == result else None
    except Exception:
        return None


def _float(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _match_format(value: Any) -> str:
    return "singles" if str(value or "").strip().casefold() == "singles" else "doubles"


def _player_ids(
    row: Mapping[str, Any], *, match_format: str = "doubles"
) -> tuple[list[int], list[int]] | None:
    if _match_format(match_format) == "singles":
        team_one = [_int(row.get("t1_p1"))]
        team_two = [_int(row.get("t2_p1"))]
    else:
        team_one = [_int(row.get("t1_p1")), _int(row.get("t1_p2"))]
        team_two = [_int(row.get("t2_p1")), _int(row.get("t2_p2"))]
    if any(player_id is None for player_id in team_one + team_two):
        return None
    one = [int(player_id) for player_id in team_one if player_id is not None]
    two = [int(player_id) for player_id in team_two if player_id is not None]
    expected_side = 1 if _match_format(match_format) == "singles" else 2
    if (
        len(set(one)) != expected_side
        or len(set(two)) != expected_side
        or set(one) & set(two)
    ):
        return None
    return one, two


def canonical_league_matches(
    matches: Any,
    *,
    club_id: str,
    league_name: str,
    match_format: str = "doubles",
) -> CanonicalLeagueMatches:
    clean_match_format = _match_format(match_format)
    included: list[dict[str, Any]] = []
    excluded: defaultdict[str, int] = defaultdict(int)
    discovered = 0
    for row in _rows(matches):
        if str(row.get("club_id") or "") != str(club_id):
            continue
        if str(row.get("league") or "").strip() != str(league_name).strip():
            continue
        discovered += 1
        if row.get("deleted_at"):
            excluded["deleted"] += 1
            continue
        if bool(row.get("excluded_from_ratings")):
            excluded["excluded_from_ratings"] += 1
            continue
        scores = (_int(row.get("score_t1")), _int(row.get("score_t2")))
        if scores[0] is None or scores[1] is None or min(scores) < 0:
            excluded["invalid_score"] += 1
            continue
        if scores[0] == scores[1]:
            excluded["tied_score"] += 1
            continue
        if _player_ids(row, match_format=clean_match_format) is None:
            excluded["invalid_player_sides"] += 1
            continue
        normalized = dict(row)
        normalized["score_t1"], normalized["score_t2"] = scores
        included.append(normalized)
    included.sort(
        key=lambda row: (
            str(row.get("date") or row.get("played_on") or ""),
            _int(row.get("id")) or 0,
        )
    )
    return CanonicalLeagueMatches(
        included=included,
        exclusion_counts=dict(excluded),
        discovered_count=discovered,
    )


def expected_elo_win_probability(
    team_rating: float, opponent_rating: float
) -> float:
    return 1.0 / (1.0 + pow(10.0, (float(opponent_rating) - float(team_rating)) / 400.0))


def _pre_match_team_ratings(
    match: Mapping[str, Any],
    *,
    match_format: str = "doubles",
) -> tuple[float, float] | None:
    if _match_format(match_format) == "singles":
        values = [
            _float(match.get("t1_p1_r")),
            _float(match.get("t2_p1_r")),
        ]
        if any(value is None for value in values):
            return None
        return float(values[0]), float(values[1])
    values = [
        _float(match.get("t1_p1_r")),
        _float(match.get("t1_p2_r")),
        _float(match.get("t2_p1_r")),
        _float(match.get("t2_p2_r")),
    ]
    if any(value is None for value in values):
        return None
    return (
        (float(values[0]) + float(values[1])) / 2.0,
        (float(values[2]) + float(values[3])) / 2.0,
    )


def _week_key(match: Mapping[str, Any]) -> str:
    week = str(match.get("week_tag") or "").strip()
    if week:
        return week
    raw = str(match.get("date") or match.get("played_on") or "")[:10]
    return raw or "unscheduled"


def compute_league_player_analytics(
    matches: Any,
    *,
    club_id: str,
    league_name: str,
    players: Any = None,
    league_ratings: Any = None,
    match_format: str = "doubles",
    expected_weeks: int | None = None,
    close_game_margin: int = 2,
    upset_threshold_jupr: float = 0.25,
) -> dict[str, Any]:
    clean_match_format = _match_format(match_format)
    canonical = canonical_league_matches(
        matches,
        club_id=str(club_id),
        league_name=str(league_name),
        match_format=clean_match_format,
    )
    names: dict[int, str] = {}
    current_ratings: dict[int, float] = {}
    for row in _rows(players):
        player_id = _int(row.get("id"))
        if player_id is None:
            continue
        names[player_id] = str(row.get("name") or f"Player {player_id}")
        rating = _float(row.get("rating"))
        if rating is not None:
            current_ratings[player_id] = rating
    starting_ratings: dict[int, float] = {}
    for row in _rows(league_ratings):
        if str(row.get("club_id") or club_id) != str(club_id):
            continue
        if str(row.get("league_name") or "").strip() != str(league_name).strip():
            continue
        player_id = _int(row.get("player_id"))
        if player_id is None:
            continue
        rating = _float(row.get("rating"))
        start = _float(row.get("starting_rating"))
        if rating is not None:
            current_ratings[player_id] = rating
        if start is not None:
            starting_ratings[player_id] = start

    stats: defaultdict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "games": 0,
            "wins": 0,
            "losses": 0,
            "points_for": 0,
            "points_against": 0,
            "longest_win_streak": 0,
            "current_win_streak": 0,
            "close_games": 0,
            "close_wins": 0,
            "largest_upset_jupr": 0.0,
            "upset_wins": 0,
            "opponent_rating_total": 0.0,
            "opponent_rating_games": 0,
            "expected_wins": 0.0,
            "expected_games": 0,
            "partners": defaultdict(lambda: {"games": 0, "wins": 0}),
            "weeks": set(),
        }
    )
    for match in canonical.included:
        sides = _player_ids(match, match_format=clean_match_format)
        if sides is None:
            continue
        team_one, team_two = sides
        score_one, score_two = int(match["score_t1"]), int(match["score_t2"])
        one_won = score_one > score_two
        pre_ratings = _pre_match_team_ratings(
            match, match_format=clean_match_format
        )
        for team, opponents, won, points_for, points_against in (
            (team_one, team_two, one_won, score_one, score_two),
            (team_two, team_one, not one_won, score_two, score_one),
        ):
            for player_id in team:
                current = stats[player_id]
                current["games"] += 1
                current["wins" if won else "losses"] += 1
                current["points_for"] += points_for
                current["points_against"] += points_against
                current["weeks"].add(_week_key(match))
                current["current_win_streak"] = (
                    current["current_win_streak"] + 1 if won else 0
                )
                current["longest_win_streak"] = max(
                    current["longest_win_streak"],
                    current["current_win_streak"],
                )
                if abs(points_for - points_against) <= int(close_game_margin):
                    current["close_games"] += 1
                    current["close_wins"] += int(won)
                if len(team) > 1:
                    partner_id = next(
                        candidate for candidate in team if candidate != player_id
                    )
                    current["partners"][partner_id]["games"] += 1
                    current["partners"][partner_id]["wins"] += int(won)
                if pre_ratings is not None:
                    team_rating, opponent_rating = (
                        pre_ratings if team is team_one else pre_ratings[::-1]
                    )
                    current["opponent_rating_total"] += opponent_rating
                    current["opponent_rating_games"] += 1
                    expected = expected_elo_win_probability(
                        team_rating, opponent_rating
                    )
                    current["expected_wins"] += expected
                    current["expected_games"] += 1
                    upset_size = opponent_rating - team_rating
                    if won and upset_size >= float(upset_threshold_jupr) * 400.0:
                        current["upset_wins"] += 1
                        current["largest_upset_jupr"] = max(
                            current["largest_upset_jupr"], upset_size / 400.0
                        )

    output: list[dict[str, Any]] = []
    for player_id, current in stats.items():
        games = int(current["games"])
        partnerships = list(current["partners"].items())
        partnerships.sort(
            key=lambda item: (
                -(item[1]["wins"] / item[1]["games"]),
                -item[1]["games"],
                names.get(item[0], "").lower(),
                item[0],
            )
        )
        best_partner_id, best_partner = partnerships[0] if partnerships else (None, {"games": 0, "wins": 0})
        expected_complete = int(current["expected_games"]) == games
        weeks_played = len(current["weeks"])
        rating = current_ratings.get(player_id)
        starting = starting_ratings.get(player_id)
        output.append(
            {
                "player_id": player_id,
                "player_name": names.get(player_id, f"Player {player_id}"),
                "rating_jupr": round(rating / 400.0, 4) if rating is not None and rating > 20 else rating,
                "rating_gain_jupr": (
                    round((rating - starting) / 400.0, 4)
                    if rating is not None
                    and starting is not None
                    and max(abs(rating), abs(starting)) > 20
                    else round(rating - starting, 4)
                    if rating is not None and starting is not None
                    else None
                ),
                "games": games,
                "wins": int(current["wins"]),
                "losses": int(current["losses"]),
                "win_pct": round(current["wins"] / games, 6) if games else None,
                "points_for": int(current["points_for"]),
                "points_against": int(current["points_against"]),
                "point_differential": int(current["points_for"] - current["points_against"]),
                "average_margin": round((current["points_for"] - current["points_against"]) / games, 4) if games else None,
                "longest_win_streak": int(current["longest_win_streak"]),
                "current_win_streak": int(current["current_win_streak"]),
                "close_games": int(current["close_games"]),
                "close_wins": int(current["close_wins"]),
                "close_win_pct": round(current["close_wins"] / current["close_games"], 6) if current["close_games"] else None,
                "largest_upset_jupr": round(current["largest_upset_jupr"], 4) if current["upset_wins"] else None,
                "upset_wins": int(current["upset_wins"]),
                "average_opponent_jupr": round(current["opponent_rating_total"] / current["opponent_rating_games"] / 400.0, 4) if current["opponent_rating_games"] else None,
                "expected_wins": round(current["expected_wins"], 6) if expected_complete else None,
                "wins_above_expected": round(current["wins"] - current["expected_wins"], 6) if expected_complete else None,
                "expected_model": (
                    "canonical_elo_pre_match_singles_v1"
                    if clean_match_format == "singles"
                    else "canonical_elo_pre_match_team_average_v1"
                ) if expected_complete else None,
                "best_partner_id": best_partner_id,
                "best_partner_name": names.get(best_partner_id) if best_partner_id else None,
                "best_partnership_games": int(best_partner["games"]),
                "best_partnership_win_pct": round(best_partner["wins"] / best_partner["games"], 6) if best_partner["games"] else None,
                "partner_variety": len(partnerships),
                "weeks_played": weeks_played,
                "attendance_pct": round(weeks_played / int(expected_weeks), 6) if expected_weeks and expected_weeks > 0 else None,
            }
        )
    output.sort(key=lambda row: (str(row["player_name"]).lower(), int(row["player_id"])))
    provenance = canonical.provenance()
    provenance.update(
        {
            "close_game_margin": int(close_game_margin),
            "upset_threshold_jupr": float(upset_threshold_jupr),
            "expected_weeks": expected_weeks,
        }
    )
    return {"players": output, "provenance": provenance}


def compute_team_league_standings(
    fixtures: Any,
    teams: Any,
    *,
    include_playoffs: bool = False,
) -> list[dict[str, Any]]:
    team_rows = {
        str(row.get("id")): row
        for row in _rows(teams)
        if row.get("id") and str(row.get("status") or "") == "confirmed"
    }
    stats = {
        team_id: {
            "team_id": team_id,
            "team_name": str(row.get("team_name") or "Team"),
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "points_for": 0,
            "points_against": 0,
            "head_to_head_results": defaultdict(lambda: {"wins": 0, "losses": 0}),
        }
        for team_id, row in team_rows.items()
    }
    for fixture in _rows(fixtures):
        if not include_playoffs and str(fixture.get("phase") or "") != "regular":
            continue
        fixture_status = str(fixture.get("status") or "")
        if fixture_status not in {"complete", "forfeit"}:
            continue
        team_a, team_b = str(fixture.get("team_a_id") or ""), str(fixture.get("team_b_id") or "")
        score_a, score_b = _int(fixture.get("team_a_score")), _int(fixture.get("team_b_score"))
        if team_a not in stats or team_b not in stats:
            continue
        if fixture_status == "forfeit":
            winner = str(fixture.get("winner_team_id") or "")
            if winner not in {team_a, team_b}:
                continue
            score_a = score_b = 0
        elif score_a is None or score_b is None or score_a == score_b:
            continue
        for team_id, opponent_id, score_for, score_against in (
            (team_a, team_b, score_a, score_b),
            (team_b, team_a, score_b, score_a),
        ):
            won = (
                team_id == winner
                if fixture_status == "forfeit"
                else score_for > score_against
            )
            current = stats[team_id]
            current["games_played"] += 1
            current["wins" if won else "losses"] += 1
            current["points_for"] += score_for
            current["points_against"] += score_against
            current["head_to_head_results"][opponent_id]["wins" if won else "losses"] += 1
    values = []
    for current in stats.values():
        games = current["games_played"]
        values.append(
            {
                **current,
                "win_pct": round(current["wins"] / games, 6) if games else 0.0,
                "point_differential": current["points_for"] - current["points_against"],
                "standing_score": current["wins"],
                "head_to_head_results": {
                    key: dict(value)
                    for key, value in sorted(current["head_to_head_results"].items())
                },
            }
        )
    # Tied teams first use their direct meeting record, then point differential.
    # Precompute this before sorting: CPython temporarily empties a list while
    # evaluating its in-place sort keys, so consulting ``values`` from the key
    # function would silently erase the head-to-head tie-break.
    def h2h_score(row: Mapping[str, Any]) -> int:
        tied_ids = {
            str(other["team_id"])
            for other in values
            if other["team_id"] != row["team_id"]
            and other["wins"] == row["wins"]
            and other["losses"] == row["losses"]
        }
        return sum(
            int(result.get("wins") or 0) - int(result.get("losses") or 0)
            for opponent_id, result in row["head_to_head_results"].items()
            if opponent_id in tied_ids
        )

    for row in values:
        row["head_to_head_score"] = h2h_score(row)
    values.sort(
        key=lambda row: (
            -int(row["wins"]),
            int(row["losses"]),
            -int(row["head_to_head_score"]),
            -int(row["point_differential"]),
            str(row["team_name"]).lower(),
            str(row["team_id"]),
        )
    )
    for rank, row in enumerate(values, start=1):
        row["rank"] = rank
        # Team Champion must follow the official, fully tie-broken standings
        # rather than treating every equal-win record as a co-winner.
        row["standing_score"] = len(values) - rank + 1
    return values


def compute_team_league_analytics(
    fixtures: Any, teams: Any
) -> dict[str, Any]:
    standings = compute_team_league_standings(fixtures, teams)
    return {
        "teams": standings,
        "measurable_stats": (
            "games_played",
            "wins",
            "losses",
            "win_pct",
            "points_for",
            "points_against",
            "point_differential",
            "head_to_head_score",
            "standing_score",
        ),
    }


def award_category_catalog() -> list[dict[str, Any]]:
    return [dict(row) for row in AWARD_CATEGORY_CATALOG]


def build_league_award_catalog() -> list[dict[str, Any]]:
    return award_category_catalog()
