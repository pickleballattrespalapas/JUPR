from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterable


class LeagueMatchStructureError(ValueError):
    """A scored league series does not match the configured match structure."""


DEFAULT_LEAGUE_MATCH_STRUCTURE = {
    "kind": "fixed_games",
    "games": 1,
    "result_counting": "each_game",
    "completion": "all_games",
}


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def normalize_league_match_structure(value: Any) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    kind = str(raw.get("kind") or "fixed_games").strip().lower()
    if kind not in {"fixed_games", "best_of"}:
        kind = "fixed_games"
    games = _safe_int(raw.get("games"), 1) or 1
    games = max(1, min(int(games), 9))
    if kind == "best_of" and (games < 3 or games % 2 == 0):
        return dict(DEFAULT_LEAGUE_MATCH_STRUCTURE)
    return {
        "kind": kind,
        "games": games,
        # Every completed pickleball game remains an official game for ratings,
        # minimum-game qualification, and league statistics. Best-of changes
        # only when the series is complete.
        "result_counting": "each_game",
        "completion": "clinch" if kind == "best_of" else "all_games",
    }


def league_match_structure_label(value: Any) -> str:
    structure = normalize_league_match_structure(value)
    games = int(structure["games"])
    if structure["kind"] == "best_of":
        return f"Best {games // 2 + 1} out of {games}"
    return "1 game" if games == 1 else f"{games} games"


def validate_league_series_matches(
    matches: Iterable[dict[str, Any]] | None,
    *,
    match_structure: Any,
) -> list[dict[str, Any]]:
    """Validate complete fixed-game or best-of series and preserve game rows.

    A fixed series requires every configured game. A best-of series requires
    consecutive game numbers and stops as soon as one team clinches. Each
    returned row remains an individual official pickleball game.
    """

    structure = normalize_league_match_structure(match_structure)
    rows = [dict(row) for row in (matches or []) if isinstance(row, dict)]
    if not rows:
        return []

    needs_series_metadata = int(structure["games"]) > 1 or structure["kind"] == "best_of"
    if not needs_series_metadata and not any(row.get("series_key") for row in rows):
        return rows

    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for index, row in enumerate(rows, start=1):
        series_key = str(row.get("series_key") or "").strip()
        if not series_key:
            raise LeagueMatchStructureError(
                f"Game {index} is missing its generated series reference. Regenerate the round before publishing."
            )
        if len(series_key) > 160:
            raise LeagueMatchStructureError(f"Game {index} has an invalid series reference.")
        supplied_kind = str(row.get("series_kind") or structure["kind"]).strip().lower()
        supplied_games = _safe_int(row.get("series_games"), int(structure["games"]))
        if supplied_kind != structure["kind"] or supplied_games != int(structure["games"]):
            raise LeagueMatchStructureError(
                "The scored series does not match the league settings. Reload the league before entering scores."
            )
        game_number = _safe_int(row.get("game_number"))
        if game_number is None or game_number < 1 or game_number > int(structure["games"]):
            raise LeagueMatchStructureError(f"Series {series_key} has an invalid game number.")
        canonical = {
            **row,
            "series_key": series_key,
            "series_kind": structure["kind"],
            "series_games": int(structure["games"]),
            "game_number": int(game_number),
        }
        grouped.setdefault(series_key, []).append(canonical)

    validated: list[dict[str, Any]] = []
    for series_key, games in grouped.items():
        games.sort(key=lambda row: int(row["game_number"]))
        game_numbers = [int(row["game_number"]) for row in games]
        if game_numbers != list(range(1, len(games) + 1)):
            raise LeagueMatchStructureError(
                f"Series {series_key} must contain consecutive games beginning with Game 1."
            )
        identities = {
            (
                _safe_int(row.get("court", row.get("court_number"))),
                _safe_int(row.get("t1_p1")),
                _safe_int(row.get("t1_p2")),
                _safe_int(row.get("t2_p1")),
                _safe_int(row.get("t2_p2")),
            )
            for row in games
        }
        if len(identities) != 1:
            raise LeagueMatchStructureError(
                f"Series {series_key} changes teams or courts between games. Regenerate the round."
            )

        if structure["kind"] == "fixed_games":
            if len(games) != int(structure["games"]):
                raise LeagueMatchStructureError(
                    f"Series {series_key} requires all {structure['games']} configured games."
                )
        else:
            wins_needed = int(structure["games"]) // 2 + 1
            team_one_wins = 0
            team_two_wins = 0
            clinched_at: int | None = None
            for game in games:
                score_one = _safe_int(game.get("score_t1", game.get("s1")))
                score_two = _safe_int(game.get("score_t2", game.get("s2")))
                if score_one is None or score_two is None or score_one == score_two:
                    raise LeagueMatchStructureError(
                        f"Series {series_key}, Game {game['game_number']} needs a complete non-tied score."
                    )
                if score_one > score_two:
                    team_one_wins += 1
                else:
                    team_two_wins += 1
                if max(team_one_wins, team_two_wins) == wins_needed:
                    clinched_at = int(game["game_number"])
                    break
            if clinched_at is None:
                raise LeagueMatchStructureError(
                    f"Series {series_key} is not complete; one team must win {wins_needed} games."
                )
            if clinched_at != len(games):
                raise LeagueMatchStructureError(
                    f"Series {series_key} includes a game after the series was already clinched."
                )

        validated.extend(games)
    return validated
