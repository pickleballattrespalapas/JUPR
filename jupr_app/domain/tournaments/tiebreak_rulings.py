"""Explicit director rulings, scoped to the recorded draw, teams, and record."""

from functools import lru_cache
import json
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def _rulings() -> list[dict[str, Any]]:
    path = Path(__file__).resolve().parents[3] / "config" / "tournament_tiebreak_rulings.json"
    return json.loads(path.read_text(encoding="utf-8"))


def find_tiebreak_ruling(group: list[Any], games: list[dict[str, Any]]) -> dict[str, Any] | None:
    team_ids = {standing.team_id for standing in group}
    relevant_games = [
        game for game in games
        if game.get("team_a_id") in team_ids or game.get("team_b_id") in team_ids
    ]
    for ruling in _rulings():
        if team_ids != set(ruling["team_ids"]):
            continue
        if not all(s.wins == ruling["wins"] and s.losses == ruling["losses"] for s in group):
            continue
        if relevant_games and all(
            game.get("draw_id") == ruling["draw_id"]
            and game.get("tournament_id") == ruling["tournament_id"]
            for game in relevant_games
        ):
            return ruling
    return None
