from __future__ import annotations

from datetime import datetime, timezone


def build_tournament_match_payload(tournament, game, teams_by_id, *, score_a: int, score_b: int) -> dict:
    team_a = teams_by_id.get(game.get("team_a_id"))
    team_b = teams_by_id.get(game.get("team_b_id"))

    return {
        "t1_p1": team_a.get("player1_id") if team_a else None,
        "t1_p2": team_a.get("player2_id") if team_a else None,
        "t2_p1": team_b.get("player1_id") if team_b else None,
        "t2_p2": team_b.get("player2_id") if team_b else None,
        "s1": int(score_a),
        "s2": int(score_b),
        "score_t1": int(score_a),
        "score_t2": int(score_b),
        "date": datetime.now(timezone.utc).isoformat(),
        "league": tournament.get("name", "Tournament"),
        "match_type": "Tournament",
        "week_tag": "Tournament",
        "is_popup": False,
        "context_type": "TOURNAMENT",
        "context_id": tournament["id"],
        "tournament_id": tournament["id"],
        "tournament_game_id": game["id"],
    }
