from __future__ import annotations

# SCHEMA STRICT MODE ENABLED
# All environments must match migrations

from typing import Any

from jupr_app.domain.gamification.fact_engine import update_match_facts_for_players


def rebuild_facts_from_history(supabase: Any, club_id: str) -> dict[str, int]:
    """Rebuild ``player_badge_facts`` by replaying all matches in historical order.

    The replay is idempotent because ``update_match_facts_for_players`` enforces
    per-player match guards via ``processed_match_facts``.
    """
    if supabase is None or not str(club_id).strip():
        return {"matches_seen": 0, "matches_processed": 0, "players_touched": 0}

    rows = (
        supabase.table("matches")
        .select("id,date,score_t1,score_t2,t1_p1,t1_p2,t2_p1,t2_p2")
        .eq("club_id", str(club_id))
        .order("date", desc=False)
        .execute()
        .data
        or []
    )

    matches_processed = 0
    touched_players: set[int] = set()

    for row in rows:
        players = _extract_players(row)
        if not players:
            continue

        winner_team = _winner_team(row)
        payload = {
            "match_id": row.get("id"),
            "id": row.get("id"),
            "score_t1": row.get("score_t1"),
            "score_t2": row.get("score_t2"),
            "t1_p1": row.get("t1_p1"),
            "t1_p2": row.get("t1_p2"),
            "t2_p1": row.get("t2_p1"),
            "t2_p2": row.get("t2_p2"),
            "winner_team": winner_team,
            "loser_team": 2 if winner_team == 1 else 1 if winner_team == 2 else None,
        }

        update_match_facts_for_players(
            supabase,
            str(club_id),
            players,
            payload,
            match_id=str(row.get("id") or ""),
        )
        matches_processed += 1
        touched_players.update(players)

    return {
        "matches_seen": len(rows),
        "matches_processed": matches_processed,
        "players_touched": len(touched_players),
    }


def _extract_players(row: dict[str, Any]) -> list[int]:
    players: set[int] = set()
    for key in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
        pid = _safe_int(row.get(key))
        if pid is not None:
            players.add(pid)
    return sorted(players)


def _winner_team(row: dict[str, Any]) -> int | None:
    score_t1 = _safe_int(row.get("score_t1"))
    score_t2 = _safe_int(row.get("score_t2"))
    if score_t1 is None or score_t2 is None:
        return None
    if score_t1 > score_t2:
        return 1
    if score_t2 > score_t1:
        return 2
    return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
