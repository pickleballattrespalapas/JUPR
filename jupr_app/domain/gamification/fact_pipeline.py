from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from jupr_app.data.sb_write import sb_upsert


def update_player_badge_facts_for_match_commit(
    supabase: Any,
    *,
    club_id: str,
    overall_updates: dict[int, dict[str, Any]],
) -> None:
    """Update V3 badge facts after match writes and before queue enqueue.

    Facts updated:
    - total_matches
    - best_win_streak
    - rating_delta
    - upset_wins
    """
    if supabase is None or not club_id or not overall_updates:
        return

    affected_players = {int(pid) for pid in overall_updates.keys()}
    if not affected_players:
        return

    match_rows = (
        supabase.table("matches")
        .select(
            "id,date,score_t1,score_t2,t1_p1,t1_p2,t2_p1,t2_p2,t1_p1_r,t1_p2_r,t2_p1_r,t2_p2_r"
        )
        .eq("club_id", str(club_id))
        .order("date", desc=False)
        .execute()
        .data
        or []
    )

    streaks: dict[int, int] = {pid: 0 for pid in affected_players}
    best_streaks: dict[int, int] = {pid: 0 for pid in affected_players}
    upset_wins: dict[int, int] = {pid: 0 for pid in affected_players}

    for row in match_rows:
        score_t1 = _safe_int(row.get("score_t1"))
        score_t2 = _safe_int(row.get("score_t2"))
        if score_t1 is None or score_t2 is None or (score_t1 + score_t2) <= 0:
            continue

        t1 = [pid for pid in (_safe_int(row.get("t1_p1")), _safe_int(row.get("t1_p2"))) if pid is not None]
        t2 = [pid for pid in (_safe_int(row.get("t2_p1")), _safe_int(row.get("t2_p2"))) if pid is not None]
        if not t1 or not t2:
            continue

        winner = 1 if score_t1 > score_t2 else 2 if score_t2 > score_t1 else 0
        if winner == 0:
            continue

        t1_avg = _avg([
            _safe_float(row.get("t1_p1_r")),
            _safe_float(row.get("t1_p2_r")),
        ])
        t2_avg = _avg([
            _safe_float(row.get("t2_p1_r")),
            _safe_float(row.get("t2_p2_r")),
        ])

        for pid in t1:
            if pid not in affected_players:
                continue
            if winner == 1:
                streaks[pid] += 1
                best_streaks[pid] = max(best_streaks[pid], streaks[pid])
                if _is_upset_win(winner_team_avg=t1_avg, loser_team_avg=t2_avg):
                    upset_wins[pid] += 1
            else:
                streaks[pid] = 0

        for pid in t2:
            if pid not in affected_players:
                continue
            if winner == 2:
                streaks[pid] += 1
                best_streaks[pid] = max(best_streaks[pid], streaks[pid])
                if _is_upset_win(winner_team_avg=t2_avg, loser_team_avg=t1_avg):
                    upset_wins[pid] += 1
            else:
                streaks[pid] = 0

    now_iso = datetime.now(timezone.utc).isoformat()
    for pid in sorted(affected_players):
        player_stats = overall_updates.get(pid) or {}
        total_matches = int(player_stats.get("mp") or 0)
        current_rating = float(player_stats.get("r") or 1200.0)
        rating_delta = current_rating - 1200.0

        _upsert_fact(supabase, str(club_id), pid, "total_matches", float(total_matches), now_iso)
        _upsert_fact(supabase, str(club_id), pid, "best_win_streak", float(best_streaks.get(pid, 0)), now_iso)
        _upsert_fact(supabase, str(club_id), pid, "rating_delta", float(rating_delta), now_iso)
        _upsert_fact(supabase, str(club_id), pid, "upset_wins", float(upset_wins.get(pid, 0)), now_iso)


def _upsert_fact(supabase: Any, club_id: str, player_id: int, fact_key: str, value: float, updated_at: str) -> None:
    sb_upsert(
        supabase,
        "player_badge_facts",
        {
            "club_id": club_id,
            "player_id": int(player_id),
            "context_id": "overall",
            "fact_key": str(fact_key),
            "fact_value_num": float(value),
            "updated_at": updated_at,
        },
        conflict="club_id,player_id,context_id,fact_key",
    )


def _avg(values: list[float | None]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _is_upset_win(winner_team_avg: float | None, loser_team_avg: float | None) -> bool:
    if winner_team_avg is None or loser_team_avg is None:
        return False
    return float(winner_team_avg) < float(loser_team_avg)


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
