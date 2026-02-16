from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from jupr_app.data.sb_write import sb_upsert

UPSET_THRESHOLD = 0.30


def update_match_facts_for_players(
    supabase: Any,
    club_id: str,
    player_ids: list[int],
    match_payload: dict,
    context_id: str = "overall",
    match_id: str | None = None,
):
    """Incrementally update badge facts for players for a single match payload.

    Facts updated:
    - total_matches
    - current_win_streak
    - best_win_streak
    - rating_delta
    - upset_wins

    The update is idempotent through a per-player/match guard table.
    """
    if supabase is None or not club_id or not player_ids:
        return

    payload = dict(match_payload or {})
    resolved_match_id = str(match_id or payload.get("match_id") or payload.get("id") or "").strip()
    if not resolved_match_id:
        return

    score_t1 = _safe_int(payload.get("score_t1", payload.get("s1")))
    score_t2 = _safe_int(payload.get("score_t2", payload.get("s2")))

    team1 = _extract_team(payload, ("t1_p1", "t1_p2"))
    team2 = _extract_team(payload, ("t2_p1", "t2_p2"))
    winner_team = _winner_team(score_t1, score_t2)

    now_iso = datetime.now(timezone.utc).isoformat()
    normalized_players = sorted({int(pid) for pid in player_ids})

    for pid in normalized_players:
        if not _insert_processed_match_guard(supabase, str(club_id), resolved_match_id, int(pid)):
            continue

        _increment_numeric_fact(supabase, str(club_id), int(pid), context_id, "total_matches", 1.0, now_iso)

        won = _player_won(int(pid), winner_team, team1, team2)
        current_streak = _read_numeric_fact(supabase, str(club_id), int(pid), context_id, "current_win_streak")
        best_streak = _read_numeric_fact(supabase, str(club_id), int(pid), context_id, "best_win_streak")

        if won is True:
            current_streak += 1.0
            best_streak = max(best_streak, current_streak)
        elif won is False:
            current_streak = 0.0

        _set_numeric_fact(supabase, str(club_id), int(pid), context_id, "current_win_streak", current_streak, now_iso)
        _set_numeric_fact(supabase, str(club_id), int(pid), context_id, "best_win_streak", best_streak, now_iso)

        current_rating = _read_player_rating(supabase, str(club_id), int(pid))
        starting_rating = _read_player_starting_rating(supabase, str(club_id), int(pid), fallback=current_rating)
        _set_numeric_fact(
            supabase,
            str(club_id),
            int(pid),
            context_id,
            "rating_delta",
            float(current_rating - starting_rating),
            now_iso,
        )

        if won and _is_upset_win_for_player(int(pid), payload, team1, team2):
            _increment_numeric_fact(supabase, str(club_id), int(pid), context_id, "upset_wins", 1.0, now_iso)


def _extract_team(payload: dict[str, Any], keys: tuple[str, str]) -> set[int]:
    members: set[int] = set()
    for key in keys:
        pid = _safe_int(payload.get(key))
        if pid is not None:
            members.add(int(pid))
    return members


def _winner_team(score_t1: int | None, score_t2: int | None) -> int | None:
    if score_t1 is None or score_t2 is None:
        return None
    if score_t1 > score_t2:
        return 1
    if score_t2 > score_t1:
        return 2
    return None


def _player_won(player_id: int, winner_team: int | None, team1: set[int], team2: set[int]) -> bool | None:
    if winner_team is None:
        return None
    if player_id in team1:
        return winner_team == 1
    if player_id in team2:
        return winner_team == 2
    return None


def _is_upset_win_for_player(player_id: int, payload: dict[str, Any], team1: set[int], team2: set[int]) -> bool:
    t1_avg = _avg([
        _safe_float(payload.get("t1_p1_r")),
        _safe_float(payload.get("t1_p2_r")),
    ])
    t2_avg = _avg([
        _safe_float(payload.get("t2_p1_r")),
        _safe_float(payload.get("t2_p2_r")),
    ])
    if t1_avg is None or t2_avg is None:
        return False

    if player_id in team1:
        return ((t2_avg - t1_avg) / 400.0) >= UPSET_THRESHOLD
    if player_id in team2:
        return ((t1_avg - t2_avg) / 400.0) >= UPSET_THRESHOLD
    return False


def _insert_processed_match_guard(supabase: Any, club_id: str, match_id: str, player_id: int) -> bool:
    payload = {
        "club_id": str(club_id),
        "match_id": str(match_id),
        "player_id": int(player_id),
    }
    existed_before = _processed_match_guard_exists(supabase, str(club_id), str(match_id), int(player_id))
    try:
        resp = (
            supabase.table("processed_match_facts")
            .insert(
                payload,
                on_conflict="club_id,match_id,player_id",
                ignore_duplicates=True,
                returning="representation",
            )
            .execute()
        )
    except TypeError:
        resp = (
            supabase.table("processed_match_facts")
            .upsert(payload, on_conflict="club_id,match_id,player_id")
            .execute()
        )
    inserted_rows = _response_rowcount(resp)
    if inserted_rows is not None:
        return inserted_rows > 0
    exists_after = _processed_match_guard_exists(supabase, str(club_id), str(match_id), int(player_id))
    return (not existed_before) and exists_after


def _processed_match_guard_exists(supabase: Any, club_id: str, match_id: str, player_id: int) -> bool:
    rows = (
        supabase.table("processed_match_facts")
        .select("player_id")
        .eq("club_id", str(club_id))
        .eq("match_id", str(match_id))
        .eq("player_id", int(player_id))
        .limit(1)
        .execute()
        .data
        or []
    )
    return bool(rows)


def _response_rowcount(resp: Any) -> int | None:
    data = getattr(resp, "data", None)
    if isinstance(data, list):
        return len(data)
    count = getattr(resp, "count", None)
    if isinstance(count, int):
        return count
    return None


def _read_fact_rows(supabase: Any, club_id: str, player_id: int, context_id: str, fact_key: str) -> list[dict[str, Any]]:
    resp = (
        supabase.table("player_badge_facts")
        .select("fact_value_num")
        .eq("club_id", club_id)
        .eq("player_id", int(player_id))
        .eq("context_id", context_id)
        .eq("fact_key", fact_key)
        .limit(1)
        .execute()
    )
    return resp.data or []


def _read_numeric_fact(supabase: Any, club_id: str, player_id: int, context_id: str, fact_key: str) -> float:
    rows = _read_fact_rows(supabase, club_id, player_id, context_id, fact_key)
    if not rows:
        return 0.0
    return float(rows[0].get("fact_value_num") or 0.0)


def _increment_numeric_fact(
    supabase: Any,
    club_id: str,
    player_id: int,
    context_id: str,
    fact_key: str,
    delta: float,
    now_iso: str,
) -> None:
    current = _read_numeric_fact(supabase, club_id, player_id, context_id, fact_key)
    _set_numeric_fact(supabase, club_id, player_id, context_id, fact_key, current + float(delta), now_iso)


def _set_numeric_fact(
    supabase: Any,
    club_id: str,
    player_id: int,
    context_id: str,
    fact_key: str,
    value: float,
    now_iso: str,
) -> None:
    sb_upsert(
        supabase,
        "player_badge_facts",
        {
            "club_id": club_id,
            "player_id": int(player_id),
            "context_id": context_id,
            "fact_key": fact_key,
            "fact_value_num": float(value),
            "updated_at": now_iso,
        },
        conflict="club_id,player_id,context_id,fact_key",
    )


def _read_player_rating(supabase: Any, club_id: str, player_id: int) -> float:
    resp = (
        supabase.table("players")
        .select("rating")
        .eq("club_id", club_id)
        .eq("id", int(player_id))
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return 1200.0
    return float(rows[0].get("rating") or 1200.0)


def _read_player_starting_rating(supabase: Any, club_id: str, player_id: int, *, fallback: float) -> float:
    resp = (
        supabase.table("players")
        .select("starting_rating,rating")
        .eq("club_id", club_id)
        .eq("id", int(player_id))
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return float(fallback)
    row = rows[0]
    starting = row.get("starting_rating")
    if starting is None:
        starting = row.get("rating")
    if starting is None:
        return float(fallback)
    return float(starting)


def _avg(values: list[float | None]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


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
