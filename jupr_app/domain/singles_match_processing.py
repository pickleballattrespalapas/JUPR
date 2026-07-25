from __future__ import annotations

from datetime import datetime, timezone
import logging
import math
from typing import Any, Callable

from jupr_app.domain.matches import (
    as_player_id,
    build_match_row,
    compute_outcomes,
    compute_team_deltas,
    extract_scores,
    insert_match_chunks_with_rating_scope_fallback,
    normalize_rating_scope,
)
from jupr_app.domain.notifications.player_profile_update_repo import queue_player_updates_for_affected_subscribers
from jupr_app.domain.player_activity import build_player_activity_update, coerce_utc_datetime, max_activity_time

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float = 1200.0) -> float:
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_positive_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return max(0.0, float(value))
    except Exception:
        return 0.0


def _is_singles_match(match: dict[str, Any]) -> bool:
    return str(match.get("match_format") or match.get("format") or "").strip().lower() == "singles"


def _fallback_player_frame_lookup(df_players_all: Any, pid: int) -> dict[str, Any] | None:
    try:
        row = df_players_all[df_players_all["id"] == int(pid)]
        if row.empty:
            return None
        return dict(row.iloc[0])
    except Exception:
        return None


def _fetch_player_rows(supabase: Any, *, club_id: str, player_ids: set[int]) -> dict[int, dict[str, Any]]:
    if not player_ids:
        return {}
    try:
        rows = (
            supabase.table("players")
            .select(
                "id,name,rating,wins,losses,matches_played,last_game_at,"
                "inactive_at,active,singles_rating,singles_wins,singles_losses,"
                "singles_matches_played,singles_last_game_at,"
                "singles_replay_baseline"
            )
            .eq("club_id", str(club_id))
            .in_("id", sorted(int(pid) for pid in player_ids))
            .execute()
            .data
            or []
        )
    except Exception as exc:
        raise RuntimeError(
            "Authoritative singles player baselines are unavailable; "
            "no singles match was written."
        ) from exc
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            result[int(row.get("id"))] = dict(row)
        except Exception:
            continue
    return result


def _singles_replay_baseline(row: dict[str, Any] | None) -> dict[str, Any]:
    baseline = (row or {}).get("singles_replay_baseline")
    return dict(baseline) if isinstance(baseline, dict) else {}


def _seed_singles_rating(row: dict[str, Any] | None) -> float:
    row = row or {}
    if row.get("singles_rating") not in (None, ""):
        raw_rating = row.get("singles_rating")
    else:
        baseline = _singles_replay_baseline(row)
        raw_rating = baseline.get("rating")
        if raw_rating in (None, ""):
            raise RuntimeError(
                "A preserved singles replay baseline is required before the "
                "first managed singles match."
            )
    try:
        rating = float(raw_rating)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("The singles rating seed is invalid.") from exc
    if isinstance(raw_rating, bool) or not math.isfinite(rating):
        raise RuntimeError("The singles rating seed is invalid.")
    return rating


def _seed_stat(row: dict[str, Any] | None, key: str) -> int:
    try:
        return int(float((row or {}).get(key) or 0))
    except Exception:
        return 0


def _update_player_singles_row(supabase: Any, *, club_id: str, player_id: int, payload: dict[str, Any], sb_retry: Callable) -> None:
    def _update(update_payload: dict[str, Any]):
        return (
            supabase.table("players")
            .update(update_payload)
            .eq("club_id", str(club_id))
            .eq("id", int(player_id))
            .execute()
        )

    result = sb_retry(lambda: _update(payload))
    updated_ids = {
        int(row.get("id"))
        for row in (getattr(result, "data", None) or [])
        if row.get("id") is not None
    }
    if updated_ids != {int(player_id)}:
        raise RuntimeError(
            "Singles player aggregate update did not affect exactly one "
            f"authoritative row for player {int(player_id)}."
        )


def process_singles_matches(
    match_list: list[dict[str, Any]],
    *,
    supabase: Any,
    club_id: str,
    name_to_id: dict[str, int],
    df_players_all: Any = None,
    sb_retry: Callable | None = None,
    default_k_factor: int = 32,
    min_win_delta_elo: float = 1.0,
    cap_loser_gain_elo: float | None = 16.0,
    build_write_plan_only: bool = False,
) -> dict[str, Any]:
    """Persist and rate one-on-one singles matches using players.singles_* fields.

    Singles rating is intentionally separate from doubles/overall JUPR. It writes official
    match-history rows with match_format='singles', but only updates singles rating counters.
    """

    if sb_retry is None:
        def sb_retry(fn):
            return fn()

    prepared_singles_rows: list[tuple[datetime, int, dict[str, Any]]] = []
    for row_index, row in enumerate(match_list or []):
        if not _is_singles_match(row):
            continue
        match_dt = coerce_utc_datetime(row.get("date")) or datetime.now(timezone.utc)
        prepared_singles_rows.append((match_dt, row_index, row))
    prepared_singles_rows.sort(key=lambda item: (item[0], item[1]))

    db_matches: list[dict[str, Any]] = []
    player_updates: dict[int, dict[str, Any]] = {}
    last_game_updates: dict[int, datetime] = {}
    affected_players: set[int] = set()
    successful_match_dates: list[str] = []

    skipped_incomplete = 0
    skipped_empty = 0
    skipped_unrated = 0
    bonus_match_count = 0
    bonus_player_elo_total = 0.0

    candidate_ids: set[int] = set()
    for _match_dt, _row_index, match in prepared_singles_rows:
        for key in ("t1_p1", "t2_p1"):
            pid = as_player_id(match.get(key), name_to_id)
            if pid is not None:
                candidate_ids.add(int(pid))
    live_players = _fetch_player_rows(supabase, club_id=str(club_id), player_ids=candidate_ids)
    missing_player_ids = sorted(candidate_ids - set(live_players))
    if missing_player_ids:
        raise RuntimeError(
            "Authoritative singles player rows are incomplete; "
            f"no singles match was written: {missing_player_ids[:10]}"
        )

    def player_row(pid: int) -> dict[str, Any] | None:
        if int(pid) in live_players:
            return live_players[int(pid)]
        return _fallback_player_frame_lookup(df_players_all, int(pid))

    def ensure_entry(pid: int) -> None:
        pid = int(pid)
        if pid in player_updates:
            return
        row = player_row(pid)
        start = _seed_singles_rating(row)
        player_updates[pid] = {
            "r": start,
            "start": start,
            "w": _seed_stat(row, "singles_wins"),
            "l": _seed_stat(row, "singles_losses"),
            "mp": _seed_stat(row, "singles_matches_played"),
        }

    def get_singles_r(pid: int) -> float:
        ensure_entry(pid)
        return float(player_updates[int(pid)]["r"])

    def apply_update(pid: int, delta: float, outcome: bool | None, *, bonus_elo: float = 0.0) -> float:
        pid = int(pid)
        ensure_entry(pid)
        player_updates[pid]["r"] += float(delta) + _safe_positive_float(bonus_elo)
        player_updates[pid]["mp"] += 1
        if outcome is True:
            player_updates[pid]["w"] += 1
        elif outcome is False:
            player_updates[pid]["l"] += 1
        return float(player_updates[pid]["r"])

    for match_dt, _row_index, match in prepared_singles_rows:
        p1_raw = as_player_id(match.get("t1_p1"), name_to_id)
        p2_raw = as_player_id(match.get("t2_p1"), name_to_id)
        if p1_raw is None or p2_raw is None:
            skipped_incomplete += 1
            continue
        p1, p2 = int(p1_raw), int(p2_raw)
        if p1 == p2:
            skipped_incomplete += 1
            continue

        score_t1, score_t2 = extract_scores(match)
        if (score_t1 + score_t2) <= 0:
            skipped_empty += 1
            continue
        rating_scope = normalize_rating_scope(match)
        is_unrated = rating_scope == "unrated"
        winner_bonus_elo = 0.0 if is_unrated else _safe_positive_float(match.get("rating_bonus_elo", match.get("winner_bonus_elo")))
        dt_val = match_dt.isoformat()
        r1, r2 = get_singles_r(p1), get_singles_r(p2)
        d1, d2 = (0.0, 0.0) if is_unrated else compute_team_deltas(
            r1,
            r2,
            score_t1,
            score_t2,
            k_factor=float(default_k_factor),
            min_win_delta=float(min_win_delta_elo),
            cap_loser_gain=cap_loser_gain_elo,
        )
        p1_outcome, p2_outcome = compute_outcomes(score_t1, score_t2)
        p1_bonus = winner_bonus_elo if p1_outcome is True else 0.0
        p2_bonus = winner_bonus_elo if p2_outcome is True else 0.0

        if is_unrated:
            end_r1, end_r2 = r1, r2
            stored_delta = 0.0
            skipped_unrated += 1
        else:
            end_r1 = apply_update(p1, d1, p1_outcome, bonus_elo=p1_bonus)
            end_r2 = apply_update(p2, d2, p2_outcome, bonus_elo=p2_bonus)
            for pid in (p1, p2):
                last_game_updates[pid] = max_activity_time(last_game_updates.get(pid), match_dt)
                affected_players.add(int(pid))
            if winner_bonus_elo > 0 and (p1_outcome is True or p2_outcome is True):
                bonus_match_count += 1
                bonus_player_elo_total += winner_bonus_elo
            stored_delta = (abs(d1) if p1_outcome is True else abs(d2)) + (winner_bonus_elo if (p1_outcome is True or p2_outcome is True) else 0.0)

        db_match = build_match_row(
            club_id=str(club_id),
            dt_val=dt_val,
            league_name=str(match.get("league") or "Singles").strip() or "Singles",
            pids=(p1, None, p2, None),
            scores=(int(score_t1), int(score_t2)),
            stored_elo_delta=stored_delta,
            match_type=str(match.get("match_type") or "Singles"),
            week_tag=str(match.get("week_tag") or "Singles"),
            start_ratings=(r1, None, r2, None),
            end_ratings=(end_r1, None, end_r2, None),
            context={**match, "match_format": "singles"},
            rating_scope=rating_scope,
            match_format="singles",
        )
        db_match["singles_replay_managed"] = True
        db_matches.append(db_match)
        successful_match_dates.append(dt_val)

    if build_write_plan_only:
        planned_player_updates: list[dict[str, Any]] = []
        for pid, stats in sorted(player_updates.items()):
            if pid not in affected_players:
                continue
            if int(pid) not in live_players:
                raise RuntimeError(
                    f"Official publish requires one authoritative player snapshot for player {int(pid)}."
                )
            current = dict(live_players[int(pid)])
            latest_match_at = last_game_updates.get(int(pid))
            singles_last_game_at = max_activity_time(
                current.get("singles_last_game_at")
                or _singles_replay_baseline(current).get("last_game_at"),
                latest_match_at,
            )
            activity_update = build_player_activity_update(current.get("last_game_at"), latest_match_at)
            expected = {
                "singles_rating": _seed_singles_rating(current),
                "singles_wins": _seed_stat(current, "singles_wins"),
                "singles_losses": _seed_stat(current, "singles_losses"),
                "singles_matches_played": _seed_stat(current, "singles_matches_played"),
                "singles_last_game_at": current.get("singles_last_game_at"),
                "last_game_at": current.get("last_game_at"),
                "inactive_at": current.get("inactive_at"),
                "active": bool(current.get("active", True)) if current.get("active") is not None else None,
            }
            after = {
                "singles_rating": float(stats["r"]),
                "singles_wins": int(stats["w"]),
                "singles_losses": int(stats["l"]),
                "singles_matches_played": int(stats["mp"]),
                "singles_last_game_at": (
                    singles_last_game_at.isoformat()
                    if singles_last_game_at
                    else None
                ),
                **activity_update,
            }
            planned_player_updates.append(
                {"player_id": int(pid), "rating_mode": "singles", "expected": expected, "after": after}
            )
        return {
            "inserted": len(db_matches),
            "match_format": "singles",
            "skipped_incomplete": int(skipped_incomplete),
            "skipped_empty": int(skipped_empty),
            "skipped_unrated": int(skipped_unrated),
            "winner_bonus_summary": {
                "match_count": int(bonus_match_count),
                "player_elo_total": float(bonus_player_elo_total),
            },
            "write_plan": {
                "match_rows": db_matches,
                "player_updates": planned_player_updates,
                "league_rating_updates": [],
                "league_metadata_expectations": [],
            },
            "side_effect_context": {
                "affected_player_ids": sorted(affected_players),
                "successful_match_dates": successful_match_dates,
                "has_badge_eligible_match": False,
                "match_payloads": [
                    {
                        "league": str(row.get("league") or ""),
                        "date": str(row.get("date") or ""),
                        "score_t1": row.get("score_t1"),
                        "score_t2": row.get("score_t2"),
                    }
                    for row in db_matches
                ],
            },
        }

    if db_matches:
        insert_match_chunks_with_rating_scope_fallback(db_matches=db_matches, supabase=supabase, sb_retry=sb_retry)

    for pid, stats in player_updates.items():
        if pid not in affected_players:
            continue
        existing_row = player_row(int(pid)) or {}
        existing_last_game_at = existing_row.get("last_game_at")
        latest_match_at = last_game_updates.get(int(pid))
        singles_last_game_at = max_activity_time(
            existing_row.get("singles_last_game_at")
            or _singles_replay_baseline(existing_row).get("last_game_at"),
            latest_match_at,
        )
        activity_update = build_player_activity_update(existing_last_game_at, latest_match_at)
        payload = {
            "singles_rating": float(stats["r"]),
            "singles_wins": int(stats["w"]),
            "singles_losses": int(stats["l"]),
            "singles_matches_played": int(stats["mp"]),
            "singles_last_game_at": (
                singles_last_game_at.isoformat()
                if singles_last_game_at
                else None
            ),
        }
        if activity_update:
            payload.update(activity_update)
        sb_retry(lambda pid=pid, payload=payload: _update_player_singles_row(supabase, club_id=str(club_id), player_id=int(pid), payload=payload, sb_retry=sb_retry))

    player_update_queue: dict[str, Any] = {"mode": "skipped", "affected_players": len(affected_players), "week_windows": 0, "queued": 0, "already_queued": 0, "no_active_subscription": 0, "failed": 0}
    if db_matches and affected_players:
        try:
            queue_summary = queue_player_updates_for_affected_subscribers(
                supabase,
                club_id=str(club_id),
                affected_player_ids=sorted(affected_players),
                match_dates=successful_match_dates,
            )
            player_update_queue = {"mode": "queued", **queue_summary}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Player update queueing failed after singles match processing: %s", exc)
            player_update_queue = {**player_update_queue, "mode": "error", "error": str(exc)}

    return {
        "inserted": len(db_matches),
        "match_format": "singles",
        "skipped_incomplete": int(skipped_incomplete),
        "skipped_empty": int(skipped_empty),
        "skipped_unrated": int(skipped_unrated),
        "winner_bonus_summary": {"match_count": int(bonus_match_count), "player_elo_total": float(bonus_player_elo_total)},
        "player_update_queue": player_update_queue,
    }
