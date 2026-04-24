# jupr_app/domain/match_processing.py

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Callable

from jupr_app.domain.ratings import calculate_hybrid_elo
from jupr_app.domain.player_activity import (
    build_player_activity_update,
    coerce_utc_datetime,
    max_activity_time,
)
from jupr_app.domain.gamification.badge_queue import enqueue_badge_eval
from jupr_app.domain.gamification.badge_worker import process_badge_eval_queue
from jupr_app.domain.gamification.live_awards import run_live_badge_awards
from jupr_app.domain.notifications.player_profile_update_repo import (
    queue_player_updates_for_affected_subscribers,
)
from jupr_app.domain.player_ratings_source import build_seed_rating_maps, current_seed_rating


logger = logging.getLogger(__name__)


def process_matches(
    match_list: list[dict[str, Any]],
    *,
    supabase,
    club_id: str,
    name_to_id: dict[str, int],
    df_players_all,
    df_leagues,
    df_meta,
    sb_retry: Callable | None = None,
    default_k_factor: int = 32,
    min_win_delta_elo: float = 1.0,
    cap_loser_gain_elo: float | None = 16.0,
) -> dict[str, Any]:
    """
    - Applies overall rating updates to players table
    - Applies league rating updates to league_ratings table (skips PopUp)
    - Inserts match rows with snapshot start/end ratings for each player in that match

    match_list rows may contain player ids (int) or names (str). Supported score keys:
      - s1/s2 (preferred; used by live ladder + uploader)
      - score_t1/score_t2 (legacy)
    """

    # Default retry wrapper: just run the callable
    if sb_retry is None:
        def sb_retry(fn):
            return fn()

    db_matches: list[dict[str, Any]] = []
    overall_updates: dict[int, dict[str, Any]] = {}   # pid -> {"r","w","l","mp"}
    island_updates: dict[tuple[int, str], dict[str, Any]] = {}  # (pid, league) -> {"r","start","w","l","mp"}
    last_game_updates: dict[int, datetime] = {}
    affected_players: set[int] = set()
    match_payloads: list[dict[str, Any]] = []
    successful_match_dates: list[str] = []

    skipped_incomplete = 0
    skipped_empty = 0
    has_badge_eligible_match = False

    def as_pid(x):
        """Accept int IDs OR numeric strings OR exact names. Returns int player_id or None."""
        if x is None:
            return None
        if isinstance(x, int):
            return int(x)

        s = str(x).strip()
        if not s:
            return None
        if s.isdigit():
            return int(s)

        return name_to_id.get(s)

    candidate_player_ids: set[int] = set()
    candidate_league_names: set[str] = set()
    for m in match_list:
        for col in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
            pid = as_pid(m.get(col))
            if pid is not None:
                candidate_player_ids.add(int(pid))
        league_name = str(m.get("league", "") or "").strip()
        if league_name:
            candidate_league_names.add(league_name)

    overall_seed_map, league_seed_map, ratings_from_live_tables = build_seed_rating_maps(
        supabase=supabase,
        club_id=str(club_id),
        player_ids=candidate_player_ids,
        league_names=candidate_league_names,
        df_players_all=df_players_all,
        df_leagues=df_leagues,
    )
    if not ratings_from_live_tables:
        logger.warning("process_matches is using fallback rating DataFrames; live seed rating query failed.")

    def get_k(league_name: str) -> int:
        if df_meta is None or getattr(df_meta, "empty", True):
            return int(default_k_factor)
        row = df_meta[df_meta["league_name"] == league_name]
        if not row.empty:
            try:
                return int(row.iloc[0].get("k_factor", default_k_factor) or default_k_factor)
            except Exception:
                return int(default_k_factor)
        return int(default_k_factor)

    def get_player_row(pid: int):
        row = df_players_all[df_players_all["id"] == pid]
        if row.empty:
            return None
        return row.iloc[0]

    def ensure_overall_entry(pid: int):
        pid = int(pid)
        if pid in overall_updates:
            return
        pr = get_player_row(pid)
        if pr is None:
            overall_updates[pid] = {
                "r": float(overall_seed_map.get(pid, 1200.0)),
                "w": 0,
                "l": 0,
                "mp": 0,
            }
            return
        overall_updates[pid] = {
            "r": float(overall_seed_map.get(pid, pr.get("rating", 1200.0) or 1200.0)),
            "w": int(pr.get("wins", 0) or 0),
            "l": int(pr.get("losses", 0) or 0),
            "mp": int(pr.get("matches_played", 0) or 0),
        }

    def get_overall_r(pid: int) -> float:
        pid = int(pid)
        if pid in overall_updates:
            return float(overall_updates[pid]["r"])
        pr = get_player_row(pid)
        if pr is None:
            return float(overall_seed_map.get(pid, 1200.0))
        return float(overall_seed_map.get(pid, pr.get("rating", 1200.0) or 1200.0))

    def get_island_r(pid: int, league_name: str) -> float:
        key = (int(pid), str(league_name))
        if key in island_updates:
            return float(island_updates[key]["r"])

        return current_seed_rating(
            player_id=int(pid),
            league_name=str(league_name),
            overall_map=overall_seed_map,
            league_map=league_seed_map,
            default_rating=get_overall_r(int(pid)),
        )

    def ensure_island_entry(pid: int, league_name: str):
        key = (int(pid), str(league_name))
        if key in island_updates:
            return
        start = float(get_island_r(int(pid), str(league_name)))
        island_updates[key] = {"r": start, "start": start, "w": 0, "l": 0, "mp": 0}

    def apply_updates(
        pid: int,
        d_ov: float,
        d_isl: float,
        outcome,
        *,
        update_island: bool,
        league_name: str,
    ) -> float:
        """
        outcome:
          - True  => this player won
          - False => this player lost
          - None  => tie/unknown (no W/L change)
        """
        pid = int(pid)
        ensure_overall_entry(pid)

        overall_updates[pid]["r"] += float(d_ov)
        overall_updates[pid]["mp"] += 1
        if outcome is True:
            overall_updates[pid]["w"] += 1
        elif outcome is False:
            overall_updates[pid]["l"] += 1

        if update_island:
            ensure_island_entry(pid, league_name)
            key = (pid, league_name)
            island_updates[key]["r"] += float(d_isl)
            island_updates[key]["mp"] += 1
            if outcome is True:
                island_updates[key]["w"] += 1
            elif outcome is False:
                island_updates[key]["l"] += 1

        return float(overall_updates[pid]["r"])

    # -------------------------
    # Main match loop
    # -------------------------
    for m in match_list:
        p1 = as_pid(m.get("t1_p1"))
        p2 = as_pid(m.get("t1_p2"))
        p3 = as_pid(m.get("t2_p1"))
        p4 = as_pid(m.get("t2_p2"))

        if any(pid is None for pid in (p1, p2, p3, p4)):
            skipped_incomplete += 1
            continue

        p1, p2, p3, p4 = int(p1), int(p2), int(p3), int(p4)

        s1 = int(m.get("s1", m.get("score_t1", 0) or 0) or 0)
        s2 = int(m.get("s2", m.get("score_t2", 0) or 0) or 0)
        if (s1 + s2) <= 0:
            skipped_empty += 1
            continue

        league_name = str(m.get("league", "") or "").strip()
        week_tag = str(m.get("week_tag", "") or "")
        match_type = str(m.get("match_type", "") or "")
        rating_scope = str(m.get("rating_scope", "") or "").strip().lower()
        if rating_scope not in {"overall_only", "unrated"}:
            rating_scope = ""
        is_popup = bool(m.get("is_popup", False)) or (match_type == "PopUp")
        is_unrated = rating_scope == "unrated"
        update_island = (not is_popup) and (rating_scope != "overall_only") and (not is_unrated)
        if (not is_popup) and (not is_unrated):
            has_badge_eligible_match = True

        dt_val = m.get("date", None)
        match_dt = coerce_utc_datetime(dt_val)
        if match_dt is None:
            match_dt = datetime.now(timezone.utc)
        dt_val = match_dt.isoformat()

        ro1, ro2, ro3, ro4 = get_overall_r(p1), get_overall_r(p2), get_overall_r(p3), get_overall_r(p4)

        do1, do2 = 0.0, 0.0
        if not is_unrated:
            do1, do2 = calculate_hybrid_elo(
                (ro1 + ro2) / 2.0,
                (ro3 + ro4) / 2.0,
                s1,
                s2,
                k_factor=float(default_k_factor),
                min_win_delta=float(min_win_delta_elo),
                cap_loser_gain=cap_loser_gain_elo,
            )

        di1, di2 = 0.0, 0.0
        if update_island:
            k_val = get_k(league_name)
            ri1, ri2, ri3, ri4 = (
                get_island_r(p1, league_name),
                get_island_r(p2, league_name),
                get_island_r(p3, league_name),
                get_island_r(p4, league_name),
            )
            di1, di2 = calculate_hybrid_elo(
                (ri1 + ri2) / 2.0,
                (ri3 + ri4) / 2.0,
                s1,
                s2,
                k_factor=float(k_val),
                min_win_delta=float(min_win_delta_elo),
                cap_loser_gain=cap_loser_gain_elo,
            )

        if s1 == s2:
            t1_outcome = None
            t2_outcome = None
        else:
            t1_outcome = (s1 > s2)
            t2_outcome = (s2 > s1)

        if is_unrated:
            end_r1, end_r2, end_r3, end_r4 = ro1, ro2, ro3, ro4
            stored_elo_delta = 0.0
        else:
            end_r1 = apply_updates(
                p1, do1, di1, t1_outcome, update_island=update_island, league_name=league_name
            )
            end_r2 = apply_updates(
                p2, do1, di1, t1_outcome, update_island=update_island, league_name=league_name
            )
            end_r3 = apply_updates(
                p3, do2, di2, t2_outcome, update_island=update_island, league_name=league_name
            )
            end_r4 = apply_updates(
                p4, do2, di2, t2_outcome, update_island=update_island, league_name=league_name
            )
            for pid in (p1, p2, p3, p4):
                last_game_updates[pid] = max_activity_time(last_game_updates.get(pid), match_dt)
                affected_players.add(int(pid))
            stored_elo_delta = abs(do1) if (t1_outcome is True) else abs(do2)

        db_matches.append(
            {
                "club_id": club_id,
                "date": dt_val,
                "league": league_name,
                "t1_p1": p1,
                "t1_p2": p2,
                "t2_p1": p3,
                "t2_p2": p4,
                "score_t1": s1,
                "score_t2": s2,
                "elo_delta": float(stored_elo_delta),
                "match_type": match_type,
                "week_tag": week_tag,
                "t1_p1_r": float(ro1),
                "t1_p2_r": float(ro2),
                "t2_p1_r": float(ro3),
                "t2_p2_r": float(ro4),
                "t1_p1_r_end": float(end_r1),
                "t1_p2_r_end": float(end_r2),
                "t2_p1_r_end": float(end_r3),
                "t2_p2_r_end": float(end_r4),
                "context_type": m.get("context_type"),
                "context_id": m.get("context_id"),
                "tournament_id": m.get("tournament_id"),
                "tournament_game_id": m.get("tournament_game_id"),
            }
        )
        match_payloads.append({"league": league_name, "date": dt_val, "score_t1": s1, "score_t2": s2})
        successful_match_dates.append(dt_val)

    # -------------------------
    # Write match rows
    # -------------------------
    badge_summary: dict[str, Any] = {"mode": "skipped", "awarded_count": 0, "candidate_count": 0, "badge_ids": []}
    if db_matches:
        CHUNK_M = 300
        for i in range(0, len(db_matches), CHUNK_M):
            chunk = db_matches[i : i + CHUNK_M]
            insert_result = sb_retry(lambda chunk=chunk: supabase.table("matches").insert(chunk).execute())
            inserted_rows = getattr(insert_result, "data", None) if insert_result is not None else None
            if not inserted_rows:
                error_obj = getattr(insert_result, "error", None) if insert_result is not None else None
                sample = chunk[0] if chunk else {}
                debug_payload = {
                    "club_id": sample.get("club_id"),
                    "league": sample.get("league"),
                    "t1_p1": sample.get("t1_p1"),
                    "t1_p2": sample.get("t1_p2"),
                    "t2_p1": sample.get("t2_p1"),
                    "t2_p2": sample.get("t2_p2"),
                    "score_t1": sample.get("score_t1"),
                    "score_t2": sample.get("score_t2"),
                    "tournament_game_id": sample.get("tournament_game_id"),
                }
                raise RuntimeError(
                    "Failed to insert match rows into matches table; "
                    f"error={error_obj!r}; sample_row={debug_payload}"
                )

        if supabase is not None and has_badge_eligible_match:
            enqueue_result = enqueue_badge_eval(
                supabase,
                club_id=str(club_id),
                event_type="match_recorded",
                player_ids=sorted(affected_players),
                payload={"match_count": len(db_matches), "matches": match_payloads[:10]},
            )
            worker_result = None
            should_fallback = not bool(enqueue_result.get("queued"))
            if not should_fallback:
                try:
                    worker_result = process_badge_eval_queue(supabase, max_jobs=1, time_budget_seconds=2)
                    should_fallback = bool(worker_result.get("errored")) or (
                        int(worker_result.get("processed") or 0) == 0 and int(worker_result.get("errored") or 0) > 0
                    )
                    badge_summary = {"mode": "queue", **worker_result}
                except Exception as exc:  # noqa: BLE001
                    should_fallback = True
                    logger.warning("Badge queue worker failed during match processing: %s", exc)
            else:
                logger.warning(
                    "Badge queue enqueue unavailable during match processing; falling back inline. reason=%s",
                    enqueue_result.get("reason"),
                )
            if should_fallback:
                try:
                    badge_summary = run_live_badge_awards(
                        supabase,
                        club_id=str(club_id),
                        player_ids=sorted(affected_players),
                        event_type="match_recorded",
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Inline live badge fallback failed after match processing: %s", exc)
                    badge_summary = {"mode": "inline_error", "error": str(exc)}

    # -------------------------
    # Update overall player rows
    # -------------------------
    def update_player_row(row, activity_update: dict):
        pid = int(row["id"])
        payload = {
            "rating": float(row["rating"]),
            "wins": int(row["wins"]),
            "losses": int(row["losses"]),
            "matches_played": int(row["matches_played"]),
        }
        if activity_update:
            payload.update(activity_update)
        res = supabase.table("players").update(payload).eq("club_id", club_id).eq("id", pid).execute()
        if not res.data:
            payload_ins = {"club_id": club_id, "id": pid, **payload}
            supabase.table("players").insert(payload_ins).execute()

    for pid, stats in overall_updates.items():
        row = {
            "id": int(pid),
            "rating": float(stats["r"]),
            "wins": int(stats["w"]),
            "losses": int(stats["l"]),
            "matches_played": int(stats["mp"]),
        }
        existing_last_game_at = None
        pr = get_player_row(int(pid))
        if pr is not None:
            existing_last_game_at = pr.get("last_game_at")
        latest_match_at = last_game_updates.get(int(pid))
        activity_update = build_player_activity_update(existing_last_game_at, latest_match_at)
        sb_retry(lambda row=row, activity_update=activity_update: update_player_row(row, activity_update))

    # -------------------------
    # Update league ratings
    # -------------------------
    if island_updates:
        for (pid, league_name), stats in island_updates.items():
            payload = {
                "club_id": club_id,
                "player_id": int(pid),
                "league_name": str(league_name),
                "rating": float(stats["r"]),
                "wins": int(stats["w"]),
                "losses": int(stats["l"]),
                "matches_played": int(stats["mp"]),
            }

            existing = sb_retry(lambda pid=pid, league_name=league_name: (
                supabase.table("league_ratings")
                .select("id,wins,losses,matches_played,starting_rating")
                .eq("club_id", club_id)
                .eq("player_id", int(pid))
                .eq("league_name", str(league_name))
                .limit(1)
                .execute()
            ))

            if existing.data:
                cur = existing.data[0]
                payload["wins"] += int(cur.get("wins", 0) or 0)
                payload["losses"] += int(cur.get("losses", 0) or 0)
                payload["matches_played"] += int(cur.get("matches_played", 0) or 0)
                payload["is_active"] = True
                payload["inactive_at"] = None

                if cur.get("starting_rating") is not None:
                    payload["starting_rating"] = float(cur["starting_rating"])
                else:
                    payload["starting_rating"] = float(stats.get("start", 1200.0))

                sb_retry(lambda payload=payload, rid=int(cur["id"]): (
                    supabase.table("league_ratings").update(payload).eq("id", rid).execute()
                ))
            else:
                payload["starting_rating"] = float(stats.get("start", 1200.0))
                payload["is_active"] = True
                payload["inactive_at"] = None
                sb_retry(lambda payload=payload: supabase.table("league_ratings").insert(payload).execute())

    player_update_queue: dict[str, Any] = {
        "mode": "skipped",
        "affected_players": len(affected_players),
        "week_windows": 0,
        "queued": 0,
        "already_queued": 0,
        "no_active_subscription": 0,
        "failed": 0,
    }
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
            logger.warning("Player update queueing failed after match processing: %s", exc)
            player_update_queue = {
                **player_update_queue,
                "mode": "error",
                "error": str(exc),
            }

    return {
        "inserted": len(db_matches),
        "skipped_incomplete": int(skipped_incomplete),
        "skipped_empty": int(skipped_empty),
        "badge_summary": badge_summary,
        "player_update_queue": player_update_queue,
    }
