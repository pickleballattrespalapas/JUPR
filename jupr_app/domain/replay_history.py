from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import pandas as pd
import unicodedata

from jupr_app.domain.constants import DEFAULT_K_FACTOR
from jupr_app.domain.ratings import calculate_hybrid_elo

FULL_RESET_LABEL = "ALL (Full System Reset)"

def replay_history(
    *,
    supabase,
    club_id: str,
    df_meta: Optional[pd.DataFrame],
    target_reset: str,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """
    Replays match history in chronological order and:

      - rewrites match snapshot columns (RPC bulk UPDATE)
      - rebuilds league_ratings (delete + insert)
      - updates players table ONLY when doing FULL reset (RPC bulk UPDATE)

    No per-row .execute() loops for writes.
    No upsert() for partial updates (avoids NOT NULL failures like matches.league).
    """
    import random
    import time

    import httpx

    READ_PAGE_SIZE = 1000
    WRITE_BATCH_SIZE = 500
    RETRIES = 5

    RPC_MATCH_SNAPSHOTS = "bulk_update_match_snapshots"
    RPC_PLAYERS_STATS = "bulk_update_players_stats"

    # ------------------------------
    # Helpers
    # ------------------------------
    def _norm_league(val: Any) -> str:
        s = str(val or "")
        s = unicodedata.normalize("NFKC", s)
        s = s.replace("’", "'")
        s = " ".join(s.split())
        return s

    def _chunk(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
        if not rows:
            return []
        return [rows[i : i + size] for i in range(0, len(rows), size)]

    def _retry(fn, *, label: str = ""):
        for attempt in range(RETRIES):
            try:
                return fn()
            except (
                httpx.ReadError,
                httpx.ConnectError,
                httpx.WriteError,
                httpx.RemoteProtocolError,
                httpx.TimeoutException,
            ):
                if attempt == RETRIES - 1:
                    raise
                time.sleep(0.35 * (2**attempt) + random.random() * 0.15)

    # ------------------------------
    # Reset mode
    # ------------------------------
    target_reset_raw = str(target_reset).strip()
    full_reset = target_reset_raw == FULL_RESET_LABEL
    target_league_norm = _norm_league(target_reset_raw)

    # ---------------------------------------------------------
    # Load players (only what we need)
    # ---------------------------------------------------------
    players_resp = _retry(
        lambda: (
            supabase.table("players")
            .select("id,starting_rating,rating")
            .eq("club_id", club_id)
            .execute()
        ),
        label="select_players",
    )
    all_players = players_resp.data or []

    valid_player_ids: set[int] = set()
    start_base_by_pid: dict[int, float] = {}

    for p in all_players:
        try:
            pid = int(p["id"])
        except Exception:
            continue

        valid_player_ids.add(pid)

        base = p.get("starting_rating")
        if base is None:
            base = p.get("rating", 1200.0)

        try:
            start_base_by_pid[pid] = float(base or 1200.0)
        except Exception:
            start_base_by_pid[pid] = 1200.0

    # ---------------------------------------------------------
    # Build K-factor map (normalized league names)
    # ---------------------------------------------------------
    k_map: Dict[str, int] = {}
    if df_meta is not None and not df_meta.empty:
        for _, r in df_meta.iterrows():
            try:
                lg_name = _norm_league(r["league_name"])
                k_map[lg_name] = int(r.get("k_factor", DEFAULT_K_FACTOR) or DEFAULT_K_FACTOR)
            except Exception:
                pass

    def k_for(lg: str) -> int:
        return int(k_map.get(_norm_league(lg), DEFAULT_K_FACTOR))

    # ---------------------------------------------------------
    # Initialize overall player state (in-memory)
    # ---------------------------------------------------------
    p_map: Dict[int, Dict[str, Any]] = {}
    for p in all_players:
        try:
            pid = int(p["id"])
        except Exception:
            continue

        base = p.get("starting_rating")
        if base is None:
            base = p.get("rating", 1200.0)

        try:
            p_map[pid] = {"r": float(base or 1200.0), "w": 0, "l": 0, "mp": 0}
        except Exception:
            p_map[pid] = {"r": 1200.0, "w": 0, "l": 0, "mp": 0}

    def ensure_player(pid: int) -> None:
        if pid not in p_map:
            p_map[pid] = {"r": 1200.0, "w": 0, "l": 0, "mp": 0}

    def gr(pid: int) -> float:
        ensure_player(int(pid))
        return float(p_map[int(pid)]["r"])

    island_map: Dict[tuple[int, str], Dict[str, Any]] = {}

    def gir(pid: int, lg: str, *, seed_rating: Optional[float] = None) -> float:
        pid_i = int(pid)
        lg_s = _norm_league(lg)
        ensure_player(pid_i)

        key = (pid_i, lg_s)
        if key not in island_map:
            base_r = seed_rating if seed_rating is not None else float(p_map[pid_i]["r"])
            island_map[key] = {"r": float(base_r), "w": 0, "l": 0, "mp": 0}
        return float(island_map[key]["r"])

    # ---------------------------------------------------------
    # Read matches with pagination
    # ---------------------------------------------------------
    match_cols = (
        "id,date,league,match_type,"
        "t1_p1,t1_p2,t2_p1,t2_p2,"
        "score_t1,score_t2,deleted_at,rating_scope"
    )

    matches_to_update: list[dict[str, Any]] = []
    skipped_incomplete_scope = 0
    matches_scanned_total = 0

    match_query_mode = "with_deleted_and_scope"
    offset = 0
    while True:
        start = offset
        end = offset + READ_PAGE_SIZE - 1

        def _load_page_with_deleted_and_scope():
            return (
                supabase.table("matches")
                .select(match_cols)
                .eq("club_id", club_id)
                .is_("deleted_at", None)
                .order("date", desc=False)
                .order("id", desc=False)
                .range(start, end)
                .execute()
            )

        def _load_page_with_deleted_only():
            return (
                supabase.table("matches")
                .select("id,date,league,match_type,t1_p1,t1_p2,t2_p1,t2_p2,score_t1,score_t2,deleted_at")
                .eq("club_id", club_id)
                .is_("deleted_at", None)
                .order("date", desc=False)
                .order("id", desc=False)
                .range(start, end)
                .execute()
            )

        def _load_page_legacy_fallback():
            return (
                supabase.table("matches")
                .select("id,date,league,match_type,t1_p1,t1_p2,t2_p1,t2_p2,score_t1,score_t2")
                .eq("club_id", club_id)
                .order("date", desc=False)
                .order("id", desc=False)
                .range(start, end)
                .execute()
            )

        try:
            if match_query_mode == "with_deleted_and_scope":
                page_resp = _retry(_load_page_with_deleted_and_scope, label=f"select_matches_{start}_{end}")
            elif match_query_mode == "with_deleted_only":
                page_resp = _retry(_load_page_with_deleted_only, label=f"select_matches_deleted_only_{start}_{end}")
            else:
                page_resp = _retry(_load_page_legacy_fallback, label=f"select_matches_legacy_{start}_{end}")
        except Exception:
            # Backward-compat fallback for deployments with partial schema rollout.
            if match_query_mode == "with_deleted_and_scope":
                match_query_mode = "with_deleted_only"
                page_resp = _retry(_load_page_with_deleted_only, label=f"select_matches_deleted_only_{start}_{end}")
            elif match_query_mode == "with_deleted_only":
                match_query_mode = "legacy"
                page_resp = _retry(_load_page_legacy_fallback, label=f"select_matches_legacy_{start}_{end}")
            else:
                raise

        page = page_resp.data or []
        if not page:
            break

        matches_scanned_total += len(page)

        for m in page:
            if m.get("deleted_at") is not None:
                continue
            if "rating_scope" in m and str(m.get("rating_scope", "") or "").strip().lower() == "unrated":
                continue
            lg = _norm_league(m.get("league", "") or "")
            in_scope = full_reset or (lg == target_league_norm)

            p1, p2, p3, p4 = m.get("t1_p1"), m.get("t1_p2"), m.get("t2_p1"), m.get("t2_p2")
            if None in (p1, p2, p3, p4):
                if in_scope:
                    skipped_incomplete_scope += 1
                continue

            try:
                p1, p2, p3, p4 = int(p1), int(p2), int(p3), int(p4)
            except Exception:
                if in_scope:
                    skipped_incomplete_scope += 1
                continue

            s1 = int(m.get("score_t1", 0) or 0)
            s2 = int(m.get("score_t2", 0) or 0)

            sr1, sr2, sr3, sr4 = gr(p1), gr(p2), gr(p3), gr(p4)

            # league-specific only if in_scope and not PopUp
            do_league = (str(m.get("match_type", "")) != "PopUp") and in_scope
            if do_league:
                ir1 = gir(p1, lg, seed_rating=sr1)
                ir2 = gir(p2, lg, seed_rating=sr2)
                ir3 = gir(p3, lg, seed_rating=sr3)
                ir4 = gir(p4, lg, seed_rating=sr4)

                di1, di2 = calculate_hybrid_elo(
                    (ir1 + ir2) / 2,
                    (ir3 + ir4) / 2,
                    s1,
                    s2,
                    k_factor=k_for(lg),
                )
            else:
                di1 = di2 = 0.0

            do1, do2 = calculate_hybrid_elo(
                (sr1 + sr2) / 2,
                (sr3 + sr4) / 2,
                s1,
                s2,
                k_factor=DEFAULT_K_FACTOR,
            )

            win = s1 > s2

            # overall updates
            for pid, delta, won_flag in [
                (p1, do1, win),
                (p2, do1, win),
                (p3, do2, not win),
                (p4, do2, not win),
            ]:
                ensure_player(pid)
                p_map[pid]["r"] += float(delta)
                p_map[pid]["mp"] += 1
                if won_flag:
                    p_map[pid]["w"] += 1
                else:
                    p_map[pid]["l"] += 1

            # league updates
            if do_league:
                for pid, delta, won_flag in [
                    (p1, di1, win),
                    (p2, di1, win),
                    (p3, di2, not win),
                    (p4, di2, not win),
                ]:
                    key = (int(pid), lg)
                    if key not in island_map:
                        island_map[key] = {"r": float(gr(pid)), "w": 0, "l": 0, "mp": 0}

                    island_map[key]["r"] += float(delta)
                    island_map[key]["mp"] += 1
                    if won_flag:
                        island_map[key]["w"] += 1
                    else:
                        island_map[key]["l"] += 1

            er1, er2, er3, er4 = gr(p1), gr(p2), gr(p3), gr(p4)

            if in_scope:
                stored_elo_delta = abs(do1) if win else abs(do2)
                matches_to_update.append(
                    {
                        "id": int(m["id"]),
                        "club_id": club_id,
                        "elo_delta": float(stored_elo_delta),
                        "t1_p1_r": float(sr1),
                        "t1_p2_r": float(sr2),
                        "t2_p1_r": float(sr3),
                        "t2_p2_r": float(sr4),
                        "t1_p1_r_end": float(er1),
                        "t1_p2_r_end": float(er2),
                        "t2_p1_r_end": float(er3),
                        "t2_p2_r_end": float(er4),
                    }
                )

        offset += READ_PAGE_SIZE

    # ---------------------------------------------------------
    # Build league_ratings rows
    # ---------------------------------------------------------
    new_rows: list[dict[str, Any]] = []
    for (pid, lg), s in island_map.items():
        if pid not in valid_player_ids:
            continue

        start_base = float(start_base_by_pid.get(int(pid), 1200.0))
        new_rows.append(
            {
                "club_id": club_id,
                "player_id": int(pid),
                "league_name": str(lg),
                "rating": float(s["r"]),
                "wins": int(s["w"]),
                "losses": int(s["l"]),
                "matches_played": int(s["mp"]),
                "starting_rating": float(start_base),
            }
        )

    # ---------------------------------------------------------
    # Writes
    # ---------------------------------------------------------
    players_updated = False

    # 1) FULL reset: update players via RPC bulk UPDATE
    if full_reset:
        players_updated = True

        player_updates: list[dict[str, Any]] = []
        for pid, s in p_map.items():
            if pid not in valid_player_ids:
                continue
            player_updates.append(
                {
                    "id": int(pid),
                    "club_id": club_id,
                    "rating": float(s["r"]),
                    "wins": int(s["w"]),
                    "losses": int(s["l"]),
                    "matches_played": int(s["mp"]),
                }
            )

        for batch in _chunk(player_updates, WRITE_BATCH_SIZE):
            _retry(
                lambda b=batch: supabase.rpc(RPC_PLAYERS_STATS, {"rows": b}).execute(),
                label="rpc_bulk_update_players_stats",
            )

    # 2) league_ratings: delete then insert (insert is fine because rows are complete)
    if not full_reset:
        # delete both raw and normalized league names if they differ
        if target_reset_raw and target_reset_raw != target_league_norm:
            _retry(
                lambda: supabase.table("league_ratings")
                .delete(returning="minimal")
                .eq("club_id", club_id)
                .eq("league_name", target_reset_raw)
                .execute(),
                label="delete_league_ratings_raw",
            )

        _retry(
            lambda: supabase.table("league_ratings")
            .delete(returning="minimal")
            .eq("club_id", club_id)
            .eq("league_name", target_league_norm)
            .execute(),
            label="delete_league_ratings_norm",
        )
    else:
        _retry(
            lambda: supabase.table("league_ratings")
            .delete(returning="minimal")
            .eq("club_id", club_id)
            .execute(),
            label="delete_league_ratings_all",
        )

    for batch in _chunk(new_rows, WRITE_BATCH_SIZE):
        _retry(
            lambda b=batch: supabase.table("league_ratings")
            .insert(b, returning="minimal")
            .execute(),
            label="insert_league_ratings",
        )

    # 3) match snapshots: update via RPC bulk UPDATE (never touches league)
    total = max(1, len(matches_to_update))
    rewritten = 0
    updated_rows_total = 0

    for batch in _chunk(matches_to_update, WRITE_BATCH_SIZE):
        resp = _retry(
            lambda b=batch: supabase.rpc(RPC_MATCH_SNAPSHOTS, {"rows": b}).execute(),
            label="rpc_bulk_update_match_snapshots",
        )
        # RPC returns integer in resp.data (usually). Be defensive.
        try:
            updated_rows_total += int(resp.data or 0)
        except Exception:
            pass

        rewritten += len(batch)
        if progress_cb:
            try:
                progress_cb(rewritten / total)
            except Exception:
                pass

    return {
        "target_reset": target_reset,
        "players_updated": players_updated,
        "skipped_incomplete": int(skipped_incomplete_scope),
        "matches_rewritten": int(len(matches_to_update)),
        "matches_snapshots_updated_rows": int(updated_rows_total),
        "league_ratings_rows": int(len(new_rows)),
        "matches_scanned_total": int(matches_scanned_total),
    }
