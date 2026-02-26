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

      - rewrites match snapshot columns
      - rebuilds league_ratings
      - updates players table ONLY when doing FULL reset

    Players are UPDATED only (never inserted).
    """
    import random
    import time

    import httpx

    BATCH_SIZE = 500

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

    def _retry_execute(fn, *, retries: int = 4, base_sleep: float = 0.35):
        """
        Retry only transient transport errors (httpx/httpcore).
        Do NOT retry PostgREST API errors / validation errors.
        """
        for attempt in range(retries):
            try:
                return fn()
            except (
                httpx.ReadError,
                httpx.ConnectError,
                httpx.WriteError,
                httpx.RemoteProtocolError,
                httpx.TimeoutException,
            ):
                if attempt >= retries - 1:
                    raise
                # exp backoff + jitter
                time.sleep(base_sleep * (2**attempt) + random.random() * 0.15)

    # ------------------------------
    # Reset mode + target normalization
    # ------------------------------
    target_reset_raw = str(target_reset).strip()
    full_reset = target_reset_raw == FULL_RESET_LABEL
    target_league_norm = _norm_league(target_reset_raw)

    # ---------------------------------------------------------
    # Load source-of-truth data
    # ---------------------------------------------------------
    all_players = (
        supabase.table("players").select("*").eq("club_id", club_id).execute().data or []
    )

    all_matches = (
        supabase.table("matches")
        .select("*")
        .eq("club_id", club_id)
        .order("date", desc=False)
        .order("id", desc=False)
        .execute()
        .data
        or []
    )

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
            # Never create phantom DB players — just track in memory
            p_map[pid] = {"r": 1200.0, "w": 0, "l": 0, "mp": 0}

    def gr(pid: int) -> float:
        ensure_player(int(pid))
        return float(p_map[int(pid)]["r"])

    # league-specific map: (player_id, league_name) -> state
    island_map: Dict[tuple[int, str], Dict[str, Any]] = {}

    def gir(pid: int, lg: str, *, seed_rating: Optional[float] = None) -> float:
        """
        Get island rating for (pid, lg). If it doesn't exist yet, initialize it to:
          - seed_rating (pre-match overall rating) if provided
          - otherwise the player's current overall rating
        """
        pid_i = int(pid)
        lg_s = _norm_league(lg)
        ensure_player(pid_i)

        key = (pid_i, lg_s)
        if key not in island_map:
            base_r = seed_rating if seed_rating is not None else float(p_map[pid_i]["r"])
            island_map[key] = {"r": float(base_r), "w": 0, "l": 0, "mp": 0}
        return float(island_map[key]["r"])

    matches_to_update: list[dict[str, Any]] = []
    skipped_incomplete_scope = 0

    # ---------------------------------------------------------
    # Replay Loop (Pure Computation)
    #
    # Key correction:
    #   We ALWAYS replay ALL matches for overall ratings so that overall snapshots
    #   (t*_r, t*_r_end) are correct even when resetting only one league.
    #
    # When target_reset is NOT full reset, we only:
    #   - collect match updates for the target league
    #   - rebuild league_ratings for the target league
    # ---------------------------------------------------------
    for m in all_matches:
        lg = _norm_league(m.get("league", "") or "")
        in_scope = full_reset or (lg == target_league_norm)

        p1 = m.get("t1_p1")
        p2 = m.get("t1_p2")
        p3 = m.get("t2_p1")
        p4 = m.get("t2_p2")

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

        # pre-match overall ratings
        sr1, sr2, sr3, sr4 = gr(p1), gr(p2), gr(p3), gr(p4)

        # league-specific (only tracked if in-scope AND not PopUp)
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

        # overall delta
        do1, do2 = calculate_hybrid_elo(
            (sr1 + sr2) / 2,
            (sr3 + sr4) / 2,
            s1,
            s2,
            k_factor=DEFAULT_K_FACTOR,
        )

        win = s1 > s2

        # update overall state
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

        # update league-specific state
        if do_league:
            for pid, delta, won_flag in [
                (p1, di1, win),
                (p2, di1, win),
                (p3, di2, not win),
                (p4, di2, not win),
            ]:
                key = (int(pid), lg)  # lg is already normalized
                if key not in island_map:
                    # Shouldn't happen (gir seeds), but keep safe
                    island_map[key] = {"r": float(gr(pid)), "w": 0, "l": 0, "mp": 0}

                island_map[key]["r"] += float(delta)
                island_map[key]["mp"] += 1
                if won_flag:
                    island_map[key]["w"] += 1
                else:
                    island_map[key]["l"] += 1

        # post-match overall ratings
        er1, er2, er3, er4 = gr(p1), gr(p2), gr(p3), gr(p4)

        # only write match snapshots for the target league (or all matches on full reset)
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

    # ---------------------------------------------------------
    # Build league_ratings rows (this was missing in your version)
    # ---------------------------------------------------------
    new_rows: list[dict[str, Any]] = []
    for (pid, lg), s in island_map.items():
        if pid not in valid_player_ids:
            continue  # never recreate deleted/merged players

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
    # Writes (BATCHED ONLY) — no per-row .execute()
    # ---------------------------------------------------------
    players_updated = False

    # 1) Update players (FULL reset only) — batched upsert
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

        for batch in _chunk(player_updates, BATCH_SIZE):
            _retry_execute(
                lambda b=batch: supabase.table("players").upsert(b, on_conflict="id").execute()
            )

    # 2) Rebuild league_ratings (delete then insert)
    if not full_reset:
        # defensive cleanup: delete both raw and normalized league names if they differ
        if target_reset_raw and target_reset_raw != target_league_norm:
            _retry_execute(
                lambda: supabase.table("league_ratings")
                .delete()
                .eq("club_id", club_id)
                .eq("league_name", target_reset_raw)
                .execute()
            )

        _retry_execute(
            lambda: supabase.table("league_ratings")
            .delete()
            .eq("club_id", club_id)
            .eq("league_name", target_league_norm)
            .execute()
        )
    else:
        _retry_execute(
            lambda: supabase.table("league_ratings").delete().eq("club_id", club_id).execute()
        )

    for batch in _chunk(new_rows, BATCH_SIZE):
        _retry_execute(lambda b=batch: supabase.table("league_ratings").insert(b).execute())

    # 3) Rewrite match snapshots (batched upsert ONCE)
    total = max(1, len(matches_to_update))
    rewritten = 0

    for batch in _chunk(matches_to_update, BATCH_SIZE):
        _retry_execute(
            lambda b=batch: supabase.table("matches").upsert(b, on_conflict="id").execute()
        )

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
        "league_ratings_rows": int(len(new_rows)),
        "matches_scanned_total": int(len(all_matches)),
    }
