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

      - rewrites match snapshot columns (for target league, or all on FULL reset)
      - rebuilds league_ratings (for target league, or all on FULL reset)
      - updates players table ONLY when doing FULL reset

    Players are UPDATED only (never inserted). (We enforce this by filtering to valid_player_ids.)
    """

    import random
    import time

    import httpx

    # Tune these if needed
    READ_PAGE_SIZE = 1000   # matches pagination size
    WRITE_BATCH_SIZE = 250  # upsert/insert batch size
    RETRIES = 5

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
        """
        Retry only transient transport failures. If it keeps failing, raise.
        """
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
                # exponential backoff + jitter
                time.sleep(0.35 * (2**attempt) + random.random() * 0.15)

    def _execute(fn, *, label: str = ""):
        # convenience wrapper to keep call sites short
        return _retry(fn, label=label)

    # Returning="minimal" is supported by Supabase insert/upsert/delete APIs (docs),
    # but to be extra-safe across versions we fall back if a TypeError occurs.
    def _delete_minimal(builder_fn, *, label: str):
        def _do():
            return builder_fn(returning="minimal").execute()

        try:
            return _execute(_do, label=label)
        except TypeError:
            return _execute(lambda: builder_fn().execute(), label=label)

    def _insert_minimal(table: str, rows: list[dict[str, Any]], *, label: str):
        def _do():
            return supabase.table(table).insert(rows, returning="minimal").execute()

        try:
            return _execute(_do, label=label)
        except TypeError:
            return _execute(lambda: supabase.table(table).insert(rows).execute(), label=label)

    def _upsert_minimal(table: str, rows: list[dict[str, Any]], *, on_conflict: str, label: str):
        def _do():
            return (
                supabase.table(table)
                .upsert(rows, on_conflict=on_conflict, returning="minimal")
                .execute()
            )

        try:
            return _execute(_do, label=label)
        except TypeError:
            return _execute(
                lambda: supabase.table(table).upsert(rows, on_conflict=on_conflict).execute(),
                label=label,
            )

    # ------------------------------
    # Reset mode
    # ------------------------------
    target_reset_raw = str(target_reset).strip()
    full_reset = target_reset_raw == FULL_RESET_LABEL
    target_league_norm = _norm_league(target_reset_raw)

    # ---------------------------------------------------------
    # Load players (select only what we need)
    # ---------------------------------------------------------
    players_cols = "id, starting_rating, rating"
    players_resp = _execute(
        lambda: (
            supabase.table("players")
            .select(players_cols)
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

    # league-specific state: (player_id, league_name) -> state
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
    # Iterate matches via pagination (select only what we need)
    # ---------------------------------------------------------
    match_cols = "id, date, league, match_type, t1_p1, t1_p2, t2_p1, t2_p2, score_t1, score_t2"

    matches_to_update: list[dict[str, Any]] = []
    skipped_incomplete_scope = 0
    matches_scanned_total = 0

    offset = 0
    while True:
        start = offset
        end = offset + READ_PAGE_SIZE - 1  # inclusive

        page_resp = _execute(
            lambda s=start, e=end: (
                supabase.table("matches")
                .select(match_cols)
                .eq("club_id", club_id)
                .order("date", desc=False)
                .order("id", desc=False)
                .range(s, e)
                .execute()
            ),
            label=f"select_matches_range_{start}_{end}",
        )

        page = page_resp.data or []
        if not page:
            break

        matches_scanned_total += len(page)

        # ---------------------------------------------------------
        # Replay Loop (computation only)
        #
        # We ALWAYS replay all matches for overall state so overall snapshots
        # for the target league are correct even on league-only reset.
        # ---------------------------------------------------------
        for m in page:
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

            # (optional) league-specific deltas only when in scope & not PopUp
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
                    key = (int(pid), lg)  # lg already normalized
                    if key not in island_map:
                        island_map[key] = {"r": float(gr(pid)), "w": 0, "l": 0, "mp": 0}

                    island_map[key]["r"] += float(delta)
                    island_map[key]["mp"] += 1
                    if won_flag:
                        island_map[key]["w"] += 1
                    else:
                        island_map[key]["l"] += 1

            # post-match overall ratings
            er1, er2, er3, er4 = gr(p1), gr(p2), gr(p3), gr(p4)

            # only write match snapshots for target league (or all on full reset)
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
    # Writes (BATCHED)
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

        for batch in _chunk(player_updates, WRITE_BATCH_SIZE):
            _upsert_minimal("players", batch, on_conflict="id", label="upsert_players")

    # 2) Rebuild league_ratings (delete then insert)
    if not full_reset:
        # delete target league
        _delete_minimal(
            lambda **kwargs: (
                supabase.table("league_ratings")
                .delete(**kwargs)
                .eq("club_id", club_id)
                .eq("league_name", target_league_norm)
            ),
            label="delete_league_ratings_target",
        )
    else:
        # delete all club rows
        _delete_minimal(
            lambda **kwargs: supabase.table("league_ratings").delete(**kwargs).eq("club_id", club_id),
            label="delete_league_ratings_all",
        )

    for batch in _chunk(new_rows, WRITE_BATCH_SIZE):
        _insert_minimal("league_ratings", batch, label="insert_league_ratings")

    # 3) Rewrite match snapshots (batched upsert ONCE)
    total = max(1, len(matches_to_update))
    rewritten = 0

    for batch in _chunk(matches_to_update, WRITE_BATCH_SIZE):
        _upsert_minimal("matches", batch, on_conflict="id", label="upsert_matches")
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
        "matches_scanned_total": int(matches_scanned_total),
    }
