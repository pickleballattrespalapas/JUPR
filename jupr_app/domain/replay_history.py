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

    # ---------------------------------------------------------
    # Load source-of-truth data
    # ---------------------------------------------------------
    all_players = (
        supabase.table("players")
        .select("*")
        .eq("club_id", club_id)
        .execute()
        .data
        or []
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

    valid_player_ids = {int(p["id"]) for p in all_players}

    # ---------------------------------------------------------
    # Build K-factor map
    # ---------------------------------------------------------
    k_map: Dict[str, int] = {}
    if df_meta is not None and not df_meta.empty:
        for _, r in df_meta.iterrows():
            try:
                k_map[str(r["league_name"])] = int(
                    r.get("k_factor", DEFAULT_K_FACTOR) or DEFAULT_K_FACTOR
                )
            except Exception:
                pass

    def k_for(lg: str) -> int:
        return int(k_map.get(str(lg), DEFAULT_K_FACTOR))

    # ---------------------------------------------------------
    # Initialize player overall state
    # ---------------------------------------------------------
    p_map: Dict[int, Dict[str, Any]] = {}

    for p in all_players:
        base = p.get("starting_rating")
        if base is None:
            base = p.get("rating", 1200.0)

        try:
            p_map[int(p["id"])] = {
                "r": float(base),
                "w": 0,
                "l": 0,
                "mp": 0,
            }
        except Exception:
            continue

    def ensure_player(pid: int) -> None:
        if pid not in p_map:
            # Never create phantom DB players — just track in memory
            p_map[pid] = {"r": 1200.0, "w": 0, "l": 0, "mp": 0}

    def gr(pid: int) -> float:
        ensure_player(int(pid))
        return float(p_map[int(pid)]["r"])

    # league-specific map
    island_map: Dict[tuple[int, str], Dict[str, Any]] = {}

    def gir(pid: int, lg: str) -> float:
        ensure_player(int(pid))
        key = (int(pid), str(lg))
        if key not in island_map:
            island_map[key] = {
                "r": float(p_map[int(pid)]["r"]),
                "w": 0,
                "l": 0,
                "mp": 0,
            }
        return float(island_map[key]["r"])

    matches_to_update: list[dict[str, Any]] = []
    skipped_incomplete = 0

    # ---------------------------------------------------------
    # Replay Loop (Pure Computation)
    # ---------------------------------------------------------
    for m in all_matches:
        lg = str(m.get("league", "") or "")
        lg = unicodedata.normalize("NFKC", lg)
        lg = lg.replace("’", "'")
        lg = " ".join(lg.split())

        if (
            str(target_reset).strip() != FULL_RESET_LABEL
            and lg != str(target_reset).strip()
        ):
            continue

        p1 = m.get("t1_p1")
        p2 = m.get("t1_p2")
        p3 = m.get("t2_p1")
        p4 = m.get("t2_p2")

        if None in (p1, p2, p3, p4):
            skipped_incomplete += 1
            continue

        p1, p2, p3, p4 = int(p1), int(p2), int(p3), int(p4)

        s1 = int(m.get("score_t1", 0) or 0)
        s2 = int(m.get("score_t2", 0) or 0)

        # overall start ratings
        sr1, sr2, sr3, sr4 = gr(p1), gr(p2), gr(p3), gr(p4)

        do1, do2 = calculate_hybrid_elo(
            (sr1 + sr2) / 2,
            (sr3 + sr4) / 2,
            s1,
            s2,
            k_factor=DEFAULT_K_FACTOR,
        )

        win = s1 > s2

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

        er1, er2, er3, er4 = gr(p1), gr(p2), gr(p3), gr(p4)

        # league-specific (skip PopUp)
        if str(m.get("match_type", "")) != "PopUp":
            ir1, ir2, ir3, ir4 = gir(p1, lg), gir(p2, lg), gir(p3, lg), gir(p4, lg)

            di1, di2 = calculate_hybrid_elo(
                (ir1 + ir2) / 2,
                (ir3 + ir4) / 2,
                s1,
                s2,
                k_factor=k_for(lg),
            )

            for pid, delta, won_flag in [
                (p1, di1, win),
                (p2, di1, win),
                (p3, di2, not win),
                (p4, di2, not win),
            ]:
                key = (pid, lg)
                if key not in island_map:
                    island_map[key] = {
                        "r": float(p_map[pid]["r"]),
                        "w": 0,
                        "l": 0,
                        "mp": 0,
                    }

                island_map[key]["r"] += float(delta)
                island_map[key]["mp"] += 1
                if won_flag:
                    island_map[key]["w"] += 1
                else:
                    island_map[key]["l"] += 1

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
    # Update Players (UPDATE only, no inserts)
    # ---------------------------------------------------------
    players_updated = False

    if str(target_reset).strip() == FULL_RESET_LABEL:
        players_updated = True

        for pid, s in p_map.items():
            if pid not in valid_player_ids:
                continue  # never recreate missing players

            supabase.table("players").update(
                {
                    "rating": float(s["r"]),
                    "wins": int(s["w"]),
                    "losses": int(s["l"]),
                    "matches_played": int(s["mp"]),
                }
            ).eq("club_id", club_id).eq("id", pid).execute()

    # ---------------------------------------------------------
    # Rebuild league_ratings
    # ---------------------------------------------------------
    if str(target_reset).strip() != FULL_RESET_LABEL:
        supabase.table("league_ratings").delete().eq(
            "club_id", club_id
        ).eq("league_name", str(target_reset)).execute()
    else:
        supabase.table("league_ratings").delete().eq(
            "club_id", club_id
        ).execute()

    new_rows: list[dict[str, Any]] = []

    for (pid, lg), s in island_map.items():
        if (
            str(target_reset).strip() == FULL_RESET_LABEL
            or str(lg).strip() == str(target_reset).strip()
        ):
            start_base = 1200.0
            for p in all_players:
                try:
                    if int(p["id"]) == int(pid):
                        start_base = float(
                            p.get("starting_rating", p.get("rating", 1200.0))
                            or 1200.0
                        )
                        break
                except Exception:
                    continue

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

    for i in range(0, len(new_rows), 1000):
        supabase.table("league_ratings").insert(
            new_rows[i : i + 1000]
        ).execute()

    # ---------------------------------------------------------
    # Rewrite match snapshots (BATCHED)
    # ---------------------------------------------------------
    total = max(1, len(matches_to_update))
    rewritten = 0

    for i in range(0, len(matches_to_update), 500):
        batch = matches_to_update[i : i + 500]

        supabase.table("matches").upsert(
            batch,
            on_conflict="id",
        ).execute()

        rewritten += len(batch)

        if progress_cb:
            try:
                progress_cb(rewritten / total)
            except Exception:
                pass

    return {
        "target_reset": target_reset,
        "players_updated": players_updated,
        "skipped_incomplete": int(skipped_incomplete),
        "matches_rewritten": int(len(matches_to_update)),
        "league_ratings_rows": int(len(new_rows)),
        "matches_scanned_total": int(len(all_matches)),
    }
