from __future__ import annotations

# Match writes must go through match_pipeline.

from jupr_app.data.sb_write import sb_rpc, sb_update

import json
import time
from typing import Any, Callable, Dict, Optional

import pandas as pd

from jupr_app.domain.ratings import calculate_hybrid_elo
from jupr_app.domain.constants import DEFAULT_K_FACTOR
FULL_RESET_LABEL = "ALL (Full System Reset)"


def _exec_with_retry(fn, *, retries: int = 6, base_sleep: float = 0.5):
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt >= retries:
                raise
            msg = str(exc)
            if "Resource temporarily unavailable" not in msg and "ReadError" not in msg:
                raise
            delay = base_sleep * (2**attempt)
            time.sleep(delay)


def _set_replay_lock_running(*, supabase: Any, club_id: str) -> dict[str, Any]:
    lock_rows = (
        supabase.table("replay_lock")
        .select("club_id,status")
        .eq("club_id", str(club_id))
        .execute()
        .data
    ) or []

    existing = lock_rows[0] if lock_rows else None
    if existing and str(existing.get("status") or "").lower() == "running":
        raise RuntimeError(f"Replay already running for club_id={club_id}.")

    if existing:
        supabase.table("replay_lock").update({"status": "running"}).eq("club_id", str(club_id)).execute()
    else:
        supabase.table("replay_lock").insert({"club_id": str(club_id), "status": "running"}).execute()

    return {
        "club_id": str(club_id),
        "status_before": (str(existing.get("status")) if existing else None),
        "status_after": "running",
        "created": bool(existing is None),
    }


def _set_replay_lock_status(*, supabase: Any, club_id: str, status: str) -> None:
    supabase.table("replay_lock").update({"status": str(status)}).eq("club_id", str(club_id)).execute()


def _json_size_bytes(payload: Any) -> int:
    return len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def _chunk_rows_by_payload_limit(rows: list[Dict[str, Any]], *, max_payload_bytes: int) -> list[list[Dict[str, Any]]]:
    if not rows:
        return []

    chunks: list[list[Dict[str, Any]]] = []
    current: list[Dict[str, Any]] = []
    current_size = 2  # []

    for row in rows:
        row_size = _json_size_bytes(row)
        if row_size + 2 > max_payload_bytes:
            raise RuntimeError("replace_league_ratings row exceeds max payload size")

        delim = 1 if current else 0
        if current and current_size + delim + row_size > max_payload_bytes:
            chunks.append(current)
            current = [row]
            current_size = 2 + row_size
            continue

        current.append(row)
        current_size += delim + row_size

    if current:
        chunks.append(current)

    return chunks


def replay_history(
    *,
    supabase,
    club_id: str,
    df_meta: Optional[pd.DataFrame],
    target_reset: str,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """
    Extracted replay logic from jupr_app/ui/pages/admin_tools.py.

    This replays match history in chronological order and:
      - rewrites match snapshot columns (t1_p1_r, ..., t2_p2_r_end)
      - rebuilds league_ratings for either a single league or ALL
      - updates players table only when doing ALL (Full System Reset)

    progress_cb: called with fraction [0..1] during match snapshot rewrites.
    """

    lock_info = _set_replay_lock_running(supabase=supabase, club_id=club_id)
    try:
        all_players = supabase.table("players").select("*").eq("club_id", club_id).execute().data or []
        all_matches = (
            supabase.table("matches")
            .select("*")
            .eq("club_id", club_id)
            .order("date", desc=False)
            .order("id", desc=False)
            .execute()
            .data
        ) or []
        row_by_id: Dict[int, Dict[str, Any]] = {}
        for row in all_matches:
            try:
                row_by_id[int(row["id"])] = dict(row)
            except Exception:
                continue

        k_map: Dict[str, int] = {}
        if df_meta is not None and not df_meta.empty:
            for _, r in df_meta.iterrows():
                try:
                    k_map[str(r["league_name"])] = int(r.get("k_factor", DEFAULT_K_FACTOR) or DEFAULT_K_FACTOR)
                except Exception:
                    pass

        def k_for(lg: str) -> int:
            return int(k_map.get(str(lg), DEFAULT_K_FACTOR))

        p_map: Dict[int, Dict[str, Any]] = {}
        for p in all_players:
            base = p.get("starting_rating", None)
            if base is None:
                base = p.get("rating", 1200.0)
            try:
                p_map[int(p["id"])] = {"r": float(base), "w": 0, "l": 0, "mp": 0}
            except Exception:
                continue

        def ensure_player(pid: int) -> None:
            if pid not in p_map:
                p_map[pid] = {"r": 1200.0, "w": 0, "l": 0, "mp": 0}

        def gr(pid: int) -> float:
            ensure_player(int(pid))
            return float(p_map[int(pid)]["r"])

        island_map: Dict[tuple[int, str], Dict[str, Any]] = {}

        def gir(pid: int, lg: str) -> float:
            ensure_player(int(pid))
            key = (int(pid), str(lg))
            if key not in island_map:
                island_map[key] = {"r": float(p_map[int(pid)]["r"]), "w": 0, "l": 0, "mp": 0}
            return float(island_map[key]["r"])

        matches_to_update = []
        skipped_incomplete = 0
        for m in all_matches:
            lg = str(m.get("league", "") or "").strip()
            if str(target_reset).strip() != FULL_RESET_LABEL and lg != str(target_reset).strip():
                continue

            p1, p2, p3, p4 = m.get("t1_p1"), m.get("t1_p2"), m.get("t2_p1"), m.get("t2_p2")
            if p1 is None or p2 is None or p3 is None or p4 is None:
                skipped_incomplete += 1
                continue
            p1, p2, p3, p4 = int(p1), int(p2), int(p3), int(p4)

            s1 = int(m.get("score_t1", 0) or 0)
            s2 = int(m.get("score_t2", 0) or 0)

            sr1, sr2, sr3, sr4 = gr(p1), gr(p2), gr(p3), gr(p4)
            do1, do2 = calculate_hybrid_elo((sr1 + sr2) / 2, (sr3 + sr4) / 2, s1, s2, k_factor=DEFAULT_K_FACTOR)

            win = s1 > s2
            for pid, d, won_flag in [(p1, do1, win), (p2, do1, win), (p3, do2, not win), (p4, do2, not win)]:
                ensure_player(int(pid))
                p_map[int(pid)]["r"] += float(d)
                p_map[int(pid)]["mp"] += 1
                if won_flag:
                    p_map[int(pid)]["w"] += 1
                else:
                    p_map[int(pid)]["l"] += 1

            er1, er2, er3, er4 = gr(p1), gr(p2), gr(p3), gr(p4)

            if str(m.get("match_type", "")) != "PopUp":
                ir1, ir2, ir3, ir4 = gir(p1, lg), gir(p2, lg), gir(p3, lg), gir(p4, lg)
                di1, di2 = calculate_hybrid_elo((ir1 + ir2) / 2, (ir3 + ir4) / 2, s1, s2, k_factor=k_for(lg))
                for pid, d, won_flag in [(p1, di1, win), (p2, di1, win), (p3, di2, not win), (p4, di2, not win)]:
                    key = (int(pid), lg)
                    if key not in island_map:
                        island_map[key] = {"r": float(p_map[int(pid)]["r"]), "w": 0, "l": 0, "mp": 0}
                    island_map[key]["r"] += float(d)
                    island_map[key]["mp"] += 1
                    if won_flag:
                        island_map[key]["w"] += 1
                    else:
                        island_map[key]["l"] += 1

            stored_elo_delta = abs(do1) if win else abs(do2)
            matches_to_update.append(
                {
                    "id": int(m["id"]),
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

        players_updated = False
        if str(target_reset).strip() == FULL_RESET_LABEL:
            players_updated = True
            for pid, s in p_map.items():
                _exec_with_retry(
                    lambda pid=pid, s=s: sb_update(
                        supabase,
                        "players",
                        {"rating": s["r"], "wins": s["w"], "losses": s["l"], "matches_played": s["mp"]},
                        filters={"club_id": club_id, "id": int(pid)},
                        derived_from_match_history=True,
                    )
                )

        new_rows = []
        for (pid, lg), s in island_map.items():
            if str(target_reset).strip() == FULL_RESET_LABEL or str(lg).strip() == str(target_reset).strip():
                start_base = 1200.0
                for p in all_players:
                    try:
                        if int(p["id"]) == int(pid):
                            start_base = float(p.get("starting_rating", p.get("rating", 1200.0)) or 1200.0)
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

        new_rows = sorted(new_rows, key=lambda row: (str(row.get("league_name") or ""), int(row.get("player_id") or 0)))
        payload_chunks = _chunk_rows_by_payload_limit(new_rows, max_payload_bytes=5 * 1024 * 1024)

        if not payload_chunks:
            if str(target_reset).strip() != FULL_RESET_LABEL:
                supabase.table("league_ratings").delete().eq("club_id", club_id).eq("league_name", str(target_reset)).execute()
            else:
                supabase.table("league_ratings").delete().eq("club_id", club_id).execute()
        else:
            for idx, chunk in enumerate(payload_chunks):
                _exec_with_retry(
                    lambda chunk=chunk, idx=idx: sb_rpc(
                        supabase,
                        "replace_league_ratings",
                        {
                            "p_club_id": club_id,
                            "p_rows": chunk,
                            "p_reset": bool(idx == 0),
                        },
                    )
                )

        snapshot_columns = {
            "elo_delta",
            "t1_p1_r",
            "t1_p2_r",
            "t2_p1_r",
            "t2_p2_r",
            "t1_p1_r_end",
            "t1_p2_r_end",
            "t2_p1_r_end",
            "t2_p2_r_end",
        }

        total = len(matches_to_update)
        chunk_size = 500
        completed = 0
        for chunk_index, start in enumerate(range(0, total, chunk_size), start=1):
            chunk = matches_to_update[start : start + chunk_size]
            try:
                for row in chunk:
                    match_id = int(row["id"])
                    if match_id not in row_by_id:
                        continue
                    update_payload = {k: row[k] for k in snapshot_columns if k in row}
                    _exec_with_retry(
                        lambda update_payload=update_payload, match_id=match_id: sb_update(
                            supabase,
                            "matches",
                            update_payload,
                            filters={"club_id": club_id, "id": match_id},
                        )
                    )
            except Exception as exc:
                raise RuntimeError(f"Failed matches update chunk {chunk_index}: {exc}") from exc

            completed += len(chunk)
            if progress_cb is not None:
                try:
                    progress_cb(completed / max(1, total))
                except Exception:
                    pass

        summary = {
            "target_reset": target_reset,
            "players_updated": players_updated,
            "skipped_incomplete": int(skipped_incomplete),
            "matches_rewritten": int(len(matches_to_update)),
            "league_ratings_rows": int(len(new_rows)),
            "matches_scanned_total": int(len(all_matches)),
            "log_summary": {
                "club_id": str(club_id),
                "lock": {**lock_info, "final_status": "success"},
                "constraints": {
                    "match_player_ids_changed": False,
                    "scores_changed": False,
                    "league_changed": False,
                    "matches_deleted": False,
                },
            },
        }
        _set_replay_lock_status(supabase=supabase, club_id=club_id, status="success")
        return summary
    except Exception:
        _set_replay_lock_status(supabase=supabase, club_id=club_id, status="failed")
        raise
