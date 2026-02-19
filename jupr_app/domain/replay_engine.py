from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from jupr_app.domain.constants import DEFAULT_K_FACTOR
from jupr_app.domain.ratings import calculate_hybrid_elo

SNAPSHOT_FIELDS = (
    "elo_delta",
    "t1_p1_r",
    "t1_p2_r",
    "t2_p1_r",
    "t2_p2_r",
    "t1_p1_r_end",
    "t1_p2_r_end",
    "t2_p1_r_end",
    "t2_p2_r_end",
)


def _to_int(value: Any, *, field: str) -> int:
    if value is None:
        raise ValueError(f"Missing required integer for field={field}")
    return int(value)


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    return float(value)


def _match_order_key(row: Dict[str, Any]) -> Tuple[str, int]:
    return (str(row.get("date") or row.get("played_at") or ""), int(row["id"]))


def _normalize_snapshot(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": _to_int(payload["id"], field="id"),
        "elo_delta": float(payload["elo_delta"]),
        "t1_p1_r": float(payload["t1_p1_r"]),
        "t1_p2_r": float(payload["t1_p2_r"]),
        "t2_p1_r": float(payload["t2_p1_r"]),
        "t2_p2_r": float(payload["t2_p2_r"]),
        "t1_p1_r_end": float(payload["t1_p1_r_end"]),
        "t1_p2_r_end": float(payload["t1_p2_r_end"]),
        "t2_p1_r_end": float(payload["t2_p1_r_end"]),
        "t2_p2_r_end": float(payload["t2_p2_r_end"]),
    }


def _rpc_transactional_replace(
    *,
    supabase: Any,
    club_id: str,
    new_league_rows: List[Dict[str, Any]],
    updated_match_snapshots: List[Dict[str, Any]],
) -> None:
    payload = {
        "p_club_id": club_id,
        "p_league_rows": new_league_rows,
        "p_match_snapshot_rows": updated_match_snapshots,
        "p_match_batch_size": 500,
    }
    last_exc: Exception | None = None
    for rpc_name in ("replay_transactional_replace", "replay_transactional_replace_v1"):
        try:
            supabase.rpc(rpc_name, payload).execute()
            return
        except Exception as exc:  # pragma: no cover - depends on db runtime
            last_exc = exc

    raise RuntimeError(
        "Replay requires transactional bulk RPC (replay_transactional_replace). "
        "No compatible RPC was available."
    ) from last_exc


def _batch(iterable: List[Dict[str, Any]], batch_size: int = 500) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def replay_engine(*, supabase: Any, club_id: str) -> Dict[str, Any]:
    start = time.perf_counter()

    if not club_id:
        raise ValueError("club_id is required")

    # PHASE A — Load History (compute-only, no writes)
    matches = (
        supabase.table("matches")
        .select("*")
        .eq("club_id", club_id)
        .order("date", desc=False)
        .order("id", desc=False)
        .execute()
        .data
        or []
    )
    players = supabase.table("players").select("id,starting_rating,rating").eq("club_id", club_id).execute().data or []
    leagues = (
        supabase.table("leagues_metadata")
        .select("league_name,k_factor")
        .eq("club_id", club_id)
        .execute()
        .data
        or []
    )

    player_base: Dict[int, float] = {}
    for row in players:
        pid = _to_int(row.get("id"), field="players.id")
        player_base[pid] = _to_float(row.get("starting_rating"), _to_float(row.get("rating"), 1200.0))

    k_map = {str(r.get("league_name") or ""): int(r.get("k_factor") or DEFAULT_K_FACTOR) for r in leagues}

    overall = {pid: player_base[pid] for pid in player_base}
    league_state: Dict[Tuple[str, int], Dict[str, float]] = {}
    snapshots: List[Dict[str, Any]] = []

    for match in sorted(matches, key=_match_order_key):
        m_id = _to_int(match.get("id"), field="matches.id")
        p1 = _to_int(match.get("t1_p1"), field="t1_p1")
        p2 = _to_int(match.get("t1_p2"), field="t1_p2")
        p3 = _to_int(match.get("t2_p1"), field="t2_p1")
        p4 = _to_int(match.get("t2_p2"), field="t2_p2")
        s1 = _to_int(match.get("score_t1") or 0, field="score_t1")
        s2 = _to_int(match.get("score_t2") or 0, field="score_t2")
        league_name = str(match.get("league") or "")

        for pid in (p1, p2, p3, p4):
            overall.setdefault(pid, player_base.get(pid, 1200.0))

        sr1, sr2, sr3, sr4 = overall[p1], overall[p2], overall[p3], overall[p4]
        d1, d2 = calculate_hybrid_elo((sr1 + sr2) / 2.0, (sr3 + sr4) / 2.0, s1, s2, k_factor=DEFAULT_K_FACTOR)

        overall[p1] += float(d1)
        overall[p2] += float(d1)
        overall[p3] += float(d2)
        overall[p4] += float(d2)

        win_t1 = s1 > s2
        snapshots.append(
            _normalize_snapshot(
                {
                    "id": m_id,
                    "elo_delta": abs(float(d1) if win_t1 else float(d2)),
                    "t1_p1_r": sr1,
                    "t1_p2_r": sr2,
                    "t2_p1_r": sr3,
                    "t2_p2_r": sr4,
                    "t1_p1_r_end": overall[p1],
                    "t1_p2_r_end": overall[p2],
                    "t2_p1_r_end": overall[p3],
                    "t2_p2_r_end": overall[p4],
                }
            )
        )

        for pid in (p1, p2, p3, p4):
            key = (league_name, pid)
            if key not in league_state:
                league_state[key] = {
                    "rating": player_base.get(pid, 1200.0),
                    "wins": 0,
                    "losses": 0,
                    "matches_played": 0,
                }

        lr1 = league_state[(league_name, p1)]["rating"]
        lr2 = league_state[(league_name, p2)]["rating"]
        lr3 = league_state[(league_name, p3)]["rating"]
        lr4 = league_state[(league_name, p4)]["rating"]
        lk = int(k_map.get(league_name, DEFAULT_K_FACTOR))
        ld1, ld2 = calculate_hybrid_elo((lr1 + lr2) / 2.0, (lr3 + lr4) / 2.0, s1, s2, k_factor=lk)

        for pid, delta, won in ((p1, ld1, win_t1), (p2, ld1, win_t1), (p3, ld2, not win_t1), (p4, ld2, not win_t1)):
            row = league_state[(league_name, pid)]
            row["rating"] += float(delta)
            row["matches_played"] += 1
            if won:
                row["wins"] += 1
            else:
                row["losses"] += 1

    # PHASE B — Compute League Ratings (in memory only)
    new_league_rows = [
        {
            "club_id": club_id,
            "league_name": league_name,
            "player_id": pid,
            "rating": float(stats["rating"]),
            "wins": int(stats["wins"]),
            "losses": int(stats["losses"]),
            "matches_played": int(stats["matches_played"]),
            "starting_rating": float(player_base.get(pid, 1200.0)),
        }
        for (league_name, pid), stats in league_state.items()
    ]
    new_league_rows.sort(key=lambda r: (str(r.get("league_id") or r["league_name"]), int(r["player_id"])))
    updated_match_snapshots = sorted(snapshots, key=lambda r: int(r["id"]))

    # PHASE C — Transactional Replace (single DB transaction via RPC)
    _rpc_transactional_replace(
        supabase=supabase,
        club_id=club_id,
        new_league_rows=new_league_rows,
        updated_match_snapshots=updated_match_snapshots,
    )

    # PHASE D — Verification (post-commit)
    db_league_rows = (
        supabase.table("league_ratings")
        .select("player_id,rating")
        .eq("club_id", club_id)
        .execute()
        .data
        or []
    )
    player_ratings = defaultdict(list)
    for row in db_league_rows:
        player_ratings[_to_int(row.get("player_id"), field="league_ratings.player_id")].append(_to_float(row.get("rating"), 1200.0))
    recomputed_player_ratings = {pid: round(sum(vals) / len(vals), 6) for pid, vals in player_ratings.items() if vals}

    match_ids = [r["id"] for r in updated_match_snapshots]
    db_snapshots: Dict[int, Dict[str, Any]] = {}
    for chunk in _batch([{"id": m_id} for m_id in match_ids], 500):
        ids = [r["id"] for r in chunk]
        rows = supabase.table("matches").select("id," + ",".join(SNAPSHOT_FIELDS)).eq("club_id", club_id).in_("id", ids).execute().data or []
        for row in rows:
            db_snapshots[_to_int(row.get("id"), field="matches.id")] = row

    for expected in updated_match_snapshots:
        actual = db_snapshots.get(expected["id"])
        if not actual:
            raise AssertionError(f"Missing snapshot row for match_id={expected['id']}")
        for field in SNAPSHOT_FIELDS:
            ev = round(float(expected[field]), 6)
            av = round(float(actual.get(field) or 0.0), 6)
            if ev != av:
                raise AssertionError(
                    f"Snapshot mismatch match_id={expected['id']} field={field}: expected={ev} actual={av}"
                )

    runtime = time.perf_counter() - start
    return {
        "matches_processed": int(len(matches)),
        "league_rows_rebuilt": int(len(new_league_rows)),
        "players_updated": int(len(recomputed_player_ratings)),
        "runtime_seconds": float(round(runtime, 6)),
    }
