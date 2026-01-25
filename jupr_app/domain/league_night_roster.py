from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


class RosterChangeError(ValueError):
    pass


@dataclass(frozen=True)
class RosterChangeResult:
    roster_df: pd.DataFrame
    court_sizes: list[int]
    bench_ids: list[int]
    note: str | None = None


def suggest_court_sizes(total_players: int) -> dict:
    """
    Courts may be size 4 or 5 only.
    Prefer exact fit (bench=0), then fewer 5s, then fewer courts.
    """
    n = int(total_players or 0)
    if n <= 0:
        return {"ok": False, "sizes": [], "bench": 0, "note": "No players."}

    if n in (20, 40) and n % 4 == 0:
        sizes = [4] * (n // 4)
        return {"ok": True, "sizes": sizes, "bench": 0, "note": f"Preferred all-4s for total {n}."}

    candidates = []
    min_courts = (n + 5 - 1) // 5
    max_courts = max(1, n // 4)

    for courts in range(max(1, min_courts - 2), max_courts + 3):
        for fives in range(0, courts + 1):
            fours = courts - fives
            capacity = 4 * fours + 5 * fives
            if capacity > n:
                continue
            bench = n - capacity
            score = (bench, fives, courts)
            candidates.append((score, fours, fives, bench))

    if not candidates:
        return {"ok": False, "sizes": [], "bench": n, "note": "No feasible setup with 4/5 courts."}

    candidates.sort(key=lambda x: x[0])
    _, fours, fives, bench = candidates[0]
    sizes = ([4] * int(fours)) + ([5] * int(fives))

    if bench == 0:
        note = f"Exact fit: {fours} court(s) of 4 and {fives} court(s) of 5."
    else:
        note = f"Closest fit: {fours} court(s) of 4 and {fives} court(s) of 5, with {bench} bench."

    return {"ok": True, "sizes": sizes, "bench": int(bench), "note": note}


def roster_change_availability(
    ladder_state: str,
    current_round: int,
    total_rounds: int,
    is_admin: bool = True,
) -> tuple[bool, str]:
    if not is_admin:
        return False, "Admin login required."
    if str(ladder_state or "").strip().upper() != "CONFIRM_MOVEMENT":
        return False, "Roster changes are available between rounds only."
    if int(current_round) >= int(total_rounds):
        return False, "Roster changes are unavailable because the league night is complete."
    return True, ""


def _ordered_roster(roster_df: pd.DataFrame) -> pd.DataFrame:
    df = roster_df.copy()
    if df.empty:
        return df

    order_cols = []
    if "court" in df.columns:
        order_cols.append("court")
    if "slot" in df.columns:
        order_cols.append("slot")

    if order_cols:
        return df.sort_values(order_cols).reset_index(drop=True)

    if "rating" in df.columns:
        return df.sort_values("rating", ascending=False).reset_index(drop=True)

    return df.reset_index(drop=True)


def _apply_court_sizes(roster_df: pd.DataFrame, court_sizes: Iterable[int]) -> pd.DataFrame:
    df = _ordered_roster(roster_df)
    sizes = [int(x) for x in court_sizes]
    total = int(sum(sizes))
    if len(df) != total:
        raise RosterChangeError("Court sizes must match roster size.")

    assignments = []
    idx = 0
    for c_idx, size in enumerate(sizes, start=1):
        for slot in range(1, int(size) + 1):
            row = df.iloc[idx].copy()
            row["court"] = int(c_idx)
            row["slot"] = int(slot)
            assignments.append(row)
            idx += 1

    return pd.DataFrame(assignments)


def _select_bench_ids(
    roster_df: pd.DataFrame,
    bench_count: int,
    prefer_keep_ids: set[int] | None = None,
) -> list[int]:
    if bench_count <= 0:
        return []

    prefer_keep_ids = prefer_keep_ids or set()
    df = roster_df.copy()
    df["rating"] = pd.to_numeric(df.get("rating", 1200.0), errors="coerce").fillna(1200.0)
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").fillna(-1).astype(int)

    sorted_df = df.sort_values("rating", ascending=True).reset_index(drop=True)
    bench_ids: list[int] = []

    for _, row in sorted_df.iterrows():
        pid = int(row["player_id"])
        if pid in prefer_keep_ids:
            continue
        bench_ids.append(pid)
        if len(bench_ids) >= bench_count:
            break

    if len(bench_ids) < bench_count:
        for _, row in sorted_df.iterrows():
            pid = int(row["player_id"])
            if pid in bench_ids:
                continue
            bench_ids.append(pid)
            if len(bench_ids) >= bench_count:
                break

    return bench_ids


def rebalance_roster(
    roster_df: pd.DataFrame,
    court_sizes: list[int] | None = None,
    prefer_keep_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, list[int], list[int], str | None]:
    df = roster_df.copy()
    total = len(df)
    sizes = [int(x) for x in (court_sizes or []) if int(x) > 0]

    note = None
    bench_ids: list[int] = []

    if not sizes or sum(sizes) != total:
        suggestion = suggest_court_sizes(total)
        sizes = [int(x) for x in suggestion.get("sizes", [])]
        bench_count = int(suggestion.get("bench", 0) or 0)
        note = suggestion.get("note")
        if bench_count > 0:
            bench_ids = _select_bench_ids(df, bench_count, prefer_keep_ids=prefer_keep_ids)
            df = df[~df["player_id"].astype(int).isin(bench_ids)].copy()
            total = len(df)
            suggestion = suggest_court_sizes(total)
            sizes = [int(x) for x in suggestion.get("sizes", [])]
            note = suggestion.get("note")

    if sum(sizes) != len(df):
        raise RosterChangeError("Unable to rebalance courts with current roster size.")

    df = _apply_court_sizes(df, sizes)
    return df, sizes, bench_ids, note


def apply_roster_change(
    roster_df: pd.DataFrame,
    change_type: str,
    new_player: dict,
    replaced_player_id: int | None = None,
    court_sizes: list[int] | None = None,
    roster_locked: bool = False,
) -> RosterChangeResult:
    if roster_locked:
        raise RosterChangeError("Round in progress; roster changes are locked.")

    if roster_df is None or roster_df.empty:
        raise RosterChangeError("Roster is empty.")

    df = roster_df.copy()
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").fillna(-1).astype(int)

    change_type = str(change_type or "").strip().lower()
    if change_type not in {"substitute", "add"}:
        raise RosterChangeError("Invalid roster change type.")

    new_pid = int(new_player.get("id", -1))
    if new_pid <= 0:
        raise RosterChangeError("New player is invalid.")

    if change_type == "substitute":
        if replaced_player_id is None:
            raise RosterChangeError("Replacement player is required.")
        replaced_player_id = int(replaced_player_id)
        if replaced_player_id == new_pid:
            raise RosterChangeError("Cannot substitute with the same player.")
        if new_pid in df["player_id"].tolist():
            raise RosterChangeError("Player already active.")
        if replaced_player_id not in df["player_id"].tolist():
            raise RosterChangeError("Replacement player is not active.")

        df.loc[df["player_id"] == replaced_player_id, "player_id"] = new_pid
        df.loc[df["player_id"] == new_pid, "name"] = str(new_player.get("name", ""))
        df.loc[df["player_id"] == new_pid, "rating"] = float(new_player.get("rating", 1200.0))

        new_df, new_sizes, bench_ids, note = rebalance_roster(df, court_sizes=court_sizes)
        return RosterChangeResult(roster_df=new_df, court_sizes=new_sizes, bench_ids=bench_ids, note=note)

    if new_pid in df["player_id"].tolist():
        raise RosterChangeError("Player already active.")

    new_row = {
        "player_id": new_pid,
        "name": str(new_player.get("name", "")),
        "rating": float(new_player.get("rating", 1200.0)),
        "court": int(new_player.get("court", 0) or 0),
        "slot": int(new_player.get("slot", 0) or 0),
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    prefer_keep_ids = {new_pid}
    new_df, new_sizes, bench_ids, note = rebalance_roster(df, court_sizes=court_sizes, prefer_keep_ids=prefer_keep_ids)
    return RosterChangeResult(roster_df=new_df, court_sizes=new_sizes, bench_ids=bench_ids, note=note)
