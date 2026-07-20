from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


def validate_courts(
    roster_df: pd.DataFrame,
    min_players_per_court: int = 4,
    target_sizes: Optional[List[int]] = None,
) -> dict:
    problems: List[str] = []
    warnings: List[str] = []

    if roster_df is None or roster_df.empty or "court" not in roster_df.columns:
        return {
            "can_start": False,
            "problems": ["Roster is empty or missing 'court' column."],
            "warnings": [],
            "court_counts": {},
        }

    df = roster_df.copy()
    df["court"] = pd.to_numeric(df["court"], errors="coerce").fillna(-1).astype(int)
    df = df[df["court"] > 0].copy()

    court_counts = df.groupby("court").size().to_dict()

    for c, n in sorted(court_counts.items()):
        if int(n) < int(min_players_per_court):
            problems.append(f"Court {c} has {n} players (min {min_players_per_court}).")

    if isinstance(target_sizes, list) and len(target_sizes) > 0:
        for c, n in sorted(court_counts.items()):
            idx = int(c) - 1
            if 0 <= idx < len(target_sizes):
                tgt = int(target_sizes[idx])
                if int(n) != tgt:
                    warnings.append(f"Court {c} has {n} players (target {tgt}).")
    else:
        for c, n in sorted(court_counts.items()):
            if int(n) != 4:
                warnings.append(f"Court {c} has {n} players (target 4).")

    return {
        "can_start": len(problems) == 0,
        "problems": problems,
        "warnings": warnings,
        "court_counts": {int(k): int(v) for k, v in court_counts.items()},
    }


def compute_round_stats(
    valid_matches: List[dict],
    roster_pids: List[int],
) -> Dict[int, Dict[str, int]]:
    stats: Dict[int, Dict[str, int]] = {int(pid): {"w": 0, "diff": 0, "pts": 0} for pid in roster_pids}

    for r in valid_matches:
        s1 = int(r.get("s1", 0) or 0)
        s2 = int(r.get("s2", 0) or 0)
        if (s1 + s2) <= 0:
            continue

        t1 = [int(r["t1_p1"]), int(r["t1_p2"])]
        t2 = [int(r["t2_p1"]), int(r["t2_p2"])]

        win_team1 = s1 > s2
        diff = abs(s1 - s2)

        for pid in t1:
            pid = int(pid)
            if pid not in stats:
                stats[pid] = {"w": 0, "diff": 0, "pts": 0}
            stats[pid]["pts"] += int(s1)
            stats[pid]["diff"] += int(diff if win_team1 else -diff)
            if win_team1:
                stats[pid]["w"] += 1

        for pid in t2:
            pid = int(pid)
            if pid not in stats:
                stats[pid] = {"w": 0, "diff": 0, "pts": 0}
            stats[pid]["pts"] += int(s2)
            stats[pid]["diff"] += int(-diff if win_team1 else diff)
            if not win_team1:
                stats[pid]["w"] += 1

    return stats


def build_movement_preview(
    roster_df: pd.DataFrame,
    round_stats: Dict[int, Dict[str, int]],
    max_court: int,
) -> pd.DataFrame:
    if roster_df is None or roster_df.empty:
        return pd.DataFrame()

    df = roster_df.copy()

    if "player_id" not in df.columns:
        raise KeyError("roster_df must include 'player_id' column.")
    if "court" not in df.columns:
        raise KeyError("roster_df must include 'court' column.")

    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").fillna(-1).astype(int)
    df["court"] = pd.to_numeric(df["court"], errors="coerce").fillna(-1).astype(int)

    df["Round Wins"] = df["player_id"].map(lambda pid: int(round_stats.get(int(pid), {}).get("w", 0)))
    df["Round Diff"] = df["player_id"].map(lambda pid: int(round_stats.get(int(pid), {}).get("diff", 0)))
    df["Round Pts"] = df["player_id"].map(lambda pid: int(round_stats.get(int(pid), {}).get("pts", 0)))

    if "slot" not in df.columns:
        df["slot"] = 0
    df["slot"] = pd.to_numeric(df["slot"], errors="coerce").fillna(0).astype(int)
    # Stable final tie-breakers are part of the League Live contract. The old
    # browser implementation used the current court slot; Python must make the
    # same choice deterministically so retries and different JS engines cannot
    # select different movers.
    df = df.sort_values(
        by=["court", "Round Wins", "Round Diff", "Round Pts", "slot", "player_id"],
        ascending=[True, False, False, False, True, True],
        kind="mergesort",
    ).copy()

    df["Proposed Court"] = df["court"].astype(int)

    for c_num in sorted(df["court"].unique().tolist()):
        court_group = df[df["court"] == int(c_num)]
        if court_group.empty:
            continue

        top_idx = court_group.index[0]
        bot_idx = court_group.index[-1]

        if int(c_num) > 1:
            df.loc[top_idx, "Proposed Court"] = int(c_num) - 1
        if int(c_num) < int(max_court):
            df.loc[bot_idx, "Proposed Court"] = int(c_num) + 1

    return df
