# jupr_app/domain/tier_movement.py
from __future__ import annotations

from datetime import datetime
import pandas as pd

from jupr_app.domain.challenge_ladder import tier_for_jupr


def match_end_elo_for_pid(m: dict, pid: int) -> float | None:
    try:
        pid = int(pid)
    except Exception:
        return None

    if int(m.get("t1_p1") or -1) == pid:
        return m.get("t1_p1_r_end")
    if int(m.get("t1_p2") or -1) == pid:
        return m.get("t1_p2_r_end")
    if int(m.get("t2_p1") or -1) == pid:
        return m.get("t2_p1_r_end")
    if int(m.get("t2_p2") or -1) == pid:
        return m.get("t2_p2_r_end")
    return None


def compute_out_of_tier_streak(
    pid: int,
    joined_at_utc: datetime | None,
    current_tier_id: str,
    df_matches: pd.DataFrame,
) -> dict:
    """
    Returns:
      {
        "dest_tier": str|None,
        "count": int,
        "latest_match_at": datetime|None
      }

    Rule: streak counts consecutive matches where post-match rating tier == same dest tier != current tier.
    If any match returns to current tier, streak resets (break).
    """
    if df_matches is None or df_matches.empty:
        return {"dest_tier": None, "count": 0, "latest_match_at": None}

    pid = int(pid)
    cur = str(current_tier_id)

    m = df_matches.copy()

    # Ensure date parse
    if "date" in m.columns:
        m["date_dt"] = pd.to_datetime(m["date"], utc=True, errors="coerce")
    else:
        m["date_dt"] = pd.NaT

    # Filter to matches containing pid
    mask = (
        (m.get("t1_p1") == pid) |
        (m.get("t1_p2") == pid) |
        (m.get("t2_p1") == pid) |
        (m.get("t2_p2") == pid)
    )
    m = m[mask].copy()
    if m.empty:
        return {"dest_tier": None, "count": 0, "latest_match_at": None}

    # Only matches after joined_at
    if joined_at_utc is not None:
        m = m[m["date_dt"] >= pd.to_datetime(joined_at_utc, utc=True)].copy()
        if m.empty:
            return {"dest_tier": None, "count": 0, "latest_match_at": None}

    # Most recent first
    sort_cols = ["date_dt"]
    if "id" in m.columns:
        sort_cols.append("id")
        m = m.sort_values(sort_cols, ascending=[False, False])
    else:
        m = m.sort_values(sort_cols, ascending=[False])

    dest = None
    count = 0
    latest_dt = None

    for _, row in m.iterrows():
        if latest_dt is None:
            latest_dt = row["date_dt"].to_pydatetime() if pd.notna(row["date_dt"]) else None

        end_elo = match_end_elo_for_pid(row.to_dict(), pid)
        if end_elo is None:
            continue

        end_jupr = float(end_elo) / 400.0
        t = tier_for_jupr(end_jupr)

        if t == cur:
            break

        if dest is None:
            dest = t
            count = 1
        else:
            if t == dest:
                count += 1
            else:
                break

        if count >= 10:
            break

    return {"dest_tier": dest, "count": int(count), "latest_match_at": latest_dt}
