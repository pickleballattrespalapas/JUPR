# jupr_app/domain/challenge_ladder.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd


# -------------------------
# Challenge Ladder Status Sets
# -------------------------
LADDER_OPEN_STATUSES = {
    "PENDING_ACCEPTANCE",
    "ACCEPTED_SCHEDULING",
    "IN_PROGRESS",
    "AWAITING_VERIFICATION",
    "OVERDUE_PLAY",
}
LADDER_FINAL_STATUSES = {"COMPLETED", "FORFEITED"}


# -------------------------
# Time helpers (pure)
# -------------------------
def dt_utc_now() -> datetime:
    return datetime.now(timezone.utc)


def month_key_utc(dt: datetime) -> str:
    d = dt.astimezone(timezone.utc)
    return f"{d.year:04d}-{d.month:02d}"


def ladder_parse_dt(x) -> Optional[pd.Timestamp]:
    if x is None or str(x).strip() == "":
        return None
    try:
        return pd.to_datetime(x, utc=True)
    except Exception:
        return None


# -------------------------
# Name helper (pure)
# -------------------------
def ladder_nm(pid: int, id_to_name: Dict[int, str]) -> str:
    try:
        return str(id_to_name.get(int(pid), f"#{int(pid)}"))
    except Exception:
        return "—"


# -------------------------
# Tier definitions (Option A, range-labeled)
# -------------------------
TIER_ORDER = ["DEV", "INT", "ADV", "PREM"]

TIER_DEFS = {
    "DEV":  {"label": "Developing",    "min": None,  "max": 3.25, "range": "< 3.25"},
    "INT":  {"label": "Intermediate",  "min": 3.25,  "max": 3.75, "range": "3.25–3.75"},
    "ADV":  {"label": "Advanced",      "min": 3.75,  "max": 4.25, "range": "3.75–4.25"},
    "PREM": {"label": "Premier",       "min": 4.25,  "max": None, "range": "4.25+"},
}


def tier_for_jupr(jupr: float) -> str:
    try:
        x = float(jupr)
    except Exception:
        return "INT"
    if x < 3.25:
        return "DEV"
    if x < 3.75:
        return "INT"
    if x < 4.25:
        return "ADV"
    return "PREM"


def tier_title(tier_id: str) -> str:
    t = TIER_DEFS.get(str(tier_id), {"label": str(tier_id), "range": ""})
    rng = t.get("range", "")
    return f"{t.get('label','Tier')} — {rng}".strip(" —")


def tier_idx(tier_id: str) -> int:
    tid = str(tier_id)
    return TIER_ORDER.index(tid) if tid in TIER_ORDER else 999


def is_promotion(from_tier: str, to_tier: str) -> bool:
    return tier_idx(to_tier) > tier_idx(from_tier)


def is_demotion(from_tier: str, to_tier: str) -> bool:
    return tier_idx(to_tier) < tier_idx(from_tier)


# -------------------------
# Status Map (pure rules)
# -------------------------
def ladder_compute_status_map(
    df_roster: pd.DataFrame,
    df_flags: pd.DataFrame,
    df_ch: pd.DataFrame,
    df_pass: pd.DataFrame,
    settings: Dict[str, Any],
    id_to_name: Dict[int, str],
    now_utc: Optional[datetime] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Computes status for each ACTIVE roster player.

    Returns:
      status_map[pid] = {
        "status": str,
        "until": pd.Timestamp|None,
        "detail": str,
        "challenge_id": int|None (only for Locked)
      }
    """
    now = (now_utc or dt_utc_now())

    accept_h = int(settings.get("accept_window_hours", 48) or 48)
    play_d = int(settings.get("play_window_days", 7) or 7)
    cooldown_h = int(settings.get("cooldown_hours", 72) or 72)
    protected_h = int(settings.get("protected_hours", 72) or 72)
    passhold_h = int(settings.get("pass_hold_hours", 72) or 72)

    # Normalize datetime columns
    if df_flags is not None and not df_flags.empty:
        df_flags = df_flags.copy()
        df_flags["vacation_until_dt"] = df_flags["vacation_until"].apply(ladder_parse_dt)
    else:
        df_flags = pd.DataFrame(columns=["player_id", "vacation_until_dt", "reinstate_required", "reinstate_notes"])

    if df_ch is not None and not df_ch.empty:
        df_ch = df_ch.copy()
        df_ch["created_at_dt"] = df_ch["created_at"].apply(ladder_parse_dt)
        df_ch["accept_by_dt"] = df_ch["accept_by"].apply(ladder_parse_dt)
        df_ch["accepted_at_dt"] = df_ch["accepted_at"].apply(ladder_parse_dt)
        df_ch["play_by_dt"] = df_ch["play_by"].apply(ladder_parse_dt)
        df_ch["completed_at_dt"] = df_ch["completed_at"].apply(ladder_parse_dt)
    else:
        df_ch = pd.DataFrame()

    if df_pass is not None and not df_pass.empty:
        df_pass = df_pass.copy()
        df_pass["used_at_dt"] = df_pass["used_at"].apply(ladder_parse_dt)
    else:
        df_pass = pd.DataFrame()

    # Map: player -> flags
    flags_map: Dict[int, Dict[str, Any]] = {}
    for _, r in df_flags.iterrows():
        pid = int(r.get("player_id"))
        flags_map[pid] = {
            "vacation_until": r.get("vacation_until_dt"),
            "reinstate_required": bool(r.get("reinstate_required", False)),
            "reinstate_notes": str(r.get("reinstate_notes", "") or ""),
        }

    # Map: player -> open challenge row (for Locked context)
    open_map: Dict[int, Dict[str, Any]] = {}
    if not df_ch.empty and "status" in df_ch.columns:
        open_df = df_ch[df_ch["status"].isin(list(LADDER_OPEN_STATUSES))].copy()
        for _, r in open_df.iterrows():
            rr = r.to_dict()
            for pid in (rr.get("challenger_id"), rr.get("defender_id")):
                if pid is None:
                    continue
                pid = int(pid)
                if pid not in open_map:
                    open_map[pid] = rr

    # Map: player -> last finalized challenge row (for Protected/Cooldown)
    final_map: Dict[int, Dict[str, Any]] = {}
    if not df_ch.empty and "status" in df_ch.columns:
        fin_df = df_ch[df_ch["status"].isin(list(LADDER_FINAL_STATUSES))].copy()
        if not fin_df.empty:
            fin_df = fin_df.sort_values("completed_at_dt", ascending=False, na_position="last")
            for _, r in fin_df.iterrows():
                rr = r.to_dict()
                for pid in (rr.get("challenger_id"), rr.get("defender_id")):
                    if pid is None:
                        continue
                    pid = int(pid)
                    if pid not in final_map:
                        final_map[pid] = rr

    # Map: player -> last pass used
    pass_map: Dict[int, Dict[str, Any]] = {}
    if not df_pass.empty:
        for _, r in df_pass.sort_values("used_at_dt", ascending=False).iterrows():
            pid = r.get("player_id")
            if pid is None:
                continue
            pid = int(pid)
            if pid not in pass_map:
                pass_map[pid] = r.to_dict()

    # Compute status
    status_map: Dict[int, Dict[str, Any]] = {}
    if df_roster is None or df_roster.empty:
        return status_map

    for _, rr in df_roster.iterrows():
        if not bool(rr.get("is_active", True)):
            continue

        pid = int(rr["player_id"])
        f = flags_map.get(pid, {})
        vacation_until = f.get("vacation_until")
        reinstate_required = bool(f.get("reinstate_required", False))

        # 1) Reinstate Required
        if reinstate_required:
            status_map[pid] = {"status": "Reinstate Required", "until": None, "detail": f.get("reinstate_notes", "")}
            continue

        # 2) Vacation
        if vacation_until is not None and vacation_until.to_pydatetime() >= now:
            status_map[pid] = {"status": "Vacation", "until": vacation_until, "detail": "Admin-only hold"}
            continue

        # 3) Pass Hold
        p_last = pass_map.get(pid)
        if p_last:
            used_at = p_last.get("used_at_dt")
            if used_at is not None:
                until = used_at.to_pydatetime() + timedelta(hours=passhold_h)
                if until >= now:
                    status_map[pid] = {"status": "Pass Hold", "until": pd.to_datetime(until, utc=True), "detail": "72h after Pass Used"}
                    continue

        # 4) Locked (any open challenge)
        oc = open_map.get(pid)
        if oc:
            opp = None
            ch_id = oc.get("id")
            ch_status = str(oc.get("status", "") or "")
            if int(oc.get("challenger_id") or -1) == pid:
                opp = int(oc.get("defender_id"))
                role = "Challenger"
            else:
                opp = int(oc.get("challenger_id"))
                role = "Defender"

            accept_by = oc.get("accept_by_dt")
            play_by = oc.get("play_by_dt")

            if ch_status == "PENDING_ACCEPTANCE" and accept_by is not None:
                detail = f"{role} vs {ladder_nm(opp, id_to_name)} • Accept by {accept_by.strftime('%Y-%m-%d %H:%M UTC')}"
            elif play_by is not None:
                detail = f"{role} vs {ladder_nm(opp, id_to_name)} • Play by {play_by.strftime('%Y-%m-%d %H:%M UTC')}"
            else:
                detail = f"{role} vs {ladder_nm(opp, id_to_name)}"

            status_map[pid] = {"status": "Locked", "until": None, "detail": detail, "challenge_id": ch_id}
            continue

        # 5/6) Protected or Cooldown based on last finalized
        last_fin = final_map.get(pid)
        if last_fin:
            completed_at = last_fin.get("completed_at_dt")
            winner_id = last_fin.get("winner_id")
            if completed_at is not None and winner_id is not None:
                completed_dt = completed_at.to_pydatetime()
                if int(winner_id) == pid:
                    until = completed_dt + timedelta(hours=protected_h)
                    if until >= now:
                        status_map[pid] = {"status": "Protected", "until": pd.to_datetime(until, utc=True), "detail": "72h after win"}
                        continue
                else:
                    until = completed_dt + timedelta(hours=cooldown_h)
                    if until >= now:
                        status_map[pid] = {"status": "Cooldown", "until": pd.to_datetime(until, utc=True), "detail": "72h after loss"}
                        continue

        # 7) Ready
        status_map[pid] = {"status": "Ready to Defend", "until": None, "detail": ""}

    return status_map


def ladder_bucket_challenge(row: Dict[str, Any], now_utc: Optional[datetime] = None) -> str:
    now = (now_utc or dt_utc_now())
    status = str(row.get("status", "") or "")
    accept_by = ladder_parse_dt(row.get("accept_by"))
    play_by = ladder_parse_dt(row.get("play_by"))

    if status == "PENDING_ACCEPTANCE":
        if accept_by is not None and accept_by.to_pydatetime() < now:
            return "Acceptance Overdue"
        return "Pending Acceptance"

    if status in ("ACCEPTED_SCHEDULING", "IN_PROGRESS", "AWAITING_VERIFICATION", "OVERDUE_PLAY"):
        if play_by is not None and play_by.to_pydatetime() < now:
            return "Play Overdue"
        return "Accepted / In Window"

    if status in ("COMPLETED", "FORFEITED"):
        return "Recently Completed"

    if status in ("CANCELED", "EXPIRED_ACCEPTANCE"):
        return "Closed (No Result)"

    return "Other"


def ladder_compute_challenge_outcome(df_match_rows: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Returns dict:
      winner_side: 'DEF' or 'CHAL'
      match_wins_def/chal
      games_def/chal
      points_def/chal
      point_diff_def (def - chal)

    Tie-break: match wins -> games -> point diff -> defender holds
    """
    def parse_int(x):
        try:
            if x is None:
                return None
            return int(x)
        except Exception:
            return None

    def match_summary(row):
        games = []
        for i in (1, 2, 3):
            a = parse_int(row.get(f"g{i}_def"))
            b = parse_int(row.get(f"g{i}_chal"))
            if a is None or b is None:
                continue
            games.append((a, b))

        def_g = sum(1 for a, b in games if a > b)
        chal_g = sum(1 for a, b in games if b > a)

        def_pts = sum(a for a, b in games)
        chal_pts = sum(b for a, b in games)

        if def_g > chal_g:
            win = "DEF"
        elif chal_g > def_g:
            win = "CHAL"
        else:
            win = "TIE"

        return {"win": win, "def_g": def_g, "chal_g": chal_g, "def_pts": def_pts, "chal_pts": chal_pts}

    if df_match_rows is None or df_match_rows.empty:
        return None

    ms = []
    for _, r in df_match_rows.sort_values("match_no").iterrows():
        ms.append(match_summary(r.to_dict()))

    if len(ms) < 2:
        return None

    match_wins_def = sum(1 for x in ms if x["win"] == "DEF")
    match_wins_chal = sum(1 for x in ms if x["win"] == "CHAL")

    games_def = sum(x["def_g"] for x in ms)
    games_chal = sum(x["chal_g"] for x in ms)

    pts_def = sum(x["def_pts"] for x in ms)
    pts_chal = sum(x["chal_pts"] for x in ms)
    pdiff = pts_def - pts_chal

    if match_wins_def > match_wins_chal:
        winner_side = "DEF"
    elif match_wins_chal > match_wins_def:
        winner_side = "CHAL"
    else:
        if games_def > games_chal:
            winner_side = "DEF"
        elif games_chal > games_def:
            winner_side = "CHAL"
        else:
            if pdiff > 0:
                winner_side = "DEF"
            elif pdiff < 0:
                winner_side = "CHAL"
            else:
                winner_side = "DEF"

    return {
        "winner_side": winner_side,
        "match_wins_def": match_wins_def,
        "match_wins_chal": match_wins_chal,
        "games_def": games_def,
        "games_chal": games_chal,
        "points_def": pts_def,
        "points_chal": pts_chal,
        "point_diff_def": pdiff,
    }
