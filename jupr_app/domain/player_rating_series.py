from __future__ import annotations

from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import pandas as pd


def _safe_int(x, default=None):
    try:
        if x is None or str(x).strip() == "":
            return default
        return int(x)
    except Exception:
        return default


def _safe_float(x, default=None):
    try:
        if x is None or str(x).strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def _score_for_player(r: dict, player_id: int) -> str:
    t1p1 = _safe_int(r.get("t1_p1"))
    t1p2 = _safe_int(r.get("t1_p2"))
    s1 = _safe_int(r.get("score_t1"), 0) or 0
    s2 = _safe_int(r.get("score_t2"), 0) or 0
    if t1p1 == player_id or t1p2 == player_id:
        return f"{s1}-{s2}"
    return f"{s2}-{s1}"


def _result_for_player(r: dict, player_id: int) -> str:
    t1p1 = _safe_int(r.get("t1_p1"))
    t1p2 = _safe_int(r.get("t1_p2"))
    s1 = _safe_int(r.get("score_t1"), 0) or 0
    s2 = _safe_int(r.get("score_t2"), 0) or 0

    if s1 == s2:
        return "DRAW"
    on_t1 = player_id in {t1p1, t1p2}
    winner = "WIN" if s1 > s2 else "LOSS"
    if not on_t1:
        winner = "WIN" if s2 > s1 else "LOSS"
    return winner


def _get_overall_snap(r: dict, player_id: int) -> tuple[float | None, float | None]:
    t1p1 = _safe_int(r.get("t1_p1"))
    t1p2 = _safe_int(r.get("t1_p2"))
    t2p1 = _safe_int(r.get("t2_p1"))
    t2p2 = _safe_int(r.get("t2_p2"))

    if t1p1 == player_id:
        return _safe_float(r.get("t1_p1_r")), _safe_float(r.get("t1_p1_r_end"))
    if t1p2 == player_id:
        return _safe_float(r.get("t1_p2_r")), _safe_float(r.get("t1_p2_r_end"))
    if t2p1 == player_id:
        return _safe_float(r.get("t2_p1_r")), _safe_float(r.get("t2_p1_r_end"))
    if t2p2 == player_id:
        return _safe_float(r.get("t2_p2_r")), _safe_float(r.get("t2_p2_r_end"))
    return None, None


def _signed_delta_from_elo_delta(r: dict, player_id: int) -> float | None:
    raw = _safe_float(r.get("elo_delta"), None)
    if raw is None:
        return None

    s1 = _safe_int(r.get("score_t1"), 0) or 0
    s2 = _safe_int(r.get("score_t2"), 0) or 0
    if s1 == s2:
        return 0.0

    t1 = {_safe_int(r.get("t1_p1")), _safe_int(r.get("t1_p2"))}
    t2 = {_safe_int(r.get("t2_p1")), _safe_int(r.get("t2_p2"))}
    on_t1 = player_id in t1
    on_t2 = player_id in t2
    if not on_t1 and not on_t2:
        return None

    winner_team = 1 if s1 > s2 else 2
    my_team = 1 if on_t1 else 2
    return abs(float(raw)) if winner_team == my_team else -abs(float(raw))


def _load_player_matches(supabase, club_id: str, player_id: int, limit: int = 600) -> pd.DataFrame:
    base_select = (
        "id,date,league,match_type,score_t1,score_t2,"
        "t1_p1,t1_p2,t2_p1,t2_p2,"
        "elo_delta"
    )
    snap_select = (
        base_select
        + ",t1_p1_r,t1_p1_r_end,t1_p2_r,t1_p2_r_end,"
        "t2_p1_r,t2_p1_r_end,t2_p2_r,t2_p2_r_end"
    )

    def _run(cols: str) -> pd.DataFrame:
        resp = (
            supabase.table("matches")
            .select(cols)
            .eq("club_id", str(club_id))
            .or_(f"t1_p1.eq.{player_id},t1_p2.eq.{player_id},t2_p1.eq.{player_id},t2_p2.eq.{player_id}")
            .order("date", desc=True)
            .order("id", desc=True)
            .limit(int(limit))
            .execute()
        )
        return pd.DataFrame(resp.data or [])

    try:
        return _run(snap_select)
    except Exception:
        return _run(base_select)


def build_player_overall_rating_series(supabase, club_id: str, player_id: int, limit: int = 600) -> pd.DataFrame:
    matches = _load_player_matches(supabase, club_id, int(player_id), limit=limit)
    if matches.empty:
        return pd.DataFrame(columns=["id", "Date", "League", "Score", "Result", "Overall Δ", "Overall After"])

    matches = matches.copy()
    matches["date_dt"] = pd.to_datetime(matches.get("date", None), errors="coerce", utc=True)
    matches = matches.dropna(subset=["date_dt"]).copy()
    matches["league"] = matches.get("league", "").fillna("").astype(str).str.strip()
    matches["match_type"] = matches.get("match_type", "").fillna("").astype(str).str.strip()

    processed = []
    for _, r0 in matches.iterrows():
        r = dict(r0)
        start_elo, end_elo = _get_overall_snap(r, int(player_id))
        after_jupr = None
        delta_jupr = None

        if start_elo is not None and end_elo is not None:
            try:
                delta_jupr = (float(end_elo) - float(start_elo)) / 400.0
                after_jupr = float(end_elo) / 400.0
            except Exception:
                pass
        else:
            d_elo = _signed_delta_from_elo_delta(r, int(player_id))
            if d_elo is not None:
                delta_jupr = float(d_elo) / 400.0

        processed.append(
            {
                "id": _safe_int(r.get("id")),
                "Date": r.get("date_dt"),
                "League": str(r.get("league", "") or "").strip(),
                "match_type": str(r.get("match_type", "") or "").strip(),
                "Score": _score_for_player(r, int(player_id)),
                "Result": _result_for_player(r, int(player_id)),
                "Overall Δ": delta_jupr,
                "Overall After": after_jupr,
                "t1_p1": _safe_int(r.get("t1_p1")),
                "t1_p2": _safe_int(r.get("t1_p2")),
                "t2_p1": _safe_int(r.get("t2_p1")),
                "t2_p2": _safe_int(r.get("t2_p2")),
                "score_t1": _safe_int(r.get("score_t1"), 0) or 0,
                "score_t2": _safe_int(r.get("score_t2"), 0) or 0,
            }
        )

    df = pd.DataFrame(processed)
    if df.empty:
        return df

    df = df.sort_values(["Date", "id"], ascending=[True, True]).reset_index(drop=True)
    df["Overall Δ"] = pd.to_numeric(df["Overall Δ"], errors="coerce")
    df["Overall After"] = pd.to_numeric(df["Overall After"], errors="coerce")

    if df["Overall After"].notna().any():
        for i in range(len(df)):
            if pd.isna(df.loc[i, "Overall After"]):
                if i > 0 and pd.notna(df.loc[i - 1, "Overall After"]) and pd.notna(df.loc[i, "Overall Δ"]):
                    df.loc[i, "Overall After"] = float(df.loc[i - 1, "Overall After"]) + float(df.loc[i, "Overall Δ"])
        for i in range(len(df) - 2, -1, -1):
            if pd.isna(df.loc[i, "Overall After"]):
                if pd.notna(df.loc[i + 1, "Overall After"]) and pd.notna(df.loc[i + 1, "Overall Δ"]):
                    df.loc[i, "Overall After"] = float(df.loc[i + 1, "Overall After"]) - float(df.loc[i + 1, "Overall Δ"])

    return df


def _coerce_window_bounds(start_date, end_date, tz_name: str = "America/Mazatlan") -> tuple[pd.Timestamp, pd.Timestamp]:
    tz = ZoneInfo(tz_name)

    if isinstance(start_date, datetime):
        start_d = start_date.date()
    elif isinstance(start_date, date):
        start_d = start_date
    else:
        start_d = pd.to_datetime(start_date).date()

    if isinstance(end_date, datetime):
        end_d = end_date.date()
    elif isinstance(end_date, date):
        end_d = end_date
    else:
        end_d = pd.to_datetime(end_date).date()

    start_utc = datetime.combine(start_d, time.min).replace(tzinfo=tz).astimezone(timezone.utc)
    end_utc = datetime.combine(end_d, time.max).replace(tzinfo=tz).astimezone(timezone.utc)
    return pd.Timestamp(start_utc), pd.Timestamp(end_utc)


def filter_rating_series_for_window(df: pd.DataFrame, start_date, end_date, tz_name: str = "America/Mazatlan") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=list(df.columns) if isinstance(df, pd.DataFrame) else [])

    start_ts, end_ts = _coerce_window_bounds(start_date, end_date, tz_name)
    out = df.copy()
    out["Date"] = pd.to_datetime(out.get("Date"), utc=True, errors="coerce")
    out = out.dropna(subset=["Date"])
    out = out[(out["Date"] >= start_ts) & (out["Date"] <= end_ts)].copy()
    return out.sort_values(["Date", "id"], ascending=[True, True]).reset_index(drop=True)
