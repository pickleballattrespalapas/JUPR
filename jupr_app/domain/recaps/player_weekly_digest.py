from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from jupr_app.domain.gamification.trophies import get_player_tournament_trophies
from jupr_app.domain.player_rating_series import (
    build_player_overall_rating_series,
    filter_rating_series_for_window,
)
from jupr_app.domain.recaps.weekly_recap import get_date_range_bounds, normalize_date_range


def _safe_float(value) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except Exception:
        return None


def _display_range(start_date: date, end_date: date) -> str:
    return f"{start_date.isoformat()} – {end_date.isoformat()}"


def _player_name(ctx, player_id: int) -> str:
    id_to_name = getattr(ctx, "id_to_name", {}) or {}
    name = str(id_to_name.get(int(player_id), "")).strip()
    if name:
        return name

    supabase = getattr(ctx, "supabase", None)
    club_id = str(getattr(ctx, "club_id", "") or "")
    if not supabase or not club_id:
        return f"Player #{int(player_id)}"

    try:
        resp = (
            supabase.table("players")
            .select("id,name")
            .eq("club_id", club_id)
            .eq("id", int(player_id))
            .limit(1)
            .execute()
        )
        row = (resp.data or [{}])[0]
        loaded = str(row.get("name") or "").strip()
        if loaded:
            return loaded
    except Exception:
        pass
    return f"Player #{int(player_id)}"


def _load_badges_earned(supabase, club_id: str, player_id: int, start_dt_utc: datetime, end_dt_utc: datetime) -> list[dict]:
    if not supabase:
        return []

    try:
        badge_rows = (
            supabase.table("player_badges")
            .select("badge_id,earned_at,context_type,context_id,value_num,value_json")
            .eq("club_id", club_id)
            .eq("player_id", int(player_id))
            .gte("earned_at", start_dt_utc.isoformat())
            .lte("earned_at", end_dt_utc.isoformat())
            .order("earned_at", desc=True)
            .execute()
            .data
            or []
        )
    except Exception:
        return []

    badge_ids = sorted({str(row.get("badge_id") or "").strip() for row in badge_rows if row.get("badge_id")})
    badge_name_map: dict[str, str] = {}
    if badge_ids:
        try:
            badge_defs = (
                supabase.table("badges")
                .select("badge_id,name")
                .in_("badge_id", badge_ids)
                .execute()
                .data
                or []
            )
            badge_name_map = {
                str(row.get("badge_id")): str(row.get("name") or row.get("badge_id") or "Badge")
                for row in badge_defs
            }
        except Exception:
            badge_name_map = {}

    out: list[dict] = []
    for row in badge_rows:
        bid = str(row.get("badge_id") or "").strip()
        if not bid:
            continue
        out.append(
            {
                "badge_id": bid,
                "name": badge_name_map.get(bid, bid),
                "earned_at": row.get("earned_at"),
                "context_type": row.get("context_type"),
                "context_id": row.get("context_id"),
            }
        )
    return out


def _load_trophies_earned(supabase, club_id: str, player_id: int, start_dt_utc: datetime, end_dt_utc: datetime) -> list[dict]:
    trophies = get_player_tournament_trophies(supabase, club_id, int(player_id))
    if not trophies:
        return []

    start_ts = pd.Timestamp(start_dt_utc)
    end_ts = pd.Timestamp(end_dt_utc)
    kept: list[dict] = []
    for item in trophies:
        earned_at = pd.to_datetime(item.get("earned_at"), utc=True, errors="coerce")
        if pd.isna(earned_at):
            continue
        if start_ts <= earned_at <= end_ts:
            kept.append(item)
    return kept


def _build_highlights(summary: dict, badges_earned: list[dict], trophies_earned: list[dict]) -> list[str]:
    lines: list[str] = []
    before_val = _safe_float(summary.get("overall_jupr_before"))
    after_val = _safe_float(summary.get("overall_jupr_after"))
    if before_val is not None and after_val is not None:
        lines.append(f"Overall JUPR moved from {before_val:.3f} to {after_val:.3f}.")

    lines.append(
        f"Finished the week {summary.get('record', '0-0')} across {int(summary.get('matches_played', 0) or 0)} matches."
    )

    if badges_earned:
        lines.append(f"Earned {str(badges_earned[0].get('name') or 'a badge')}.")
    elif trophies_earned:
        t_name = str(trophies_earned[0].get("tournament_name") or "a tournament")
        lines.append(f"Reached the podium in {t_name}.")

    return lines[:3]


def compute_player_weekly_digest(ctx, player_id: int, start_date: date, end_date: date, tz_name: str = "America/Mazatlan") -> dict:
    start_date, end_date = normalize_date_range(start_date, end_date)
    supabase = getattr(ctx, "supabase", None)
    club_id = str(getattr(ctx, "club_id", "") or "")

    overall_series = build_player_overall_rating_series(supabase, club_id, int(player_id), limit=1200)
    window_series = filter_rating_series_for_window(overall_series, start_date, end_date, tz_name=tz_name)

    before = _safe_float(window_series["Overall After"].iloc[0]) if not window_series.empty else None
    after = _safe_float(window_series["Overall After"].iloc[-1]) if not window_series.empty else None
    overall_delta = None
    if before is not None and after is not None:
        overall_delta = after - before

    wins = int((window_series.get("Result") == "WIN").sum()) if not window_series.empty else 0
    losses = int((window_series.get("Result") == "LOSS").sum()) if not window_series.empty else 0
    matches_played = int(len(window_series.index))

    summary = {
        "overall_jupr_before": round(before, 3) if before is not None else None,
        "overall_jupr_after": round(after, 3) if after is not None else None,
        "overall_delta": round(overall_delta, 4) if overall_delta is not None else None,
        "matches_played": matches_played,
        "wins": wins,
        "losses": losses,
        "record": f"{wins}-{losses}",
    }

    points = []
    for _, row in window_series.sort_values(["Date", "id"], ascending=[True, True]).iterrows():
        points.append(
            {
                "id": int(row.get("id")) if pd.notna(row.get("id")) else None,
                "date": pd.to_datetime(row.get("Date"), utc=True, errors="coerce").isoformat()
                if pd.notna(pd.to_datetime(row.get("Date"), utc=True, errors="coerce"))
                else None,
                "overall_after": round(float(row.get("Overall After")), 3)
                if pd.notna(row.get("Overall After"))
                else None,
                "overall_delta": round(float(row.get("Overall Δ")), 4) if pd.notna(row.get("Overall Δ")) else None,
                "league": str(row.get("League") or ""),
                "score": str(row.get("Score") or ""),
                "result": str(row.get("Result") or ""),
            }
        )

    start_dt_utc, end_dt_utc = get_date_range_bounds(start_date, end_date, tz_name)
    badges_earned = _load_badges_earned(supabase, club_id, int(player_id), start_dt_utc, end_dt_utc)
    trophies_earned = _load_trophies_earned(supabase, club_id, int(player_id), start_dt_utc, end_dt_utc)

    player_name = _player_name(ctx, int(player_id))
    display_range = _display_range(start_date, end_date)

    digest = {
        "club_id": club_id,
        "player_id": int(player_id),
        "player_name": player_name,
        "week_start": start_date.isoformat(),
        "week_end": end_date.isoformat(),
        "display_range": display_range,
        "subject_line": f"{player_name} weekly digest · {display_range}",
        "summary": summary,
        "chart": {
            "title": "Overall JUPR Trend",
            "window_label": display_range,
            "series_key": "overall_after",
            "points": points,
        },
        "badges_earned": badges_earned,
        "trophies_earned": trophies_earned,
        "highlights": _build_highlights(summary, badges_earned, trophies_earned),
        "links": {
            "player_profile": f"/?page=players&pid={int(player_id)}",
        },
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tz_name": tz_name,
            "chart_points": len(points),
        },
    }
    return digest
