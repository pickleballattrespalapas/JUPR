from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone

import pandas as pd

from jupr_app.domain.gamification.badge_registry import badge_schema_by_id
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


def _safe_int(value) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(value))
    except Exception:
        return None


def _display_range(start_date: date, end_date: date) -> str:
    return f"{_format_display_date(start_date)} – {_format_display_date(end_date)}"


def _ordinal_suffix(day: int) -> str:
    if 11 <= (day % 100) <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def _format_display_date(value: date | datetime | pd.Timestamp | None) -> str:
    if value is None:
        return "Unknown date"
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return "Unknown date"
        day_value = value.date()
    elif isinstance(value, datetime):
        day_value = value.date()
    elif isinstance(value, date):
        day_value = value
    else:
        dt = pd.to_datetime(value, errors="coerce")
        if pd.isna(dt):
            return "Unknown date"
        day_value = dt.date()
    return f"{day_value.strftime('%B')} {day_value.day}{_ordinal_suffix(day_value.day)}"


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


def _format_record(wins: int, losses: int) -> str:
    return f"{int(wins)}-{int(losses)}"


def _format_delta(delta: float | None, decimals: int = 3) -> str:
    if delta is None:
        return "—"
    return f"{delta:+.{decimals}f}"


def _format_pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.0%"
    return f"{(100.0 * float(numerator) / float(denominator)):.1f}%"


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
    badge_desc_map: dict[str, str] = {}
    badge_prestige_map: dict[str, int] = {}
    for badge_id, schema in badge_schema_by_id().items():
        schema_id = str(badge_id)
        badge_name_map[schema_id] = str(schema.title or badge_id)
        badge_desc_map[schema_id] = str(
            schema.display.flavor or schema.display.requirements or "Award earned during this update window."
        ).strip()
        badge_prestige_map[schema_id] = int(getattr(schema, "prestige", 0) or 0)
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
            badge_name_map.update(
                {
                    str(row.get("badge_id")): str(row.get("name") or row.get("badge_id") or "Badge")
                    for row in badge_defs
                }
            )
        except Exception:
            pass

    out: list[dict] = []
    for row in badge_rows:
        bid = str(row.get("badge_id") or "").strip()
        if not bid:
            continue
        out.append(
            {
                "badge_id": bid,
                "name": badge_name_map.get(bid, bid),
                "description": badge_desc_map.get(bid, "Award earned during this update window."),
                "prestige": int(badge_prestige_map.get(bid, 0) or 0),
                "earned_at": row.get("earned_at"),
                "context_type": row.get("context_type"),
                "context_id": row.get("context_id"),
            }
        )
    return out


def _group_badges_for_display(badges: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for badge in badges or []:
        badge_id = str(badge.get("badge_id") or "").strip()
        name = str(badge.get("name") or badge_id or "Badge").strip() or "Badge"
        key = badge_id or name.lower()
        if key not in grouped:
            grouped[key] = {
                "badge_id": badge_id or None,
                "name": name,
                "description": str(badge.get("description") or "Award earned during this update window.").strip(),
                "prestige": int(badge.get("prestige") or 0),
                "count": 0,
                "last_earned_at": None,
            }
        grouped[key]["prestige"] = max(
            int(grouped[key].get("prestige") or 0),
            int(badge.get("prestige") or 0),
        )
        grouped[key]["count"] = int(grouped[key]["count"]) + 1
        earned_at = badge.get("earned_at")
        if earned_at and (grouped[key]["last_earned_at"] is None or str(earned_at) > str(grouped[key]["last_earned_at"])):
            grouped[key]["last_earned_at"] = earned_at

    grouped_items = list(grouped.values())
    grouped_items.sort(key=lambda item: str(item.get("last_earned_at") or ""), reverse=True)
    return grouped_items


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


def _league_breakdown(window_series: pd.DataFrame) -> list[dict]:
    if window_series.empty:
        return []

    rows: list[dict] = []
    grouped = window_series.copy()
    grouped["League"] = grouped.get("League", "").fillna("Unspecified").astype(str)

    for league_name, league_df in grouped.groupby("League"):
        matches = int(len(league_df.index))
        wins = int((league_df.get("Result") == "WIN").sum())
        losses = int((league_df.get("Result") == "LOSS").sum())
        delta = pd.to_numeric(league_df.get("Overall Δ"), errors="coerce").fillna(0.0).sum()
        rows.append(
            {
                "league_name": str(league_name or "Unspecified"),
                "matches": matches,
                "wins": wins,
                "losses": losses,
                "record": _format_record(wins, losses),
                "overall_delta": round(float(delta), 4),
            }
        )

    rows.sort(key=lambda item: (-int(item.get("matches", 0)), -int(item.get("wins", 0)), str(item.get("league_name", ""))))
    return rows


def _is_player_on_t1(row: pd.Series, player_id: int) -> bool | None:
    p = int(player_id)
    t1 = {_safe_int(row.get("t1_p1")), _safe_int(row.get("t1_p2"))}
    t2 = {_safe_int(row.get("t2_p1")), _safe_int(row.get("t2_p2"))}
    if p in t1:
        return True
    if p in t2:
        return False
    return None


def _aggregate_people(ctx, window_series: pd.DataFrame, player_id: int) -> dict:
    partner_stats: dict[int, dict] = defaultdict(lambda: {"matches": 0, "wins": 0, "losses": 0, "point_diff": 0})
    opponent_stats: dict[int, dict] = defaultdict(lambda: {"matches": 0, "wins": 0, "losses": 0, "point_diff": 0})

    if window_series.empty:
        return {
            "top_partners": [],
            "top_opponents": [],
            "best_partner": None,
            "most_faced_opponent": None,
        }

    for _, row in window_series.iterrows():
        player_on_t1 = _is_player_on_t1(row, player_id)
        if player_on_t1 is None:
            continue

        if player_on_t1:
            partner_id = _safe_int(row.get("t1_p2")) if _safe_int(row.get("t1_p1")) == int(player_id) else _safe_int(row.get("t1_p1"))
            opponent_ids = [_safe_int(row.get("t2_p1")), _safe_int(row.get("t2_p2"))]
            my_score = _safe_int(row.get("score_t1"))
            opp_score = _safe_int(row.get("score_t2"))
        else:
            partner_id = _safe_int(row.get("t2_p2")) if _safe_int(row.get("t2_p1")) == int(player_id) else _safe_int(row.get("t2_p1"))
            opponent_ids = [_safe_int(row.get("t1_p1")), _safe_int(row.get("t1_p2"))]
            my_score = _safe_int(row.get("score_t2"))
            opp_score = _safe_int(row.get("score_t1"))

        result = str(row.get("Result") or "").upper()
        won = result == "WIN"
        if result not in {"WIN", "LOSS"} and my_score is not None and opp_score is not None:
            won = my_score > opp_score
        point_diff = (my_score - opp_score) if (my_score is not None and opp_score is not None) else 0

        if partner_id is not None and partner_id != int(player_id):
            entry = partner_stats[int(partner_id)]
            entry["matches"] += 1
            entry["wins"] += int(won)
            entry["losses"] += int(not won)
            entry["point_diff"] += int(point_diff)

        for opp_id in opponent_ids:
            if opp_id is None or opp_id == int(player_id):
                continue
            entry = opponent_stats[int(opp_id)]
            entry["matches"] += 1
            entry["wins"] += int(won)
            entry["losses"] += int(not won)
            entry["point_diff"] += int(point_diff)

    def _finalize(stat_map: dict[int, dict]) -> list[dict]:
        out: list[dict] = []
        for pid, vals in stat_map.items():
            wins = int(vals.get("wins", 0))
            losses = int(vals.get("losses", 0))
            out.append(
                {
                    "player_id": int(pid),
                    "player_name": _player_name(ctx, int(pid)),
                    "matches": int(vals.get("matches", 0)),
                    "wins": wins,
                    "losses": losses,
                    "record": _format_record(wins, losses),
                    "point_diff": int(vals.get("point_diff", 0)),
                }
            )
        out.sort(key=lambda item: (-int(item.get("matches", 0)), -int(item.get("wins", 0)), -int(item.get("point_diff", 0))))
        return out

    top_partners = _finalize(partner_stats)[:3]
    top_opponents = _finalize(opponent_stats)[:3]

    best_partner = None
    for entry in _finalize(partner_stats):
        if int(entry.get("matches", 0)) >= 3:
            best_partner = entry
            break

    most_faced_opponent = top_opponents[0] if top_opponents else None
    return {
        "top_partners": top_partners,
        "top_opponents": top_opponents,
        "best_partner": best_partner,
        "most_faced_opponent": most_faced_opponent,
    }


def _build_week_at_a_glance(
    *,
    player_name: str,
    summary: dict,
    league_breakdown: list[dict],
    people: dict,
) -> list[str]:
    lines: list[str] = []
    matches_played = int(summary.get("matches_played", 0) or 0)
    lines.append(f"{player_name} logged {matches_played} matches in this window.")

    before_val = _safe_float(summary.get("overall_jupr_before"))
    after_val = _safe_float(summary.get("overall_jupr_after"))
    delta_val = _safe_float(summary.get("overall_delta"))
    if after_val is not None:
        if before_val is not None and delta_val is not None:
            lines.append(f"Overall JUPR moved {_format_delta(delta_val)} to {after_val:.3f}.")
        else:
            lines.append(f"Overall JUPR closed at {after_val:.3f}.")

    if league_breakdown:
        top = league_breakdown[0]
        lines.append(
            f"Most of the volume came in {top.get('league_name')} ({int(top.get('matches', 0))} matches, {top.get('record')})."
        )

    top_partner = (people or {}).get("top_partners") or []
    top_opp = (people or {}).get("top_opponents") or []
    if top_partner:
        p = top_partner[0]
        lines.append(
            f"Most played with {p.get('player_name')} ({int(p.get('matches', 0))} matches, {p.get('record')})."
        )
    elif top_opp:
        o = top_opp[0]
        lines.append(f"Most often faced {o.get('player_name')} ({int(o.get('matches', 0))} meetings).")

    return lines[:4]


def _build_notable_results(window_series: pd.DataFrame) -> list[dict]:
    if window_series.empty:
        return []

    df = window_series.copy()
    df["overall_delta"] = pd.to_numeric(df.get("Overall Δ"), errors="coerce")
    df["player_on_t1"] = df.apply(lambda row: _is_player_on_t1(row, _safe_int(row.get("player_id") or -1) or -1), axis=1)
    # margin from player perspective
    def _row_margin(row) -> float | None:
        on_t1 = _is_player_on_t1(row, _safe_int(row.get("player_id") or -1) or -1)
        s1 = _safe_float(row.get("score_t1"))
        s2 = _safe_float(row.get("score_t2"))
        if s1 is None or s2 is None or on_t1 is None:
            return None
        return float(s1 - s2) if on_t1 else float(s2 - s1)

    df["margin"] = df.apply(_row_margin, axis=1)

    def _note(row: pd.Series, title: str, detail: str) -> dict:
        played_at = pd.to_datetime(row.get("Date"), utc=True, errors="coerce")
        date_text = _format_display_date(played_at) if pd.notna(played_at) else "Unknown date"
        league = str(row.get("League") or "Unspecified league")
        score = str(row.get("Score") or "")
        suffix = f" · {score}" if score else ""
        return {"title": title, "detail": f"{date_text} · {league}{suffix}. {detail}"}

    notes: list[dict] = []
    if not df["overall_delta"].dropna().empty:
        best_jump = df.loc[df["overall_delta"].idxmax()]
        if _safe_float(best_jump.get("overall_delta")) and float(best_jump.get("overall_delta")) > 0:
            notes.append(
                _note(
                    best_jump,
                    "Biggest ratings lift",
                    f"Overall moved {_format_delta(float(best_jump.get('overall_delta')), 4)} in this match.",
                )
            )

    losses = df[df.get("Result") == "LOSS"]
    if not losses.empty and not losses["margin"].dropna().empty:
        tough = losses.loc[losses["margin"].idxmin()]
        notes.append(_note(tough, "Toughest loss", "Largest negative margin in the date window."))

    wins = df[df.get("Result") == "WIN"]
    if not wins.empty and not wins["margin"].dropna().empty:
        strong = wins.loc[wins["margin"].idxmax()]
        notes.append(_note(strong, "Strongest scoreline win", "Best positive margin in the date window."))

    close = df[df["margin"].notna()]
    if not close.empty:
        close_row = close.iloc[(close["margin"].abs()).argsort().iloc[0]]
        result = str(close_row.get("Result") or "match").lower()
        notes.append(_note(close_row, "Closest finish", f"A one-score style {result} by margin."))

    unique: list[dict] = []
    seen = set()
    for item in notes:
        key = (item.get("title"), item.get("detail"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique[:5]


def _build_highlights(
    *,
    summary: dict,
    badges_earned: list[dict],
    trophies_earned: list[dict],
    people: dict,
    league_breakdown: list[dict],
) -> list[str]:
    lines: list[str] = []
    before_val = _safe_float(summary.get("overall_jupr_before"))
    after_val = _safe_float(summary.get("overall_jupr_after"))
    delta_val = _safe_float(summary.get("overall_delta"))
    if before_val is not None and after_val is not None and delta_val is not None:
        lines.append(f"Overall JUPR moved {_format_delta(delta_val)} from {before_val:.3f} to {after_val:.3f}.")

    matches_played = int(summary.get("matches_played", 0) or 0)
    lines.append(f"Finished {_format_record(int(summary.get('wins', 0) or 0), int(summary.get('losses', 0) or 0))} across {matches_played} matches.")

    top_partner = (people or {}).get("top_partners") or []
    if top_partner:
        p = top_partner[0]
        lines.append(
            f"Most played with {p.get('player_name')} — {int(p.get('matches', 0))} matches, {p.get('record')}."
        )

    top_opp = (people or {}).get("top_opponents") or []
    if top_opp:
        o = top_opp[0]
        lines.append(f"Most faced {o.get('player_name')} — {int(o.get('matches', 0))} meetings, {o.get('record')}.")

    if league_breakdown:
        top_league = league_breakdown[0]
        lines.append(
            f"Heaviest league segment: {top_league.get('league_name')} ({int(top_league.get('matches', 0))} matches)."
        )

    if badges_earned:
        lines.append(f"Earned badge: {str(badges_earned[0].get('name') or 'Badge')}.")
    if trophies_earned:
        t_name = str(trophies_earned[0].get("tournament_name") or trophies_earned[0].get("league_name") or "Tournament")
        lines.append(f"Podium finish recorded in {t_name}.")

    return lines[:6]


def compute_player_weekly_digest(ctx, player_id: int, start_date: date, end_date: date, tz_name: str = "America/Mazatlan") -> dict:
    start_date, end_date = normalize_date_range(start_date, end_date)
    supabase = getattr(ctx, "supabase", None)
    club_id = str(getattr(ctx, "club_id", "") or "")

    overall_series = build_player_overall_rating_series(supabase, club_id, int(player_id), limit=1200)
    window_series = filter_rating_series_for_window(overall_series, start_date, end_date, tz_name=tz_name).copy()
    if not window_series.empty:
        window_series["player_id"] = int(player_id)

    before = None
    after = _safe_float(window_series["Overall After"].iloc[-1]) if not window_series.empty else None
    if not window_series.empty:
        first_window_row = window_series.iloc[0]
        first_window_id = first_window_row.get("id")
        overall_sorted = overall_series.copy()
        overall_sorted["Date"] = pd.to_datetime(overall_sorted.get("Date"), utc=True, errors="coerce")
        overall_sorted["id"] = pd.to_numeric(overall_sorted.get("id"), errors="coerce")
        overall_sorted = overall_sorted.dropna(subset=["Date"]).sort_values(["Date", "id"], ascending=[True, True])

        if pd.notna(first_window_id):
            first_window_id_num = pd.to_numeric(first_window_id, errors="coerce")
            prior_rows = overall_sorted[
                (overall_sorted["Date"] < first_window_row.get("Date"))
                | ((overall_sorted["Date"] == first_window_row.get("Date")) & (overall_sorted["id"] < first_window_id_num))
            ]
        else:
            prior_rows = overall_sorted[overall_sorted["Date"] < first_window_row.get("Date")]

        if not prior_rows.empty:
            before = _safe_float(prior_rows["Overall After"].iloc[-1])
        if before is None:
            first_after = _safe_float(first_window_row.get("Overall After"))
            first_delta = _safe_float(first_window_row.get("Overall Δ"))
            if first_after is not None and first_delta is not None:
                before = first_after - first_delta

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

    start_dt_utc, end_dt_utc = get_date_range_bounds(start_date, end_date, tz_name)
    points = []
    if before is not None and not window_series.empty:
        points.append(
            {
                "id": "anchor",
                "match_number": 0,
                "date": start_dt_utc.isoformat(),
                "overall_after": round(float(before), 3),
                "overall_delta": 0.0,
                "league": "",
                "score": "",
                "result": "ANCHOR",
                "is_anchor": True,
            }
        )
    for idx, (_, row) in enumerate(window_series.sort_values(["Date", "id"], ascending=[True, True]).iterrows(), start=1):
        dt = pd.to_datetime(row.get("Date"), utc=True, errors="coerce")
        points.append(
            {
                "id": int(row.get("id")) if pd.notna(row.get("id")) else None,
                "match_number": idx,
                "date": dt.isoformat() if pd.notna(dt) else None,
                "overall_after": round(float(row.get("Overall After")), 3)
                if pd.notna(row.get("Overall After"))
                else None,
                "overall_delta": round(float(row.get("Overall Δ")), 4) if pd.notna(row.get("Overall Δ")) else None,
                "league": str(row.get("League") or ""),
                "score": str(row.get("Score") or ""),
                "result": str(row.get("Result") or ""),
            }
        )

    badges_earned = _load_badges_earned(supabase, club_id, int(player_id), start_dt_utc, end_dt_utc)
    badges_grouped = _group_badges_for_display(badges_earned)
    trophies_earned = _load_trophies_earned(supabase, club_id, int(player_id), start_dt_utc, end_dt_utc)

    player_name = _player_name(ctx, int(player_id))
    display_range = _display_range(start_date, end_date)

    league_breakdown = _league_breakdown(window_series)
    people = _aggregate_people(ctx, window_series, int(player_id))
    highlights = _build_highlights(
        summary=summary,
        badges_earned=badges_earned,
        trophies_earned=trophies_earned,
        people=people,
        league_breakdown=league_breakdown,
    )
    week_at_a_glance = _build_week_at_a_glance(
        player_name=player_name,
        summary=summary,
        league_breakdown=league_breakdown,
        people=people,
    )
    notable_results = _build_notable_results(window_series)
    matches_played_rows: list[dict] = []
    for _, row in window_series.sort_values(["Date", "id"], ascending=[False, False]).iterrows():
        player_on_t1 = _is_player_on_t1(row, int(player_id))
        if player_on_t1 is None:
            continue

        if player_on_t1:
            partner_ids = [pid for pid in [_safe_int(row.get("t1_p1")), _safe_int(row.get("t1_p2"))] if pid not in {None, int(player_id)}]
            opponent_ids = [pid for pid in [_safe_int(row.get("t2_p1")), _safe_int(row.get("t2_p2"))] if pid is not None]
        else:
            partner_ids = [pid for pid in [_safe_int(row.get("t2_p1")), _safe_int(row.get("t2_p2"))] if pid not in {None, int(player_id)}]
            opponent_ids = [pid for pid in [_safe_int(row.get("t1_p1")), _safe_int(row.get("t1_p2"))] if pid is not None]

        result = str(row.get("Result") or "").upper()
        result_short = "W" if result == "WIN" else ("L" if result == "LOSS" else (result[:1] if result else "—"))
        score = str(row.get("Score") or "").strip() or None
        played_at = pd.to_datetime(row.get("Date"), utc=True, errors="coerce")
        date_text = _format_display_date(played_at) if pd.notna(played_at) else "Unknown date"
        partners = " / ".join(_player_name(ctx, int(pid)) for pid in partner_ids) if partner_ids else "No partner listed"
        opponents = " / ".join(_player_name(ctx, int(pid)) for pid in opponent_ids) if opponent_ids else "Unknown opponents"
        summary_line = f"{date_text} — with {partners} vs {opponents} — {result_short}"
        if score:
            summary_line = f"{summary_line} {score}"
        matches_played_rows.append(
            {
                "date": played_at.isoformat() if pd.notna(played_at) else None,
                "date_display": date_text,
                "partners": partners,
                "opponents": opponents,
                "result": result_short,
                "score": score,
                "summary": summary_line,
            }
        )

    win_pct = _format_pct(wins, matches_played)
    numbers_cards = [
        {"key": "matches", "label": "Matches", "value": matches_played},
        {"key": "record", "label": "Record", "value": summary.get("record")},
        {"key": "win_pct", "label": "Win %", "value": win_pct},
        {"key": "overall_delta", "label": "Overall Δ", "value": _format_delta(summary.get("overall_delta"), 4)},
        {"key": "leagues", "label": "Leagues", "value": len(league_breakdown)},
        {
            "key": "awards",
            "label": "Badges/Trophies",
            "value": f"{len(badges_earned)}/{len(trophies_earned)}",
        },
    ]

    digest = {
        "club_id": club_id,
        "player_id": int(player_id),
        "player_name": player_name,
        "week_start": start_date.isoformat(),
        "week_end": end_date.isoformat(),
        "display_range": display_range,
        "subject_line": f"{player_name} weekly digest · {display_range}",
        "summary": summary,
        "numbers_cards": numbers_cards,
        "week_at_a_glance": week_at_a_glance,
        "league_breakdown": league_breakdown,
        "people": people,
        "notable_results": notable_results,
        "chart": {
            "title": "Overall JUPR by Match",
            "window_label": display_range,
            "series_key": "overall_after",
            "points": points,
        },
        "badges_earned": badges_earned,
        "badges_grouped": badges_grouped,
        "trophies_earned": trophies_earned,
        "matches_played_rows": matches_played_rows,
        "highlights": highlights,
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
