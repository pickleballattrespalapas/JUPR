from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd


DEFAULT_MAX_FEATURED = 1
RECAP_CATEGORY_CONFIG = {
    "TOP_PERFORMER": {
        "label": "Top Performer",
        "max_featured": 1,
    },
    "BIGGEST_JUMP": {
        "label": "Biggest Jump",
        "max_featured": 1,
    },
}


@dataclass
class SpotlightCandidate:
    candidate_id: str
    key: str
    label: str
    display: str
    player_ids: list[int]
    event_key: tuple[str, str] | None
    value_json: dict
    band: str | None = None


def normalize_date_range(start_date: date, end_date: date) -> tuple[date, date]:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    return start_date, end_date


def get_date_range_bounds(start_date: date, end_date: date, tz_name: str) -> tuple[datetime, datetime]:
    start_date, end_date = normalize_date_range(start_date, end_date)
    tz = ZoneInfo(tz_name)
    start_local = datetime.combine(start_date, time.min).replace(tzinfo=tz)
    end_local = datetime.combine(end_date, time.max).replace(tzinfo=tz)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def get_week_bounds(week_start: date, tz_name: str) -> tuple[datetime, datetime]:
    week_end = week_start + timedelta(days=6)
    return get_date_range_bounds(week_start, week_end, tz_name)


def compute_weekly_recap(
    ctx,
    week_start: date | None = None,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    tz_name: str = "America/Mazatlan",
) -> dict:
    start_date, end_date = _resolve_range_inputs(week_start, start_date, end_date)
    recap, _ = _compute_weekly_recap_payload(ctx, start_date, end_date, tz_name=tz_name)
    return recap


def get_spotlight_candidates(
    ctx,
    week_start: date | None = None,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    tz_name: str = "America/Mazatlan",
) -> dict[str, list[dict]]:
    start_date, end_date = _resolve_range_inputs(week_start, start_date, end_date)
    _, candidates = _compute_weekly_recap_payload(ctx, start_date, end_date, tz_name=tz_name)
    return {
        key: [candidate.__dict__ for candidate in items]
        for key, items in candidates.items()
    }


def _resolve_range_inputs(
    week_start: date | None,
    start_date: date | None,
    end_date: date | None,
) -> tuple[date, date]:
    if start_date is not None and end_date is not None:
        return normalize_date_range(start_date, end_date)
    if week_start is None:
        raise ValueError("Either week_start or both start_date and end_date are required")
    return normalize_date_range(week_start, week_start + timedelta(days=6))


def _compute_weekly_recap_payload(ctx, start_date: date, end_date: date, *, tz_name: str) -> tuple[dict, dict[str, list[SpotlightCandidate]]]:
    club_id = str(ctx.club_id)
    supabase = getattr(ctx, "supabase", None)
    df_matches = getattr(ctx, "df_matches", pd.DataFrame())
    id_to_name = getattr(ctx, "id_to_name", {}) or {}
    df_players_all = getattr(ctx, "df_players_all", pd.DataFrame())

    start_date, end_date = normalize_date_range(start_date, end_date)
    start_dt_utc, end_dt_utc = get_date_range_bounds(start_date, end_date, tz_name)
    df_week = _load_week_matches(df_matches, supabase, club_id, start_dt_utc, end_dt_utc)
    df_week = _filter_week_matches(df_week)

    rating_map = _build_rating_map(df_players_all)

    stats, event_stats, event_meta, giant_slayer_candidates = _compute_stats(
        df_week, rating_map
    )

    spotlight_candidates = _build_spotlight_candidates(
        stats, giant_slayer_candidates, id_to_name
    )
    spotlight = _select_spotlight_items(spotlight_candidates)

    numbers = _build_numbers(df_week, stats, event_meta, supabase, club_id, start_dt_utc, end_dt_utc)

    around_club = _build_around_club(
        event_stats,
        event_meta,
        id_to_name,
        supabase,
    )

    recap = {
        "club_id": club_id,
        "week_start": start_date.isoformat(),
        "week_end": end_date.isoformat(),
        "numbers": numbers,
        "spotlight": [item.__dict__ for item in spotlight],
        "around_club": around_club,
        "looking_ahead": ["", "", ""],
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tz_name": tz_name,
        },
    }
    return recap, spotlight_candidates


def _load_week_matches(
    df_matches: pd.DataFrame | None,
    supabase,
    club_id: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    if df_matches is not None and not df_matches.empty:
        df_local = df_matches.copy()
        df_local["date_dt"] = pd.to_datetime(df_local.get("date", None), utc=True, errors="coerce")
        in_range = df_local[(df_local["date_dt"] >= start_dt) & (df_local["date_dt"] <= end_dt)].copy()
        if not in_range.empty:
            return in_range

    if supabase is None:
        return pd.DataFrame()

    select_cols = [
        "id",
        "date",
        "league",
        "match_type",
        "week_tag",
        "t1_p1",
        "t1_p2",
        "t2_p1",
        "t2_p2",
        "score_t1",
        "score_t2",
        "t1_p1_r",
        "t1_p2_r",
        "t2_p1_r",
        "t2_p2_r",
        "t1_p1_r_end",
        "t1_p2_r_end",
        "t2_p1_r_end",
        "t2_p2_r_end",
        "context_type",
        "context_id",
        "tournament_id",
    ]
    response = (
        supabase.table("matches")
        .select(",".join(select_cols))
        .eq("club_id", club_id)
        .gte("date", start_dt.isoformat())
        .lte("date", end_dt.isoformat())
        .execute()
    )
    return pd.DataFrame(response.data or [])


def _filter_week_matches(df_matches: pd.DataFrame) -> pd.DataFrame:
    if df_matches is None or df_matches.empty:
        return pd.DataFrame()
    df = df_matches.copy()
    df["score_t1"] = pd.to_numeric(df.get("score_t1", 0), errors="coerce").fillna(0).astype(int)
    df["score_t2"] = pd.to_numeric(df.get("score_t2", 0), errors="coerce").fillna(0).astype(int)
    df = df[(df["score_t1"] + df["score_t2"]) > 0].copy()
    if "context_type" in df.columns:
        df = df[df["context_type"].fillna("").astype(str).str.upper() != "TOURNAMENT"].copy()
    if "tournament_id" in df.columns:
        df = df[df["tournament_id"].isna()].copy()
    if "match_type" in df.columns:
        df = df[df["match_type"].fillna("").astype(str) != "Tournament"].copy()
    df["league"] = df.get("league", "").fillna("").astype(str).str.strip()
    df["match_type"] = df.get("match_type", "").fillna("").astype(str).str.strip()
    df["week_tag"] = df.get("week_tag", "").fillna("").astype(str).str.strip()
    df["date_dt"] = pd.to_datetime(df.get("date", None), utc=True, errors="coerce")
    return df


def _build_rating_map(df_players_all: pd.DataFrame | None) -> dict[int, float]:
    if df_players_all is None or df_players_all.empty:
        return {}
    if "id" not in df_players_all.columns:
        return {}
    rating_col = "rating" if "rating" in df_players_all.columns else None
    if rating_col is None:
        return {}
    ids = df_players_all["id"].astype(int)
    ratings = pd.to_numeric(df_players_all[rating_col], errors="coerce").fillna(1200.0).astype(float)
    return dict(zip(ids, ratings))


def _safe_int(value) -> int | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return int(value)
    except Exception:
        return None


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return float(value)
    except Exception:
        return default


def _resolve_event_key(match: dict) -> tuple[str, str] | None:
    league = str(match.get("league", "") or "").strip()
    match_type = str(match.get("match_type", "") or "").strip()
    week_tag = str(match.get("week_tag", "") or "").strip()
    if match_type == "PopUp":
        context_id = match.get("context_id")
        if context_id is None or str(context_id).strip() == "":
            fallback = ":".join([x for x in [league, week_tag] if x]) or "POPUP"
            return ("RR", fallback)
        return ("RR", str(context_id))
    if not league or league.upper() in {"OVERALL", "POPUP"}:
        return None
    return ("LEAGUE", league)


def _compute_stats(
    df_week: pd.DataFrame,
    rating_map: dict[int, float],
) -> tuple[dict, dict, dict, list[dict]]:
    stats: dict[int, dict] = {}
    event_stats: dict[tuple[str, str], dict[int, dict]] = {}
    event_meta: dict[str, dict[str, dict]] = {"leagues": {}, "round_robins": {}}
    giant_slayer_candidates: list[dict] = []

    if df_week is None or df_week.empty:
        return stats, event_stats, event_meta, giant_slayer_candidates

    df_sorted = df_week.sort_values(["date_dt", "id"], ascending=[True, True])

    for _, row in df_sorted.iterrows():
        match = row.to_dict()
        p1 = _safe_int(match.get("t1_p1"))
        p2 = _safe_int(match.get("t1_p2"))
        p3 = _safe_int(match.get("t2_p1"))
        p4 = _safe_int(match.get("t2_p2"))
        if any(pid is None for pid in (p1, p2, p3, p4)):
            continue
        s1 = int(match.get("score_t1", 0) or 0)
        s2 = int(match.get("score_t2", 0) or 0)
        if (s1 + s2) <= 0:
            continue
        t1_win = s1 > s2
        t2_win = s2 > s1

        r1 = _safe_float(match.get("t1_p1_r"), rating_map.get(p1, 1200.0))
        r2 = _safe_float(match.get("t1_p2_r"), rating_map.get(p2, 1200.0))
        r3 = _safe_float(match.get("t2_p1_r"), rating_map.get(p3, 1200.0))
        r4 = _safe_float(match.get("t2_p2_r"), rating_map.get(p4, 1200.0))

        r1_end = _safe_float(match.get("t1_p1_r_end"), r1)
        r2_end = _safe_float(match.get("t1_p2_r_end"), r2)
        r3_end = _safe_float(match.get("t2_p1_r_end"), r3)
        r4_end = _safe_float(match.get("t2_p2_r_end"), r4)

        team1_avg = (r1 + r2) / 2.0 if r1 is not None and r2 is not None else None
        team2_avg = (r3 + r4) / 2.0 if r3 is not None and r4 is not None else None

        event_key = _resolve_event_key(match)
        if event_key is not None:
            if event_key[0] == "LEAGUE":
                event_meta["leagues"].setdefault(event_key[1], {})
            elif event_key[0] == "RR":
                event_meta["round_robins"].setdefault(event_key[1], {})

        def update_player(pid: int, win: int, loss: int, opp_avg: float | None, pre: float | None, end: float | None):
            if pid not in stats:
                stats[pid] = {
                    "games": 0,
                    "wins": 0,
                    "losses": 0,
                    "opponent_ratings": [],
                    "start_rating": None,
                    "end_rating": None,
                    "event_keys": [],
                }
            entry = stats[pid]
            entry["games"] += 1
            entry["wins"] += win
            entry["losses"] += loss
            if opp_avg is not None:
                entry["opponent_ratings"].append(float(opp_avg))
            if event_key is not None:
                entry["event_keys"].append(event_key)
            if entry["start_rating"] is None and pre is not None:
                entry["start_rating"] = float(pre)
            if end is not None:
                entry["end_rating"] = float(end)

        def update_event(pid: int, win: int, loss: int, opp_avg: float | None, pre: float | None, end: float | None):
            if event_key is None:
                return
            if event_key not in event_stats:
                event_stats[event_key] = {}
            if pid not in event_stats[event_key]:
                event_stats[event_key][pid] = {
                    "games": 0,
                    "wins": 0,
                    "losses": 0,
                    "opponent_ratings": [],
                    "start_rating": None,
                    "end_rating": None,
                }
            entry = event_stats[event_key][pid]
            entry["games"] += 1
            entry["wins"] += win
            entry["losses"] += loss
            if opp_avg is not None:
                entry["opponent_ratings"].append(float(opp_avg))
            if entry["start_rating"] is None and pre is not None:
                entry["start_rating"] = float(pre)
            if end is not None:
                entry["end_rating"] = float(end)

        update_player(p1, int(t1_win), int(t2_win), team2_avg, r1, r1_end)
        update_player(p2, int(t1_win), int(t2_win), team2_avg, r2, r2_end)
        update_player(p3, int(t2_win), int(t1_win), team1_avg, r3, r3_end)
        update_player(p4, int(t2_win), int(t1_win), team1_avg, r4, r4_end)

        update_event(p1, int(t1_win), int(t2_win), team2_avg, r1, r1_end)
        update_event(p2, int(t1_win), int(t2_win), team2_avg, r2, r2_end)
        update_event(p3, int(t2_win), int(t1_win), team1_avg, r3, r3_end)
        update_event(p4, int(t2_win), int(t1_win), team1_avg, r4, r4_end)

        if t1_win and team1_avg is not None and team2_avg is not None:
            gap = team2_avg - team1_avg
            if gap > 0:
                giant_slayer_candidates.append(
                    {
                        "player_ids": [p1, p2],
                        "gap_elo": gap,
                        "gap_jupr": gap / 400.0,
                        "match_id": match.get("id"),
                        "event_key": event_key,
                    }
                )
        if t2_win and team1_avg is not None and team2_avg is not None:
            gap = team1_avg - team2_avg
            if gap > 0:
                giant_slayer_candidates.append(
                    {
                        "player_ids": [p3, p4],
                        "gap_elo": gap,
                        "gap_jupr": gap / 400.0,
                        "match_id": match.get("id"),
                        "event_key": event_key,
                    }
                )

    _finalize_rating_deltas(stats, rating_map)
    for event_players in event_stats.values():
        _finalize_rating_deltas(event_players, rating_map)

    return stats, event_stats, event_meta, giant_slayer_candidates


def _finalize_rating_deltas(stats: dict[int, dict], rating_map: dict[int, float]) -> None:
    for pid, entry in stats.items():
        start = entry.get("start_rating")
        end = entry.get("end_rating")
        if start is None:
            start = rating_map.get(pid, 1200.0)
        if end is None:
            end = rating_map.get(pid, start)
        entry["start_rating"] = float(start)
        entry["end_rating"] = float(end)
        delta = float(end) - float(start)
        entry["raw_elo_delta"] = delta
        entry["delta_jupr"] = delta / 400.0


def _build_spotlight_candidates(
    stats: dict[int, dict],
    giant_slayer_candidates: list[dict],
    id_to_name: dict[int, str],
) -> dict[str, list[SpotlightCandidate]]:
    if not stats:
        return {}

    start_ratings = [entry.get("start_rating") for entry in stats.values() if entry.get("start_rating") is not None]
    median = pd.Series(start_ratings).median() if start_ratings else None

    def band_for_player(pid: int) -> str | None:
        if median is None:
            return None
        rating = stats.get(pid, {}).get("start_rating")
        if rating is None:
            return None
        return "top" if float(rating) >= float(median) else "bottom"

    candidates: dict[str, list[SpotlightCandidate]] = {
        "TOP_PERFORMER_WEEK": [],
        "BIGGEST_JUMP_WEEK": [],
        "GIANT_SLAYER_WEEK": [],
        "GRIND_WEEK": [],
        "PERFECT_RUN": [],
    }

    for pid, entry in stats.items():
        games = int(entry.get("games", 0))
        wins = int(entry.get("wins", 0))
        losses = int(entry.get("losses", 0))
        opp_ratings = entry.get("opponent_ratings", [])
        avg_opp = sum(opp_ratings) / len(opp_ratings) if opp_ratings else 0.0
        delta = float(entry.get("delta_jupr", 0.0))
        name = id_to_name.get(pid, f"#{pid}")

        event_key = None
        event_keys = entry.get("event_keys", [])
        if event_keys:
            event_key = max(set(event_keys), key=event_keys.count)

        if games >= 4:
            candidates["TOP_PERFORMER_WEEK"].append(
                SpotlightCandidate(
                    candidate_id=f"player:{pid}",
                    key="TOP_PERFORMER_WEEK",
                    label="Top Performer",
                    display=f"{name} — {wins}-{losses} (+{wins - losses})",
                    player_ids=[pid],
                    event_key=event_key,
                    value_json={"wins": wins, "losses": losses, "games": games, "avg_opponent_rating": avg_opp},
                    band=band_for_player(pid),
                )
            )
        if games >= 3 and delta > 0:
            candidates["BIGGEST_JUMP_WEEK"].append(
                SpotlightCandidate(
                    candidate_id=f"player:{pid}",
                    key="BIGGEST_JUMP_WEEK",
                    label="Biggest Jump",
                    display=f"{name} — +{delta:.2f} JUPR",
                    player_ids=[pid],
                    event_key=event_key,
                    value_json={"delta_jupr": delta, "games": games},
                    band=band_for_player(pid),
                )
            )
        candidates["GRIND_WEEK"].append(
            SpotlightCandidate(
                candidate_id=f"player:{pid}",
                key="GRIND_WEEK",
                label="Grind Week",
                display=f"{name} — {games} games",
                player_ids=[pid],
                event_key=event_key,
                value_json={"games": games},
                band=band_for_player(pid),
            )
        )
        if games >= 5 and losses == 0 and wins > 0:
            candidates["PERFECT_RUN"].append(
                SpotlightCandidate(
                    candidate_id=f"player:{pid}",
                    key="PERFECT_RUN",
                    label="Perfect Run",
                    display=f"{name} — {wins}-0",
                    player_ids=[pid],
                    event_key=event_key,
                    value_json={"wins": wins, "games": games},
                    band=band_for_player(pid),
                )
            )

    if giant_slayer_candidates:
        for candidate in giant_slayer_candidates:
            players = candidate["player_ids"]
            name = " + ".join([id_to_name.get(pid, f"#{pid}") for pid in players])
            gap = float(candidate["gap_jupr"])
            avg_band = None
            if median is not None:
                avg_rating = sum(stats.get(pid, {}).get("start_rating", median) for pid in players) / len(players)
                avg_band = "top" if avg_rating >= float(median) else "bottom"
            candidates["GIANT_SLAYER_WEEK"].append(
                SpotlightCandidate(
                    candidate_id=f"team:{'-'.join(str(pid) for pid in players)}:match:{candidate.get('match_id')}",
                    key="GIANT_SLAYER_WEEK",
                    label="Giant Slayer",
                    display=f"{name} — +{gap:.2f} JUPR gap",
                    player_ids=players,
                    event_key=candidate.get("event_key"),
                    value_json={
                        "gap_elo": candidate.get("gap_elo"),
                        "gap_jupr": gap,
                        "match_id": candidate.get("match_id"),
                    },
                    band=avg_band,
                )
            )

    candidates["TOP_PERFORMER_WEEK"].sort(
        key=lambda x: (x.value_json.get("wins", 0) - x.value_json.get("losses", 0), x.value_json.get("avg_opponent_rating", 0)),
        reverse=True,
    )
    candidates["BIGGEST_JUMP_WEEK"].sort(key=lambda x: x.value_json.get("delta_jupr", 0), reverse=True)
    candidates["GRIND_WEEK"].sort(key=lambda x: x.value_json.get("games", 0), reverse=True)
    candidates["PERFECT_RUN"].sort(key=lambda x: (x.value_json.get("wins", 0), x.value_json.get("games", 0)), reverse=True)
    candidates["GIANT_SLAYER_WEEK"].sort(key=lambda x: x.value_json.get("gap_jupr", 0), reverse=True)

    return candidates


def _select_spotlight_items(candidates: dict[str, list[SpotlightCandidate]]) -> list[SpotlightCandidate]:
    selected: list[SpotlightCandidate] = []
    used_events: set[tuple[str, str]] = set()
    bands: set[str] = set()

    order = [
        "TOP_PERFORMER_WEEK",
        "BIGGEST_JUMP_WEEK",
        "GIANT_SLAYER_WEEK",
        "GRIND_WEEK",
        "PERFECT_RUN",
    ]

    for key in order:
        if key not in candidates or not candidates[key]:
            continue
        missing_band = None
        if bands == {"top"}:
            missing_band = "bottom"
        elif bands == {"bottom"}:
            missing_band = "top"

        options = candidates[key]
        filtered = [c for c in options if c.event_key is None or c.event_key not in used_events]
        if filtered:
            options = filtered
        if missing_band:
            banded = [c for c in options if c.band == missing_band]
            if banded:
                options = banded
        choice = options[0]
        selected.append(choice)
        if choice.event_key is not None:
            used_events.add(choice.event_key)
        if choice.band:
            bands.add(choice.band)
        if len(selected) >= 6:
            break

    return selected


def _build_numbers(
    df_week: pd.DataFrame,
    stats: dict[int, dict],
    event_meta: dict,
    supabase,
    club_id: str,
    start_dt: datetime,
    end_dt: datetime,
) -> dict:
    match_count = int(len(df_week)) if df_week is not None else 0
    player_ids = list(stats.keys())
    league_count = len(event_meta.get("leagues", {}))
    rr_count = len(event_meta.get("round_robins", {}))

    new_faces = _compute_new_faces(supabase, club_id, player_ids, start_dt, end_dt)

    return {
        "matches": match_count,
        "players": len(player_ids),
        "leagues": league_count,
        "round_robins": rr_count,
        "new_faces": len(new_faces),
    }


def _compute_new_faces(
    supabase,
    club_id: str,
    player_ids: list[int],
    start_dt: datetime,
    end_dt: datetime,
) -> list[int]:
    if not player_ids or supabase is None:
        return []
    ids_str = ",".join(str(pid) for pid in player_ids)
    or_filter = ",".join(
        [
            f"t1_p1.in.({ids_str})",
            f"t1_p2.in.({ids_str})",
            f"t2_p1.in.({ids_str})",
            f"t2_p2.in.({ids_str})",
        ]
    )
    response = (
        supabase.table("matches")
        .select("date,t1_p1,t1_p2,t2_p1,t2_p2")
        .eq("club_id", club_id)
        .or_(or_filter)
        .execute()
    )
    df = pd.DataFrame(response.data or [])
    if df.empty:
        return []
    df["date_dt"] = pd.to_datetime(df.get("date", None), utc=True, errors="coerce")
    first_dates: dict[int, datetime] = {}
    for _, row in df.iterrows():
        dt = row.get("date_dt")
        if dt is None or pd.isna(dt):
            continue
        for col in ["t1_p1", "t1_p2", "t2_p1", "t2_p2"]:
            pid = _safe_int(row.get(col))
            if pid is None or pid not in player_ids:
                continue
            current = first_dates.get(pid)
            if current is None or dt < current:
                first_dates[pid] = dt

    new_faces = [pid for pid, dt in first_dates.items() if start_dt <= dt <= end_dt]
    return new_faces


def _build_around_club(
    event_stats: dict,
    event_meta: dict,
    id_to_name: dict[int, str],
    supabase,
) -> dict:
    league_events = sorted(event_meta.get("leagues", {}).keys())
    rr_events = sorted(event_meta.get("round_robins", {}).keys())
    total_events = len(league_events) + len(rr_events)
    highlight_count = 2 if total_events <= 8 else 1
    short_labels = total_events >= 15

    rr_name_map = _fetch_rr_event_names(supabase, rr_events)

    league_items = []
    for league in league_events:
        event_key = ("LEAGUE", league)
        highlights = _event_highlights(
            event_stats.get(event_key, {}),
            id_to_name,
            highlight_count,
            short_labels,
            prefer_jump=True,
        )
        league_items.append({"league_name": league, "highlights": highlights})

    rr_items = []
    for rr_event in rr_events:
        event_key = ("RR", rr_event)
        highlights = _event_highlights(
            event_stats.get(event_key, {}),
            id_to_name,
            1 if highlight_count == 1 else 2,
            short_labels,
            prefer_jump=False,
        )
        rr_items.append(
            {
                "event_id": rr_event,
                "event_name": rr_name_map.get(rr_event, "Pop-Up Event"),
                "highlights": highlights,
            }
        )

    return {"leagues": league_items, "round_robins": rr_items}


def _event_highlights(
    stats: dict[int, dict],
    id_to_name: dict[int, str],
    count: int,
    short_labels: bool,
    prefer_jump: bool,
) -> list[dict]:
    if not stats:
        return []

    ranked_by_category: dict[str, list[dict]] = {
        "TOP_PERFORMER": [],
        "BIGGEST_JUMP": [],
    }

    ranked_top = sorted(
        stats.items(),
        key=lambda item: (item[1].get("wins", 0), item[1].get("games", 0)),
        reverse=True,
    )
    for pid, entry in ranked_top:
        name = id_to_name.get(pid, f"#{pid}")
        wins = int(entry.get("wins", 0))
        losses = int(entry.get("losses", 0))
        display = f"{name} {wins}-{losses}" if short_labels else f"{name} — {wins}-{losses}"
        ranked_by_category["TOP_PERFORMER"].append({"id": pid, "name": name, "display": display})

    ranked_jump = sorted(stats.items(), key=lambda item: item[1].get("delta_jupr", 0), reverse=True)
    for pid, entry in ranked_jump:
        delta = float(entry.get("delta_jupr", 0))
        if delta <= 0:
            continue
        name = id_to_name.get(pid, f"#{pid}")
        display = f"{name} — +{delta:.2f}" if short_labels else f"{name} — +{delta:.2f} JUPR"
        ranked_by_category["BIGGEST_JUMP"].append({"id": pid, "name": name, "display": display})

    category_order = ["BIGGEST_JUMP", "TOP_PERFORMER"] if prefer_jump else ["TOP_PERFORMER", "BIGGEST_JUMP"]
    highlights: list[dict] = []
    for key in category_order:
        if len(highlights) >= count:
            break
        config = RECAP_CATEGORY_CONFIG.get(key, {})
        label = config.get("label") or ("Top Performer" if key == "TOP_PERFORMER" else "Biggest Jump")
        max_featured = max(0, int(config.get("max_featured", DEFAULT_MAX_FEATURED)))
        players = ranked_by_category.get(key, [])[:max_featured]
        if not players:
            continue
        highlights.append({"key": key, "label": label if not short_labels else label, "players": players})

    return highlights


def _fetch_rr_event_names(supabase, rr_events: list[str]) -> dict[str, str]:
    if supabase is None:
        return {}
    ids = [eid for eid in rr_events if eid and not eid.startswith("POPUP")]
    if not ids:
        return {}
    response = supabase.table("events").select("id,name").in_("id", ids).execute()
    return {row["id"]: row.get("name") or "Pop-Up Event" for row in (response.data or [])}
