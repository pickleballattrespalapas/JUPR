from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from postgrest.exceptions import APIError

from jupr_app.domain.challenge_ladder import normalize_tier_id, tier_idx, tier_title

DEFAULT_AWARD_DESCRIPTIONS = {
    "TOP_PERFORMER_WEEK": (
        "The strongest overall performance of the week, based on win–loss results across multiple matches. "
        "This award highlights players who consistently showed up, competed well, and delivered results."
    ),
    "BIGGEST_JUMP_WEEK": (
        "The largest JUPR rating improvement during the week. "
        "This reflects meaningful progress against competition — not just wins, but growth."
    ),
    "GIANT_SLAYER_WEEK": (
        "A standout win against a significantly higher-rated opponent or team. "
        "This award celebrates fearless play and capitalizing on tough matchups."
    ),
    "GRIND_WEEK": (
        "The players who logged the most matches during the week. "
        "Consistency, availability, and willingness to compete are what earn this one."
    ),
    "PERFECT_RUN": (
        "An undefeated week with a meaningful number of matches played. "
        "No losses, no shortcuts — just clean execution start to finish."
    ),
}

DEFAULT_AROUND_LEAGUE_DESCRIPTION = (
    "Weekly highlights from this league — top performer and biggest jump based on matches recorded this week."
)
DEFAULT_AROUND_RR_DESCRIPTION = (
    "Pop-up results from this event — highlights based on matches recorded this week."
)


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


def get_week_bounds(week_start: date, tz_name: str) -> tuple[datetime, datetime]:
    tz = ZoneInfo(tz_name)
    start_local = datetime.combine(week_start, time.min).replace(tzinfo=tz)
    end_local = start_local + timedelta(days=7) - timedelta(microseconds=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def compute_weekly_recap(
    ctx,
    week_start: date,
    *,
    tz_name: str = "America/Mazatlan",
    allow_ties: bool = True,
) -> dict:
    recap, _ = _compute_weekly_recap_payload(ctx, week_start, tz_name=tz_name, allow_ties=allow_ties)
    return recap


def get_spotlight_candidates(
    ctx,
    week_start: date,
    *,
    tz_name: str = "America/Mazatlan",
    allow_ties: bool = True,
) -> dict[str, list[dict]]:
    _, candidates = _compute_weekly_recap_payload(ctx, week_start, tz_name=tz_name, allow_ties=allow_ties)
    return {
        key: [candidate.__dict__ for candidate in items]
        for key, items in candidates.items()
    }


def _compute_weekly_recap_payload(
    ctx,
    week_start: date,
    *,
    tz_name: str,
    allow_ties: bool = True,
) -> tuple[dict, dict[str, list[SpotlightCandidate]]]:
    club_id = str(ctx.club_id)
    supabase = getattr(ctx, "supabase", None)
    df_matches = getattr(ctx, "df_matches", pd.DataFrame())
    id_to_name = getattr(ctx, "id_to_name", {}) or {}
    df_players_all = getattr(ctx, "df_players_all", pd.DataFrame())

    start_dt_utc, end_dt_utc = get_week_bounds(week_start, tz_name)
    df_week = _load_week_matches(df_matches, supabase, club_id, start_dt_utc, end_dt_utc)
    df_week = _filter_week_matches(df_week)

    rating_map = _build_rating_map(df_players_all)

    stats, event_stats, event_meta, giant_slayer_candidates = _compute_stats(
        df_week, rating_map
    )

    spotlight_candidates = _build_spotlight_candidates(
        stats, giant_slayer_candidates, id_to_name
    )
    spotlight = _select_spotlight_items(spotlight_candidates, allow_ties=allow_ties)

    numbers = _build_numbers(df_week, stats, event_meta, supabase, club_id, start_dt_utc, end_dt_utc)

    around_club = _build_around_club(
        event_stats,
        event_meta,
        id_to_name,
        supabase,
    )
    around_descriptions = apply_around_descriptions(around_club, None)
    challenge_ladder = _fetch_challenge_ladder_week_summary(
        supabase,
        club_id,
        start_dt_utc,
        end_dt_utc,
        id_to_name,
    )

    week_end = week_start + timedelta(days=6)
    recap = {
        "club_id": club_id,
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "numbers": numbers,
        "spotlight": [item.__dict__ for item in spotlight],
        "award_descriptions": dict(DEFAULT_AWARD_DESCRIPTIONS),
        "around_club": around_club,
        "around_descriptions": around_descriptions,
        "challenge_ladder": challenge_ladder,
        "looking_ahead": ["", "", ""],
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tz_name": tz_name,
            "allow_ties": allow_ties,
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


def _float_equal(a: float | None, b: float | None, eps: float = 1e-9) -> bool:
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= eps


def _metric_key_for_candidate(key: str, candidate: SpotlightCandidate) -> tuple | None:
    values = candidate.value_json or {}
    if key == "TOP_PERFORMER_WEEK":
        wins = int(values.get("wins", 0))
        losses = int(values.get("losses", 0))
        avg_opp = values.get("avg_opponent_rating")
        return (wins - losses, float(avg_opp) if avg_opp is not None else None)
    if key == "BIGGEST_JUMP_WEEK":
        return (float(values.get("delta_jupr", 0.0)),)
    if key == "GRIND_WEEK":
        return (int(values.get("games", 0)),)
    if key == "PERFECT_RUN":
        return (int(values.get("wins", 0)),)
    if key == "GIANT_SLAYER_WEEK":
        return (float(values.get("gap_jupr", 0.0)),)
    return None


def _make_tie_candidate(key: str, candidate: SpotlightCandidate, other: SpotlightCandidate) -> SpotlightCandidate:
    def split_display(display: str) -> tuple[str, str | None]:
        parts = display.split("—", 1)
        name = parts[0].strip()
        suffix = parts[1].strip() if len(parts) > 1 else None
        return name, suffix

    name1, suffix = split_display(candidate.display)
    name2, _ = split_display(other.display)
    combined_name = f"{name1} + {name2}"
    display = f"{combined_name} — {suffix}" if suffix else combined_name
    player_ids = list(dict.fromkeys(candidate.player_ids + other.player_ids))
    value_json = dict(candidate.value_json or {})
    value_json["tied"] = True
    return SpotlightCandidate(
        candidate_id=f"tie:{key}:{'+'.join(str(pid) for pid in player_ids)}",
        key=key,
        label=candidate.label,
        display=display,
        player_ids=player_ids,
        event_key=None,
        value_json=value_json,
        band=None,
    )


def _select_spotlight_items(
    candidates: dict[str, list[SpotlightCandidate]],
    *,
    allow_ties: bool = True,
) -> list[SpotlightCandidate]:
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
        if allow_ties and key in {
            "TOP_PERFORMER_WEEK",
            "BIGGEST_JUMP_WEEK",
            "GIANT_SLAYER_WEEK",
            "GRIND_WEEK",
            "PERFECT_RUN",
        }:
            metric_key = _metric_key_for_candidate(key, choice)
            if metric_key is not None:
                for candidate in options[1:]:
                    other_key = _metric_key_for_candidate(key, candidate)
                    if other_key is None or len(other_key) != len(metric_key):
                        continue
                    matches = True
                    for left, right in zip(metric_key, other_key):
                        if isinstance(left, float) or isinstance(right, float):
                            if not _float_equal(left, right):
                                matches = False
                                break
                        else:
                            if left != right:
                                matches = False
                                break
                    if matches:
                        choice = _make_tie_candidate(key, choice, candidate)
                        break
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


def build_around_descriptions(around_club: dict) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    for league_item in around_club.get("leagues", []) or []:
        league_name = str(league_item.get("league_name", "") or "").strip()
        if league_name:
            descriptions[f"LEAGUE:{league_name}"] = DEFAULT_AROUND_LEAGUE_DESCRIPTION
    for rr_item in around_club.get("round_robins", []) or []:
        event_id = str(rr_item.get("event_id", "") or "").strip()
        if event_id:
            descriptions[f"RR:{event_id}"] = DEFAULT_AROUND_RR_DESCRIPTION
    return descriptions


def apply_around_descriptions(around_club: dict, around_descriptions: dict | None) -> dict[str, str]:
    defaults = build_around_descriptions(around_club)
    merged = dict(defaults)
    merged.update(around_descriptions or {})

    for league_item in around_club.get("leagues", []) or []:
        league_name = str(league_item.get("league_name", "") or "").strip()
        if not league_name:
            continue
        desc_key = f"LEAGUE:{league_name}"
        league_item["desc_key"] = desc_key
        league_item["description"] = merged.get(desc_key, defaults.get(desc_key, DEFAULT_AROUND_LEAGUE_DESCRIPTION))

    for rr_item in around_club.get("round_robins", []) or []:
        event_id = str(rr_item.get("event_id", "") or "").strip()
        if not event_id:
            continue
        desc_key = f"RR:{event_id}"
        rr_item["desc_key"] = desc_key
        rr_item["description"] = merged.get(desc_key, defaults.get(desc_key, DEFAULT_AROUND_RR_DESCRIPTION))

    return merged


def _event_highlights(
    stats: dict[int, dict],
    id_to_name: dict[int, str],
    count: int,
    short_labels: bool,
    prefer_jump: bool,
) -> list[dict]:
    if not stats:
        return []

    def top_performer():
        best = max(stats.items(), key=lambda item: (item[1].get("wins", 0), item[1].get("games", 0)))
        pid, entry = best
        name = id_to_name.get(pid, f"#{pid}")
        wins = int(entry.get("wins", 0))
        losses = int(entry.get("losses", 0))
        label = "Top" if short_labels else "Top Performer"
        display = f"{name} — {wins}-{losses}" if not short_labels else f"{name} {wins}-{losses}"
        return {"key": "TOP_PERFORMER", "label": label, "display": display, "player_ids": [pid]}

    def biggest_jump():
        best = max(stats.items(), key=lambda item: item[1].get("delta_jupr", 0))
        pid, entry = best
        delta = float(entry.get("delta_jupr", 0))
        if delta <= 0:
            return None
        name = id_to_name.get(pid, f"#{pid}")
        label = "Jump" if short_labels else "Biggest Jump"
        display = f"{name} — +{delta:.2f}" if short_labels else f"{name} — +{delta:.2f} JUPR"
        return {"key": "BIGGEST_JUMP", "label": label, "display": display, "player_ids": [pid]}

    highlights = []
    jump = biggest_jump()
    top = top_performer()
    if prefer_jump and jump is not None and (jump["display"]):
        highlights.append(jump)
    else:
        highlights.append(top)
    if count > 1:
        if prefer_jump:
            if top["key"] != highlights[0]["key"]:
                highlights.append(top)
        else:
            if jump is not None:
                highlights.append(jump)
    return highlights[:count]


def _fetch_rr_event_names(supabase, rr_events: list[str]) -> dict[str, str]:
    if supabase is None:
        return {}
    ids = [eid for eid in rr_events if eid and not eid.startswith("POPUP")]
    if not ids:
        return {}
    response = supabase.table("events").select("id,name").in_("id", ids).execute()
    return {row["id"]: row.get("name") or "Pop-Up Event" for row in (response.data or [])}


def _empty_challenge_ladder_summary() -> dict:
    return {"title": "Match Results", "by_tier": []}


def _format_rank(rank_value) -> str:
    rank_int = _safe_int(rank_value)
    return f"#{rank_int}" if rank_int is not None else "#?"


def _tier_label_for_recap(tier_id: str) -> str:
    normalized_tier_id = normalize_tier_id(tier_id)
    label = str(tier_title(normalized_tier_id) or normalized_tier_id)
    primary_label = label.split("—", 1)[0].strip()
    if not primary_label:
        primary_label = normalized_tier_id or "Unknown"
    if not primary_label.lower().endswith("tier"):
        primary_label = f"{primary_label} Tier"
    return primary_label


def _summarize_challenge_ladder_rows(rows: list[dict], id_to_name: dict[int, str]) -> dict:
    grouped_lines: dict[str, list[str]] = {}
    tier_sort_values: dict[str, int] = {}

    for row in rows or []:
        winner_id = _safe_int(row.get("winner_id"))
        challenger_id = _safe_int(row.get("challenger_id"))
        defender_id = _safe_int(row.get("defender_id"))
        status = str(row.get("status") or "").strip().upper()
        if status != "COMPLETED" or winner_id is None:
            continue

        normalized_tier_id = normalize_tier_id(str(row.get("tier_id") or ""))
        tier_label = _tier_label_for_recap(normalized_tier_id)
        tier_sort_values[tier_label] = tier_idx(normalized_tier_id)

        challenger_name = id_to_name.get(challenger_id, f"#{challenger_id}") if challenger_id is not None else "Unknown"
        defender_name = id_to_name.get(defender_id, f"#{defender_id}") if defender_id is not None else "Unknown"
        challenger_rank = _format_rank(row.get("challenger_rank_at_create"))
        defender_rank = _format_rank(row.get("defender_rank_at_create"))

        if winner_id == challenger_id:
            line = f"{challenger_rank} {challenger_name} beat {defender_rank} {defender_name}"
        else:
            line = f"{defender_rank} {defender_name} defended vs {challenger_rank} {challenger_name}"

        grouped_lines.setdefault(tier_label, []).append(line)

    if not grouped_lines:
        return _empty_challenge_ladder_summary()

    by_tier = [
        {"tier": tier, "lines": lines}
        for tier, lines in sorted(
            grouped_lines.items(),
            key=lambda item: (tier_sort_values.get(item[0], 999), item[0]),
        )
    ]
    return {
        "title": "Match Results",
        "by_tier": by_tier,
    }


def _fetch_challenge_ladder_week_summary(
    supabase,
    club_id: str,
    start_dt_utc: datetime,
    end_dt_utc: datetime,
    id_to_name: dict[int, str],
) -> dict:
    if supabase is None:
        return _empty_challenge_ladder_summary()

    select_cols = (
        "id,tier_id,challenger_id,defender_id,winner_id,status,completed_at,"
        "challenger_rank_at_create,defender_rank_at_create"
    )
    try:
        response = (
            supabase.table("ladder_challenges")
            .select(select_cols)
            .eq("club_id", club_id)
            .gte("completed_at", start_dt_utc.isoformat())
            .lte("completed_at", end_dt_utc.isoformat())
            .not_.is_("winner_id", "null")
            .execute()
        )
        rows = response.data or []
    except APIError:
        return _empty_challenge_ladder_summary()
    except Exception:
        try:
            fallback_response = (
                supabase.table("ladder_challenges")
                .select(select_cols)
                .eq("club_id", club_id)
                .gte("completed_at", start_dt_utc.isoformat())
                .lte("completed_at", end_dt_utc.isoformat())
                .execute()
            )
            rows = fallback_response.data or []
        except APIError:
            return _empty_challenge_ladder_summary()

    return _summarize_challenge_ladder_rows(rows, id_to_name)
