from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from statistics import pstdev
from typing import Any, Iterable
from uuid import uuid4

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BadgeDefinition:
    badge_id: str
    name: str
    prestige: int
    category: str
    is_stackable: bool
    is_active: bool = True


BADGE_DEFINITIONS: list[BadgeDefinition] = [
    BadgeDefinition("participant", "Participant", 10, "Participation", False),
    BadgeDefinition("dedicated_participant", "Dedicated Participant", 25, "Participation", False),
    BadgeDefinition("lifetime_participant", "Lifetime Participant", 50, "Participation", False),
    BadgeDefinition("mountain_climber", "Mountain Climber", 45, "Momentum & Progress", True),
    BadgeDefinition("breakthrough", "Breakthrough", 55, "Momentum & Progress", False),
    BadgeDefinition("above_expectations", "Above Expectations", 50, "Performance vs Expectation", False),
    BadgeDefinition("clutch_performer", "Clutch Performer", 60, "Performance vs Expectation", False),
    BadgeDefinition("dominant_run", "Dominant Run", 45, "Dominance & Quality", False),
    BadgeDefinition("high_output", "High Output", 40, "Dominance & Quality", False),
    BadgeDefinition("battle_tested", "Battle Tested", 50, "Dominance & Quality", False),
    BadgeDefinition("consistency", "Consistency", 60, "Dominance & Quality", False),
    BadgeDefinition("giant_slayer", "Giant Slayer", 75, "Prestige / Rarity", True),
    BadgeDefinition("upset_champion", "Upset Champion", 90, "Prestige / Rarity", True),
]


PARTICIPATION_DEDICATED_GAMES = 50
PARTICIPATION_LIFETIME_GAMES = 200
BREAKTHROUGH_TOP_N = 5
BREAKTHROUGH_RATING = 1600.0
EXPECTED_WIN_DELTA = 2.0
CLOSE_MARGIN = 2
MIN_CLOSE_GAMES = 5
DOMINANT_WINDOW = 5
DOMINANT_POINT_DIFF = 30
HIGH_OUTPUT_WINDOW = 5
HIGH_OUTPUT_POINTS = 60
BATTLE_TESTED_MIN_MATCHES = 10
BATTLE_TESTED_PERCENTILE = 0.9
CONSISTENCY_MIN_MATCHES = 10
CONSISTENCY_STD_MAX = 20.0
GIANT_SLAYER_RATING = 2000.0


def ensure_badges(ctx) -> None:
    if bool(getattr(ctx, "public_mode", False)):
        return

    supabase = getattr(ctx, "supabase", None)
    club_id = str(getattr(ctx, "club_id", ""))
    if supabase is None or not club_id:
        return

    try:
        _seed_badges(supabase)
        existing = _fetch_existing_badges(supabase, club_id)
        awards = _compute_badges(ctx, existing)
        if awards:
            _insert_badges(supabase, awards)
    except Exception:
        logger.exception("ensure_badges failed")


def _seed_badges(supabase) -> None:
    payload = [
        {
            "badge_id": b.badge_id,
            "name": b.name,
            "prestige": b.prestige,
            "category": b.category,
            "is_stackable": b.is_stackable,
            "is_active": b.is_active,
        }
        for b in BADGE_DEFINITIONS
    ]
    try:
        supabase.table("badges").upsert(payload, on_conflict="badge_id").execute()
    except Exception:
        logger.exception("Failed to seed badges table")


def _fetch_existing_badges(supabase, club_id: str) -> set[tuple[str, str, str | None]]:
    try:
        resp = (
            supabase.table("player_badges")
            .select("player_id,badge_id,context_id")
            .eq("club_id", club_id)
            .execute()
        )
    except Exception:
        logger.exception("Failed to fetch existing player badges")
        return set()

    existing = set()
    for row in resp.data or []:
        player_id = str(row.get("player_id"))
        badge_id = str(row.get("badge_id"))
        context_id = row.get("context_id")
        context_id = str(context_id) if context_id is not None else None
        existing.add((player_id, badge_id, context_id))
    return existing


def _insert_badges(supabase, rows: list[dict[str, Any]]) -> None:
    chunk = 200
    for i in range(0, len(rows), chunk):
        supabase.table("player_badges").insert(rows[i : i + chunk]).execute()


def _compute_badges(ctx, existing: set[tuple[str, str, str | None]]) -> list[dict[str, Any]]:
    awards: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()

    def add_badge(
        player_id: int,
        badge_id: str,
        *,
        context_type: str,
        context_id: str | None = None,
        match_id: str | None = None,
        value_num: float | None = None,
        value_json: dict[str, Any] | None = None,
    ) -> None:
        key = (str(player_id), badge_id, str(context_id) if context_id is not None else None)
        if key in existing:
            return
        existing.add(key)
        awards.append(
            {
                "id": str(uuid4()),
                "club_id": str(ctx.club_id),
                "player_id": int(player_id),
                "badge_id": badge_id,
                "earned_at": now,
                "context_type": context_type,
                "context_id": context_id,
                "match_id": match_id,
                "value_num": value_num,
                "value_json": value_json,
            }
        )

    df_players = getattr(ctx, "df_players_all", None)
    if df_players is None or df_players.empty:
        return awards

    df_players = df_players.copy()
    if "matches_played" not in df_players.columns:
        df_players["matches_played"] = 0

    for _, row in df_players.iterrows():
        try:
            player_id = int(row.get("id"))
        except Exception:
            continue
        games = int(row.get("matches_played") or 0)
        if games >= 1:
            add_badge(player_id, "participant", context_type="overall", context_id="overall", value_num=games)
        if games >= PARTICIPATION_DEDICATED_GAMES:
            add_badge(
                player_id,
                "dedicated_participant",
                context_type="overall",
                context_id="overall",
                value_num=games,
            )
        if games >= PARTICIPATION_LIFETIME_GAMES:
            add_badge(
                player_id,
                "lifetime_participant",
                context_type="overall",
                context_id="overall",
                value_num=games,
            )

    _award_league_badges(ctx, add_badge)
    _award_match_badges(ctx, add_badge)

    return awards


def _award_league_badges(ctx, add_badge) -> None:
    df_leagues = getattr(ctx, "df_leagues", None)
    if df_leagues is None or df_leagues.empty:
        return

    df = df_leagues.copy()
    if "league_name" not in df.columns:
        return

    df["league_name"] = df["league_name"].fillna("").astype(str).str.strip()

    for league_name, league_df in df.groupby("league_name"):
        if not league_name:
            continue
        league_df = league_df.copy()
        if "player_id" not in league_df.columns:
            continue

        league_df["player_id"] = league_df["player_id"].astype(int)

        if "starting_rating" not in league_df.columns:
            league_df["starting_rating"] = league_df.get("rating", 1200.0)

        league_df["starting_rating"] = league_df["starting_rating"].fillna(league_df["rating"]).astype(float)
        league_df["rating"] = league_df.get("rating", 1200.0).fillna(1200.0).astype(float)

        start_sorted = league_df.sort_values("starting_rating", ascending=False).reset_index(drop=True)
        start_sorted["start_rank"] = start_sorted.index + 1
        current_sorted = league_df.sort_values("rating", ascending=False).reset_index(drop=True)
        current_sorted["current_rank"] = current_sorted.index + 1

        ranks = start_sorted[["player_id", "start_rank"]].merge(
            current_sorted[["player_id", "current_rank"]], on="player_id", how="inner"
        )

        for _, row in ranks.iterrows():
            rank_delta = int(row["start_rank"] - row["current_rank"])
            if rank_delta >= 3:
                context_id = f"{league_name}:{int(row['current_rank'])}"
                add_badge(
                    int(row["player_id"]),
                    "mountain_climber",
                    context_type="league",
                    context_id=str(context_id),
                    value_num=float(rank_delta),
                    value_json={
                        "start_rank": int(row["start_rank"]),
                        "current_rank": int(row["current_rank"]),
                    },
                )

        top_n = current_sorted.head(BREAKTHROUGH_TOP_N)
        for _, row in top_n.iterrows():
            add_badge(
                int(row["player_id"]),
                "breakthrough",
                context_type="league",
                context_id=str(league_name),
                value_num=float(row["rating"]),
                value_json={"reason": "top_5"},
            )

        milestone = current_sorted[current_sorted["rating"] >= BREAKTHROUGH_RATING]
        for _, row in milestone.iterrows():
            add_badge(
                int(row["player_id"]),
                "breakthrough",
                context_type="league",
                context_id=str(league_name),
                value_num=float(row["rating"]),
                value_json={"reason": "rating_milestone"},
            )


def _award_match_badges(ctx, add_badge) -> None:
    df_matches = getattr(ctx, "df_matches", None)
    if df_matches is None or df_matches.empty:
        return

    df_matches = df_matches.copy()
    df_matches["date_dt"] = pd.to_datetime(df_matches.get("date", None), utc=True, errors="coerce")
    df_matches = df_matches.dropna(subset=["date_dt"]).sort_values(["date_dt", "id"], ascending=[True, True])
    if df_matches.empty:
        return

    players_df = getattr(ctx, "df_players_all", None)
    rating_map = {}
    if players_df is not None and not players_df.empty:
        try:
            rating_map = dict(zip(players_df["id"].astype(int), players_df["rating"].astype(float)))
        except Exception:
            rating_map = {}

    stats = {}
    upset_by_league: dict[str, dict[str, Any]] = {}

    def ensure_player(pid: int) -> dict[str, Any]:
        if pid not in stats:
            stats[pid] = {
                "wins": 0,
                "expected_wins": 0.0,
                "close_games": 0,
                "close_wins": 0,
                "diffs": [],
                "points": [],
                "opp_ratings": [],
                "deltas": [],
                "matches": 0,
            }
        return stats[pid]

    for _, row in df_matches.iterrows():
        try:
            match_id = str(row.get("id"))
            p1 = int(row.get("t1_p1"))
            p2 = int(row.get("t1_p2"))
            p3 = int(row.get("t2_p1"))
            p4 = int(row.get("t2_p2"))
            s1 = int(row.get("score_t1") or 0)
            s2 = int(row.get("score_t2") or 0)
        except Exception:
            continue

        if (s1 + s2) <= 0:
            continue

        r1 = _safe_rating(row.get("t1_p1_r"), rating_map.get(p1))
        r2 = _safe_rating(row.get("t1_p2_r"), rating_map.get(p2))
        r3 = _safe_rating(row.get("t2_p1_r"), rating_map.get(p3))
        r4 = _safe_rating(row.get("t2_p2_r"), rating_map.get(p4))

        t1_avg = (r1 + r2) / 2.0
        t2_avg = (r3 + r4) / 2.0
        expected_share_t1 = _expected_share(t1_avg, t2_avg)

        close_game = abs(s1 - s2) <= CLOSE_MARGIN
        winner_team = 0
        if s1 > s2:
            winner_team = 1
        elif s2 > s1:
            winner_team = 2

        delta_abs = _safe_float(row.get("elo_delta"))

        league_name = str(row.get("league", "") or "").strip() or "OVERALL"
        upset_info = upset_by_league.setdefault(league_name, {"max": None, "matches": []})
        expected_margin = _expected_margin(expected_share_t1, max(s1, s2))
        expected_for_winner = expected_margin if winner_team == 1 else -expected_margin
        actual_margin = abs(s1 - s2)
        exceed = actual_margin - expected_for_winner
        if upset_info["max"] is None or exceed > upset_info["max"]:
            upset_info["max"] = exceed
            upset_info["matches"] = [
                {
                    "match_id": match_id,
                    "winner_team": winner_team,
                    "players": (p1, p2, p3, p4),
                    "s1": s1,
                    "s2": s2,
                    "league": league_name,
                    "exceed": exceed,
                }
            ]
        elif exceed == upset_info["max"]:
            upset_info["matches"].append(
                {
                    "match_id": match_id,
                    "winner_team": winner_team,
                    "players": (p1, p2, p3, p4),
                    "s1": s1,
                    "s2": s2,
                    "league": league_name,
                    "exceed": exceed,
                }
            )

        for pid, team, my_score, opp_score, opp_avg, opp_max, expected_win in (
            (p1, 1, s1, s2, (r3 + r4) / 2.0, max(r3, r4), expected_share_t1),
            (p2, 1, s1, s2, (r3 + r4) / 2.0, max(r3, r4), expected_share_t1),
            (p3, 2, s2, s1, (r1 + r2) / 2.0, max(r1, r2), 1.0 - expected_share_t1),
            (p4, 2, s2, s1, (r1 + r2) / 2.0, max(r1, r2), 1.0 - expected_share_t1),
        ):
            st = ensure_player(pid)
            st["matches"] += 1
            st["expected_wins"] += float(expected_win)
            if winner_team == team:
                st["wins"] += 1
            if close_game:
                st["close_games"] += 1
                if winner_team == team:
                    st["close_wins"] += 1
            st["diffs"].append(int(my_score - opp_score))
            st["points"].append(int(my_score))
            st["opp_ratings"].append(float(opp_avg))
            if delta_abs is not None and winner_team in (1, 2):
                signed = float(delta_abs) if winner_team == team else -float(delta_abs)
                st["deltas"].append(signed)
            if winner_team == team and opp_max >= GIANT_SLAYER_RATING:
                add_badge(
                    pid,
                    "giant_slayer",
                    context_type="overall",
                    context_id=str(match_id),
                    match_id=str(match_id),
                    value_num=float(opp_max),
                    value_json={"opponent_rating": float(opp_max)},
                )

    for pid, st in stats.items():
        wins = st["wins"]
        expected = st["expected_wins"]
        if wins - expected >= EXPECTED_WIN_DELTA:
            add_badge(
                pid,
                "above_expectations",
                context_type="overall",
                context_id="overall",
                value_num=float(wins - expected),
                value_json={"wins": int(wins), "expected_wins": float(expected)},
            )

        if st["close_games"] >= MIN_CLOSE_GAMES:
            rate = st["close_wins"] / float(st["close_games"])
            if rate >= 0.7:
                add_badge(
                    pid,
                    "clutch_performer",
                    context_type="overall",
                    context_id="overall",
                    value_num=float(rate),
                    value_json={
                        "close_wins": int(st["close_wins"]),
                        "close_games": int(st["close_games"]),
                    },
                )

        max_diff = _max_rolling_sum(st["diffs"], DOMINANT_WINDOW)
        if max_diff is not None and max_diff >= DOMINANT_POINT_DIFF:
            add_badge(
                pid,
                "dominant_run",
                context_type="overall",
                context_id="overall",
                value_num=float(max_diff),
                value_json={"window": DOMINANT_WINDOW},
            )

        max_points = _max_rolling_sum(st["points"], HIGH_OUTPUT_WINDOW)
        if max_points is not None and max_points >= HIGH_OUTPUT_POINTS:
            add_badge(
                pid,
                "high_output",
                context_type="overall",
                context_id="overall",
                value_num=float(max_points),
                value_json={"window": HIGH_OUTPUT_WINDOW},
            )

        if st["matches"] >= CONSISTENCY_MIN_MATCHES and len(st["deltas"]) >= 2:
            volatility = pstdev(st["deltas"])
            if volatility <= CONSISTENCY_STD_MAX:
                add_badge(
                    pid,
                    "consistency",
                    context_type="overall",
                    context_id="overall",
                    value_num=float(volatility),
                    value_json={"matches": int(st["matches"])},
                )

    _award_battle_tested(stats, add_badge)
    _award_upset_champion(upset_by_league, add_badge)


def _award_battle_tested(stats: dict[int, dict[str, Any]], add_badge) -> None:
    candidates = {}
    for pid, st in stats.items():
        if st["matches"] < BATTLE_TESTED_MIN_MATCHES:
            continue
        if not st["opp_ratings"]:
            continue
        avg_opp = sum(st["opp_ratings"]) / len(st["opp_ratings"])
        candidates[pid] = avg_opp

    if not candidates:
        return

    values = sorted(candidates.values())
    cutoff_index = max(0, int(len(values) * BATTLE_TESTED_PERCENTILE) - 1)
    cutoff = values[cutoff_index]

    for pid, avg_opp in candidates.items():
        if avg_opp >= cutoff:
            add_badge(
                pid,
                "battle_tested",
                context_type="overall",
                context_id="overall",
                value_num=float(avg_opp),
                value_json={"avg_opponent_rating": float(avg_opp)},
            )


def _award_upset_champion(upset_by_league: dict[str, dict[str, Any]], add_badge) -> None:
    for league_name, data in upset_by_league.items():
        for match in data.get("matches", []):
            match_id = match["match_id"]
            winner = match["winner_team"]
            p1, p2, p3, p4 = match["players"]
            if winner == 1:
                winners = (p1, p2)
            elif winner == 2:
                winners = (p3, p4)
            else:
                continue
            for pid in winners:
                add_badge(
                    pid,
                    "upset_champion",
                    context_type="league",
                    context_id=str(match_id),
                    match_id=str(match_id),
                    value_num=float(match["exceed"]),
                    value_json={"league": league_name},
                )


def _expected_share(team_avg: float, opp_avg: float) -> float:
    try:
        return 1.0 / (1.0 + 10 ** ((float(opp_avg) - float(team_avg)) / 400.0))
    except Exception:
        return 0.5


def _expected_margin(share: float, goal_points: int) -> float:
    score1, score2, margin = _expected_scoreline_from_share(share, goal_points=goal_points)
    return float(margin)


def _expected_scoreline_from_share(p: float, goal_points: int = 11) -> tuple[int, int, int]:
    if p is None:
        return goal_points, goal_points, 0

    p = max(0.0001, min(0.9999, float(p)))
    if abs(p - 0.5) < 1e-12:
        return goal_points, goal_points, 0

    if p > 0.5:
        opp = int(round(goal_points * (1.0 - p) / p))
        opp = max(0, min(goal_points, opp))
        return goal_points, opp, goal_points - opp

    me = int(round(goal_points * p / (1.0 - p)))
    me = max(0, min(goal_points, me))
    return me, goal_points, me - goal_points


def _safe_rating(value: Any, fallback: float | None) -> float:
    v = _safe_float(value)
    if v is None:
        return float(fallback or 1200.0)
    return float(v)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except Exception:
        return None


def _max_rolling_sum(values: Iterable[int], window: int) -> float | None:
    vals = list(values)
    if len(vals) < window or window <= 0:
        return None
    current = sum(vals[:window])
    best = current
    for i in range(window, len(vals)):
        current += vals[i] - vals[i - window]
        if current > best:
            best = current
    return float(best)
