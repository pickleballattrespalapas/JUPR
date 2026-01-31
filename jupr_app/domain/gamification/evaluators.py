from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
import logging
from typing import Any

import pandas as pd

from jupr_app.domain.awards import compute_top_performer_awards
from jupr_app.domain.gamification.badge_rules import build_player_match_facts
from jupr_app.domain.gamification.badge_types import BadgeCandidate, BadgeEvaluationContext
from jupr_app.domain.gamification.participation import compute_lifetime_games
from jupr_app.domain.gamification.top_performer_awards import (
    TOP_PERFORMER_BADGE_IDS,
    _build_league_standings,
    _min_games_for_league,
)
from jupr_app.domain.leagues import get_league_meta_row, is_league_ended
from jupr_app.domain.tournament_podium import build_tournament_podium_candidates


logger = logging.getLogger(__name__)


def _inactive(badge_id: str, reason: str) -> Iterable[BadgeCandidate]:
    logger.info("Badge evaluator inactive for %s: %s", badge_id, reason)
    return []


def _as_of_filter(df: pd.DataFrame, as_of: datetime | None) -> pd.DataFrame:
    if as_of is None or df.empty or "date_dt" not in df.columns:
        return df
    as_of_dt = pd.to_datetime(as_of, utc=True, errors="coerce")
    if pd.isna(as_of_dt):
        return df
    return df[df["date_dt"] <= as_of_dt].copy()


def _league_filter(df: pd.DataFrame, league_id: str | None) -> pd.DataFrame:
    if not league_id or df.empty or "league" not in df.columns:
        return df
    return df[df["league"].astype(str) == str(league_id)].copy()


def _participant_candidates(ctx: BadgeEvaluationContext, badge_id: str, threshold: int) -> Iterable[BadgeCandidate]:
    counts = compute_lifetime_games(ctx.ctx)
    if not counts:
        return []
    candidates: list[BadgeCandidate] = []
    for player_id, games in counts.items():
        if int(games) < threshold:
            continue
        candidates.append(
            BadgeCandidate(
                badge_id=badge_id,
                player_id=int(player_id),
                club_id=ctx.club_id,
                context_type="overall",
                context_id=None,
                match_id=None,
                value_json={"games": int(games)},
                value_num=float(games),
            )
        )
    return candidates


def evaluate_participant(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return _participant_candidates(ctx, "participant", 1)


def evaluate_dedicated_participant_50(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return _participant_candidates(ctx, "dedicated_participant_50", 50)


def evaluate_lifetime_participant_200(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return _participant_candidates(ctx, "lifetime_participant_200", 200)


def evaluate_first_win(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    facts = _as_of_filter(ctx.facts, ctx.as_of)
    facts = _league_filter(facts, ctx.league_id)
    wins = facts[facts["win"] == True].copy()
    if wins.empty:
        return []
    first = wins.sort_values(["date_dt", "match_id"]).groupby("player_id", as_index=False).first()
    return [
        BadgeCandidate(
            badge_id="first_win",
            player_id=int(row.player_id),
            club_id=ctx.club_id,
            context_type="overall",
            context_id="first_win",
            match_id=str(row.match_id),
            value_json={"match_id": str(row.match_id)},
        )
        for row in first.itertuples(index=False)
    ]


def evaluate_weekly_regular(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    facts = _as_of_filter(ctx.facts, ctx.as_of)
    grouped = facts.groupby(["player_id", "league"])["week_key"].unique().reset_index()
    candidates: list[BadgeCandidate] = []
    for row in grouped.itertuples(index=False):
        weeks = sorted([w for w in row.week_key if w])
        if len(weeks) < 4:
            continue
        streak = _max_consecutive_weeks(weeks)
        if streak >= 4:
            year = weeks[-1].split("-W")[0]
            candidates.append(
                BadgeCandidate(
                    badge_id="weekly_regular",
                    player_id=int(row.player_id),
                    club_id=ctx.club_id,
                    context_type="league",
                    context_id=f"{row.league}:{year}",
                    match_id=None,
                    value_json={"league": row.league, "year": year},
                )
            )
    return candidates


def evaluate_iron_week(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    facts = _as_of_filter(ctx.facts, ctx.as_of)
    grouped = facts.groupby(["player_id", "week_key"])["league"].nunique().reset_index(name="leagues")
    candidates: list[BadgeCandidate] = []
    for row in grouped.itertuples(index=False):
        if int(row.leagues) >= 3:
            candidates.append(
                BadgeCandidate(
                    badge_id="iron_week",
                    player_id=int(row.player_id),
                    club_id=ctx.club_id,
                    context_type="week",
                    context_id=f"{row.week_key}",
                    match_id=None,
                    value_json={"week": row.week_key, "leagues": int(row.leagues)},
                )
            )
    return candidates


def evaluate_marathon_month(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    facts = _as_of_filter(ctx.facts, ctx.as_of)
    grouped = facts.groupby(["player_id", "league", "month_key"]).size().reset_index(name="matches")
    candidates: list[BadgeCandidate] = []
    for row in grouped.itertuples(index=False):
        if int(row.matches) >= 40:
            candidates.append(
                BadgeCandidate(
                    badge_id="marathon_month",
                    player_id=int(row.player_id),
                    club_id=ctx.club_id,
                    context_type="month",
                    context_id=f"{row.league}:{row.month_key}",
                    match_id=None,
                    value_json={"league": row.league, "month": row.month_key, "matches": int(row.matches)},
                )
            )
    return candidates


def evaluate_level_up(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    df_leagues = getattr(ctx.ctx, "df_leagues", None)
    if df_leagues is None or df_leagues.empty:
        return []
    if "league_name" not in df_leagues.columns or "player_id" not in df_leagues.columns:
        return []
    df = df_leagues.copy()
    df["league_name"] = df["league_name"].fillna("").astype(str).str.strip()
    df["rating"] = pd.to_numeric(df.get("rating", 0.0), errors="coerce").fillna(0.0)
    milestones = [3.0, 3.5, 4.0, 4.5, 5.0]
    candidates: list[BadgeCandidate] = []
    awarded: set[tuple[int, float]] = set()
    for row in df.itertuples(index=False):
        if not row.league_name:
            continue
        if ctx.league_id and str(row.league_name) != str(ctx.league_id):
            continue
        for milestone in milestones:
            key = (int(row.player_id), float(milestone))
            if key in awarded:
                continue
            if float(row.rating) >= milestone:
                awarded.add(key)
                candidates.append(
                    BadgeCandidate(
                        badge_id="level_up",
                        player_id=int(row.player_id),
                        club_id=ctx.club_id,
                        context_type="league",
                        context_id=f"milestone:{milestone}",
                        match_id=None,
                        value_json={"league": row.league_name, "milestone": milestone},
                    )
                )
    return candidates


def evaluate_rocket_start(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    facts = _as_of_filter(ctx.facts, ctx.as_of).sort_values(["date_dt", "match_id"])
    candidates: list[BadgeCandidate] = []
    for (player_id, league), group in facts.groupby(["player_id", "league"]):
        if ctx.league_id and str(league) != str(ctx.league_id):
            continue
        head = group.head(5)
        if len(head) < 5:
            continue
        wins = int(head["win"].sum())
        if wins >= 4:
            candidates.append(
                BadgeCandidate(
                    badge_id="rocket_start",
                    player_id=int(player_id),
                    club_id=ctx.club_id,
                    context_type="league",
                    context_id=f"{league}:rocket_start",
                    match_id=None,
                    value_json={"league": league},
                )
            )
    return candidates


def evaluate_most_improved_monthly(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    facts = _as_of_filter(ctx.facts, ctx.as_of)
    if facts["elo_delta_signed"].dropna().empty:
        return []
    monthly = facts.groupby(["league", "month_key", "player_id"])["elo_delta_signed"].sum().reset_index()
    if monthly.empty:
        return []
    winners = monthly.sort_values(["league", "month_key", "elo_delta_signed"], ascending=[True, True, False])
    candidates: list[BadgeCandidate] = []
    for (league, month_key), group in winners.groupby(["league", "month_key"]):
        if ctx.league_id and str(league) != str(ctx.league_id):
            continue
        top = group.iloc[0]
        if pd.isna(top["elo_delta_signed"]) or float(top["elo_delta_signed"]) <= 0:
            continue
        candidates.append(
            BadgeCandidate(
                badge_id="most_improved_monthly",
                player_id=int(top["player_id"]),
                club_id=ctx.club_id,
                context_type="month",
                context_id=f"{league}:month:{month_key}",
                match_id=None,
                value_json={"league": league, "month": month_key},
                value_num=float(top["elo_delta_signed"]),
            )
        )
    return candidates


def evaluate_mountain_climber(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    df_leagues = getattr(ctx.ctx, "df_leagues", None)
    if df_leagues is None or df_leagues.empty:
        return []
    if "league_name" not in df_leagues.columns or "player_id" not in df_leagues.columns:
        return []
    df = df_leagues.copy()
    df["league_name"] = df["league_name"].fillna("").astype(str).str.strip()
    df["rating"] = pd.to_numeric(df.get("rating", 1200.0), errors="coerce").fillna(1200.0)
    df["starting_rating"] = pd.to_numeric(df.get("starting_rating", df["rating"]), errors="coerce").fillna(df["rating"])
    candidates: list[BadgeCandidate] = []
    for league_name, league_df in df.groupby("league_name"):
        if not league_name:
            continue
        if ctx.league_id and str(league_name) != str(ctx.league_id):
            continue
        start_sorted = league_df.sort_values("starting_rating", ascending=False).reset_index(drop=True)
        start_sorted["start_rank"] = start_sorted.index + 1
        current_sorted = league_df.sort_values("rating", ascending=False).reset_index(drop=True)
        current_sorted["current_rank"] = current_sorted.index + 1
        ranks = start_sorted[["player_id", "start_rank"]].merge(
            current_sorted[["player_id", "current_rank"]], on="player_id", how="inner"
        )
        for _, row in ranks.iterrows():
            rank_delta = int(row["start_rank"] - row["current_rank"])
            for tier in (5, 10, 20):
                if rank_delta >= tier:
                    candidates.append(
                        BadgeCandidate(
                            badge_id="mountain_climber",
                            player_id=int(row["player_id"]),
                            club_id=ctx.club_id,
                            context_type="league",
                            context_id=f"{league_name}:tier:{tier}",
                            match_id=None,
                            value_json={"league": league_name, "tier": tier, "rank_delta": rank_delta},
                            value_num=float(rank_delta),
                        )
                    )
    return candidates


def evaluate_hot_streak(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    facts = _as_of_filter(ctx.facts, ctx.as_of).sort_values(["date_dt", "match_id"])
    candidates: list[BadgeCandidate] = []
    for (player_id, league), group in facts.groupby(["player_id", "league"]):
        if ctx.league_id and str(league) != str(ctx.league_id):
            continue
        streak = 0
        for row in group.itertuples(index=False):
            if row.win:
                streak += 1
                for tier in (5, 10, 20):
                    if streak == tier:
                        candidates.append(
                            BadgeCandidate(
                                badge_id="hot_streak",
                                player_id=int(player_id),
                                club_id=ctx.club_id,
                                context_type="league",
                                context_id=f"{league}:streak:{tier}:{row.match_id}",
                                match_id=str(row.match_id),
                                value_json={"league": league, "streak": streak, "tier": tier},
                                value_num=float(streak),
                            )
                        )
            else:
                streak = 0
    return candidates


def evaluate_bounce_back(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    facts = _as_of_filter(ctx.facts, ctx.as_of).sort_values(["date_dt", "match_id"])
    candidates: list[BadgeCandidate] = []
    for player_id, group in facts.groupby("player_id"):
        prev_win = None
        for row in group.itertuples(index=False):
            if prev_win is False and row.win:
                candidates.append(
                    BadgeCandidate(
                        badge_id="bounce_back",
                        player_id=int(player_id),
                        club_id=ctx.club_id,
                        context_type="match",
                        context_id=f"{row.match_id}:bounce_back",
                        match_id=str(row.match_id),
                        value_json={"match_id": str(row.match_id)},
                    )
                )
            prev_win = bool(row.win)
    return candidates


def evaluate_clutch_performer(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return _inactive("clutch_performer", "missing explicit clutch performance schema")


def evaluate_ice_in_veins(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    facts = _as_of_filter(ctx.facts, ctx.as_of)
    clutch = facts[(facts["win"] == True) & (facts["margin"].abs() <= 2)]
    clutch = clutch[clutch["expected_win_prob"] <= 0.4]
    if clutch.empty:
        return []
    first = clutch.sort_values(["date_dt", "match_id"]).groupby("player_id", as_index=False).first()
    return [
        BadgeCandidate(
            badge_id="ice_in_veins",
            player_id=int(row.player_id),
            club_id=ctx.club_id,
            context_type="overall",
            context_id="ice_in_veins",
            match_id=str(row.match_id),
            value_json={"match_id": str(row.match_id)},
        )
        for row in first.itertuples(index=False)
    ]


def evaluate_pickle_perfection(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    facts = _as_of_filter(ctx.facts, ctx.as_of)
    shutouts = facts[(facts["win"] == True) & (facts["points_against"] == 0)]
    return [
        BadgeCandidate(
            badge_id="pickle_perfection",
            player_id=int(row.player_id),
            club_id=ctx.club_id,
            context_type="match",
            context_id=f"{row.match_id}:pickle_perfection",
            match_id=str(row.match_id),
            value_json={"match_id": str(row.match_id), "score_against": int(row.points_against)},
        )
        for row in shutouts.itertuples(index=False)
    ]


def evaluate_blowout_artist(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    facts = _as_of_filter(ctx.facts, ctx.as_of)
    blowouts = facts[(facts["win"] == True) & (facts["margin"] >= 8)]
    return [
        BadgeCandidate(
            badge_id="blowout_artist",
            player_id=int(row.player_id),
            club_id=ctx.club_id,
            context_type="match",
            context_id=f"{row.match_id}:blowout",
            match_id=str(row.match_id),
            value_json={"match_id": str(row.match_id), "margin": int(row.margin)},
        )
        for row in blowouts.itertuples(index=False)
    ]


def evaluate_untouchable(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    facts = _as_of_filter(ctx.facts, ctx.as_of).sort_values(["date_dt", "match_id"])
    candidates: list[BadgeCandidate] = []
    base_streak = 20
    for player_id, group in facts.groupby("player_id"):
        streak = 0
        for row in group.itertuples(index=False):
            if row.win:
                streak += 1
                if streak >= base_streak:
                    candidates.append(
                        BadgeCandidate(
                            badge_id="untouchable",
                            player_id=int(player_id),
                            club_id=ctx.club_id,
                            context_type="overall",
                            context_id=f"window_end:{row.match_id}:streak:{streak}",
                            match_id=str(row.match_id),
                            value_json={"streak": streak, "match_id": str(row.match_id)},
                        )
                    )
            else:
                streak = 0
    return candidates


def evaluate_clean_sweep_week(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    facts = _as_of_filter(ctx.facts, ctx.as_of)
    grouped = facts.groupby(["player_id", "week_key"])
    candidates: list[BadgeCandidate] = []
    for (player_id, week_key), group in grouped:
        distinct_leagues = group["league"].nunique()
        if distinct_leagues < 2:
            continue
        if group["win"].all():
            candidates.append(
                BadgeCandidate(
                    badge_id="clean_sweep_week",
                    player_id=int(player_id),
                    club_id=ctx.club_id,
                    context_type="week",
                    context_id=f"{week_key}",
                    match_id=None,
                    value_json={"week": week_key, "leagues": int(distinct_leagues), "matches": int(len(group))},
                )
            )
    return candidates


def evaluate_high_roller(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    facts = _as_of_filter(ctx.facts, ctx.as_of)
    wins = facts[facts["win"] == True].dropna(subset=["match_id"])
    grouped = wins.groupby("player_id")["match_id"].nunique()
    candidates: list[BadgeCandidate] = []
    for player_id, win_count in grouped.items():
        if int(win_count) >= 100:
            candidates.append(
                BadgeCandidate(
                    badge_id="high_roller",
                    player_id=int(player_id),
                    club_id=ctx.club_id,
                    context_type="overall",
                    context_id="lifetime_wins_100",
                    match_id=None,
                    value_json={"wins": int(win_count)},
                    value_num=float(win_count),
                )
            )
    return candidates


def evaluate_social_butterfly(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    facts = _as_of_filter(ctx.facts, ctx.as_of)
    partners = facts.dropna(subset=["partner_id"]).groupby("player_id")["partner_id"].nunique()
    candidates: list[BadgeCandidate] = []
    for player_id, count in partners.items():
        if count >= 20:
            candidates.append(
                BadgeCandidate(
                    badge_id="social_butterfly",
                    player_id=int(player_id),
                    club_id=ctx.club_id,
                    context_type="overall",
                    context_id="milestone:20_partners",
                    match_id=None,
                    value_json={"partners": int(count)},
                )
            )
    return candidates


def evaluate_network_builder(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    facts = _as_of_filter(ctx.facts, ctx.as_of)
    partners = facts.dropna(subset=["partner_id"]).groupby("player_id")["partner_id"].nunique()
    candidates: list[BadgeCandidate] = []
    for player_id, count in partners.items():
        if count >= 50:
            candidates.append(
                BadgeCandidate(
                    badge_id="network_builder",
                    player_id=int(player_id),
                    club_id=ctx.club_id,
                    context_type="overall",
                    context_id="milestone:50_partners",
                    match_id=None,
                    value_json={"partners": int(count)},
                )
            )
    return candidates


def evaluate_draft_master(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    facts = _as_of_filter(ctx.facts, ctx.as_of)
    wins = facts[(facts["win"] == True) & facts["partner_id"].notna()]
    grouped = wins.groupby(["player_id", "week_key"])["partner_id"].nunique().reset_index()
    candidates: list[BadgeCandidate] = []
    for row in grouped.itertuples(index=False):
        if int(row.partner_id) >= 5:
            candidates.append(
                BadgeCandidate(
                    badge_id="draft_master",
                    player_id=int(row.player_id),
                    club_id=ctx.club_id,
                    context_type="week",
                    context_id=f"{row.week_key}",
                    match_id=None,
                    value_json={"week": row.week_key, "partners": int(row.partner_id)},
                )
            )
    return candidates


def evaluate_swiss_army_knife(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    facts = _as_of_filter(ctx.facts, ctx.as_of)
    wins = facts[facts["win"] == True]
    grouped = wins.groupby(["player_id", "season_key"])["league"].nunique().reset_index()
    candidates: list[BadgeCandidate] = []
    for row in grouped.itertuples(index=False):
        if int(row.league) >= 3:
            candidates.append(
                BadgeCandidate(
                    badge_id="swiss_army_knife",
                    player_id=int(row.player_id),
                    club_id=ctx.club_id,
                    context_type="season",
                    context_id=str(row.season_key),
                    match_id=None,
                    value_json={"season": str(row.season_key), "leagues": int(row.league)},
                )
            )
    return candidates


def evaluate_giant_slayer(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    facts = _as_of_filter(ctx.facts, ctx.as_of)
    wins = facts[(facts["win"] == True) & facts["opp_max_rating"].notna()]
    min_opponent_rating = 2000
    return [
        BadgeCandidate(
            badge_id="giant_slayer",
            player_id=int(row.player_id),
            club_id=ctx.club_id,
            context_type="match",
            context_id=f"{row.match_id}:min_rating:{min_opponent_rating}",
            match_id=str(row.match_id),
            value_json={"match_id": str(row.match_id), "min_opponent_rating": min_opponent_rating},
        )
        for row in wins.itertuples(index=False)
        if float(row.opp_max_rating) >= min_opponent_rating
    ]


def evaluate_david_vs_goliath(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    facts = _as_of_filter(ctx.facts, ctx.as_of)
    wins = facts[(facts["win"] == True) & (facts["expected_win_prob"] <= 0.25)]
    return [
        BadgeCandidate(
            badge_id="david_vs_goliath",
            player_id=int(row.player_id),
            club_id=ctx.club_id,
            context_type="match",
            context_id=f"{row.match_id}:david_vs_goliath",
            match_id=str(row.match_id),
            value_json={"match_id": str(row.match_id), "expected_prob": float(row.expected_win_prob)},
        )
        for row in wins.itertuples(index=False)
    ]


def evaluate_upset_champion(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    facts = _as_of_filter(ctx.facts, ctx.as_of)
    winners = facts[facts["win"] == True].copy()
    if winners.empty:
        return []
    match_stats = (
        winners.groupby(["league", "month_key", "match_id"])
        .agg(expected_win_prob=("expected_win_prob", "min"), player_ids=("player_id", lambda x: list(x)))
        .reset_index()
    )
    match_stats = match_stats.sort_values(["league", "month_key", "expected_win_prob"])
    candidates: list[BadgeCandidate] = []
    for (league, month_key), group in match_stats.groupby(["league", "month_key"]):
        if ctx.league_id and str(league) != str(ctx.league_id):
            continue
        top = group.iloc[0]
        for pid in top.player_ids:
            candidates.append(
                BadgeCandidate(
                    badge_id="upset_champion",
                    player_id=int(pid),
                    club_id=ctx.club_id,
                    context_type="month",
                    context_id=f"{league}:month:{month_key}:match:{top.match_id}",
                    match_id=str(top.match_id),
                    value_num=float(top.expected_win_prob),
                    value_json={"league": league, "month": month_key, "expected_prob": float(top.expected_win_prob)},
                )
            )
    return candidates


def evaluate_hall_of_fame_night(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    facts = _as_of_filter(ctx.facts, ctx.as_of)
    if facts["abs_elo_delta"].dropna().empty:
        return []
    candidates: list[BadgeCandidate] = []
    for league, group in facts.groupby("league"):
        if ctx.league_id and str(league) != str(ctx.league_id):
            continue
        values = group["abs_elo_delta"].dropna()
        if values.empty:
            continue
        threshold = values.quantile(0.95)
        heroes = group[group["abs_elo_delta"] >= threshold]
        for row in heroes.itertuples(index=False):
            candidates.append(
                BadgeCandidate(
                    badge_id="hall_of_fame_night",
                    player_id=int(row.player_id),
                    club_id=ctx.club_id,
                    context_type="league",
                    context_id=f"{league}:hall_of_fame:{row.match_id}",
                    match_id=str(row.match_id),
                    value_json={"league": league, "abs_delta": float(row.abs_elo_delta)},
                )
            )
    return candidates


def evaluate_legendary_upset(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    facts = _as_of_filter(ctx.facts, ctx.as_of)
    wins = facts[(facts["win"] == True) & (facts["expected_win_prob"] <= 0.1)]
    return [
        BadgeCandidate(
            badge_id="legendary_upset",
            player_id=int(row.player_id),
            club_id=ctx.club_id,
            context_type="match",
            context_id=f"{row.match_id}:legendary_upset",
            match_id=str(row.match_id),
            value_json={"match_id": str(row.match_id), "expected_prob": float(row.expected_win_prob)},
        )
        for row in wins.itertuples(index=False)
    ]


def evaluate_nemesis_found(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return _inactive("nemesis_found", "missing opponent match history aggregation")


def evaluate_rivalry_win(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return _inactive("rivalry_win", "missing opponent match history aggregation")


def evaluate_rivalry_streak(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return _inactive("rivalry_streak", "missing opponent match history aggregation")


def evaluate_settled_the_score(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return _inactive("settled_the_score", "missing opponent match history aggregation")


def evaluate_steady_hand(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    facts = _as_of_filter(ctx.facts, ctx.as_of)
    grouped = facts.groupby(["player_id", "season_key"])
    candidates: list[BadgeCandidate] = []
    for (player_id, season_key), group in grouped:
        matches = len(group)
        if matches < 20:
            continue
        win_pct = float(group["win"].mean())
        if win_pct >= 0.6:
            candidates.append(
                BadgeCandidate(
                    badge_id="steady_hand",
                    player_id=int(player_id),
                    club_id=ctx.club_id,
                    context_type="season",
                    context_id=str(season_key),
                    match_id=None,
                    value_json={"season": str(season_key), "win_pct": win_pct},
                )
            )
    return candidates


def evaluate_mr_reliable(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return _inactive("mr_reliable", "missing availability tracking by season")


def evaluate_league_champion(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return _inactive("league_champion", "missing league champion records")


def evaluate_podium(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return _inactive("podium", "missing league podium results")


def evaluate_good_sport(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return _inactive("good_sport", "missing sportsmanship data")


def evaluate_community_builder(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return _inactive("community_builder", "missing community engagement data")


def evaluate_mentor(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return _inactive("mentor", "missing mentorship tracking")


def evaluate_breakthrough(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return _inactive("breakthrough", "missing explicit breakthrough criteria")


def evaluate_above_expectations(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return _inactive("above_expectations", "missing expected performance baseline")


def evaluate_dominant_run(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return _inactive("dominant_run", "missing dominant run criteria")


def evaluate_high_output(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return _inactive("high_output", "missing high output definition")


def evaluate_battle_tested(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return _inactive("battle_tested", "missing battle tested criteria")


def evaluate_consistency(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return _inactive("consistency", "missing consistency definition")


def _cached_top_performer_candidates(ctx: BadgeEvaluationContext) -> list[BadgeCandidate]:
    key = str(ctx.league_id or "__all__")
    cache = getattr(ctx.ctx, "_top_performer_candidates_cache", None)
    if isinstance(cache, dict) and cache.get("key") == key:
        return cache.get("candidates", [])

    df_leagues = getattr(ctx.ctx, "df_leagues", None)
    df_meta = getattr(ctx.ctx, "df_meta", None)
    id_to_name = getattr(ctx.ctx, "id_to_name", {}) or {}
    if df_leagues is None or df_leagues.empty:
        candidates: list[BadgeCandidate] = []
        setattr(ctx.ctx, "_top_performer_candidates_cache", {"key": key, "candidates": candidates})
        return candidates

    league_ids = (
        [ctx.league_id]
        if ctx.league_id
        else sorted(df_leagues["league_name"].dropna().astype(str).unique().tolist())
    )
    candidates = []
    for league_id in league_ids:
        meta_row = get_league_meta_row(df_meta, league_id)
        if not is_league_ended(meta_row):
            continue
        min_games = _min_games_for_league(df_meta, league_id)
        if min_games <= 0:
            continue
        standings = _build_league_standings(df_leagues, league_id, id_to_name)
        if standings.empty:
            continue
        qualified = standings[standings["matches_played"] >= int(min_games)].copy()
        if qualified.empty:
            continue
        awards = compute_top_performer_awards(qualified, min_games=min_games, winners_per_category=1)
        for award in awards:
            category_key = award.get("category_key")
            badge_id = TOP_PERFORMER_BADGE_IDS.get(str(category_key))
            if not badge_id:
                continue
            rank = int(award.get("rank", 1))
            context_id = f"{league_id}:top_performer:{category_key}:{rank}"
            value_json = {
                "league_id": league_id,
                "category_key": category_key,
                "category_label": award.get("category_label"),
                "metric_value": award.get("metric_value"),
                "metric_display": award.get("metric_display"),
                "min_games": int(min_games),
                "rank": rank,
            }
            candidates.append(
                BadgeCandidate(
                    badge_id=badge_id,
                    player_id=int(award["player_id"]),
                    club_id=ctx.club_id,
                    context_type="league",
                    context_id=context_id,
                    match_id=None,
                    value_json=value_json,
                    value_num=award.get("metric_value") if award.get("metric_value") is not None else None,
                )
            )

    setattr(ctx.ctx, "_top_performer_candidates_cache", {"key": key, "candidates": candidates})
    return candidates


def _cached_tournament_podium_candidates(ctx: BadgeEvaluationContext) -> list[BadgeCandidate]:
    tournament_id = getattr(ctx.ctx, "tournament_id", None)
    tournament_name = getattr(ctx.ctx, "tournament_name", None)
    if not tournament_id:
        return []
    cache = getattr(ctx.ctx, "_tournament_podium_candidates_cache", None)
    key = (str(tournament_id), str(tournament_name or ""))
    if isinstance(cache, dict) and cache.get("key") == key:
        return cache.get("candidates", [])
    candidates = build_tournament_podium_candidates(ctx.ctx, str(tournament_id), tournament_name)
    setattr(ctx.ctx, "_tournament_podium_candidates_cache", {"key": key, "candidates": candidates})
    return candidates


def evaluate_top_performer_highest_rating(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return [c for c in _cached_top_performer_candidates(ctx) if c.badge_id == "top_performer_highest_rating"]


def evaluate_top_performer_most_improved(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return [c for c in _cached_top_performer_candidates(ctx) if c.badge_id == "top_performer_most_improved"]


def evaluate_top_performer_best_win_pct(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return [c for c in _cached_top_performer_candidates(ctx) if c.badge_id == "top_performer_best_win_pct"]


def evaluate_top_performer_most_wins(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return [c for c in _cached_top_performer_candidates(ctx) if c.badge_id == "top_performer_most_wins"]


def evaluate_tournament_champion(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return [c for c in _cached_tournament_podium_candidates(ctx) if c.badge_id == "tournament_champion"]


def evaluate_tournament_runner_up(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return [c for c in _cached_tournament_podium_candidates(ctx) if c.badge_id == "tournament_runner_up"]


def evaluate_tournament_third_place(ctx: BadgeEvaluationContext) -> Iterable[BadgeCandidate]:
    return [c for c in _cached_tournament_podium_candidates(ctx) if c.badge_id == "tournament_third_place"]


def build_evaluation_context(ctx: Any, club_id: str, league_id: str | None, as_of: datetime | None) -> BadgeEvaluationContext:
    facts = build_player_match_facts(ctx)
    if facts.empty:
        matches = getattr(ctx, "df_matches", None)
        if matches is None:
            matches = pd.DataFrame()
    else:
        matches = facts
    if league_id:
        facts = _league_filter(facts, league_id)
    if as_of:
        facts = _as_of_filter(facts, as_of)
    return BadgeEvaluationContext(
        club_id=club_id,
        league_id=league_id,
        as_of=as_of,
        ctx=ctx,
        facts=facts,
        matches=matches,
    )


def _max_consecutive_weeks(weeks: list[str]) -> int:
    if not weeks:
        return 0
    parsed = []
    for week in weeks:
        try:
            year_str, week_str = week.split("-W")
            parsed.append((int(year_str), int(week_str)))
        except Exception:
            continue
    if not parsed:
        return 0
    parsed = sorted(set(parsed))
    max_run = 1
    run = 1
    for (y1, w1), (y2, w2) in zip(parsed, parsed[1:]):
        next_week = w1 + 1
        next_year = y1
        if w1 >= 52:
            next_week = 1
            next_year = y1 + 1
        if (y2, w2) == (next_year, next_week):
            run += 1
        else:
            run = 1
        max_run = max(max_run, run)
    return max_run
