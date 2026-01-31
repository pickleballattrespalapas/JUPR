from __future__ import annotations

# DEPRECATED: do not use; the badge engine/registry is the single source of truth.

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
from typing import Any
from uuid import uuid4

import pandas as pd

from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS
from jupr_app.domain.gamification.copy_pack import (
    get_badge_copy,
    load_copy_pack,
    pick_variant,
    render_template,
)
from jupr_app.domain.match_filters import apply_match_filters, normalize_player_id, normalize_score

logger = logging.getLogger(__name__)


_FACT_COLUMNS = [
    "club_id",
    "player_id",
    "match_id",
    "league",
    "date_dt",
    "week_key",
    "month_key",
    "season_key",
    "win",
    "points_for",
    "points_against",
    "margin",
    "partner_id",
    "opponent_ids",
    "expected_win_prob",
    "elo_delta_signed",
    "abs_elo_delta",
    "opp_max_rating",
    "lobby_avg_rating",
]


def _empty_facts() -> pd.DataFrame:
    return pd.DataFrame(columns=_FACT_COLUMNS)


@dataclass(frozen=True)
class BadgeAward:
    player_id: int
    badge_id: str
    context_type: str
    context_id: str | None
    match_id: str | None
    value_num: float | None
    value_json: dict[str, Any] | None


def ensure_badges(ctx) -> None:
    """DEPRECATED: use jupr_app.domain.gamification.ensure_badges.ensure_badges."""
    from jupr_app.domain.gamification.ensure_badges import ensure_badges as ensure_engine_badges

    ensure_engine_badges(ctx)


def compute_badge_awards(
    ctx,
    existing: set[tuple[str, str, str | None]] | None = None,
) -> tuple[list[BadgeAward], pd.DataFrame]:
    existing = existing or set()
    facts = build_player_match_facts(ctx)
    if facts.empty:
        return [], facts

    awards: list[BadgeAward] = []
    now = datetime.now(timezone.utc)
    badge_name_map = {b.badge_id: b.name for b in BADGE_DEFINITIONS}

    def add_badge(
        player_id: int,
        badge_id: str,
        *,
        context_type: str,
        context_id: str | None,
        match_id: str | None = None,
        value_num: float | None = None,
        value_json: dict[str, Any] | None = None,
    ) -> None:
        key = (str(player_id), badge_id, str(context_id) if context_id is not None else None)
        if key in existing:
            return
        existing.add(key)
        badge_name = badge_name_map.get(badge_id, badge_id)
        data = dict(value_json or {})
        data.setdefault("badge_name", badge_name)
        data.setdefault("badge_id", badge_id)
        seed = f"{player_id}:{badge_id}:{context_id}"
        if not data.get("tape_excerpt"):
            excerpt = _build_tape_excerpt(badge_id, seed, data)
            if excerpt:
                data["tape_excerpt"] = excerpt
        if not data.get("tape_title"):
            title = _build_tape_title(badge_id, seed, data)
            if title:
                data["tape_title"] = title
        awards.append(
            BadgeAward(
                player_id=int(player_id),
                badge_id=badge_id,
                context_type=context_type,
                context_id=context_id,
                match_id=match_id,
                value_num=value_num,
                value_json=data,
            )
        )

    _award_first_win(facts, add_badge, now)
    _award_weekly_regular(facts, add_badge)
    _award_iron_week(facts, add_badge)
    _award_marathon_month(facts, add_badge)
    _award_level_up(ctx, add_badge)
    _award_rocket_start(facts, add_badge)
    _award_most_improved(facts, add_badge)
    _award_mountain_climber(ctx, add_badge)
    _award_hot_streaks(facts, add_badge)
    _award_bounce_back(facts, add_badge)
    _award_ice_in_veins(facts, add_badge)
    _award_pickle_perfection(facts, add_badge)
    _award_blowout_artist(facts, add_badge)
    _award_untouchable(facts, add_badge)
    _award_clean_sweep_week(facts, add_badge)
    _award_high_roller(facts, add_badge)
    _award_social_graph(facts, add_badge)
    _award_draft_master(facts, add_badge)
    _award_swiss_army_knife(facts, add_badge)
    _award_giant_slayer(facts, add_badge)
    _award_david_vs_goliath(facts, add_badge)
    _award_upset_champion(facts, add_badge)
    _award_legendary_upset(facts, add_badge)
    _award_rivalries(facts, add_badge)
    _award_steady_hand(facts, add_badge)
    _award_hall_of_fame_night(facts, add_badge)

    return awards, facts


def build_player_match_facts(ctx) -> pd.DataFrame:
    df_matches = getattr(ctx, "df_matches", None)
    if df_matches is None or df_matches.empty:
        return _empty_facts()

    df_players = getattr(ctx, "df_players_all", None)
    rating_map: dict[int, float] = {}
    if df_players is not None and not df_players.empty:
        try:
            rating_map = dict(zip(df_players["id"].astype(int), df_players["rating"].astype(float)))
        except Exception:
            rating_map = {}

    filters = {"club_id": getattr(ctx, "club_id", None), "exclude_popups": True}
    filtered = apply_match_filters(df_matches, filters)
    if filtered.empty:
        return _empty_facts()

    filtered = filtered.copy()
    if "id" not in filtered.columns:
        filtered["id"] = range(1, len(filtered) + 1)
    filtered["date_dt"] = pd.to_datetime(filtered.get("date", None), utc=True, errors="coerce")
    filtered = filtered.dropna(subset=["date_dt"]).sort_values(["date_dt", "id"], ascending=[True, True])

    records: list[dict[str, Any]] = []
    for row in filtered.itertuples(index=False):
        match_id = str(getattr(row, "id", "") or "")
        league = str(getattr(row, "league", "") or "").strip() or "OVERALL"
        club_id = str(getattr(row, "club_id", "") or "")
        date_dt = getattr(row, "date_dt", pd.NaT)
        if not match_id or pd.isna(date_dt):
            continue

        s1 = normalize_score(getattr(row, "score_t1", None))
        s2 = normalize_score(getattr(row, "score_t2", None))
        if (s1 + s2) <= 0:
            continue

        p1 = normalize_player_id(getattr(row, "t1_p1", None))
        p2 = normalize_player_id(getattr(row, "t1_p2", None))
        p3 = normalize_player_id(getattr(row, "t2_p1", None))
        p4 = normalize_player_id(getattr(row, "t2_p2", None))
        if not p1 or not p3:
            continue

        r1 = _safe_rating(getattr(row, "t1_p1_r", None), rating_map.get(p1))
        r2 = _safe_rating(getattr(row, "t1_p2_r", None), rating_map.get(p2))
        r3 = _safe_rating(getattr(row, "t2_p1_r", None), rating_map.get(p3))
        r4 = _safe_rating(getattr(row, "t2_p2_r", None), rating_map.get(p4))

        team1 = [pid for pid in (p1, p2) if pid]
        team2 = [pid for pid in (p3, p4) if pid]
        if not team1 or not team2:
            continue

        t1_avg = _avg([r1, r2] if p2 else [r1])
        t2_avg = _avg([r3, r4] if p4 else [r3])
        expected_t1 = _expected_share(t1_avg, t2_avg)

        winner_team = 1 if s1 > s2 else 2 if s2 > s1 else 0
        delta_abs = _safe_float(getattr(row, "elo_delta", None))

        lobby_avg_rating = _avg([r for r in [r1, r2, r3, r4] if r is not None])

        for pid, team, partner, opp_ids, my_score, opp_score, opp_avg, opp_max, expected_win in (
            (p1, 1, p2, team2, s1, s2, _avg([r3, r4] if p4 else [r3]), _max_rating([r3, r4]), expected_t1),
            (p2, 1, p1, team2, s1, s2, _avg([r3, r4] if p4 else [r3]), _max_rating([r3, r4]), expected_t1),
            (p3, 2, p4, team1, s2, s1, _avg([r1, r2] if p2 else [r1]), _max_rating([r1, r2]), 1.0 - expected_t1),
            (p4, 2, p3, team1, s2, s1, _avg([r1, r2] if p2 else [r1]), _max_rating([r1, r2]), 1.0 - expected_t1),
        ):
            if not pid:
                continue
            win = winner_team == team
            signed_delta = None
            if delta_abs is not None:
                signed_delta = float(delta_abs) if win else -float(delta_abs)
            records.append(
                {
                    "club_id": club_id,
                    "player_id": int(pid),
                    "match_id": match_id,
                    "league": league,
                    "date_dt": date_dt,
                    "week_key": _week_key(date_dt),
                    "month_key": _month_key(date_dt),
                    "season_key": _season_key(date_dt),
                    "win": bool(win),
                    "points_for": int(my_score),
                    "points_against": int(opp_score),
                    "margin": int(my_score - opp_score),
                    "partner_id": int(partner) if partner else None,
                    "opponent_ids": [int(x) for x in opp_ids if x],
                    "expected_win_prob": float(expected_win),
                    "elo_delta_signed": signed_delta,
                    "abs_elo_delta": abs(delta_abs) if delta_abs is not None else None,
                    "opp_max_rating": float(opp_max) if opp_max is not None else None,
                    "lobby_avg_rating": float(lobby_avg_rating) if lobby_avg_rating is not None else None,
                }
            )

    return pd.DataFrame.from_records(records)


def _seed_badges(supabase) -> None:
    force_backfill = str(os.getenv("FORCE_COPY_BACKFILL", "") or "").lower() in {"1", "true", "yes"}
    copy_pack = {b.badge_id: get_badge_copy(b.badge_id) for b in BADGE_DEFINITIONS}
    existing_by_id = {}
    if not force_backfill:
        try:
            resp = supabase.table("badges").select("badge_id").execute()
            existing_by_id = {str(row.get("badge_id")): row for row in resp.data or []}
        except Exception:
            logger.exception("Failed to fetch badges for copy backfill")

    payload = [
        {
            "badge_id": b.badge_id,
            "name": b.name,
            "prestige": b.prestige,
            "category": b.category,
            "is_stackable": b.is_stackable,
            "is_active": b.is_active,
            "rarity": copy_pack.get(b.badge_id, {}).get("rarity", b.rarity),
            "tier": copy_pack.get(b.badge_id, {}).get("tier", b.tier),
            "icon_key": copy_pack.get(b.badge_id, {}).get("icon_key", b.icon_key),
            "lore": _resolve_copy_text(
                b.badge_id,
                "lore",
                copy_pack.get(b.badge_id, {}),
                b.lore,
                existing_by_id.get(b.badge_id, {}),
                force_backfill,
            ),
            "hint": _resolve_copy_text(
                b.badge_id,
                "hint",
                copy_pack.get(b.badge_id, {}),
                b.hint,
                existing_by_id.get(b.badge_id, {}),
                force_backfill,
            ),
            "scope": copy_pack.get(b.badge_id, {}).get("scope", b.scope),
        }
        for b in BADGE_DEFINITIONS
    ]
    try:
        supabase.table("badges").upsert(payload, on_conflict="badge_id").execute()
        _backfill_copy_pack_badges(
            supabase,
            force_backfill=force_backfill,
            existing_by_id=existing_by_id,
        )
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


def _insert_badges(supabase, club_id: str, awards: list[BadgeAward]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for award in awards:
        rows.append(
            {
                "id": str(uuid4()),
                "club_id": club_id,
                "player_id": int(award.player_id),
                "badge_id": award.badge_id,
                "earned_at": now,
                "context_type": award.context_type,
                "context_id": award.context_id,
                "match_id": award.match_id,
                "value_num": award.value_num,
                "value_json": award.value_json,
            }
        )

    chunk = 200
    for i in range(0, len(rows), chunk):
        supabase.table("player_badges").upsert(
            rows[i : i + chunk],
            on_conflict="club_id,player_id,badge_id,context_id",
        ).execute()


def _award_first_win(facts: pd.DataFrame, add_badge, now: datetime) -> None:
    wins = facts[facts["win"] == True].copy()
    if wins.empty:
        return
    first = wins.sort_values(["date_dt", "match_id"]).groupby("player_id", as_index=False).first()
    for row in first.itertuples(index=False):
        add_badge(
            int(row.player_id),
            "first_win",
            context_type="overall",
            context_id="first_win",
            match_id=str(row.match_id),
            value_json={"match_id": str(row.match_id)},
        )


def _award_weekly_regular(facts: pd.DataFrame, add_badge) -> None:
    grouped = (
        facts.groupby(["player_id", "league"])["week_key"]
        .unique()
        .reset_index()
    )
    for row in grouped.itertuples(index=False):
        weeks = sorted([w for w in row.week_key if w])
        if len(weeks) < 4:
            continue
        streak = _max_consecutive_weeks(weeks)
        if streak >= 4:
            year = weeks[-1].split("-W")[0]
            add_badge(
                int(row.player_id),
                "weekly_regular",
                context_type="league",
                context_id=f"{row.league}:{year}",
                value_json={"league": row.league, "year": year},
            )


def _award_iron_week(facts: pd.DataFrame, add_badge) -> None:
    counts = (
        facts.groupby(["player_id", "league", "week_key"])
        .size()
        .reset_index(name="matches")
    )
    for row in counts.itertuples(index=False):
        if int(row.matches) >= 5:
            add_badge(
                int(row.player_id),
                "iron_week",
                context_type="week",
                context_id=f"{row.league}:{row.week_key}",
                value_json={"league": row.league, "week": row.week_key, "matches": int(row.matches)},
            )


def _award_marathon_month(facts: pd.DataFrame, add_badge) -> None:
    counts = (
        facts.groupby(["player_id", "league", "month_key"])
        .size()
        .reset_index(name="matches")
    )
    for row in counts.itertuples(index=False):
        if int(row.matches) >= 40:
            add_badge(
                int(row.player_id),
                "marathon_month",
                context_type="month",
                context_id=f"{row.league}:{row.month_key}",
                value_json={"league": row.league, "month": row.month_key, "matches": int(row.matches)},
            )


def _award_level_up(ctx, add_badge) -> None:
    df_leagues = getattr(ctx, "df_leagues", None)
    if df_leagues is None or df_leagues.empty:
        return
    if "league_name" not in df_leagues.columns or "player_id" not in df_leagues.columns:
        return
    df = df_leagues.copy()
    df["league_name"] = df["league_name"].fillna("").astype(str).str.strip()
    df["rating"] = pd.to_numeric(df.get("rating", 1200.0), errors="coerce").fillna(1200.0)
    milestones = [1400, 1600, 1800, 2000]
    for row in df.itertuples(index=False):
        if not row.league_name:
            continue
        for milestone in milestones:
            if float(row.rating) >= milestone:
                add_badge(
                    int(row.player_id),
                    "level_up",
                    context_type="league",
                    context_id=f"{row.league_name}:milestone:{milestone}",
                    value_json={"league": row.league_name, "milestone": milestone},
                )


def _award_rocket_start(facts: pd.DataFrame, add_badge) -> None:
    facts = facts.sort_values(["date_dt", "match_id"])
    for (player_id, league), group in facts.groupby(["player_id", "league"]):
        head = group.head(5)
        if len(head) < 5:
            continue
        wins = int(head["win"].sum())
        if wins >= 4:
            add_badge(
                int(player_id),
                "rocket_start",
                context_type="league",
                context_id=f"{league}:rocket_start",
                value_json={"league": league},
            )


def _award_most_improved(facts: pd.DataFrame, add_badge) -> None:
    if facts["elo_delta_signed"].dropna().empty:
        return
    monthly = (
        facts.groupby(["league", "month_key", "player_id"])["elo_delta_signed"]
        .sum()
        .reset_index()
    )
    if monthly.empty:
        return
    winners = monthly.sort_values(["league", "month_key", "elo_delta_signed"], ascending=[True, True, False])
    for (league, month_key), group in winners.groupby(["league", "month_key"]):
        top = group.iloc[0]
        if pd.isna(top["elo_delta_signed"]) or float(top["elo_delta_signed"]) <= 0:
            continue
        add_badge(
            int(top["player_id"]),
            "most_improved_monthly",
            context_type="month",
            context_id=f"{league}:month:{month_key}",
            value_num=float(top["elo_delta_signed"]),
            value_json={"league": league, "month": month_key},
        )


def _award_mountain_climber(ctx, add_badge) -> None:
    df_leagues = getattr(ctx, "df_leagues", None)
    if df_leagues is None or df_leagues.empty:
        return
    if "league_name" not in df_leagues.columns or "player_id" not in df_leagues.columns:
        return
    df = df_leagues.copy()
    df["league_name"] = df["league_name"].fillna("").astype(str).str.strip()
    df["rating"] = pd.to_numeric(df.get("rating", 1200.0), errors="coerce").fillna(1200.0)
    df["starting_rating"] = pd.to_numeric(df.get("starting_rating", df["rating"]), errors="coerce").fillna(df["rating"])
    for league_name, league_df in df.groupby("league_name"):
        if not league_name:
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
                    add_badge(
                        int(row["player_id"]),
                        "mountain_climber",
                        context_type="league",
                        context_id=f"{league_name}:tier:{tier}",
                        value_num=float(rank_delta),
                        value_json={"league": league_name, "tier": tier, "rank_delta": rank_delta},
                    )


def _award_hot_streaks(facts: pd.DataFrame, add_badge) -> None:
    facts = facts.sort_values(["date_dt", "match_id"])
    for (player_id, league), group in facts.groupby(["player_id", "league"]):
        streak = 0
        for row in group.itertuples(index=False):
            if row.win:
                streak += 1
                for tier in (5, 10, 20):
                    if streak == tier:
                        add_badge(
                            int(player_id),
                            "hot_streak",
                            context_type="league",
                            context_id=f"{league}:streak:{tier}:{row.match_id}",
                            match_id=str(row.match_id),
                            value_num=float(streak),
                            value_json={"league": league, "streak": streak, "tier": tier},
                        )
            else:
                streak = 0


def _award_bounce_back(facts: pd.DataFrame, add_badge) -> None:
    facts = facts.sort_values(["date_dt", "match_id"])
    for player_id, group in facts.groupby("player_id"):
        prev_win = None
        for row in group.itertuples(index=False):
            if prev_win is False and row.win:
                add_badge(
                    int(player_id),
                    "bounce_back",
                    context_type="match",
                    context_id=f"{row.match_id}:bounce_back",
                    match_id=str(row.match_id),
                    value_json={"match_id": str(row.match_id)},
                )
            prev_win = bool(row.win)


def _award_ice_in_veins(facts: pd.DataFrame, add_badge) -> None:
    clutch = facts[(facts["win"] == True) & (facts["margin"].abs() <= 2)]
    clutch = clutch[clutch["expected_win_prob"] <= 0.4]
    if clutch.empty:
        return
    first = clutch.sort_values(["date_dt", "match_id"]).groupby("player_id", as_index=False).first()
    for row in first.itertuples(index=False):
        add_badge(
            int(row.player_id),
            "ice_in_veins",
            context_type="overall",
            context_id="ice_in_veins",
            match_id=str(row.match_id),
            value_json={"match_id": str(row.match_id)},
        )


def _award_pickle_perfection(facts: pd.DataFrame, add_badge) -> None:
    shutouts = facts[(facts["win"] == True) & (facts["points_against"] == 0)]
    for row in shutouts.itertuples(index=False):
        add_badge(
            int(row.player_id),
            "pickle_perfection",
            context_type="match",
            context_id=f"{row.match_id}:pickle_perfection",
            match_id=str(row.match_id),
            value_json={"match_id": str(row.match_id), "score_against": int(row.points_against)},
        )


def _award_blowout_artist(facts: pd.DataFrame, add_badge) -> None:
    blowouts = facts[(facts["win"] == True) & (facts["margin"] >= 8)]
    for row in blowouts.itertuples(index=False):
        add_badge(
            int(row.player_id),
            "blowout_artist",
            context_type="match",
            context_id=f"{row.match_id}:blowout",
            match_id=str(row.match_id),
            value_json={"match_id": str(row.match_id), "margin": int(row.margin)},
        )


def _award_untouchable(facts: pd.DataFrame, add_badge) -> None:
    facts = facts.sort_values(["date_dt", "match_id"])
    for player_id, group in facts.groupby("player_id"):
        streak = 0
        for row in group.itertuples(index=False):
            if row.win:
                streak += 1
                if streak >= 8:
                    add_badge(
                        int(player_id),
                        "untouchable",
                        context_type="overall",
                        context_id=f"window_end:{row.match_id}:streak:{streak}",
                        match_id=str(row.match_id),
                        value_json={"streak": streak, "match_id": str(row.match_id)},
                    )
            else:
                streak = 0


def _award_clean_sweep_week(facts: pd.DataFrame, add_badge) -> None:
    grouped = facts.groupby(["player_id", "league", "week_key"])
    for (player_id, league, week_key), group in grouped:
        if len(group) < 3:
            continue
        if group["win"].all():
            add_badge(
                int(player_id),
                "clean_sweep_week",
                context_type="week",
                context_id=f"{league}:{week_key}",
                value_json={"league": league, "week": week_key},
            )


def _award_high_roller(facts: pd.DataFrame, add_badge) -> None:
    high = facts[(facts["win"] == True) & (facts["points_for"] >= 15) & (facts["margin"] >= 6)]
    for row in high.itertuples(index=False):
        add_badge(
            int(row.player_id),
            "high_roller",
            context_type="match",
            context_id=f"{row.match_id}:high_roller",
            match_id=str(row.match_id),
            value_json={"match_id": str(row.match_id), "points_for": int(row.points_for)},
        )


def _award_social_graph(facts: pd.DataFrame, add_badge) -> None:
    partners = facts.dropna(subset=["partner_id"]).groupby("player_id")["partner_id"].nunique()
    for player_id, count in partners.items():
        if count >= 20:
            add_badge(
                int(player_id),
                "social_butterfly",
                context_type="overall",
                context_id="milestone:20_partners",
                value_json={"partners": int(count)},
            )
        if count >= 50:
            add_badge(
                int(player_id),
                "network_builder",
                context_type="overall",
                context_id="milestone:50_partners",
                value_json={"partners": int(count)},
            )


def _award_draft_master(facts: pd.DataFrame, add_badge) -> None:
    wins = facts[(facts["win"] == True) & facts["partner_id"].notna()]
    grouped = wins.groupby(["player_id", "month_key"])["partner_id"].nunique().reset_index()
    for row in grouped.itertuples(index=False):
        if int(row.partner_id) >= 5:
            add_badge(
                int(row.player_id),
                "draft_master",
                context_type="month",
                context_id=f"{row.month_key}",
                value_json={"month": row.month_key, "partners": int(row.partner_id)},
            )


def _award_swiss_army_knife(facts: pd.DataFrame, add_badge) -> None:
    wins = facts[facts["win"] == True]
    grouped = wins.groupby(["player_id", "season_key"])["league"].nunique().reset_index()
    for row in grouped.itertuples(index=False):
        if int(row.league) >= 3:
            add_badge(
                int(row.player_id),
                "swiss_army_knife",
                context_type="season",
                context_id=str(row.season_key),
                value_json={"season": str(row.season_key), "leagues": int(row.league)},
            )


def _award_giant_slayer(facts: pd.DataFrame, add_badge) -> None:
    tiers = [1800, 2000, 2200]
    wins = facts[(facts["win"] == True) & facts["opp_max_rating"].notna()]
    for row in wins.itertuples(index=False):
        for tier in tiers:
            if float(row.opp_max_rating) >= tier:
                add_badge(
                    int(row.player_id),
                    "giant_slayer",
                    context_type="match",
                    context_id=f"{row.match_id}:tier:{tier}",
                    match_id=str(row.match_id),
                    value_json={"match_id": str(row.match_id), "tier": tier},
                )


def _award_david_vs_goliath(facts: pd.DataFrame, add_badge) -> None:
    wins = facts[(facts["win"] == True) & (facts["expected_win_prob"] <= 0.25)]
    for row in wins.itertuples(index=False):
        add_badge(
            int(row.player_id),
            "david_vs_goliath",
            context_type="match",
            context_id=f"{row.match_id}:david_vs_goliath",
            match_id=str(row.match_id),
            value_json={"match_id": str(row.match_id), "expected_prob": float(row.expected_win_prob)},
        )


def _award_upset_champion(facts: pd.DataFrame, add_badge) -> None:
    winners = facts[facts["win"] == True].copy()
    if winners.empty:
        return
    match_stats = (
        winners.groupby(["league", "month_key", "match_id"])
        .agg(expected_win_prob=("expected_win_prob", "min"), player_ids=("player_id", lambda x: list(x)))
        .reset_index()
    )
    match_stats = match_stats.sort_values(["league", "month_key", "expected_win_prob"])
    for (league, month_key), group in match_stats.groupby(["league", "month_key"]):
        top = group.iloc[0]
        for pid in top.player_ids:
            add_badge(
                int(pid),
                "upset_champion",
                context_type="month",
                context_id=f"{league}:month:{month_key}:match:{top.match_id}",
                match_id=str(top.match_id),
                value_num=float(top.expected_win_prob),
                value_json={
                    "league": league,
                    "month": month_key,
                    "expected_prob": float(top.expected_win_prob),
                },
            )


def _award_legendary_upset(facts: pd.DataFrame, add_badge) -> None:
    wins = facts[(facts["win"] == True) & (facts["expected_win_prob"] <= 0.15)]
    for row in wins.itertuples(index=False):
        add_badge(
            int(row.player_id),
            "legendary_upset",
            context_type="match",
            context_id=f"{row.match_id}:legendary_upset",
            match_id=str(row.match_id),
            value_json={"match_id": str(row.match_id), "expected_prob": float(row.expected_win_prob)},
        )


def _award_rivalries(facts: pd.DataFrame, add_badge) -> None:
    opponent_rows = []
    for row in facts.itertuples(index=False):
        for opp in row.opponent_ids:
            opponent_rows.append(
                {
                    "player_id": int(row.player_id),
                    "opponent_id": int(opp),
                    "match_id": row.match_id,
                    "date_dt": row.date_dt,
                    "win": bool(row.win),
                }
            )
    if not opponent_rows:
        return
    opp_df = pd.DataFrame(opponent_rows).sort_values(["date_dt", "match_id"])
    grouped = opp_df.groupby(["player_id", "opponent_id"])
    nemesis_map: set[tuple[int, int]] = set()
    for (player_id, opponent_id), group in grouped:
        games = len(group)
        wins = int(group["win"].sum())
        win_pct = wins / float(games) if games else 0.0
        if games >= 6 and win_pct <= 0.4:
            nemesis_map.add((player_id, opponent_id))
            add_badge(
                int(player_id),
                "nemesis_found",
                context_type="opponent",
                context_id=f"opponent:{opponent_id}",
                value_json={"opponent_id": opponent_id, "games": games},
            )

        streak = 0
        prev_wins = 0
        prev_losses = 0
        for row in group.itertuples(index=False):
            if row.win:
                streak += 1
                for tier in (3,):
                    if streak == tier:
                        add_badge(
                            int(player_id),
                            "rivalry_streak",
                            context_type="opponent",
                            context_id=f"{opponent_id}:streak:{row.match_id}",
                            match_id=str(row.match_id),
                            value_json={"opponent_id": opponent_id, "streak": streak},
                        )
            else:
                streak = 0
            if prev_losses > prev_wins and row.win and prev_wins + 1 == prev_losses:
                add_badge(
                    int(player_id),
                    "settled_the_score",
                    context_type="opponent",
                    context_id=f"{opponent_id}:settled:{row.match_id}",
                    match_id=str(row.match_id),
                    value_json={"opponent_id": opponent_id},
                )
            prev_wins += 1 if row.win else 0
            prev_losses += 0 if row.win else 1

    if nemesis_map:
        for row in opp_df.itertuples(index=False):
            if (int(row.player_id), int(row.opponent_id)) in nemesis_map and row.win:
                add_badge(
                    int(row.player_id),
                    "rivalry_win",
                    context_type="match",
                    context_id=f"{row.match_id}:rivalry_win",
                    match_id=str(row.match_id),
                    value_json={"opponent_id": int(row.opponent_id)},
                )


def _award_steady_hand(facts: pd.DataFrame, add_badge) -> None:
    grouped = facts.groupby(["player_id", "season_key"])
    for (player_id, season_key), group in grouped:
        matches = len(group)
        if matches < 20:
            continue
        win_pct = float(group["win"].mean())
        if win_pct >= 0.6:
            add_badge(
                int(player_id),
                "steady_hand",
                context_type="season",
                context_id=str(season_key),
                value_json={"season": str(season_key), "win_pct": win_pct},
            )


def _award_hall_of_fame_night(facts: pd.DataFrame, add_badge) -> None:
    if facts["abs_elo_delta"].dropna().empty:
        return
    for league, group in facts.groupby("league"):
        values = group["abs_elo_delta"].dropna()
        if values.empty:
            continue
        threshold = values.quantile(0.95)
        for row in group.itertuples(index=False):
            if row.win and row.abs_elo_delta is not None and float(row.abs_elo_delta) >= float(threshold):
                add_badge(
                    int(row.player_id),
                    "hall_of_fame_night",
                    context_type="match",
                    context_id=f"{row.match_id}:hall_of_fame",
                    match_id=str(row.match_id),
                    value_json={"match_id": str(row.match_id), "elo_delta": float(row.abs_elo_delta)},
                )

def _build_tape_excerpt(badge_id: str, seed: str, data: dict[str, Any]) -> str:
    copy = get_badge_copy(badge_id)
    excerpts = copy.get("tape_excerpts", []) if isinstance(copy, dict) else []
    template = pick_variant(excerpts, seed)
    rendered = render_template(template, data)
    lines = [line.strip() for line in rendered.splitlines() if line.strip()]
    return "\n".join(lines[:4])


def _build_tape_title(badge_id: str, seed: str, data: dict[str, Any]) -> str:
    copy = get_badge_copy(badge_id)
    highlight = copy.get("highlight", {}) if isinstance(copy, dict) else {}
    titles = highlight.get("titles", []) if isinstance(highlight, dict) else []
    template = pick_variant(titles, f"{seed}:title")
    rendered = render_template(template, data)
    return rendered or str(data.get("badge_name") or badge_id)


def _resolve_copy_text(
    badge_id: str,
    field: str,
    copy_pack: dict[str, Any],
    fallback: str,
    existing: dict[str, Any],
    force_backfill: bool,
) -> str:
    existing_text = str(existing.get(field) or "").strip()
    if existing_text and not force_backfill:
        return existing_text
    candidate = str(copy_pack.get(field) or "").strip()
    if candidate:
        return candidate
    return fallback


def _backfill_copy_pack_badges(
    supabase,
    *,
    force_backfill: bool,
    existing_by_id: dict[str, Any],
) -> None:
    pack = load_copy_pack()
    badges = pack.get("badges", {}) if isinstance(pack, dict) else {}
    if not isinstance(badges, dict):
        return
    defined_ids = {b.badge_id for b in BADGE_DEFINITIONS}
    for badge_id, entry in badges.items():
        if badge_id in defined_ids:
            continue
        if not isinstance(entry, dict):
            continue
        existing = existing_by_id.get(str(badge_id), {})
        if not existing and not force_backfill:
            continue
        lore = _resolve_copy_text(badge_id, "lore", entry, "", existing, force_backfill)
        hint = _resolve_copy_text(badge_id, "hint", entry, "", existing, force_backfill)
        if not lore and not hint:
            continue
        if not force_backfill and existing and (existing.get("lore") or existing.get("hint")):
            continue
        try:
            supabase.table("badges").update({"lore": lore, "hint": hint}).eq("badge_id", badge_id).execute()
        except Exception:
            logger.exception("Failed to backfill badge copy", extra={"badge_id": badge_id})


def _avg(values: Iterable[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return float(sum(nums) / len(nums))


def _max_rating(values: Iterable[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return float(max(nums))


def _safe_float(value) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except Exception:
        return None


def _safe_rating(value, fallback: float | None) -> float | None:
    v = _safe_float(value)
    if v is not None:
        return v
    if fallback is not None:
        return float(fallback)
    return None


def _expected_share(team_avg: float | None, opp_avg: float | None) -> float:
    try:
        if team_avg is None or opp_avg is None:
            return 0.5
        return 1.0 / (1.0 + 10 ** ((float(opp_avg) - float(team_avg)) / 400.0))
    except Exception:
        return 0.5


def _week_key(date_dt: pd.Timestamp) -> str:
    iso = date_dt.isocalendar()
    return f"{iso.year}-W{int(iso.week):02d}"


def _month_key(date_dt: pd.Timestamp) -> str:
    return date_dt.strftime("%Y-%m")


def _season_key(date_dt: pd.Timestamp) -> str:
    return date_dt.strftime("%Y")


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
