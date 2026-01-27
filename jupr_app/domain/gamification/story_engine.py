from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from typing import Any
from uuid import uuid4

import pandas as pd

from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS
from jupr_app.domain.gamification.copy_pack import get_badge_copy, pick_variant, render_template

logger = logging.getLogger(__name__)


def ensure_player_stories(ctx, facts: pd.DataFrame, awards) -> None:
    if bool(getattr(ctx, "public_mode", False)):
        return
    supabase = getattr(ctx, "supabase", None)
    club_id = str(getattr(ctx, "club_id", "") or "")
    if supabase is None or not club_id:
        return
    try:
        rows = compute_story_cards(ctx, facts, awards)
        if not rows:
            return
        supabase.table("player_stories").upsert(
            rows,
            on_conflict="club_id,player_id,story_type,context_id",
        ).execute()
        _trim_story_feed(supabase, club_id, {row["player_id"] for row in rows})
    except Exception:
        logger.exception("ensure_player_stories failed")


def compute_story_cards(ctx, facts: pd.DataFrame, awards) -> list[dict[str, Any]]:
    if facts is None or facts.empty:
        return []

    badge_map = {b.badge_id: b for b in BADGE_DEFINITIONS}
    now = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    club_id = str(getattr(ctx, "club_id", "") or "")

    def add_story(
        player_id: int,
        story_type: str,
        context_type: str,
        context_id: str,
        title: str,
        body: str,
        *,
        match_id: str | None = None,
        importance: int = 50,
        expires_at: datetime | None = None,
        value_json: dict[str, Any] | None = None,
    ) -> None:
        rows.append(
            {
                "id": str(uuid4()),
                "club_id": club_id,
                "player_id": int(player_id),
                "created_at": now.isoformat(),
                "story_type": story_type,
                "context_type": context_type,
                "context_id": context_id,
                "match_id": match_id,
                "title": title,
                "body": body,
                "importance": int(importance),
                "expires_at": expires_at.isoformat() if expires_at else None,
                "value_json": value_json,
            }
        )

    for award in awards or []:
        badge = badge_map.get(award.badge_id)
        if not badge:
            continue
        tape = ""
        if award.value_json and award.value_json.get("tape_excerpt"):
            tape = str(award.value_json["tape_excerpt"])
        story_data = dict(award.value_json or {})
        story_data.setdefault("badge_name", badge.name)
        story_data.setdefault("tape_excerpt", tape)
        title, body = _build_highlight_copy(
            award.badge_id,
            award.player_id,
            award.context_id or "",
            badge.name,
            tape,
            story_data,
        )
        importance = _badge_importance(badge.rarity, badge.prestige)
        add_story(
            award.player_id,
            f"highlight.badge.{award.badge_id}",
            award.context_type,
            f"{award.badge_id}:{award.context_id}",
            title,
            body,
            match_id=award.match_id,
            importance=importance,
            value_json={
                "badge_id": award.badge_id,
                "rarity": badge.rarity,
                "tape_excerpt": tape,
            },
        )

    rows.extend(_signature_win_stories(facts, now, add_story))
    rows.extend(_foreshadow_weekly_regular(facts, add_story))
    rows.extend(_foreshadow_hot_streak(facts, add_story))
    rows.extend(_foreshadow_marathon_month(facts, now, add_story))
    rows.extend(_foreshadow_social_butterfly(facts, add_story))
    rows.extend(_foreshadow_draft_master(facts, now, add_story))

    return rows


def _signature_win_stories(facts: pd.DataFrame, now: datetime, add_story) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    recent_cutoff = now - timedelta(days=30)
    recent = facts[facts["date_dt"] >= recent_cutoff]
    wins = recent[recent["win"] == True]
    if wins.empty:
        return rows
    wins = wins.sort_values(["player_id", "expected_win_prob", "date_dt"])
    for player_id, group in wins.groupby("player_id"):
        row = group.iloc[0]
        title, body = _build_highlight_copy(
            "signature_win",
            int(player_id),
            f"signature_win:{row.match_id}",
            "Signature Win",
            "",
            {"expected_prob": float(row.expected_win_prob)},
        )
        add_story(
            int(player_id),
            "highlight.signature_win",
            "match",
            f"signature_win:{row.match_id}",
            title,
            body,
            match_id=str(row.match_id),
            importance=85,
            value_json={"expected_win_prob": float(row.expected_win_prob)},
        )
    return rows


def _foreshadow_weekly_regular(facts: pd.DataFrame, add_story) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = (
        facts.groupby(["player_id", "league"])["week_key"]
        .unique()
        .reset_index()
    )
    for row in grouped.itertuples(index=False):
        weeks = sorted([w for w in row.week_key if w])
        if len(weeks) < 3:
            continue
        streak, end_week = _current_week_streak(weeks)
        if streak == 3:
            title, body = _build_foreshadow_copy(
                "weekly_regular",
                int(row.player_id),
                f"{row.league}:{end_week}",
                {"league": row.league, "streak": streak, "week": end_week},
            )
            add_story(
                int(row.player_id),
                "foreshadow.weekly_regular",
                "week",
                f"{row.league}:{end_week}:foreshadow",
                title,
                body,
                importance=60,
            )
    return rows


def _foreshadow_hot_streak(facts: pd.DataFrame, add_story) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    facts = facts.sort_values(["date_dt", "match_id"])
    for (player_id, league), group in facts.groupby(["player_id", "league"]):
        streak = 0
        for row in group.itertuples(index=False):
            if row.win:
                streak += 1
                if streak in {4, 9, 19}:
                    title, body = _build_foreshadow_copy(
                        "hot_streak",
                        int(player_id),
                        f"{league}:{streak}:{row.match_id}",
                        {"league": league, "streak": streak, "match_id": str(row.match_id)},
                    )
                    add_story(
                        int(player_id),
                        "foreshadow.hot_streak",
                        "match",
                        f"{league}:streak:{streak}:{row.match_id}",
                        title,
                        body,
                        match_id=str(row.match_id),
                        importance=65,
                    )
            else:
                streak = 0
    return rows


def _foreshadow_marathon_month(facts: pd.DataFrame, now: datetime, add_story) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_month = now.strftime("%Y-%m")
    current = facts[facts["month_key"] == current_month]
    if current.empty:
        return rows
    counts = current.groupby(["player_id", "month_key"]).size().reset_index(name="matches")
    for row in counts.itertuples(index=False):
        if int(row.matches) >= 30:
            title, body = _build_foreshadow_copy(
                "marathon_month",
                int(row.player_id),
                f"{row.month_key}",
                {"month": row.month_key, "matches": int(row.matches)},
            )
            add_story(
                int(row.player_id),
                "foreshadow.marathon_month",
                "month",
                f"{row.month_key}:foreshadow",
                title,
                body,
                importance=60,
            )
    return rows


def _foreshadow_social_butterfly(facts: pd.DataFrame, add_story) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    partners = facts.dropna(subset=["partner_id"]).groupby("player_id")["partner_id"].nunique()
    for player_id, count in partners.items():
        if 15 <= int(count) <= 19:
            title, body = _build_foreshadow_copy(
                "social_butterfly",
                int(player_id),
                "milestone:20_partners",
                {"partners": int(count)},
            )
            add_story(
                int(player_id),
                "foreshadow.social_butterfly",
                "overall",
                "milestone:20_partners:foreshadow",
                title,
                body,
                importance=55,
            )
    return rows


def _foreshadow_draft_master(facts: pd.DataFrame, now: datetime, add_story) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_month = now.strftime("%Y-%m")
    wins = facts[(facts["win"] == True) & facts["partner_id"].notna()]
    wins = wins[wins["month_key"] == current_month]
    if wins.empty:
        return rows
    grouped = wins.groupby("player_id")["partner_id"].nunique().reset_index()
    for row in grouped.itertuples(index=False):
        if 3 <= int(row.partner_id) <= 4:
            title, body = _build_foreshadow_copy(
                "draft_master",
                int(row.player_id),
                f"{current_month}",
                {"month": current_month, "partners": int(row.partner_id)},
            )
            add_story(
                int(row.player_id),
                "foreshadow.draft_master",
                "month",
                f"{current_month}:foreshadow",
                title,
                body,
                importance=55,
            )
    return rows


def _badge_importance(rarity: str, prestige: int) -> int:
    base = {"common": 55, "rare": 70, "epic": 80, "legendary": 90}.get(rarity, 60)
    return min(100, base + int(prestige / 10))


def _current_week_streak(weeks: list[str]) -> tuple[int, str]:
    if not weeks:
        return 0, ""
    parsed = []
    for week in weeks:
        try:
            year_str, week_str = week.split("-W")
            parsed.append((int(year_str), int(week_str), week))
        except Exception:
            continue
    if not parsed:
        return 0, ""
    parsed = sorted(parsed)
    streak = 1
    for (y1, w1, _), (y2, w2, _) in zip(parsed, parsed[1:]):
        next_week = w1 + 1
        next_year = y1
        if w1 >= 52:
            next_week = 1
            next_year = y1 + 1
        if (y2, w2) == (next_year, next_week):
            streak += 1
        else:
            streak = 1
    return streak, parsed[-1][2]


def _trim_story_feed(supabase, club_id: str, player_ids: set[int]) -> None:
    for player_id in player_ids:
        try:
            resp = (
                supabase.table("player_stories")
                .select("id")
                .eq("club_id", club_id)
                .eq("player_id", int(player_id))
                .order("created_at", desc=True)
                .range(50, 200)
                .execute()
            )
            ids = [row["id"] for row in resp.data or [] if row.get("id")]
            if ids:
                supabase.table("player_stories").delete().in_("id", ids).execute()
        except Exception:
            logger.exception("Failed trimming story feed", extra={"player_id": player_id})


def _build_highlight_copy(
    badge_id: str,
    player_id: int,
    context_id: str,
    badge_name: str,
    tape_excerpt: str,
    data: dict[str, Any],
) -> tuple[str, str]:
    copy = get_badge_copy(badge_id)
    highlight = copy.get("highlight", {}) if isinstance(copy, dict) else {}
    titles = highlight.get("titles", []) if isinstance(highlight, dict) else []
    bodies = highlight.get("bodies", []) if isinstance(highlight, dict) else []
    seed = f"{player_id}:{badge_id}:{context_id}:highlight"
    title_template = pick_variant(titles, f"{seed}:title")
    body_template = pick_variant(bodies, f"{seed}:body")
    story_data = dict(data)
    story_data.setdefault("badge_name", badge_name)
    story_data.setdefault("tape_excerpt", tape_excerpt)
    title = render_template(title_template, story_data) or f"Highlight — {badge_name}"
    body = render_template(body_template, story_data)
    if not body:
        body = f"{tape_excerpt}\nThe record adds {badge_name} to the reel.".strip()
    return title, body


def _build_foreshadow_copy(
    badge_id: str,
    player_id: int,
    context_id: str,
    data: dict[str, Any],
) -> tuple[str, str]:
    copy = get_badge_copy(badge_id)
    foreshadow = copy.get("foreshadow", {}) if isinstance(copy, dict) else {}
    titles = foreshadow.get("titles", []) if isinstance(foreshadow, dict) else []
    bodies = foreshadow.get("bodies", []) if isinstance(foreshadow, dict) else []
    seed = f"{player_id}:{badge_id}:{context_id}:foreshadow"
    title_template = pick_variant(titles, f"{seed}:title")
    body_template = pick_variant(bodies, f"{seed}:body")
    story_data = dict(data)
    story_data.setdefault("badge_name", copy.get("name", badge_id))
    title = render_template(title_template, story_data) or f"Foreshadowing — {story_data['badge_name']}"
    body = render_template(body_template, story_data)
    if not body:
        body = "The reel is leaning close.\nThe next frame could change the story."
    return title, body
