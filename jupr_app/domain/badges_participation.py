from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any
from uuid import uuid4

import pandas as pd

logger = logging.getLogger(__name__)


PARTICIPATION_BADGES = (
    ("participant", 1),
    ("dedicated_participant_50", 50),
    ("lifetime_participant_200", 200),
)


def compute_lifetime_games(ctx) -> dict[int, int]:
    df_matches = getattr(ctx, "df_matches", None)
    if df_matches is None or df_matches.empty:
        return {}

    df = df_matches.copy()
    club_id = str(getattr(ctx, "club_id", "") or "")
    if club_id and "club_id" in df.columns:
        df = df[df["club_id"].astype(str) == club_id]

    if df.empty:
        return {}

    score_cols = _score_columns(df)
    if "player_id" in df.columns:
        pid = pd.to_numeric(df["player_id"], errors="coerce")
        valid = pid.notna()
        if score_cols:
            score_total = pd.to_numeric(df[score_cols[0]], errors="coerce").fillna(0) + pd.to_numeric(
                df[score_cols[1]], errors="coerce"
            ).fillna(0)
            valid &= score_total > 0
        pid = pid[valid].astype(int)
        if pid.empty:
            return {}
        return pid.value_counts().to_dict()

    player_cols = [c for c in ("t1_p1", "t1_p2", "t2_p1", "t2_p2") if c in df.columns]
    if len(player_cols) < 1:
        return {}

    players = df[player_cols].apply(pd.to_numeric, errors="coerce")
    valid = players.notna().all(axis=1)
    if score_cols:
        score_total = pd.to_numeric(df[score_cols[0]], errors="coerce").fillna(0) + pd.to_numeric(
            df[score_cols[1]], errors="coerce"
        ).fillna(0)
        valid &= score_total > 0

    players = players[valid]
    counts: dict[int, int] = {}
    for _, row in players.iterrows():
        ids = {int(pid) for pid in row.tolist() if pd.notna(pid)}
        for pid in ids:
            counts[pid] = counts.get(pid, 0) + 1
    return counts


def ensure_participation_badges(ctx) -> None:
    if bool(getattr(ctx, "public_mode", False)):
        return

    supabase = getattr(ctx, "supabase", None)
    club_id = str(getattr(ctx, "club_id", "") or "")
    if supabase is None or not club_id:
        return

    try:
        lifetime_games = compute_lifetime_games(ctx)
        if not lifetime_games:
            return

        badge_ids = [badge_id for badge_id, _ in PARTICIPATION_BADGES]
        existing = _fetch_existing_badges(supabase, club_id, badge_ids)

        now = datetime.now(timezone.utc).isoformat()
        rows: list[dict[str, Any]] = []
        for player_id, games in lifetime_games.items():
            for badge_id, threshold in PARTICIPATION_BADGES:
                if games < threshold:
                    continue
                key = (int(player_id), badge_id)
                if key in existing:
                    continue
                existing.add(key)
                rows.append(
                    {
                        "id": str(uuid4()),
                        "club_id": club_id,
                        "player_id": int(player_id),
                        "badge_id": badge_id,
                        "earned_at": now,
                        "context_type": "overall",
                        "context_id": None,
                        "value_num": float(games),
                    }
                )

        if rows:
            _insert_badges(supabase, rows)
    except Exception:
        logger.exception("ensure_participation_badges failed")


def _score_columns(df: pd.DataFrame) -> tuple[str, str] | None:
    if "score_t1" in df.columns and "score_t2" in df.columns:
        return "score_t1", "score_t2"
    if "s1" in df.columns and "s2" in df.columns:
        return "s1", "s2"
    return None


def _fetch_existing_badges(supabase, club_id: str, badge_ids: list[str]) -> set[tuple[int, str]]:
    try:
        resp = (
            supabase.table("player_badges")
            .select("player_id,badge_id")
            .eq("club_id", club_id)
            .in_("badge_id", badge_ids)
            .execute()
        )
    except Exception:
        logger.exception("Failed to fetch participation badges")
        return set()

    existing = set()
    for row in resp.data or []:
        try:
            player_id = int(row.get("player_id"))
        except Exception:
            continue
        badge_id = str(row.get("badge_id"))
        existing.add((player_id, badge_id))
    return existing


def _insert_badges(supabase, rows: list[dict[str, Any]]) -> None:
    chunk = 200
    for i in range(0, len(rows), chunk):
        supabase.table("player_badges").insert(rows[i : i + chunk]).execute()
