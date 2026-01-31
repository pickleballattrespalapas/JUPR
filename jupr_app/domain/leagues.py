from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Mapping

import pandas as pd

from jupr_app.domain.awards import compute_top_performer_awards
from jupr_app.domain.gamification.top_performer_awards import (
    _build_league_standings,
    _min_games_for_league,
)


_COMPLETED_STATUSES = {"archived", "completed", "complete", "done"}


def get_league_meta_row(df_meta: pd.DataFrame | None, league_id: str) -> dict[str, Any] | None:
    if df_meta is None or df_meta.empty or "league_name" not in df_meta.columns:
        return None
    league_key = str(league_id or "").strip()
    if not league_key:
        return None
    meta = df_meta.copy()
    meta["league_name"] = meta["league_name"].fillna("").astype(str).str.strip()
    hit = meta[meta["league_name"] == league_key]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()


def is_league_ended(meta_row: Mapping[str, Any] | None) -> bool:
    if not meta_row:
        return False
    ended_at = meta_row.get("ended_at")
    if ended_at is not None and not pd.isna(ended_at):
        return True
    status = str(meta_row.get("status") or "").strip().lower()
    if status in _COMPLETED_STATUSES:
        return True
    is_active = meta_row.get("is_active")
    if is_active is None or (isinstance(is_active, float) and pd.isna(is_active)):
        return False
    return not bool(is_active)


def compute_top_performer_preview(
    df_leagues: pd.DataFrame | None,
    df_meta: pd.DataFrame | None,
    id_to_name: Mapping[int, str] | None,
    league_id: str,
    *,
    winners_per_category: int = 1,
) -> list[dict[str, Any]]:
    min_games = _min_games_for_league(df_meta, league_id)
    if min_games <= 0:
        return []
    standings = _build_league_standings(df_leagues, league_id, dict(id_to_name or {}))
    if standings.empty:
        return []
    qualified = standings[standings["matches_played"] >= int(min_games)].copy()
    if qualified.empty:
        return []
    awards = compute_top_performer_awards(qualified, min_games=min_games, winners_per_category=winners_per_category)
    name_map = {
        int(row._pid): str(getattr(row, "name", "") or "")
        for row in qualified.itertuples(index=False)
        if getattr(row, "_pid", None) is not None
    }
    for award in awards:
        pid = award.get("player_id")
        award["player_name"] = name_map.get(int(pid)) if pid is not None else ""
    return awards


def _apply_league_end_to_meta(
    df_meta: pd.DataFrame | None,
    league_id: str,
    *,
    ended_at: str,
    ended_by: str | None,
    status: str,
    end_awards: dict[str, Any],
) -> pd.DataFrame | None:
    if df_meta is None or df_meta.empty or "league_name" not in df_meta.columns:
        return df_meta
    meta = df_meta.copy()
    meta["league_name"] = meta["league_name"].fillna("").astype(str).str.strip()
    league_key = str(league_id or "").strip()
    if not league_key:
        return meta
    mask = meta["league_name"] == league_key
    if not mask.any():
        return meta
    if "ended_at" not in meta.columns:
        meta["ended_at"] = pd.NA
    if "ended_by" not in meta.columns:
        meta["ended_by"] = pd.NA
    if "status" not in meta.columns:
        meta["status"] = pd.NA
    if "is_active" not in meta.columns:
        meta["is_active"] = pd.NA
    if "end_awards" not in meta.columns:
        meta["end_awards"] = pd.NA
    meta.loc[mask, "ended_at"] = ended_at
    meta.loc[mask, "ended_by"] = ended_by
    meta.loc[mask, "status"] = status
    meta.loc[mask, "is_active"] = False
    meta.loc[mask, "end_awards"] = end_awards
    return meta


def end_league_and_award_top_performers(
    ctx: Any,
    league_id: str,
    *,
    admin_id: str | None = None,
) -> dict[str, Any]:
    from jupr_app.domain.gamification.ensure_badges import ensure_badges

    supabase = getattr(ctx, "supabase", None)
    club_id = str(getattr(ctx, "club_id", "") or "")
    if supabase is None or not club_id:
        return {"ended": False, "awards": [], "created": []}

    league_key = str(league_id or "").strip()
    if not league_key:
        return {"ended": False, "awards": [], "created": []}

    meta_row: dict[str, Any] | None = None
    try:
        resp = (
            supabase.table("leagues_metadata")
            .select("*")
            .eq("club_id", club_id)
            .eq("league_name", league_key)
            .execute()
        )
        meta_row = (resp.data or [None])[0]
    except Exception:
        meta_row = get_league_meta_row(getattr(ctx, "df_meta", None), league_key)

    now_iso = datetime.now(timezone.utc).isoformat()
    awards = compute_top_performer_preview(
        getattr(ctx, "df_leagues", None),
        getattr(ctx, "df_meta", None),
        getattr(ctx, "id_to_name", None),
        league_key,
        winners_per_category=1,
    )
    end_awards = {"top_performers": awards}
    already_ended = is_league_ended(meta_row)
    if not already_ended:
        update_payload = {
            "is_active": False,
            "status": "completed",
            "ended_at": now_iso,
            "ended_by": admin_id,
            "end_awards": end_awards,
        }
        supabase.table("leagues_metadata").update(update_payload).eq("club_id", club_id).eq(
            "league_name", league_key
        ).execute()

    updated_meta = _apply_league_end_to_meta(
        getattr(ctx, "df_meta", None),
        league_key,
        ended_at=now_iso,
        ended_by=admin_id,
        status="completed",
        end_awards=end_awards,
    )
    attrs = dict(vars(ctx)) if hasattr(ctx, "__dict__") else {}
    attrs["df_meta"] = updated_meta
    evaluation_ctx = SimpleNamespace(**attrs)
    created = ensure_badges(
        evaluation_ctx,
        club_id=club_id,
        league_id=league_key,
        status="seasonal",
        award_timing="on_league_close",
    )
    return {"ended": True, "awards": awards, "created": created}
