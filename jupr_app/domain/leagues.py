from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Mapping

import pandas as pd

from jupr_app.domain.awards import TOP_PERFORMER_SPECS, compute_top_performer_awards
from jupr_app.domain.event_tags import get_event_tags
from jupr_app.domain.gamification.top_performer_awards import (
    TOP_PERFORMER_BADGE_IDS,
    _build_league_standings,
    _min_games_for_league,
)
from jupr_app.domain.gamification.badge_types import BadgeCandidate
from jupr_app.domain.gamification.badges_repo import upsert_player_badges


_COMPLETED_STATUSES = {"archived", "completed", "complete", "done", "ended"}
_ACTIVE_STATUSES = {"active", "running", "live"}
_DRAFT_STATUSES = {"draft", "planned"}
_PAUSED_STATUSES = {"paused"}
_ARCHIVED_STATUSES = {"archived"}
_ENDED_STATUSES = {"ended", "completed", "complete", "done"}


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



def get_league_event_tags(meta_row: Mapping[str, Any] | None) -> dict[str, list[str]]:
    payload = meta_row if isinstance(meta_row, Mapping) else {}
    return get_event_tags(dict(payload), default_skill_all=False)

def normalize_league_status(meta_row: Mapping[str, Any] | None) -> str:
    if not meta_row:
        return "draft"
    status = str(meta_row.get("status") or "").strip().lower()
    if status:
        if status in _ARCHIVED_STATUSES:
            return "archived"
        if status in _ENDED_STATUSES:
            return "ended"
        if status in _ACTIVE_STATUSES:
            return "active"
        if status in _PAUSED_STATUSES:
            return "paused"
        if status in _DRAFT_STATUSES:
            return "draft"
    ended_at = meta_row.get("ended_at")
    if ended_at is not None and not pd.isna(ended_at):
        return "ended"
    is_active = meta_row.get("is_active")
    if is_active is None or (isinstance(is_active, float) and pd.isna(is_active)):
        return "draft"
    if bool(is_active):
        return "active"
    if not status:
        return "ended"
    return "draft"


def is_league_ended(meta_row: Mapping[str, Any] | None) -> bool:
    status = normalize_league_status(meta_row)
    return status in {"ended", "archived"}


def compute_top_performer_awards_for_config(
    df_leagues: pd.DataFrame | None,
    df_meta: pd.DataFrame | None,
    id_to_name: Mapping[int, str] | None,
    league_id: str,
    *,
    awards_config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    standings = _build_league_standings(df_leagues, league_id, dict(id_to_name or {}))
    if standings.empty:
        return []
    cfg = dict(awards_config or {})
    categories_cfg = cfg.get("categories") or {}
    default_min_games = int(cfg.get("default_min_games") or _min_games_for_league(df_meta, league_id) or 0)
    default_depth = int(cfg.get("default_depth") or 1)
    awards: list[dict[str, Any]] = []
    for spec in TOP_PERFORMER_SPECS:
        cat_cfg = categories_cfg.get(spec.category_key, {}) if isinstance(categories_cfg, dict) else {}
        if cat_cfg.get("enabled") is False:
            continue
        min_games = int(cat_cfg.get("min_games") or default_min_games or 0)
        depth = int(cat_cfg.get("depth") or default_depth or 1)
        if min_games <= 0 or depth <= 0:
            continue
        qualified = standings[standings["matches_played"] >= int(min_games)].copy()
        if qualified.empty:
            continue
        if spec.sort_key not in qualified.columns:
            sort_df = qualified.copy()
            sort_df[spec.sort_key] = 0
        else:
            sort_df = qualified
        top = sort_df.sort_values(spec.sort_key, ascending=False).head(int(depth))
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            player_id = row.get("_pid")
            if pd.isna(player_id):
                player_id = row.get("player_id")
            if player_id is None or pd.isna(player_id):
                continue
            metric_value = spec.value_fn(row)
            awards.append(
                {
                    "category_key": spec.category_key,
                    "category_label": spec.label,
                    "player_id": int(player_id),
                    "player_name": str(row.get("name", "") or ""),
                    "metric_value": metric_value,
                    "metric_display": spec.display_fn(row, metric_value),
                    "rank": int(rank),
                    "min_games": int(min_games),
                }
            )
    return awards


def build_top_performer_badge_candidates(
    awards: list[dict[str, Any]],
    *,
    club_id: str,
    league_id: str,
    ended_at: str | None,
    override_notes: Mapping[str, str] | None = None,
) -> list[BadgeCandidate]:
    candidates: list[BadgeCandidate] = []
    overrides = dict(override_notes or {})
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
            "rank": rank,
            "metric_value": award.get("metric_value"),
            "metric_display": award.get("metric_display"),
            "ended_at": ended_at,
        }
        override_key = f"{category_key}:{rank}"
        if override_key in overrides:
            value_json["override_note"] = overrides[override_key]
        candidates.append(
            BadgeCandidate(
                badge_id=badge_id,
                player_id=int(award["player_id"]),
                club_id=club_id,
                context_type="league",
                context_id=context_id,
                match_id=None,
                value_json=value_json,
                value_num=award.get("metric_value") if award.get("metric_value") is not None else None,
            )
        )
    return candidates


def mint_top_performer_badges(
    supabase: Any,
    *,
    club_id: str,
    league_id: str,
    awards: list[dict[str, Any]],
    ended_at: str | None,
    override_notes: Mapping[str, str] | None = None,
) -> list[BadgeCandidate]:
    candidates = build_top_performer_badge_candidates(
        awards,
        club_id=club_id,
        league_id=league_id,
        ended_at=ended_at,
        override_notes=override_notes,
    )
    return upsert_player_badges(supabase, club_id, candidates)


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
            "status": "ended",
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
        status="ended",
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
