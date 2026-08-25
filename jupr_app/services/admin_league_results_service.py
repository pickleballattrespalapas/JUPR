from __future__ import annotations

from typing import Any

from jupr_app.services.public_league_results_service import (
    DEFAULT_WEEKLY_HIGHLIGHT_MIN_GAMES,
    _build_resolved_league_results,
    _public_league_meta,
)
from jupr_app.services.public_league_visibility import league_is_public

ADMIN_LEAGUE_RESULTS_META_SELECT = (
    "id,club_id,league_name,league_type,match_format,is_active,status,min_games,k_factor,"
    "schedule_config,awards_config"
)


def _safe_rows(response: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (response.data or [])]
    except Exception:
        return []


def _exact_league_metadata(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
) -> dict[str, Any]:
    requested = str(league_name or "").strip()
    if not requested or requested.upper() in {"OVERALL", "POPUP"}:
        raise ValueError("League results were not found.")
    try:
        response = (
            supabase.table("leagues_metadata")
            .select(ADMIN_LEAGUE_RESULTS_META_SELECT)
            .eq("club_id", str(club_id))
            .eq("league_name", requested)
            .limit(2)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Unable to load the selected league metadata.") from exc
    rows = _safe_rows(response)
    if len(rows) != 1:
        raise ValueError("League results were not found.")
    return rows[0]


def _is_publicly_visible(metadata: dict[str, Any]) -> bool:
    return league_is_public(metadata)


def build_admin_league_results(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    week_num: int | None = None,
    player_id: int | None = None,
    weekly_min_games: int = DEFAULT_WEEKLY_HIGHLIGHT_MIN_GAMES,
) -> dict[str, Any]:
    """Build exact league results after the admin route has authorized the caller."""

    cid = str(club_id).strip()
    requested = str(league_name or "").strip()
    metadata = _exact_league_metadata(
        supabase,
        club_id=cid,
        league_name=requested,
    )
    exact_name = str(metadata.get("league_name") or "").strip()
    if exact_name != requested:
        raise ValueError("League results were not found.")

    historical = str(metadata.get("status") or "").strip().lower() in {
        "ended",
        "archived",
    }
    results = _build_resolved_league_results(
        supabase,
        club_id=cid,
        overview={"leagues": [_public_league_meta(exact_name, metadata)]},
        selected=exact_name,
        week_num=week_num,
        player_id=player_id,
        weekly_min_games=weekly_min_games,
        include_inactive_players=historical,
        include_inactive_ratings=historical,
        league_metadata=metadata,
    )
    league_id = metadata.get("id") or metadata.get("league_id") or exact_name
    return {
        "ok": True,
        "mode": "league_manager_results",
        "league_id": str(league_id),
        "league_name": exact_name,
        "league_type": str(metadata.get("league_type") or "Individual"),
        "league_status": str(metadata.get("status") or "active"),
        "publicly_visible": _is_publicly_visible(metadata),
        **results,
    }
