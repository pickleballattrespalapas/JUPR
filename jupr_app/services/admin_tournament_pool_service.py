from __future__ import annotations

import os
from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.tournaments.pools import normalize_pool_config
from jupr_app.services.admin_tournament_draw_service import _draw_payload
from jupr_app.services.admin_tournament_game_service import (
    _canonical_timestamp,
    _require_reviewed_row_versions,
)
from jupr_app.services.admin_tournament_guarded_operation import StaleTournamentAdminStateError
from jupr_app.services.admin_tournament_service import (
    TOURNAMENT_SELECT,
    _first_row,
    is_admin_tournament_admin_enabled,
)

CONFIRM_SAVE_POOLS = "SAVE POOLS"


def _draw_rows(supabase: Any, table: str, tournament_id: str, draw_id: str) -> list[dict[str, Any]]:
    try:
        query = supabase.table(table).select("*").eq("tournament_id", tournament_id)
        query = query.eq("id" if table == "tournament_event_draws" else "draw_id", draw_id)
        response = query.execute()
        return [dict(row) for row in (response.data or [])]
    except Exception as exc:
        raise RuntimeError("Could not verify the draw's current state. Reload before saving pools.") from exc


def save_admin_tournament_draw_pools(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    pool_config: dict[str, Any] | None,
    expected_draw_updated_at: str,
    expected_team_versions: list[dict[str, Any]],
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_tournament_admin_save_pools",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Save one reviewed division pool assignment before any games are created.

    The database RPC is mandatory, including when the operation wrapper is not
    enabled. It repeats ownership, exact-roster and no-play checks under locks.
    """
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_SAVE_POOLS:
        raise ValueError(f"Type {CONFIRM_SAVE_POOLS} to save the pool assignments.")
    tournament = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=str(tournament_id))
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    if str(tournament.get("status") or "").upper() in {"COMPLETED", "ARCHIVED"}:
        raise ValueError("Pool assignments cannot change after tournament closeout.")
    draws = _draw_rows(supabase, "tournament_event_draws", str(tournament_id), str(draw_id))
    if len(draws) != 1:
        raise ValueError("draw not found for this tournament")
    draw = draws[0]
    if not expected_draw_updated_at or _canonical_timestamp(draw.get("updated_at")) != _canonical_timestamp(expected_draw_updated_at):
        raise StaleTournamentAdminStateError("This draw changed after review. Reload before saving pools.")
    teams = _draw_rows(supabase, "tournament_teams", str(tournament_id), str(draw_id))
    reviewed = _require_reviewed_row_versions(teams, expected_team_versions, label="pool team set", atomic=True)
    canonical = normalize_pool_config(pool_config, teams)
    for table in ("tournament_games", "tournament_podium", "tournament_day_live_draws", "tournament_day_live_queue"):
        if _draw_rows(supabase, table, str(tournament_id), str(draw_id)):
            raise ValueError("Pool assignments and advancement counts must be saved before games or live play exist. Use playoff review to override qualifiers after play.")
    if str(draw.get("status") or "draft").upper() in {"CANCELLED", "CANCELED", "INACTIVE", "ARCHIVED", "COMPLETED"}:
        raise ValueError("Pool assignments cannot change for an inactive or completed draw.")
    if dry_run:
        return {"ok": True, "mode": "tournament_draw_pools_preview", "dry_run": True, "write_count": 0,
                "draw": _draw_payload({**draw, "pool_config": canonical}), "pool_config": canonical, "warnings": []}
    try:
        response = supabase.rpc("admin_save_tournament_draw_pools_cas", {
            "p_club_id": str(club_id), "p_tournament_id": str(tournament_id), "p_draw_id": str(draw_id),
            "p_expected_draw_updated_at": expected_draw_updated_at, "p_expected_teams": reviewed,
            "p_pool_config": canonical,
        }).execute()
    except Exception as exc:
        message = str(exc)
        if any(marker in message for marker in ("JUPR_TOURNAMENT_DRAW_STALE", "JUPR_TOURNAMENT_TEAM_SNAPSHOT_STALE", "JUPR_TOURNAMENT_POOLS_LOCKED")):
            raise StaleTournamentAdminStateError("The draw or team list changed while pools were being saved. Reload before continuing.") from exc
        if "JUPR_TOURNAMENT_POOL_CONFIG_INVALID" in message:
            raise ValueError("The pool assignments do not match the current team list. Review every team and advancement count.") from exc
        raise RuntimeError("Unable to confirm the pool save. Reload the draw before retrying.") from exc
    result = getattr(response, "data", None)
    if isinstance(result, list) and len(result) == 1:
        result = result[0]
    if not isinstance(result, dict) or not isinstance(result.get("draw"), dict):
        raise RuntimeError("Pool save returned no authoritative draw. Reload before continuing.")
    saved = _draw_payload(result["draw"])
    audit = write_admin_activity_log(supabase, build_activity_payload(
        club_id=str(club_id), actor_email=actor_email, actor_role=actor_role,
        action_type="save_tournament_draw_pools_admin", entity_type="tournament_event_draw", entity_id=str(draw_id),
        before_json={"draw": _draw_payload(draw)},
        after_json={"draw": saved, "source_client": "fastapi/nextjs", "source_page": source},
        source_page=source, flagged_for_review=True,
    ))
    warnings = [audit.warning] if audit.warning else []
    if not audit.ok and os.getenv("JUPR_REQUIRE_API_AUDIT_LOG", "").strip().lower() in {"1", "true", "yes", "y", "on"}:
        raise RuntimeError("Pool save completed but its required audit record could not be confirmed. Reload before continuing.")
    return {"ok": True, "mode": "tournament_draw_pools_save", "draw": saved,
            "pool_config": saved.get("pool_config"), "warnings": warnings}
