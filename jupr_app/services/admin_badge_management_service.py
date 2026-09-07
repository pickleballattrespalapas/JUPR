from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from uuid import UUID

from jupr_app.data.paged_reads import read_all_rows
from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS
from jupr_app.domain.gamification.community_awards import COMMUNITY_CRITERIA, build_community_award
from jupr_app.domain.gamification.presentation import badge_requirement
from jupr_app.domain.gamification.seasons import BadgeSeason
from jupr_app.services.admin_guarded_write_service import require_staging_service_role_write
from jupr_app.services.staging_write_guard import staging_write_wave_allows


def badge_management_options(supabase: Any, club_id: str) -> dict:
    players = read_all_rows(lambda: supabase.table("players").select("id,name").eq("club_id", club_id), order="id")
    seasons = read_all_rows(lambda: supabase.table("badge_seasons").select("*").eq("club_id", club_id), order="start_date")
    recent = supabase.table("player_badges").select("id,player_id,badge_id,earned_at,value_json,awarded_by,revoked_at").eq("club_id", club_id).in_("badge_id", list(COMMUNITY_CRITERIA)).order("earned_at", desc=True).limit(100).execute().data or []
    states = {row["badge_id"]: row for row in supabase.table("badges").select("badge_id,is_active,state").in_("badge_id", list(COMMUNITY_CRITERIA)).execute().data or []}
    badges = [{"id": badge.badge_id, "name": badge.name, "requirement": badge_requirement(badge.badge_id),
               "available": bool(states.get(badge.badge_id, {}).get("is_active")) and states.get(badge.badge_id, {}).get("state", "live") == "live",
               "criteria": COMMUNITY_CRITERIA[badge.badge_id]}
              for badge in BADGE_DEFINITIONS if badge.badge_id in COMMUNITY_CRITERIA]
    state_rows = supabase.table("badge_program_state").select("pending_ties,review,checked_at,last_error,revision,applied_revision").eq("club_id", club_id).execute().data or []
    program = state_rows[0] if state_rows else {}
    return {"ok": True, "program": program, "players": sorted(players, key=lambda row: str(row.get("name", "")).casefold()), "seasons": seasons,
            "badges": badges, "recent_awards": recent,
            "write_enabled": staging_write_wave_allows("badge-diagnostics")}


def save_badge_management(
    supabase: Any, *, club_id: str, actor_email: str, actor_user_id: str,
    actor_role: str, operation_id: str, action: str, payload: dict,
) -> dict:
    if actor_role not in {"administrator", "super_admin", "club_owner"}:
        raise PermissionError("Only club administrators can manage badge awards and seasons.")
    if not staging_write_wave_allows("badge-diagnostics"):
        raise PermissionError("Badge changes are currently paused.")
    require_staging_service_role_write(supabase, workflow="badge_management", required_tables=("badge_seasons", "admin_badge_operations"))
    operation_id = str(UUID(operation_id))
    actor_user_id = str(UUID(actor_user_id))
    if action == "award_community":
        contribution_date = date.fromisoformat(payload["contribution_date"])
        if contribution_date > datetime.now(timezone.utc).date():
            raise ValueError("The contribution date cannot be in the future.")
        award = build_community_award(club_id=club_id, player_id=payload["player_id"], badge_id=payload["badge_id"],
                                      recognition_id=operation_id, criteria=payload["criteria"], note=payload["note"], contribution_date=contribution_date)
        payload = {"player_id": award.player_id, "badge_id": award.badge_id, "criteria": award.value_json["criteria"],
                   "note": award.value_json["recognition_note"], "contribution_date": contribution_date.isoformat()}
    elif action == "save_season":
        season = BadgeSeason.from_row(payload | {"club_id": club_id})
        payload = {"id": str(UUID(season.id)), "name": season.name, "start_date": season.start_date.isoformat(),
                   "end_date": season.end_date.isoformat(), "timezone": season.timezone,
                   "expected_revision": int(payload["expected_revision"])}
    elif action == "resolve_round_robin":
        require_staging_service_role_write(supabase, workflow="round_robin_winner", required_tables=("badge_round_robin_decisions", "badge_program_state"))
        payload = {"source_key": str(payload["source_key"]), "source_fingerprint": str(payload["source_fingerprint"]),
                   "winner_player_id": int(payload["winner_player_id"])}
        result = supabase.rpc("admin_resolve_round_robin_v1", {"p_club_id": club_id, "p_actor_email": actor_email,
            "p_actor_user_id": actor_user_id, "p_actor_role": actor_role, "p_operation_id": operation_id, "p_payload": payload}).execute().data
        if not isinstance(result, dict) or not result.get("ok"):
            raise RuntimeError("Unable to confirm the round-robin decision. Retry the same request.")
        return result
    else:
        raise ValueError("Unknown badge action.")
    response = supabase.rpc("admin_manage_badges_v1", {"p_club_id": club_id, "p_actor_email": actor_email,
        "p_actor_user_id": actor_user_id, "p_actor_role": actor_role, "p_operation_id": operation_id,
        "p_action": action, "p_payload": payload}).execute()
    if not isinstance(response.data, dict) or not response.data.get("ok"):
        raise RuntimeError("The save could not be verified. Retry the same request to check its result.")
    return response.data
