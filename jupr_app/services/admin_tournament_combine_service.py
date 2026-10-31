from __future__ import annotations

import os
from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.tournament_admin_operations import stable_tournament_admin_fingerprint
from jupr_app.domain.tournament_draw_sources import draw_source_event_option_ids, validate_draw_source_events
from jupr_app.services.admin_tournament_draw_service import _draw_payload
from jupr_app.services.admin_tournament_guarded_operation import StaleTournamentAdminStateError
from jupr_app.services.admin_tournament_ops_service import _load_admin_tournament_ops_state, _strict_paginated_rows
from jupr_app.services.admin_tournament_service import is_admin_tournament_admin_enabled


CONFIRM_COMBINE_DIVISIONS = "COMBINE DIVISIONS"


def _active_draw(draw: dict[str, Any]) -> bool:
    return not draw.get("hidden_from_primary_ops") and not draw.get("combined_into_draw_id") and str(draw.get("status") or "draft").lower() not in {
        "cancelled", "canceled", "inactive", "disabled", "archived", "deleted",
    }


def _versions(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    if any(not row.get("id") or not row.get("updated_at") for row in rows):
        raise ValueError("A division, draw, or team is missing its review version. Reload before combining divisions.")
    return sorted([{"id": str(row["id"]), "updated_at": str(row["updated_at"])} for row in rows], key=lambda row: row["id"])


def combine_admin_tournament_draw_divisions(
    supabase: Any, *, club_id: str, tournament_id: str, draw_id: str,
    event_option_ids: list[str], name: str, expected_state_fingerprint: str,
    actor_email: str, actor_role: str, confirmation_text: str,
    source: str = "next_tournament_ops_combine_divisions", dry_run: bool = False,
) -> dict[str, Any]:
    """Combine reviewed pre-game draws without changing registration entries.

    Snapshot versions come from the exact state the operator reviewed. The RPC
    repeats all checks under locks and moves the complete team set atomically.
    """
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_COMBINE_DIVISIONS:
        raise ValueError(f"Type {CONFIRM_COMBINE_DIVISIONS} to combine these divisions.")
    selected_ids = [str(value or "").strip() for value in event_option_ids]
    if not 2 <= len(selected_ids) <= 16 or any(not value for value in selected_ids) or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Select between two and sixteen distinct registration divisions.")
    clean_name = str(name or "").strip()
    if not clean_name or len(clean_name) > 180:
        raise ValueError("Enter a combined draw name of at most 180 characters.")

    state = _load_admin_tournament_ops_state(supabase, club_id=str(club_id), tournament_id=str(tournament_id))
    if not expected_state_fingerprint or stable_tournament_admin_fingerprint(state) != expected_state_fingerprint:
        raise StaleTournamentAdminStateError("Tournament setup changed after review. Reload before combining divisions.")
    if str(state["tournament"].get("status") or "").upper() in {"COMPLETED", "ARCHIVED", "CANCELLED", "CANCELED"}:
        raise ValueError("Divisions cannot be combined after tournament closeout or cancellation.")
    tables = state["tables"]
    draws = tables["tournament_event_draws"]
    target = next((row for row in draws if str(row.get("id") or "") == str(draw_id)), None)
    if not target or not _active_draw(target):
        raise ValueError("Choose an active draft draw for the combined divisions.")
    primary_id = str(target.get("event_option_id") or "")
    if not primary_id or not set(draw_source_event_option_ids(target)).issubset(selected_ids):
        raise ValueError("Keep every division already represented by this draw selected.")
    selected_ids = [primary_id, *sorted(set(selected_ids) - {primary_id})]
    candidate = {**target, "source_event_option_ids": selected_ids, "name": clean_name}
    events = validate_draw_source_events(candidate, tables["tournament_event_options"])
    day_id = str(target.get("registration_day_id") or "")
    day = next((row for row in tables["tournament_registration_days"] if str(row.get("id") or "") == day_id), None)
    if not day or day.get("enabled") is not True:
        raise ValueError("The combined draw must belong to an enabled tournament day.")
    affected = [row for row in draws if _active_draw(row) and set(draw_source_event_option_ids(row)).intersection(selected_ids)]
    for draw in affected:
        if str(draw.get("registration_day_id") or "") != day_id:
            raise ValueError("An existing source draw belongs to another day. Review its schedule before combining.")
        if str(draw.get("draw_kind") or "STANDARD").upper() != "STANDARD" or str(draw.get("status") or "draft").lower() not in {"draft", "ready"}:
            raise ValueError("Only standard draft or ready draws can be combined before play.")
        if draw.get("pool_config") is not None:
            raise ValueError("Remove pool assignments from the affected draws before combining divisions.")
        if str(draw.get("id")) != str(draw_id) and len(draw_source_event_option_ids(draw)) > 1:
            raise ValueError("Select the existing combined draw as the destination before adding more divisions.")
    affected_ids = {str(row["id"]) for row in affected}
    teams = [row for row in tables["tournament_teams"] if str(row.get("draw_id") or "") in affected_ids]
    if len(teams) > 64:
        raise ValueError("A combined draw can contain at most 64 teams.")
    player_ids = [int(row[key]) for row in teams for key in ("player1_id", "player2_id") if row.get(key) is not None]
    if len(player_ids) != len(set(player_ids)):
        raise ValueError("A player appears on more than one source team. Resolve the duplicate before combining divisions.")
    for table in ("tournament_games", "tournament_podium"):
        if any(str(row.get("draw_id") or "") in affected_ids for row in tables[table]):
            raise ValueError("Combine divisions before games, results, or podiums exist in any source draw.")
    for table in ("tournament_day_live_draws", "tournament_day_live_queue"):
        rows = _strict_paginated_rows(supabase, table, filters=(("tournament_id", str(tournament_id)),))
        if any(str(row.get("draw_id") or "") in affected_ids for row in rows):
            raise ValueError("Combine divisions before any source draw has entered live operations.")
    source_draw_ids = sorted(affected_ids - {str(draw_id)})
    moved_count = sum(str(row.get("draw_id")) != str(draw_id) for row in teams)
    result = {"ok": True, "mode": "tournament_draw_combine_divisions", "draw": _draw_payload(candidate),
              "moved_team_count": moved_count, "combined_source_draw_ids": source_draw_ids, "warnings": []}
    snapshots = {"p_expected_draws": _versions(affected), "p_expected_teams": _versions(teams), "p_expected_events": _versions(events)}
    if dry_run:
        return {**result, "mode": "tournament_draw_combine_divisions_preview", "dry_run": True, "write_count": 0}
    try:
        response = supabase.rpc("admin_combine_tournament_draw_divisions_cas", {
            "p_club_id": str(club_id), "p_tournament_id": str(tournament_id), "p_target_draw_id": str(draw_id),
            **snapshots, "p_source_event_option_ids": selected_ids, "p_name": clean_name,
        }).execute()
    except Exception as exc:
        message = str(exc)
        if "STALE" in message or "LOCKED" in message:
            raise StaleTournamentAdminStateError("The divisions or their teams changed while combining. Reload before continuing.") from exc
        if "JUPR_TOURNAMENT_COMBINE" in message:
            raise ValueError("These divisions can no longer be combined. Refresh and review their teams, schedules, and draw status.") from exc
        raise RuntimeError("Unable to confirm the combined draw. Reload before retrying.") from exc
    saved = getattr(response, "data", None)
    if isinstance(saved, list) and len(saved) == 1:
        saved = saved[0]
    if not isinstance(saved, dict) or not isinstance(saved.get("draw"), dict):
        raise RuntimeError("Combine returned no authoritative draw. Reload before continuing.")
    result["draw"] = _draw_payload(saved["draw"])
    result["moved_team_count"] = saved.get("moved_team_count", moved_count)
    result["combined_source_draw_ids"] = saved.get("combined_source_draw_ids", source_draw_ids)
    audit = write_admin_activity_log(supabase, build_activity_payload(
        club_id=str(club_id), actor_email=actor_email, actor_role=actor_role,
        action_type="combine_tournament_draw_divisions_admin", entity_type="tournament_event_draw", entity_id=str(draw_id),
        before_json={"draws": [_draw_payload(row) for row in affected], "team_versions": snapshots["p_expected_teams"]},
        after_json={**result, "source_client": "fastapi/nextjs", "source_page": source}, source_page=source, flagged_for_review=True,
    ))
    result["warnings"] = [audit.warning] if audit.warning else []
    if not audit.ok and os.getenv("JUPR_REQUIRE_API_AUDIT_LOG", "").lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError("The combined draw was saved but its audit record could not be confirmed. Reload before continuing.")
    return result
