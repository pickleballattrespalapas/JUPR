from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_player_editor_service import (
    _fetch_league_ratings,
    _jupr_to_elo,
    _league_rating_payload,
    _safe_int,
    _safe_rows,
    is_admin_player_editor_enabled,
    is_api_audit_log_required,
)

REQUIRED_CONFIRMATION = "SAVE LEAGUE RATING"


def _fetch_league_rating_row(supabase: Any, *, club_id: str, player_id: int, league_rating_id: int) -> dict[str, Any] | None:
    rows = _safe_rows(
        supabase.table("league_ratings")
        .select("id,club_id,player_id,league_name,rating,starting_rating,wins,losses,matches_played,is_active,inactive_at")
        .eq("club_id", str(club_id))
        .eq("player_id", int(player_id))
        .eq("id", int(league_rating_id))
        .limit(1)
        .execute()
    )
    return rows[0] if rows else None


def update_admin_player_editor_league_rating(
    supabase: Any,
    *,
    club_id: str,
    player_id: int,
    league_rating_id: int,
    patch: dict[str, Any],
    actor_email: str,
    actor_role: str,
    confirmation_text: str = "",
    source: str = "next_player_editor_league_rating",
) -> dict[str, Any]:
    if not is_admin_player_editor_enabled():
        raise PermissionError("Next Player Editor is disabled.")
    normalized_confirmation = str(confirmation_text or "").strip().upper()
    if normalized_confirmation != REQUIRED_CONFIRMATION:
        raise ValueError(f"Type {REQUIRED_CONFIRMATION} to confirm league-rating edits.")
    before = _fetch_league_rating_row(
        supabase,
        club_id=str(club_id),
        player_id=int(player_id),
        league_rating_id=int(league_rating_id),
    )
    if before is None:
        raise ValueError("league rating not found")

    update_payload: dict[str, Any] = {}
    if "rating_jupr" in patch:
        update_payload["rating"] = _jupr_to_elo(patch.get("rating_jupr"), field_name="League JUPR")
    if "starting_jupr" in patch:
        update_payload["starting_rating"] = _jupr_to_elo(patch.get("starting_jupr"), field_name="League starting JUPR")
    if "is_active" in patch:
        next_active = bool(patch.get("is_active"))
        update_payload["is_active"] = next_active
        update_payload["inactive_at"] = None if next_active else (before.get("inactive_at") or datetime.now(timezone.utc).isoformat())
    if not update_payload:
        raise ValueError("No supported league-rating fields were provided.")

    updated_rows = _safe_rows(
        supabase.table("league_ratings")
        .update(update_payload)
        .eq("club_id", str(club_id))
        .eq("player_id", int(player_id))
        .eq("id", int(league_rating_id))
        .execute()
    )
    after = _league_rating_payload(updated_rows[0]) if updated_rows else _league_rating_payload(before | update_payload)
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="update_player_editor_league_rating",
        entity_type="league_rating",
        entity_id=str(int(league_rating_id)),
        before_json={"league_rating": _league_rating_payload(before)},
        after_json={"source_client": "fastapi/nextjs", "source_page": source, "patch": update_payload, "league_rating": after},
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and is_api_audit_log_required():
        raise RuntimeError("audit log write required but unavailable")
    return {
        "ok": True,
        "mode": "player_editor_league_rating_update",
        "league_rating": after,
        "league_ratings": _fetch_league_ratings(supabase, club_id=str(club_id), player_id=int(player_id)),
        "warnings": warnings,
    }
