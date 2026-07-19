from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.leagues import normalize_league_status
from jupr_app.services.admin_league_manager_service import (
    get_admin_league_manager_detail,
    is_admin_league_manager_enabled,
    validate_admin_league_manager_lifecycle_state,
)

TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
LIFECYCLE_CONFIRMATIONS = {
    "start": "START LEAGUE",
    "pause": "PAUSE LEAGUE",
    "resume": "RESUME LEAGUE",
    "end": "END LEAGUE",
    "archive": "ARCHIVE LEAGUE",
}
ALLOWED_TRANSITIONS = {
    "start": {"draft"},
    "pause": {"active"},
    "resume": {"paused"},
    "end": {"active", "paused"},
    "archive": {"ended"},
}


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _clean_text(value: Any, *, limit: int) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _fetch_league_meta(supabase: Any, *, club_id: str, league_name: str) -> dict[str, Any] | None:
    rows = _safe_rows(
        supabase.table("leagues_metadata")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("league_name", str(league_name))
        .limit(1)
        .execute()
    )
    return rows[0] if rows else None


def _transition_patch(
    *,
    action: str,
    before: dict[str, Any],
    actor_email: str,
    now_iso: str,
) -> dict[str, Any]:
    if action in {"start", "resume"}:
        return {
            "status": "active",
            "is_active": True,
            "started_at": before.get("started_at") or now_iso,
            "ended_at": None,
            "ended_by": None,
            "updated_at": now_iso,
        }
    if action == "pause":
        return {"status": "paused", "is_active": False, "updated_at": now_iso}
    if action == "end":
        return {
            "status": "ended",
            "is_active": False,
            "ended_at": now_iso,
            "ended_by": str(actor_email or ""),
            "updated_at": now_iso,
        }
    return {"status": "archived", "is_active": False, "updated_at": now_iso}


def _rollback_transition(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    before: dict[str, Any],
    changed_fields: set[str],
    expected_status: str,
    expected_updated_at: str,
) -> None:
    """Best-effort compensation when staging requires an audit row."""

    rollback = {field: before.get(field) for field in changed_fields}
    try:
        (
            supabase.table("leagues_metadata")
            .update(rollback)
            .eq("club_id", str(club_id))
            .eq("league_name", str(league_name))
            .eq("status", str(expected_status))
            .eq("is_active", bool(expected_status == "active"))
            .eq("updated_at", str(expected_updated_at))
            .execute()
        )
    except Exception:
        pass


def transition_admin_league_manager_lifecycle(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    action: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_league_manager_lifecycle",
) -> dict[str, Any]:
    if not is_admin_league_manager_enabled():
        raise PermissionError("Next League Manager is disabled.")

    clean_action = _clean_text(action, limit=40).lower()
    if clean_action not in LIFECYCLE_CONFIRMATIONS:
        raise ValueError("action must be one of start, pause, resume, end, or archive.")
    expected_confirmation = LIFECYCLE_CONFIRMATIONS[clean_action]
    if _clean_text(confirmation_text, limit=80).upper() != expected_confirmation:
        raise ValueError(f"Type {expected_confirmation} to {clean_action} this league.")

    clean_league = _clean_text(league_name, limit=120)
    if not clean_league:
        raise ValueError("league_name is required")
    before = _fetch_league_meta(supabase, club_id=str(club_id), league_name=clean_league)
    if before is None:
        raise ValueError("league not found")

    previous_status = validate_admin_league_manager_lifecycle_state(before)
    allowed_from = ALLOWED_TRANSITIONS[clean_action]
    if previous_status not in allowed_from:
        allowed_label = " or ".join(sorted(allowed_from))
        raise ValueError(
            f"Cannot {clean_action} a {previous_status} league; expected {allowed_label}."
        )

    patch = _transition_patch(
        action=clean_action,
        before=before,
        actor_email=actor_email,
        now_iso=_now_iso(),
    )
    update_query = (
        supabase.table("leagues_metadata")
        .update(patch)
        .eq("club_id", str(club_id))
        .eq("league_name", clean_league)
    )
    raw_previous_status = before.get("status")
    if raw_previous_status not in (None, ""):
        update_query = update_query.eq("status", str(raw_previous_status))
    update_query = update_query.eq("is_active", bool(before.get("is_active", False)))
    updated = _safe_rows(update_query.execute())
    if not updated:
        raise ValueError("League status changed before this action completed; reload and try again.")
    after = updated[0]
    new_status = normalize_league_status(after)

    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type=f"{clean_action}_league_manager_admin",
        entity_type="leagues_metadata",
        entity_id=clean_league,
        before_json=before,
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "action": clean_action,
            "previous_status": previous_status,
            "new_status": new_status,
            "patch": patch,
            "league": after,
        },
        source_page=source,
        flagged_for_review=True,
    )
    try:
        audit_write = write_admin_activity_log(supabase, audit_payload)
    except Exception:
        if _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
            _rollback_transition(
                supabase,
                club_id=str(club_id),
                league_name=clean_league,
                before=before,
                changed_fields=set(patch),
                expected_status=str(patch["status"]),
                expected_updated_at=str(patch["updated_at"]),
            )
        raise
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        _rollback_transition(
            supabase,
            club_id=str(club_id),
            league_name=clean_league,
            before=before,
            changed_fields=set(patch),
            expected_status=str(patch["status"]),
            expected_updated_at=str(patch["updated_at"]),
        )
        raise RuntimeError("audit log write required but unavailable")

    detail = get_admin_league_manager_detail(
        supabase,
        club_id=str(club_id),
        league_name=clean_league,
    )
    return {
        "ok": True,
        "mode": "league_manager_lifecycle_transition",
        "action": clean_action,
        "previous_status": previous_status,
        "new_status": new_status,
        "league": detail.get("league"),
        "detail": detail,
        "warnings": warnings,
    }
