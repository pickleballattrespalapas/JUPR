from __future__ import annotations

from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.tournament_registration_repo import archive_tournament, unarchive_tournament
from jupr_app.services.admin_tournament_service import (
    TOURNAMENT_SELECT,
    _clean_text,
    _first_row,
    _tournament_payload,
    is_admin_tournament_admin_enabled,
    is_api_audit_log_required,
)

CONFIRM_ARCHIVE = "ARCHIVE"
CONFIRM_UNARCHIVE = "UNARCHIVE"


def apply_admin_tournament_status_action(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    action: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_tournament_admin_status_action",
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    clean_tournament_id = _clean_text(tournament_id, limit=120)
    normalized_action = _clean_text(action, limit=40).lower()
    if normalized_action not in {"archive", "unarchive"}:
        raise ValueError("action must be archive or unarchive")
    expected_confirmation = CONFIRM_ARCHIVE if normalized_action == "archive" else CONFIRM_UNARCHIVE
    if str(confirmation_text or "").strip().upper() != expected_confirmation:
        raise ValueError(f"Type {expected_confirmation} to confirm tournament status change.")

    before = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id)
    if not before or str(before.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    before_payload = _tournament_payload(before)
    if normalized_action == "archive" and str(before.get("status") or "").upper() == "ARCHIVED":
        raise ValueError("Tournament is already archived.")
    if normalized_action == "unarchive" and str(before.get("status") or "").upper() != "ARCHIVED":
        raise ValueError("Only archived tournaments can be unarchived.")

    if normalized_action == "archive":
        archive_tournament(supabase, clean_tournament_id)
    else:
        unarchive_tournament(supabase, clean_tournament_id)

    after = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id) or {**before, "status": "ARCHIVED" if normalized_action == "archive" else "DRAFT"}
    tournament = _tournament_payload(after)
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type=f"{normalized_action}_tournament_admin",
        entity_type="tournament",
        entity_id=clean_tournament_id,
        before_json={"tournament": before_payload},
        after_json={"source_client": "fastapi/nextjs", "source_page": source, "action": normalized_action, "tournament": tournament},
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and is_api_audit_log_required():
        raise RuntimeError("audit log write required but unavailable")
    return {"ok": True, "mode": "tournament_status_action", "action": normalized_action, "tournament": tournament, "warnings": warnings}
