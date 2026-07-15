from __future__ import annotations

from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.tournament_registration_repo import delete_unused_draft_tournament, tournament_can_be_deleted
from jupr_app.services.admin_tournament_service import (
    TOURNAMENT_SELECT,
    _clean_text,
    _first_row,
    _tournament_payload,
    is_admin_tournament_admin_enabled,
    is_api_audit_log_required,
)

CONFIRM_DELETE_DRAFT = "DELETE DRAFT"


def delete_admin_tournament_draft(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_tournament_admin_delete_draft",
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_DELETE_DRAFT:
        raise ValueError(f"Type {CONFIRM_DELETE_DRAFT} to confirm draft deletion.")
    clean_tournament_id = _clean_text(tournament_id, limit=120)
    tournament = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id)
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    can_delete, usage_summary, reason = tournament_can_be_deleted(supabase, tournament)
    if not can_delete:
        raise ValueError(reason or "Tournament cannot be deleted.")
    before_payload = _tournament_payload(tournament)
    delete_unused_draft_tournament(supabase, tournament)
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="delete_draft_tournament_admin",
        entity_type="tournament",
        entity_id=clean_tournament_id,
        before_json={"tournament": before_payload, "usage_summary": usage_summary},
        after_json={"source_client": "fastapi/nextjs", "source_page": source, "deleted": True, "tournament_id": clean_tournament_id},
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
        "mode": "tournament_draft_deleted",
        "tournament_id": clean_tournament_id,
        "usage_summary": usage_summary,
        "warnings": warnings,
    }
