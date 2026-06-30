from __future__ import annotations

from typing import Any

from jupr_app.domain.tournament_registration_repo import delete_registration
from jupr_app.ui.pages import tournament_registration_admin_streamlined as base

BULK_DELETE_CANCELLED_ACTION = "Hard delete cancelled registrations"
BULK_ACTIONS = [
    *[action for action in base.BULK_ACTIONS if action != BULK_DELETE_CANCELLED_ACTION],
    BULK_DELETE_CANCELLED_ACTION,
]


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _unique_cancelled_registration_rows(selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for row in selected_rows:
        reg_id = _safe_text(row.get("registration_id"))
        if not reg_id or reg_id in seen:
            continue
        seen.add(reg_id)
        rows.append(row)
    return rows


def _hard_delete_cancelled_registrations(*, supabase, tournament_id: str, selected_rows: list[dict[str, Any]]) -> tuple[int, list[str]]:
    changed = 0
    skipped: list[str] = []
    for row in _unique_cancelled_registration_rows(selected_rows):
        label = _safe_text(row.get("label") or row.get("entry_key"))
        reg_id = _safe_text(row.get("registration_id"))
        if _safe_text(row.get("registration_status")).lower() != "cancelled":
            skipped.append(f"{label} — not cancelled")
            continue
        try:
            delete_registration(supabase, tournament_id=tournament_id, registration_id=reg_id)
            changed += 1
        except Exception as exc:
            skipped.append(f"{label} — {exc}")
    return changed, skipped


def _apply_bulk_action(
    *,
    supabase,
    tournament_id: str,
    selected_rows: list[dict[str, Any]],
    action: str,
    status_value: str,
    payment_value: str,
    partner_mode_value: str,
    target_event_id: str,
    event_lookup: dict[str, dict[str, Any]],
    note_text: str,
) -> tuple[int, list[str]]:
    if action == BULK_DELETE_CANCELLED_ACTION:
        return _hard_delete_cancelled_registrations(
            supabase=supabase,
            tournament_id=tournament_id,
            selected_rows=selected_rows,
        )
    return base._apply_bulk_action(
        supabase=supabase,
        tournament_id=tournament_id,
        selected_rows=selected_rows,
        action=action,
        status_value=status_value,
        payment_value=payment_value,
        partner_mode_value=partner_mode_value,
        target_event_id=target_event_id,
        event_lookup=event_lookup,
        note_text=note_text,
    )


base.BULK_ACTIONS = BULK_ACTIONS
base._apply_bulk_action = _apply_bulk_action

render = base.render
