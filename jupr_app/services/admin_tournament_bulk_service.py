from __future__ import annotations

from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.tournament_registration_repo import (
    ADMIN_PAYMENT_STATUS_OPTIONS,
    ADMIN_REGISTRATION_STATUS_OPTIONS,
    update_admin_registration,
    registration_is_imported_to_draw,
    StaleTournamentRegistrationAdminError,
)
from jupr_app.services.admin_tournament_service import (
    TOURNAMENT_SELECT,
    _clean_text,
    _fetch_registration_by_id,
    _first_row,
    _registration_payload,
    _selection_count_for_registration,
    is_admin_tournament_admin_enabled,
    is_api_audit_log_required,
)

CONFIRM_BULK_REGISTRATION_UPDATE = "BULK UPDATE REGISTRATIONS"


def _unique_ids(values: list[str] | list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        clean = _clean_text(value, limit=120)
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def _bulk_update_payload(patch: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if "registration_status" in patch and _clean_text(patch.get("registration_status"), limit=40):
        next_status = _clean_text(patch.get("registration_status"), limit=40).lower()
        if next_status not in ADMIN_REGISTRATION_STATUS_OPTIONS:
            raise ValueError(f"Invalid registration status: {patch.get('registration_status')}")
        payload["status"] = next_status
    if "payment_status" in patch and _clean_text(patch.get("payment_status"), limit=40):
        next_payment = _clean_text(patch.get("payment_status"), limit=40).lower()
        if next_payment not in ADMIN_PAYMENT_STATUS_OPTIONS:
            raise ValueError(f"Invalid payment status: {patch.get('payment_status')}")
        payload["payment_status"] = next_payment
    return payload


def bulk_update_admin_tournament_registrations(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    registration_ids: list[str],
    patch: dict[str, Any],
    expected_versions: dict[str, str] | None = None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_tournament_admin_registration_bulk_update",
    dry_run: bool = False,
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_BULK_REGISTRATION_UPDATE:
        raise ValueError(f"Type {CONFIRM_BULK_REGISTRATION_UPDATE} to confirm bulk registration changes.")
    clean_tournament_id = _clean_text(tournament_id, limit=120)
    tournament = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id)
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    ids = _unique_ids(registration_ids)
    if not ids:
        raise ValueError("Select at least one registration.")
    if len(ids) > 100:
        raise ValueError("No more than 100 registrations can be updated at once.")

    common_payload = _bulk_update_payload(patch)
    note_text = _clean_text(patch.get("append_note"), limit=1000)
    if not common_payload and not note_text:
        raise ValueError("Choose a registration status, payment status, or note to apply.")

    # Preflight every selected row before the first write so a stale or imported
    # registration cannot leave a partially applied bulk operation.
    before_by_id: dict[str, dict[str, Any]] = {}
    for registration_id in ids:
        before = _fetch_registration_by_id(
            supabase,
            tournament_id=clean_tournament_id,
            registration_id=registration_id,
        )
        if before is None:
            if expected_versions is not None:
                raise StaleTournamentRegistrationAdminError(
                    f"Registration {registration_id} is no longer present. Refresh and review the entire bulk selection."
                )
            continue
        before_by_id[registration_id] = before
        expected = str((expected_versions or {}).get(registration_id) or "").strip()
        if expected_versions is not None and (
            not expected or expected != str(before.get("updated_at") or "").strip()
        ):
            raise StaleTournamentRegistrationAdminError(
                f"Registration {registration_id} changed after the bulk selection was loaded. Refresh and review the entire selection."
            )
        if "status" in common_payload and registration_is_imported_to_draw(
            supabase,
            tournament_id=clean_tournament_id,
            registration_id=registration_id,
        ):
            raise ValueError(
                "A selected registration is already imported into a draw. Change draw membership in Tournament Ops before applying a bulk registration-status update."
            )

    if dry_run:
        return {
            "ok": True,
            "mode": "tournament_registration_bulk_update_preflight",
            "dry_run": True,
            "write_count": 0,
            "registration_ids": ids,
            "patch": {**common_payload, "append_note": note_text or None},
        }

    before_payloads: list[dict[str, Any]] = []
    after_payloads: list[dict[str, Any]] = []
    updated_ids: list[str] = []
    skipped: list[str] = []
    for registration_id in ids:
        before = before_by_id.get(registration_id)
        if before is None:
            skipped.append(f"{registration_id}: not found")
            continue
        update_payload = dict(common_payload)
        if note_text:
            existing_note = _clean_text(before.get("notes"), limit=2000)
            update_payload["notes"] = f"{existing_note}\n{note_text}".strip() if existing_note else note_text
        updated = update_admin_registration(
            supabase,
            tournament_id=clean_tournament_id,
            registration_id=registration_id,
            payload=update_payload,
            expected_updated_at=(expected_versions or {}).get(registration_id) if expected_versions is not None else None,
        )
        selection_count = _selection_count_for_registration(supabase, tournament_id=clean_tournament_id, registration_id=registration_id)
        before_payloads.append(_registration_payload(before, selection_count=selection_count))
        after_payloads.append(_registration_payload(updated, selection_count=selection_count))
        updated_ids.append(registration_id)

    if not updated_ids:
        raise ValueError("No selected registrations were updated.")

    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="bulk_update_tournament_registrations_admin",
        entity_type="tournament_registration",
        entity_id="bulk",
        before_json={"registrations": before_payloads},
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "registration_ids": updated_ids,
            "patch": {**common_payload, "append_note": note_text or None},
            "registrations": after_payloads,
            "skipped": skipped,
        },
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
        "mode": "tournament_registration_bulk_update",
        "updated_count": len(updated_ids),
        "registration_ids": updated_ids,
        "registrations": after_payloads,
        "skipped": skipped,
        "warnings": warnings,
    }
