from __future__ import annotations

import csv
import io
from typing import Any

from jupr_app.domain.notifications.tournament_registrant_broadcast_email import (
    build_tournament_registrant_broadcast_email_html,
    build_tournament_registrant_broadcast_email_text,
    build_tournament_registrant_broadcast_subject,
)
from jupr_app.services.admin_tournament_service import (
    DAY_SELECT,
    EVENT_OPTION_SELECT,
    SELECTION_SELECT,
    TOURNAMENT_SELECT,
    _clean_text,
    _display_name,
    _event_label,
    _registration_status,
    is_admin_tournament_admin_enabled,
)


REGISTRATION_EXPORT_COLUMNS = [
    "registration_id",
    "selection_id",
    "player_id",
    "display_name",
    "email",
    "phone",
    "registration_status",
    "payment_status",
    "submitted_at",
    "registration_day_id",
    "day",
    "event_option_id",
    "division",
    "partner_mode",
    "partner_name",
    "partner_email",
    "partner_phone",
    "notes",
]

RECIPIENT_EXPORT_COLUMNS = ["name", "email", "registration_status", "payment_status"]
MAX_EXPORT_REGISTRATIONS = 2000
MAX_EXPORT_SELECTIONS = 5000
MAX_TOURNAMENT_DAYS = 100
MAX_TOURNAMENT_EVENTS = 500


def _execute_complete_rows(
    query: Any,
    *,
    label: str,
    max_rows: int,
) -> list[dict[str, Any]]:
    try:
        response = query.limit(int(max_rows) + 1).execute()
    except Exception as exc:
        raise RuntimeError(f"Could not load {label}.") from exc
    try:
        rows = [dict(row) for row in (response.data or [])]
    except Exception as exc:
        raise RuntimeError(f"Could not parse {label}.") from exc

    total = getattr(response, "count", None)
    if total is not None:
        try:
            total_count = int(total)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Could not verify the {label} row count.") from exc
        if total_count > len(rows):
            raise RuntimeError(
                f"The {label} result was truncated ({len(rows)} of {total_count} rows)."
            )
    if len(rows) > max_rows:
        raise RuntimeError(
            f"The {label} result exceeds the safe export limit of {max_rows} rows."
        )
    return rows


def _require_tournament(supabase: Any, *, club_id: str, tournament_id: str) -> dict[str, Any]:
    clean_tournament_id = _clean_text(tournament_id, limit=120)
    if not clean_tournament_id:
        raise ValueError("tournament_id is required")
    rows = _execute_complete_rows(
        supabase.table("tournaments")
        .select(TOURNAMENT_SELECT, count="exact")
        .eq("id", clean_tournament_id),
        label="tournament",
        max_rows=1,
    )
    tournament = rows[0] if rows else None
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    return tournament


def _clean_email(value: Any) -> str:
    return _clean_text(value, limit=180).lower()


def _spreadsheet_safe_cell(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
        return f"'{value}"
    return value


def _csv_text(rows: list[dict[str, Any]], columns: list[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=columns,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                column: _spreadsheet_safe_cell(row.get(column))
                for column in columns
            }
        )
    return buffer.getvalue()


def _registration_export_rows(
    supabase: Any,
    *,
    tournament_id: str,
) -> list[dict[str, Any]]:
    registrations = _execute_complete_rows(
        supabase.table("tournament_registrations")
        .select(
            (
                "id,tournament_id,player_id,first_name,last_name,display_name,"
                "email,phone,status,payment_status,notes,"
                "wants_partner_board_contact,submitted_at,updated_at"
            ),
            count="exact",
        )
        .eq("tournament_id", str(tournament_id))
        .order("submitted_at", desc=True),
        label="tournament registrations",
        max_rows=MAX_EXPORT_REGISTRATIONS,
    )
    selections = _execute_complete_rows(
        supabase.table("tournament_registration_selections")
        .select(SELECTION_SELECT, count="exact")
        .eq("tournament_id", str(tournament_id)),
        label="tournament registration selections",
        max_rows=MAX_EXPORT_SELECTIONS,
    )
    days = {
        _clean_text(row.get("id"), limit=120): row
        for row in _execute_complete_rows(
            supabase.table("tournament_registration_days")
            .select(DAY_SELECT, count="exact")
            .eq("tournament_id", str(tournament_id)),
            label="tournament registration days",
            max_rows=MAX_TOURNAMENT_DAYS,
        )
    }
    events = {
        _clean_text(row.get("id"), limit=120): row
        for row in _execute_complete_rows(
            supabase.table("tournament_event_options")
            .select(EVENT_OPTION_SELECT, count="exact")
            .eq("tournament_id", str(tournament_id)),
            label="tournament event options",
            max_rows=MAX_TOURNAMENT_EVENTS,
        )
    }
    selections_by_registration: dict[str, list[dict[str, Any]]] = {}
    for selection in selections:
        registration_id = _clean_text(selection.get("registration_id"), limit=120)
        if registration_id:
            selections_by_registration.setdefault(registration_id, []).append(selection)

    rows: list[dict[str, Any]] = []
    for registration in registrations:
        registration_id = _clean_text(registration.get("id"), limit=120)
        related = selections_by_registration.get(registration_id) or [None]
        for raw_selection in related:
            selection = raw_selection or {}
            day_id = _clean_text(selection.get("registration_day_id"), limit=120)
            event_id = _clean_text(selection.get("event_option_id"), limit=120)
            day = days.get(day_id, {})
            event = events.get(event_id, {})
            rows.append(
                {
                    "registration_id": registration_id,
                    "selection_id": _clean_text(selection.get("id"), limit=120),
                    "player_id": registration.get("player_id"),
                    "display_name": _display_name(registration),
                    "email": _clean_email(registration.get("email")),
                    "phone": _clean_text(registration.get("phone"), limit=80),
                    "registration_status": _registration_status(registration),
                    "payment_status": _clean_text(
                        registration.get("payment_status") or "unpaid",
                        limit=40,
                    ).lower(),
                    "submitted_at": (
                        registration.get("submitted_at")
                        or registration.get("created_at")
                        or ""
                    ),
                    "registration_day_id": day_id,
                    "day": _clean_text(
                        day.get("label") or day.get("event_date") or day.get("date"),
                        limit=160,
                    ),
                    "event_option_id": event_id,
                    "division": _event_label(event) if event_id else "",
                    "partner_mode": _clean_text(
                        selection.get("partner_mode") or "NONE",
                        limit=40,
                    ).upper(),
                    "partner_name": _clean_text(
                        selection.get("partner_name"),
                        limit=160,
                    ),
                    "partner_email": _clean_email(selection.get("partner_email")),
                    "partner_phone": _clean_text(
                        selection.get("partner_phone"),
                        limit=80,
                    ),
                    "notes": _clean_text(registration.get("notes"), limit=2000),
                }
            )
    return rows


def _filter_export_rows(
    rows: list[dict[str, Any]],
    *,
    registration_status: str | None = None,
    payment_status: str | None = None,
    partner_mode: str | None = None,
    registration_day_id: str | None = None,
    event_option_id: str | None = None,
    search: str | None = None,
) -> list[dict[str, Any]]:
    clean_registration_status = _clean_text(
        registration_status,
        limit=40,
    ).lower()
    clean_payment_status = _clean_text(payment_status, limit=40).lower()
    clean_partner_mode = _clean_text(partner_mode, limit=40).upper()
    clean_day_id = _clean_text(registration_day_id, limit=120)
    clean_event_id = _clean_text(event_option_id, limit=120)
    clean_search = _clean_text(search, limit=200).lower()
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if (
            clean_registration_status
            and row.get("registration_status") != clean_registration_status
        ):
            continue
        if clean_payment_status and row.get("payment_status") != clean_payment_status:
            continue
        if clean_partner_mode and row.get("partner_mode") != clean_partner_mode:
            continue
        if clean_day_id and row.get("registration_day_id") != clean_day_id:
            continue
        if clean_event_id and row.get("event_option_id") != clean_event_id:
            continue
        if clean_search:
            search_blob = " ".join(
                _clean_text(row.get(field), limit=300)
                for field in [
                    "display_name",
                    "email",
                    "phone",
                    "day",
                    "division",
                    "partner_name",
                    "partner_email",
                ]
            ).lower()
            if clean_search not in search_blob:
                continue
        filtered.append(row)
    return filtered


def build_admin_tournament_registration_export(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    registration_status: str | None = None,
    payment_status: str | None = None,
    partner_mode: str | None = None,
    registration_day_id: str | None = None,
    event_option_id: str | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    clean_tournament_id = _clean_text(tournament_id, limit=120)
    _require_tournament(
        supabase,
        club_id=str(club_id),
        tournament_id=clean_tournament_id,
    )
    rows = _filter_export_rows(
        _registration_export_rows(
            supabase,
            tournament_id=clean_tournament_id,
        ),
        registration_status=registration_status,
        payment_status=payment_status,
        partner_mode=partner_mode,
        registration_day_id=registration_day_id,
        event_option_id=event_option_id,
        search=search,
    )
    return {
        "ok": True,
        "mode": "tournament_registration_csv_export",
        "rows": rows,
        "row_count": len(rows),
        "csv": _csv_text(rows, REGISTRATION_EXPORT_COLUMNS),
    }


def build_admin_tournament_broadcast_preview(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    subject: str = "",
    message: str = "",
    include_cancelled: bool = False,
    registration_status: str | None = None,
    payment_status: str | None = None,
    partner_mode: str | None = None,
    registration_day_id: str | None = None,
    event_option_id: str | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    clean_tournament_id = _clean_text(tournament_id, limit=120)
    tournament = _require_tournament(
        supabase,
        club_id=str(club_id),
        tournament_id=clean_tournament_id,
    )
    filtered_rows = _filter_export_rows(
        _registration_export_rows(
            supabase,
            tournament_id=clean_tournament_id,
        ),
        registration_status=registration_status,
        payment_status=payment_status,
        partner_mode=partner_mode,
        registration_day_id=registration_day_id,
        event_option_id=event_option_id,
        search=search,
    )
    recipients: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in filtered_rows:
        status = _clean_text(
            row.get("registration_status") or "confirmed",
            limit=40,
        ).lower()
        if status == "cancelled" and not include_cancelled:
            continue
        email = _clean_email(row.get("email"))
        if not email or email in seen:
            continue
        seen.add(email)
        recipients.append(
            {
                "name": _clean_text(row.get("display_name"), limit=160) or email,
                "email": email,
                "registration_status": status,
                "payment_status": _clean_text(
                    row.get("payment_status") or "unpaid",
                    limit=40,
                ).lower(),
            }
        )
    recipients.sort(key=lambda row: (row["name"].lower(), row["email"]))

    tournament_name = _clean_text(
        tournament.get("name") or "Tournament",
        limit=180,
    )
    clean_subject = _clean_text(subject, limit=200)
    clean_message = str(message or "").replace("\x00", "").strip()[:10000]
    preview_recipient = (
        recipients[0]
        if recipients
        else {"name": "Registrant", "email": ""}
    )
    final_subject = build_tournament_registrant_broadcast_subject(
        tournament_name=tournament_name,
        subject=clean_subject,
    )
    return {
        "ok": True,
        "mode": "tournament_broadcast_preview",
        "dry_run": True,
        "send_available": False,
        "recipient_count": len(recipients),
        "recipients": recipients,
        "recipient_csv": _csv_text(recipients, RECIPIENT_EXPORT_COLUMNS),
        "preview": {
            "to_name": preview_recipient["name"],
            "to_email": preview_recipient["email"],
            "subject": final_subject,
            "text": build_tournament_registrant_broadcast_email_text(
                tournament_name=tournament_name,
                recipient_name=preview_recipient["name"],
                subject=final_subject,
                message=clean_message,
            ),
            "html": build_tournament_registrant_broadcast_email_html(
                tournament_name=tournament_name,
                recipient_name=preview_recipient["name"],
                subject=final_subject,
                message=clean_message,
            ),
        },
        "warnings": ["Preview only. This endpoint never sends email."],
    }
