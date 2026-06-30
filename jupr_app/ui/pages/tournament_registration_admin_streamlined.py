from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
import streamlit as st

from jupr_app.domain.tournament_registration_repo import (
    ADMIN_PAYMENT_STATUS_OPTIONS,
    ADMIN_REGISTRATION_STATUS_OPTIONS,
    PARTNER_MODE_OPTIONS,
    build_public_urls,
    cancel_registration,
    create_admin_registration,
    delete_registration,
    get_registration_settings,
    list_event_options as list_registration_event_options,
    list_existing_tournaments,
    list_registration_admin_rows,
    list_registration_days,
    registration_feature_available,
    registration_is_imported_to_draw,
    update_admin_registration,
    update_admin_registration_selection,
)
from jupr_app.ui.layout import page_shell
from jupr_app.ui.public_links import navigate_same_tab


BULK_ACTIONS = [
    "Set registration status",
    "Set payment status",
    "Set partner mode",
    "Move division",
    "Append admin note",
    "Cancel registrations",
]


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _coerce_int(value: Any) -> int | None:
    text = _safe_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def _admin_tournament_label(tournament: dict[str, Any]) -> str:
    name = _safe_text(tournament.get("name") or f"Tournament #{tournament.get('id')}")
    status = _safe_text(tournament.get("status") or "DRAFT")
    start_date = _safe_text(tournament.get("start_date"))
    parts = [part for part in [status, start_date] if part]
    return f"{name} ({' · '.join(parts)})" if parts else name


def _select_admin_tournament(ctx, supabase, *, page_key: str):
    club_id = _safe_text(getattr(ctx, "club_id", ""))
    show_archived = st.checkbox("Show archived tournaments", value=False, key="reg_admin_show_archived")
    tournaments = list_existing_tournaments(supabase, club_id, include_archived=show_archived)
    if not tournaments:
        st.info("No tournaments are available.")
        return None, {}, [], []

    requested_id = _safe_text(st.query_params.get("tournament_id"))
    tournament_ids = [str(row.get("id")) for row in tournaments]
    default_index = tournament_ids.index(requested_id) if requested_id in tournament_ids else 0
    selected_id = st.selectbox(
        "Tournament",
        tournament_ids,
        index=default_index,
        format_func=lambda tid: _admin_tournament_label(next(row for row in tournaments if str(row.get("id")) == tid)),
        key="reg_admin_tournament_select",
    )
    if _safe_text(st.query_params.get("page")) != page_key or _safe_text(st.query_params.get("tournament_id")) != selected_id:
        st.query_params["page"] = page_key
        st.query_params["admin"] = "1"
        st.query_params["tournament_id"] = selected_id
        st.query_params.pop("public", None)
        st.rerun()

    tournament = next(row for row in tournaments if str(row.get("id")) == selected_id)
    settings = get_registration_settings(supabase, selected_id, tournament_name=_safe_text(tournament.get("name")))
    days = list_registration_days(supabase, selected_id)
    event_options = list_registration_event_options(supabase, selected_id)
    return tournament, settings, days, event_options


def _display_name(reg: dict[str, Any]) -> str:
    display = _safe_text(reg.get("display_name"))
    if display:
        return display
    return " ".join(part for part in [_safe_text(reg.get("first_name")), _safe_text(reg.get("last_name"))] if part) or _safe_text(reg.get("email")) or "Player"


def _event_label(event: dict[str, Any] | None) -> str:
    event = event or {}
    family = _safe_text(event.get("event_family_label"))
    division = _safe_text(event.get("division_name") or event.get("label"))
    if family and division and family != division:
        return f"{family} / {division}"
    return division or family or "—"


def _day_label(day: dict[str, Any] | None) -> str:
    return _safe_text((day or {}).get("label")) or "—"


def _entry_key(row: dict[str, Any]) -> str:
    selection_id = _safe_text(row.get("selection_id"))
    if selection_id:
        return f"sel:{selection_id}"
    return f"reg:{_safe_text(row.get('registration_id'))}"


def _row_label(row: dict[str, Any]) -> str:
    reg = row.get("registration") or {}
    return f"{_display_name(reg)} — {_day_label(row.get('day'))} / {_event_label(row.get('event'))}"


def _row_search_blob(row: dict[str, Any]) -> str:
    reg = row.get("registration") or {}
    sel = row.get("selection") or {}
    return " ".join(
        [
            _display_name(reg),
            _safe_text(reg.get("email")),
            _safe_text(reg.get("phone")),
            _safe_text(sel.get("partner_name")),
            _safe_text(sel.get("partner_email")),
            _event_label(row.get("event")),
            _day_label(row.get("day")),
        ]
    ).lower()


def _imported_lookup(supabase, *, tournament_id: str, rows: list[dict[str, Any]]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for row in rows:
        key = _entry_key(row)
        try:
            out[key] = registration_is_imported_to_draw(
                supabase,
                tournament_id=tournament_id,
                selection_id=_safe_text(row.get("selection_id")) or None,
                registration_id=_safe_text(row.get("registration_id")) or None,
            )
        except Exception:
            out[key] = False
    return out


def _flatten_admin_rows(supabase, *, tournament_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    imported_by_key = _imported_lookup(supabase, tournament_id=tournament_id, rows=rows)
    out: list[dict[str, Any]] = []
    for row in rows:
        reg = row.get("registration") or {}
        sel = row.get("selection") or {}
        entry_key = _entry_key(row)
        out.append(
            {
                "entry_key": entry_key,
                "label": _row_label(row),
                "player": _display_name(reg),
                "email": _safe_text(reg.get("email")),
                "phone": _safe_text(reg.get("phone")),
                "day": _day_label(row.get("day")),
                "division": _event_label(row.get("event")),
                "registration_status": _safe_text(reg.get("status") or "confirmed").lower(),
                "payment_status": _safe_text(reg.get("payment_status") or "unpaid").lower(),
                "partner_mode": _safe_text(sel.get("partner_mode") or "NONE").upper() if sel else "NONE",
                "partner_note": _safe_text(sel.get("partner_note")),
                "registration_id": _safe_text(row.get("registration_id")),
                "selection_id": _safe_text(row.get("selection_id")),
                "locked": imported_by_key.get(entry_key, False),
                "raw": row,
            }
        )
    return out


def _apply_filters(rows: list[dict[str, Any]], *, status_filter: str, payment_filter: str, partner_filter: str, day_filter: str, search: str) -> list[dict[str, Any]]:
    search_text = _safe_text(search).lower()
    filtered: list[dict[str, Any]] = []
    for row in rows:
        raw = row.get("raw") or {}
        if status_filter != "All" and row.get("registration_status") != status_filter:
            continue
        if payment_filter != "All" and row.get("payment_status") != payment_filter:
            continue
        if partner_filter != "All" and row.get("partner_mode") != partner_filter:
            continue
        if day_filter != "All" and row.get("day") != day_filter:
            continue
        if search_text and search_text not in _row_search_blob(raw):
            continue
        filtered.append(row)
    return filtered


def _metrics(flat_rows: list[dict[str, Any]]) -> dict[str, int]:
    registrations = {row.get("registration_id") for row in flat_rows if row.get("registration_id")}
    return {
        "registrations": len(registrations),
        "entries": len(flat_rows),
        "needs_partner": sum(1 for row in flat_rows if row.get("partner_mode") == "NEEDS_PARTNER"),
        "paid": sum(1 for row in flat_rows if row.get("payment_status") == "paid"),
        "unpaid": sum(1 for row in flat_rows if row.get("payment_status") == "unpaid"),
        "locked": sum(1 for row in flat_rows if row.get("locked")),
    }


def _table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "Player": row.get("player"),
            "Email": row.get("email"),
            "Day": row.get("day"),
            "Division": row.get("division"),
            "Status": row.get("registration_status"),
            "Payment": row.get("payment_status"),
            "Partner": row.get("partner_mode"),
            "Locked": "Yes" if row.get("locked") else "",
            "Entry Key": row.get("entry_key"),
        }
        for row in rows
    ]


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    return pd.DataFrame(_table_rows(rows)).to_csv(index=False).encode("utf-8")


def _selection_payload_from_existing(selection: dict[str, Any], **changes) -> dict[str, Any]:
    payload = {
        "registration_day_id": _safe_text(selection.get("registration_day_id")),
        "event_option_id": _safe_text(selection.get("event_option_id")),
        "partner_mode": _safe_text(selection.get("partner_mode") or "NONE").upper(),
        "partner_name": _safe_text(selection.get("partner_name")),
        "partner_email": _safe_text(selection.get("partner_email")),
        "partner_phone": _safe_text(selection.get("partner_phone")),
        "partner_dupr_id": _safe_text(selection.get("partner_dupr_id")),
        "partner_skill": selection.get("partner_skill"),
        "partner_age": selection.get("partner_age"),
        "partner_note": _safe_text(selection.get("partner_note")),
        "show_on_partner_board": True,
    }
    payload.update(changes)
    if _safe_text(payload.get("partner_mode")).upper() != "NEEDS_PARTNER":
        payload["show_on_partner_board"] = bool(str(selection.get("show_on_partner_board") or "").lower() in {"true", "1", "yes", "y", "on"})
    return payload


def _unique_registration_ids(rows: Iterable[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in rows:
        reg_id = _safe_text(row.get("registration_id"))
        if reg_id and reg_id not in seen:
            seen.add(reg_id)
            out.append(reg_id)
    return out


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
    changed = 0
    skipped: list[str] = []

    if action == "Set registration status":
        for reg_id in _unique_registration_ids(selected_rows):
            update_admin_registration(supabase, tournament_id=tournament_id, registration_id=reg_id, payload={"status": status_value})
            changed += 1
        return changed, skipped

    if action == "Set payment status":
        for reg_id in _unique_registration_ids(selected_rows):
            update_admin_registration(supabase, tournament_id=tournament_id, registration_id=reg_id, payload={"payment_status": payment_value})
            changed += 1
        return changed, skipped

    if action == "Cancel registrations":
        for reg_id in _unique_registration_ids(selected_rows):
            cancel_registration(supabase, tournament_id=tournament_id, registration_id=reg_id)
            changed += 1
        return changed, skipped

    if action == "Append admin note":
        for row in selected_rows:
            reg_id = _safe_text(row.get("registration_id"))
            if not reg_id:
                skipped.append(row.get("label") or row.get("entry_key"))
                continue
            existing = _safe_text(((row.get("raw") or {}).get("registration") or {}).get("notes"))
            next_note = f"{existing}\n{note_text}".strip() if existing else note_text
            update_admin_registration(supabase, tournament_id=tournament_id, registration_id=reg_id, payload={"notes": next_note})
            changed += 1
        return changed, skipped

    for row in selected_rows:
        raw = row.get("raw") or {}
        selection = raw.get("selection") or {}
        selection_id = _safe_text(row.get("selection_id"))
        if not selection_id:
            skipped.append(f"{row.get('label')} — no event entry")
            continue
        if row.get("locked"):
            skipped.append(f"{row.get('label')} — locked in draw")
            continue
        if action == "Set partner mode":
            payload = _selection_payload_from_existing(selection, partner_mode=partner_mode_value)
        elif action == "Move division":
            target_event = event_lookup.get(target_event_id) or {}
            payload = _selection_payload_from_existing(
                selection,
                registration_day_id=_safe_text(target_event.get("registration_day_id")),
                event_option_id=target_event_id,
            )
        else:
            skipped.append(f"{row.get('label')} — unsupported action")
            continue
        update_admin_registration_selection(supabase, tournament_id=tournament_id, selection_id=selection_id, payload=payload)
        changed += 1
    return changed, skipped


def _render_bulk_tools(*, supabase, tournament_id: str, selected_rows: list[dict[str, Any]], event_options: list[dict[str, Any]]) -> None:
    with st.expander("Mass edit tools", expanded=bool(selected_rows)):
        st.caption("Select rows above, choose a bulk action, preview the count, type APPLY, then apply. Locked draw entries are skipped for event-entry changes.")
        if not selected_rows:
            st.info("Select one or more entries to enable bulk actions.")
            return

        action = st.selectbox("Bulk action", BULK_ACTIONS, key=f"bulk_action_{tournament_id}")
        status_value = payment_value = partner_mode_value = target_event_id = note_text = ""
        event_lookup = {str(row.get("id")): row for row in event_options}

        if action == "Set registration status":
            status_value = st.selectbox("New registration status", ADMIN_REGISTRATION_STATUS_OPTIONS, key=f"bulk_status_{tournament_id}")
        elif action == "Set payment status":
            payment_value = st.selectbox("New payment status", ADMIN_PAYMENT_STATUS_OPTIONS, key=f"bulk_payment_{tournament_id}")
        elif action == "Set partner mode":
            partner_mode_value = st.selectbox("New partner mode", PARTNER_MODE_OPTIONS, key=f"bulk_partner_{tournament_id}")
        elif action == "Move division":
            target_event_id = st.selectbox(
                "Move selected entries to division",
                [str(row.get("id")) for row in event_options],
                format_func=lambda event_id: _event_label(event_lookup.get(event_id)),
                key=f"bulk_event_{tournament_id}",
            )
            target_event = event_lookup.get(target_event_id) or {}
            st.caption(f"Day will be set to {_safe_text(target_event.get('registration_day_id')) or 'the division day'}.")
        elif action == "Append admin note":
            note_text = st.text_area("Note to append", key=f"bulk_note_{tournament_id}")
        elif action == "Cancel registrations":
            st.warning("This marks selected registrations cancelled. It does not hard delete records.")

        unique_regs = len(_unique_registration_ids(selected_rows))
        locked_count = sum(1 for row in selected_rows if row.get("locked"))
        st.info(f"Preview: {len(selected_rows)} selected event entries / {unique_regs} unique registrations. Locked entries in selection: {locked_count}.")
        confirm = st.text_input("Type APPLY to run this bulk action", key=f"bulk_confirm_{tournament_id}")
        if st.button("Apply bulk action", type="primary", disabled=confirm != "APPLY", key=f"bulk_apply_{tournament_id}"):
            if action == "Append admin note" and not _safe_text(note_text):
                st.error("Enter a note before applying.")
                st.stop()
            try:
                changed, skipped = _apply_bulk_action(
                    supabase=supabase,
                    tournament_id=tournament_id,
                    selected_rows=selected_rows,
                    action=action,
                    status_value=status_value,
                    payment_value=payment_value,
                    partner_mode_value=partner_mode_value,
                    target_event_id=target_event_id,
                    event_lookup=event_lookup,
                    note_text=_safe_text(note_text),
                )
                st.success(f"Bulk action complete. Updated {changed} record(s).")
                if skipped:
                    st.warning("Skipped: " + " | ".join(skipped[:10]))
                st.rerun()
            except Exception as exc:
                st.error(f"Bulk action failed: {exc}")


def _render_selected_entry_editor(*, supabase, tournament_id: str, row: dict[str, Any], days: list[dict[str, Any]], event_options: list[dict[str, Any]]) -> None:
    raw = row.get("raw") or {}
    reg = raw.get("registration") or {}
    sel = raw.get("selection") or {}
    reg_id = _safe_text(row.get("registration_id"))
    sel_id = _safe_text(row.get("selection_id"))
    day_lookup = {str(day.get("id")): day for day in days}
    event_lookup = {str(event.get("id")): event for event in event_options}
    day_ids = [str(day.get("id")) for day in days]
    event_ids = [str(event.get("id")) for event in event_options]
    locked = bool(row.get("locked"))

    st.markdown("### Entry Details")
    if locked:
        st.warning("This entry is locked because it has been imported into a draw. Division/day changes and hard delete are blocked.")

    with st.form(f"streamlined_edit_{row.get('entry_key')}"):
        c1, c2 = st.columns(2)
        first_name = c1.text_input("First name", value=_safe_text(reg.get("first_name")))
        last_name = c2.text_input("Last name", value=_safe_text(reg.get("last_name")))
        display_name = st.text_input("Display name", value=_safe_text(reg.get("display_name")))
        c3, c4 = st.columns(2)
        email = c3.text_input("Email", value=_safe_text(reg.get("email")))
        phone = c4.text_input("Phone", value=_safe_text(reg.get("phone")))
        c5, c6 = st.columns(2)
        reg_status = c5.selectbox(
            "Registration status",
            ADMIN_REGISTRATION_STATUS_OPTIONS,
            index=max(0, ADMIN_REGISTRATION_STATUS_OPTIONS.index(_safe_text(reg.get("status")).lower()) if _safe_text(reg.get("status")).lower() in ADMIN_REGISTRATION_STATUS_OPTIONS else 0),
        )
        reg_payment = c6.selectbox(
            "Payment status",
            ADMIN_PAYMENT_STATUS_OPTIONS,
            index=max(0, ADMIN_PAYMENT_STATUS_OPTIONS.index(_safe_text(reg.get("payment_status")).lower()) if _safe_text(reg.get("payment_status")).lower() in ADMIN_PAYMENT_STATUS_OPTIONS else 0),
        )
        notes = st.text_area("Internal/admin notes", value=_safe_text(reg.get("notes")), height=90)

        if sel_id:
            c7, c8 = st.columns(2)
            day_id = c7.selectbox(
                "Day",
                day_ids,
                index=max(0, day_ids.index(_safe_text(sel.get("registration_day_id"))) if _safe_text(sel.get("registration_day_id")) in day_ids else 0),
                format_func=lambda day_id: _safe_text((day_lookup.get(day_id) or {}).get("label") or day_id),
                disabled=locked,
            )
            event_id = c8.selectbox(
                "Division",
                event_ids,
                index=max(0, event_ids.index(_safe_text(sel.get("event_option_id"))) if _safe_text(sel.get("event_option_id")) in event_ids else 0),
                format_func=lambda event_id: _event_label(event_lookup.get(event_id)),
                disabled=locked,
            )
            c9, c10 = st.columns(2)
            partner_mode = c9.selectbox(
                "Partner mode",
                PARTNER_MODE_OPTIONS,
                index=max(0, PARTNER_MODE_OPTIONS.index(_safe_text(sel.get("partner_mode")).upper()) if _safe_text(sel.get("partner_mode")).upper() in PARTNER_MODE_OPTIONS else 0),
            )
            partner_name = c10.text_input("Legacy/free-text partner", value=_safe_text(sel.get("partner_name")))
            partner_email = st.text_input("Legacy/free-text partner email", value=_safe_text(sel.get("partner_email")))
            partner_note = st.text_area("Public partner note", value=_safe_text(sel.get("partner_note")), height=80)
        else:
            day_id = event_id = partner_mode = partner_name = partner_email = partner_note = ""
            st.info("This registration has no event entries yet.")

        save = st.form_submit_button("Save selected entry", type="primary", use_container_width=True)

    if save:
        try:
            update_admin_registration(
                supabase,
                tournament_id=tournament_id,
                registration_id=reg_id,
                payload={
                    "first_name": first_name,
                    "last_name": last_name,
                    "display_name": display_name,
                    "email": email,
                    "phone": phone,
                    "status": reg_status,
                    "payment_status": reg_payment,
                    "notes": notes,
                },
            )
            if sel_id and not locked:
                update_admin_registration_selection(
                    supabase,
                    tournament_id=tournament_id,
                    selection_id=sel_id,
                    payload=_selection_payload_from_existing(
                        sel,
                        registration_day_id=day_id,
                        event_option_id=event_id,
                        partner_mode=partner_mode,
                        partner_name=partner_name,
                        partner_email=partner_email,
                        partner_note=partner_note,
                    ),
                )
            st.success("Entry saved.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not save entry: {exc}")

    danger_cols = st.columns(2)
    with danger_cols[0]:
        if st.button("Cancel registration", key=f"cancel_selected_{row.get('entry_key')}"):
            cancel_registration(supabase, tournament_id=tournament_id, registration_id=reg_id)
            st.success("Registration cancelled.")
            st.rerun()
    with danger_cols[1]:
        with st.expander("Hard delete", expanded=False):
            confirm = st.text_input("Type DELETE to hard delete", key=f"delete_selected_confirm_{row.get('entry_key')}")
            if st.button("Delete permanently", key=f"delete_selected_{row.get('entry_key')}"):
                if locked:
                    st.error("This registration is locked in a draw. Remove draw/team records first.")
                elif confirm != "DELETE":
                    st.error("Type DELETE exactly.")
                else:
                    delete_registration(supabase, tournament_id=tournament_id, registration_id=reg_id)
                    st.success("Registration deleted.")
                    st.rerun()


def _render_manage_entries(*, supabase, tournament: dict[str, Any], days: list[dict[str, Any]], event_options: list[dict[str, Any]]) -> None:
    tournament_id = _safe_text(tournament.get("id"))
    raw_rows = list_registration_admin_rows(supabase, tournament_id)
    flat_rows = _flatten_admin_rows(supabase, tournament_id=tournament_id, rows=raw_rows)

    metric_values = _metrics(flat_rows)
    metric_cols = st.columns(6)
    metric_cols[0].metric("Registrations", metric_values["registrations"])
    metric_cols[1].metric("Event entries", metric_values["entries"])
    metric_cols[2].metric("Needs partner", metric_values["needs_partner"])
    metric_cols[3].metric("Paid", metric_values["paid"])
    metric_cols[4].metric("Unpaid", metric_values["unpaid"])
    metric_cols[5].metric("Locked", metric_values["locked"])

    filter_cols = st.columns(5)
    status_filter = filter_cols[0].selectbox("Status", ["All"] + ADMIN_REGISTRATION_STATUS_OPTIONS, key=f"stream_status_{tournament_id}")
    payment_filter = filter_cols[1].selectbox("Payment", ["All"] + ADMIN_PAYMENT_STATUS_OPTIONS, key=f"stream_payment_{tournament_id}")
    partner_filter = filter_cols[2].selectbox("Partner", ["All"] + PARTNER_MODE_OPTIONS, key=f"stream_partner_{tournament_id}")
    day_filter = filter_cols[3].selectbox("Day", ["All"] + sorted({row.get("day") for row in flat_rows if row.get("day")}), key=f"stream_day_{tournament_id}")
    search = filter_cols[4].text_input("Search", key=f"stream_search_{tournament_id}")

    filtered_rows = _apply_filters(flat_rows, status_filter=status_filter, payment_filter=payment_filter, partner_filter=partner_filter, day_filter=day_filter, search=search)
    st.caption(f"Showing {len(filtered_rows)} of {len(flat_rows)} event entries.")
    st.download_button(
        "Download filtered CSV",
        data=_csv_bytes(filtered_rows),
        file_name=f"registration-management-{tournament_id}.csv",
        mime="text/csv",
        disabled=not bool(filtered_rows),
    )

    if not filtered_rows:
        st.info("No entries match the current filters.")
        return

    table_df = pd.DataFrame(_table_rows(filtered_rows))
    st.dataframe(table_df.drop(columns=["Entry Key"]), hide_index=True, use_container_width=True)

    row_by_key = {row["entry_key"]: row for row in filtered_rows}
    selection_options = list(row_by_key.keys())
    selected_keys = st.multiselect(
        "Select entries for mass edit",
        selection_options,
        format_func=lambda key: row_by_key[key]["label"],
        key=f"stream_bulk_selected_{tournament_id}",
    )
    selected_rows = [row_by_key[key] for key in selected_keys if key in row_by_key]
    _render_bulk_tools(supabase=supabase, tournament_id=tournament_id, selected_rows=selected_rows, event_options=event_options)

    edit_key = st.selectbox(
        "Edit one entry",
        [""] + selection_options,
        format_func=lambda key: "Choose an entry…" if not key else row_by_key[key]["label"],
        key=f"stream_edit_key_{tournament_id}",
    )
    if edit_key and edit_key in row_by_key:
        _render_selected_entry_editor(
            supabase=supabase,
            tournament_id=tournament_id,
            row=row_by_key[edit_key],
            days=days,
            event_options=event_options,
        )


def _render_add_registration(*, supabase, tournament: dict[str, Any], days: list[dict[str, Any]], event_options: list[dict[str, Any]]) -> None:
    tournament_id = _safe_text(tournament.get("id"))
    if not days or not event_options:
        st.warning("Configure registration days and event divisions in Tournament Setup first.")
        return
    day_lookup = {str(row.get("id")): row for row in days}
    event_lookup = {str(row.get("id")): row for row in event_options}
    with st.form(f"stream_add_registration_{tournament_id}"):
        c1, c2 = st.columns(2)
        first_name = c1.text_input("First name")
        last_name = c2.text_input("Last name")
        display_name = st.text_input("Display name")
        c3, c4 = st.columns(2)
        email = c3.text_input("Email")
        phone = c4.text_input("Phone")
        c5, c6 = st.columns(2)
        status = c5.selectbox("Registration status", ADMIN_REGISTRATION_STATUS_OPTIONS, index=0)
        payment_status = c6.selectbox("Payment status", ADMIN_PAYMENT_STATUS_OPTIONS)
        day_id = st.selectbox("Day", [str(day.get("id")) for day in days], format_func=lambda day_id: _safe_text((day_lookup.get(day_id) or {}).get("label") or day_id))
        event_id = st.selectbox("Division", [str(event.get("id")) for event in event_options], format_func=lambda event_id: _event_label(event_lookup.get(event_id)))
        partner_mode = st.selectbox("Partner mode", PARTNER_MODE_OPTIONS)
        partner_note = st.text_area("Partner/public note")
        notes = st.text_area("Internal/admin notes")
        saved = st.form_submit_button("Save registration", type="primary", use_container_width=True)
    if saved:
        try:
            create_admin_registration(
                supabase,
                tournament_id=tournament_id,
                payload={
                    "first_name": first_name,
                    "last_name": last_name,
                    "display_name": display_name or " ".join([first_name, last_name]).strip(),
                    "email": email,
                    "phone": phone,
                    "status": status,
                    "payment_status": payment_status,
                    "notes": notes,
                    "selections": [
                        {
                            "registration_day_id": day_id,
                            "event_option_id": event_id,
                            "partner_mode": partner_mode,
                            "partner_note": partner_note,
                            "show_on_partner_board": True,
                        }
                    ],
                },
            )
            st.success("Registration created.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not create registration: {exc}")


def _render_links(*, tournament: dict[str, Any], settings: dict[str, Any]) -> None:
    tournament_id = _safe_text(tournament.get("id"))
    public_urls = build_public_urls(
        base_url=_safe_text(st.session_state.get("base_url")),
        tournament_id=tournament_id,
        registration_slug=settings.get("registration_slug"),
    )
    st.markdown("#### Public links")
    st.code(public_urls["registration"])
    st.code(public_urls["roster"])
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Open Public Registration Form", key=f"stream_open_registration_{tournament_id}"):
            nav_params = {"tournament_id": tournament_id}
            slug = _safe_text(settings.get("registration_slug"))
            if slug:
                nav_params["tournament"] = slug
            navigate_same_tab(page="tournament_registration", params=nav_params, public_mode=True)
    with c2:
        if st.button("Open Public Roster", key=f"stream_open_roster_{tournament_id}"):
            nav_params = {"tournament_id": tournament_id}
            slug = _safe_text(settings.get("registration_slug"))
            if slug:
                nav_params["tournament"] = slug
            navigate_same_tab(page="tournament_roster", params=nav_params, public_mode=True)


def render(ctx) -> None:
    page_shell(
        "🧾 Registration Management",
        "Manage tournament entries, payments, partner status, and bulk updates from one streamlined workspace.",
        mode_label="Admin",
    )
    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()
    supabase = getattr(ctx, "supabase", None)
    club_id = _safe_text(getattr(ctx, "club_id", ""))
    if supabase is None or not club_id:
        st.error("Missing database context.")
        st.stop()
    available, detail = registration_feature_available(supabase)
    if not available:
        st.error("Tournament registration is not enabled yet. Apply the registration SQL migration first.")
        if detail:
            st.caption(detail)
        st.stop()

    current_page_key = _safe_text(st.query_params.get("page")) or "tournament_registration_admin"
    page_key = "tournament_registration_admin" if current_page_key == "tournament_registration_admin" else "tournament_registration"
    tournament, settings, days, event_options = _select_admin_tournament(ctx, supabase, page_key=page_key)
    if not tournament:
        st.stop()

    st.subheader(_safe_text(tournament.get("name") or "Tournament"))
    tabs = st.tabs(["Manage Entries", "Add Registration", "Links"])
    with tabs[0]:
        _render_manage_entries(supabase=supabase, tournament=tournament, days=days, event_options=event_options)
    with tabs[1]:
        _render_add_registration(supabase=supabase, tournament=tournament, days=days, event_options=event_options)
    with tabs[2]:
        _render_links(tournament=tournament, settings=settings)
