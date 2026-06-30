from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from jupr_app.domain.notifications.smtp_mailer import get_smtp_config_status
from jupr_app.domain.notifications.tournament_registrant_broadcast_email import send_tournament_registrant_broadcast_email
from jupr_app.domain.tournament_registration_repo import delete_registration, list_registrations, registration_feature_available

_LEGACY_MODULE_NAME = "jupr_app.ui.pages._tournament_registration_admin_streamlined_legacy"
_LEGACY_PATH = Path(__file__).resolve().parent.parent / "tournament_registration_admin_streamlined.py"
BULK_DELETE_CANCELLED_ACTION = "Hard delete cancelled registrations"
_CANCELLED_REGISTRATION_STATUSES = {"cancelled", "canceled"}


def _load_legacy_module():
    module = sys.modules.get(_LEGACY_MODULE_NAME)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(_LEGACY_MODULE_NAME, _LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load streamlined registration admin legacy module from {_LEGACY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_LEGACY_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


_legacy = _load_legacy_module()
_LEGACY_APPLY_BULK_ACTION = _legacy._apply_bulk_action
for _name in dir(_legacy):
    if _name.startswith("__") and _name.endswith("__"):
        continue
    globals()[_name] = getattr(_legacy, _name)

BULK_ACTIONS = [
    *[action for action in _legacy.BULK_ACTIONS if action != BULK_DELETE_CANCELLED_ACTION],
    BULK_DELETE_CANCELLED_ACTION,
]


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _registration_is_cancelled(registration: dict[str, Any]) -> bool:
    return _safe_text(registration.get("status")).lower() in _CANCELLED_REGISTRATION_STATUSES


def _recipient_name(registration: dict[str, Any]) -> str:
    display = _safe_text(registration.get("display_name"))
    if display:
        return display
    name = " ".join(part for part in [_safe_text(registration.get("first_name")), _safe_text(registration.get("last_name"))] if part)
    return name or _safe_text(registration.get("email")) or "Registrant"


def _registrant_recipients(supabase, *, tournament_id: str, include_cancelled: bool = False) -> list[dict[str, str]]:
    rows = list_registrations(supabase, tournament_id)
    recipients: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if _registration_is_cancelled(row) and not include_cancelled:
            continue
        email = _safe_text(row.get("email")).lower()
        if not email or email in seen:
            continue
        seen.add(email)
        recipients.append(
            {
                "name": _recipient_name(row),
                "email": email,
                "status": _safe_text(row.get("status") or "confirmed").lower(),
                "payment_status": _safe_text(row.get("payment_status") or "unpaid").lower(),
            }
        )
    return sorted(recipients, key=lambda row: (row["name"].lower(), row["email"]))


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
    return _LEGACY_APPLY_BULK_ACTION(
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


def _render_email_registrants(*, supabase, tournament: dict[str, Any]) -> None:
    tournament_id = _safe_text(tournament.get("id"))
    tournament_name = _safe_text(tournament.get("name") or "Tournament")
    st.markdown("### Email Registrants")
    st.caption("Send an individual email to every registration email address. Recipients do not see the full recipient list.")

    smtp_status = get_smtp_config_status()
    if smtp_status.get("ok"):
        st.caption(f"Sending from {smtp_status.get('from_name')} <{smtp_status.get('from_email')}>. Replies go to {smtp_status.get('reply_to') or 'the configured sender'}.")
    else:
        missing = ", ".join(smtp_status.get("missing") or [])
        st.warning(f"SMTP is not fully configured. Sending will fail unless email mode is dry-run. Missing: {missing or smtp_status.get('port_error') or 'unknown'}")

    include_cancelled = st.checkbox("Include cancelled registrations", value=False, key=f"email_include_cancelled_{tournament_id}")
    recipients = _registrant_recipients(supabase, tournament_id=tournament_id, include_cancelled=include_cancelled)
    st.metric("Recipients", len(recipients))
    if recipients:
        with st.expander("Preview recipients", expanded=False):
            st.dataframe(pd.DataFrame(recipients), hide_index=True, use_container_width=True)
            st.download_button(
                "Download recipient CSV",
                data=pd.DataFrame(recipients).to_csv(index=False).encode("utf-8"),
                file_name=f"{tournament_id}-email-recipients.csv",
                mime="text/csv",
            )
    else:
        st.info("No recipients match the current scope.")

    with st.form(f"email_registrants_form_{tournament_id}"):
        subject = st.text_input("Subject *", value=f"{tournament_name} update")
        message = st.text_area("Message *", height=220, placeholder="Write the tournament update here...")
        confirm = st.text_input("Type SEND to confirm")
        submitted = st.form_submit_button("Send email to registrants", type="primary", use_container_width=True, disabled=not bool(recipients))

    if submitted:
        errors: list[str] = []
        if not _safe_text(subject):
            errors.append("Enter a subject.")
        if not _safe_text(message):
            errors.append("Enter a message.")
        if confirm != "SEND":
            errors.append("Type SEND exactly to confirm.")
        if not recipients:
            errors.append("There are no recipients to email.")
        if errors:
            for error in errors:
                st.error(error)
            st.stop()

        sent = 0
        dry_run = 0
        failed: list[str] = []
        for recipient in recipients:
            try:
                result = send_tournament_registrant_broadcast_email(
                    tournament_name=tournament_name,
                    recipient_email=recipient["email"],
                    recipient_name=recipient["name"],
                    subject=subject,
                    message=message,
                )
                if _safe_text(result.get("status")) == "dry_run":
                    dry_run += 1
                else:
                    sent += 1
            except Exception as exc:
                failed.append(f"{recipient['email']}: {exc}")
        if sent or dry_run:
            summary_bits = []
            if sent:
                summary_bits.append(f"sent {sent}")
            if dry_run:
                summary_bits.append(f"dry-run {dry_run}")
            st.success("Broadcast complete: " + ", ".join(summary_bits) + ".")
        if failed:
            st.error(f"Failed for {len(failed)} recipient(s).")
            with st.expander("Failure details", expanded=True):
                for item in failed[:50]:
                    st.caption(item)


def _email_tab_first() -> bool:
    return _safe_text(st.query_params.get("registration_admin_view")).lower() in {"email", "emails", "registrants_email"}


def render(ctx) -> None:
    _legacy.page_shell(
        "🧾 Registration Management",
        "Manage tournament entries, payments, partner status, emails, and bulk updates from one streamlined workspace.",
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
    tournament, settings, days, event_options = _legacy._select_admin_tournament(ctx, supabase, page_key=page_key)
    if not tournament:
        st.stop()

    st.subheader(_safe_text(tournament.get("name") or "Tournament"))
    tab_labels = ["Email Registrants", "Manage Entries", "Add Registration", "Links"] if _email_tab_first() else ["Manage Entries", "Email Registrants", "Add Registration", "Links"]
    tabs = dict(zip(tab_labels, st.tabs(tab_labels)))
    with tabs["Manage Entries"]:
        _legacy._render_manage_entries(supabase=supabase, tournament=tournament, days=days, event_options=event_options)
    with tabs["Email Registrants"]:
        _render_email_registrants(supabase=supabase, tournament=tournament)
    with tabs["Add Registration"]:
        _legacy._render_add_registration(supabase=supabase, tournament=tournament, days=days, event_options=event_options)
    with tabs["Links"]:
        _legacy._render_links(tournament=tournament, settings=settings)


_legacy.BULK_ACTIONS = BULK_ACTIONS
_legacy._apply_bulk_action = _apply_bulk_action

globals()["BULK_ACTIONS"] = BULK_ACTIONS
globals()["_apply_bulk_action"] = _apply_bulk_action
