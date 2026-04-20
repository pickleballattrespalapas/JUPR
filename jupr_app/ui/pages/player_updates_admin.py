from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from jupr_app.domain.notifications.player_profile_update_repo import (
    approve_request,
    list_active_subscriptions,
    list_pending_requests,
    mark_unsubscribed,
    reject_request,
    replace_verified_subscriber,
)
from jupr_app.ui.layout import page_shell


def _actor_label(ctx) -> str:
    for key in ["admin_email", "user_email", "admin_user", "admin_name"]:
        value = getattr(ctx, key, None)
        if value:
            return str(value)
    for key in ["admin_email", "user_email", "admin_user", "admin_name"]:
        value = st.session_state.get(key)
        if value:
            return str(value)
    return "admin"


def _player_name(ctx, player_id: int | None) -> str:
    id_to_name = getattr(ctx, "id_to_name", {}) or {}
    try:
        pid = int(player_id)
    except Exception:
        return ""
    return str(id_to_name.get(pid, ""))


def _pending_table_rows(ctx, rows: list[dict]) -> list[dict]:
    display_rows: list[dict] = []
    for row in rows:
        pid = row.get("player_id")
        display_rows.append(
            {
                "player_id": pid,
                "player_name": _player_name(ctx, pid),
                "email": row.get("email"),
                "request_note": row.get("request_note"),
                "created_at": row.get("created_at"),
            }
        )
    return display_rows


def _active_table_rows(ctx, rows: list[dict]) -> list[dict]:
    display_rows: list[dict] = []
    for row in rows:
        pid = row.get("player_id")
        display_rows.append(
            {
                "player_id": pid,
                "player_name": _player_name(ctx, pid),
                "email": row.get("email"),
                "verified_at": row.get("verified_at"),
                "verified_by": row.get("verified_by"),
                "last_digest_week_start": row.get("last_digest_week_start"),
            }
        )
    return display_rows


def _find_active_for_player(active_rows: list[dict], player_id: int | None) -> dict | None:
    try:
        pid = int(player_id)
    except Exception:
        return None
    for row in active_rows:
        try:
            if int(row.get("player_id")) == pid:
                return row
        except Exception:
            continue
    return None


def render(ctx) -> None:
    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell(
        "📬 Player Updates Admin",
        "Review verified player update requests and monitor delivery pipeline.",
        mode_label=mode_label,
    )

    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    supabase = ctx.supabase
    club_id = str(ctx.club_id)
    actor = _actor_label(ctx)

    pending_tab, active_tab, digests_tab, queue_tab = st.tabs(
        [
            "Pending Requests",
            "Active Profiles",
            "Weekly Digests",
            "Send Queue",
        ]
    )

    with pending_tab:
        st.subheader("Pending Requests")
        pending_rows = list_pending_requests(supabase, club_id, limit=200)
        active_rows = list_active_subscriptions(supabase, club_id, limit=200)

        if pending_rows:
            st.dataframe(pd.DataFrame(_pending_table_rows(ctx, pending_rows)), use_container_width=True)
        else:
            st.info("No pending requests.")

        for row in pending_rows:
            row_id = str(row.get("id") or "")
            pid = row.get("player_id")
            p_name = _player_name(ctx, pid)
            header = f"Player #{pid}"
            if p_name:
                header += f" · {p_name}"
            with st.expander(header):
                st.caption(f"Requested by {row.get('email') or 'unknown'} on {row.get('created_at') or 'n/a'}")
                st.write(f"Request note: {row.get('request_note') or '—'}")
                with st.form(f"pending_action_{row_id}"):
                    admin_note = st.text_area("Admin note", key=f"pending_admin_note_{row_id}")
                    replacement_email = st.text_input(
                        "Replacement email (for Replace Verified Subscriber)",
                        key=f"pending_replace_email_{row_id}",
                    )
                    replacement_note = st.text_area(
                        "Replacement request note (optional)",
                        key=f"pending_replace_note_{row_id}",
                    )
                    approve_clicked = st.form_submit_button("Approve")
                    reject_clicked = st.form_submit_button("Reject")
                    replace_clicked = st.form_submit_button("Replace Verified Subscriber")

                if approve_clicked:
                    try:
                        approve_request(supabase, row_id, verified_by=actor, admin_note=admin_note)
                        st.success("Request approved.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Approve failed: {exc}")

                if reject_clicked:
                    try:
                        reject_request(supabase, row_id, admin_note=admin_note, verified_by=actor)
                        st.success("Request rejected.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Reject failed: {exc}")

                if replace_clicked:
                    try:
                        if not str(replacement_email or "").strip():
                            raise ValueError("Replacement email is required")
                        current_active = _find_active_for_player(active_rows, pid)
                        if current_active is None:
                            raise ValueError("No active subscription exists for this player. Use Approve or Reject instead.")
                        replace_verified_subscriber(
                            supabase,
                            old_subscription_id=str(current_active.get("id")),
                            new_email=replacement_email,
                            new_request_note=replacement_note,
                            verified_by=actor,
                            admin_note=admin_note,
                        )
                        reject_note = (
                            f"Replaced verified subscriber by {actor}. "
                            f"{str(admin_note or '').strip()}"
                        ).strip()
                        reject_request(supabase, row_id, admin_note=reject_note, verified_by=actor)
                        st.success("Verified subscriber replaced and pending request closed.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Replace failed: {exc}")

    with active_tab:
        st.subheader("Active Profiles")
        active_rows = list_active_subscriptions(supabase, club_id, limit=200)

        if active_rows:
            st.dataframe(pd.DataFrame(_active_table_rows(ctx, active_rows)), use_container_width=True)
        else:
            st.info("No active profiles.")

        for row in active_rows:
            row_id = str(row.get("id") or "")
            pid = row.get("player_id")
            p_name = _player_name(ctx, pid)
            header = f"Player #{pid}"
            if p_name:
                header += f" · {p_name}"
            with st.expander(header):
                st.caption(f"Verified email: {row.get('email') or 'unknown'}")
                with st.form(f"active_action_{row_id}"):
                    admin_note = st.text_area("Admin note", key=f"active_admin_note_{row_id}")
                    replacement_email = st.text_input(
                        "Replacement email",
                        key=f"active_replace_email_{row_id}",
                    )
                    replacement_note = st.text_area(
                        "Replacement request note (optional)",
                        key=f"active_replace_note_{row_id}",
                    )
                    replace_clicked = st.form_submit_button("Replace Verified Subscriber")
                    unsubscribe_clicked = st.form_submit_button("Unsubscribe / Deactivate")

                if replace_clicked:
                    try:
                        if not str(replacement_email or "").strip():
                            raise ValueError("Replacement email is required")
                        replace_verified_subscriber(
                            supabase,
                            old_subscription_id=row_id,
                            new_email=replacement_email,
                            new_request_note=replacement_note,
                            verified_by=actor,
                            admin_note=admin_note,
                        )
                        st.success("Verified subscriber replaced.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Replace failed: {exc}")

                if unsubscribe_clicked:
                    try:
                        mark_unsubscribed(supabase, row_id)
                        st.success("Subscription deactivated.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Unsubscribe failed: {exc}")

    with digests_tab:
        st.subheader("Weekly Digests")
        today = date.today()
        with st.form("player_updates_digests_filters"):
            st.date_input("Start Date", value=today - timedelta(days=28), key="player_updates_digest_start")
            st.date_input("End Date", value=today, key="player_updates_digest_end")
            st.form_submit_button("Apply")
        st.info("Digest generation is implemented in a later step.")

    with queue_tab:
        st.subheader("Send Queue")
        st.info("Email sending is implemented in a later step.")
