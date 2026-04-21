from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from jupr_app.domain.notifications.player_profile_update_repo import (
    approve_request,
    list_active_subscriptions,
    list_digests_for_range,
    list_outbox_rows,
    list_pending_requests,
    list_recent_digests,
    mark_unsubscribed,
    reject_request,
    reset_outbox_rows_to_pending,
    replace_verified_subscriber,
    save_digest,
)
from jupr_app.domain.notifications.player_update_sender import (
    generate_digests_for_active_subscriptions,
    queue_saved_digest_rows,
    send_pending_player_update_emails,
)
from jupr_app.domain.notifications.smtp_mailer import get_smtp_config_status
from jupr_app.domain.recaps.player_weekly_digest import compute_player_weekly_digest
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


def _digest_player_options(ctx, active_rows: list[dict]) -> tuple[list[str], dict[str, int]]:
    labels: list[str] = []
    index: dict[str, int] = {}
    for row in active_rows:
        try:
            pid = int(row.get("player_id"))
        except Exception:
            continue
        label = f"#{pid}"
        name = _player_name(ctx, pid)
        if name:
            label = f"{name} (#{pid})"
        labels.append(label)
        index[label] = pid
    return labels, index


def _render_digest_preview(digest: dict) -> None:
    st.markdown("#### Preview")
    st.write(
        {
            "player": digest.get("player_name"),
            "range": digest.get("display_range"),
            "subject_line": digest.get("subject_line"),
        }
    )
    st.json(digest.get("summary") or {})

    points = (digest.get("chart") or {}).get("points") or []
    if points:
        st.dataframe(pd.DataFrame(points), use_container_width=True, hide_index=True)
    else:
        st.caption("No chart points available in selected date range.")

    highlights = digest.get("highlights") or []
    if highlights:
        st.markdown("**Highlights**")
        for line in highlights:
            st.write(f"• {line}")


def _friendly_error(exc: Exception) -> str:
    text = str(exc or "").strip()
    return text or "Unknown error."


def _digest_lookup_key(row: dict) -> tuple[int, str, str] | None:
    try:
        return (
            int(row.get("player_id")),
            str(row.get("week_start") or ""),
            str(row.get("week_end") or ""),
        )
    except Exception:
        return None


def _saved_digest_payload(row: dict) -> dict:
    payload = row.get("final_json") or row.get("generated_json") or {}
    return payload if isinstance(payload, dict) else {}


def _saved_digest_label(ctx, row: dict, *, queue_status: str | None = None) -> str:
    pid = row.get("player_id")
    player_label = f"#{pid}"
    name = _player_name(ctx, pid)
    if name:
        player_label = f"{name} (#{pid})"
    range_label = f"{row.get('week_start') or 'n/a'} → {row.get('week_end') or 'n/a'}"
    if queue_status:
        return f"{player_label} · {range_label} · {queue_status}"
    return f"{player_label} · {range_label}"


def render(ctx) -> None:
    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell(
        "📬 Player Updates Admin",
        "Review verified player update requests and manage player digest generation and delivery.",
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
                    admin_note = st.text_area(
                        "Admin note",
                        key=f"pending_admin_note_{row_id}",
                        help="Visible to operators for review context.",
                    )
                    replacement_email = st.text_input(
                        "Replacement email (for Replace Verified Subscriber)",
                        key=f"pending_replace_email_{row_id}",
                        help="Used only when replacing an existing active subscriber.",
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
                        st.error(f"Approve failed: {_friendly_error(exc)}")

                if reject_clicked:
                    try:
                        reject_request(supabase, row_id, admin_note=admin_note, verified_by=actor)
                        st.success("Request rejected.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Reject failed: {_friendly_error(exc)}")

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
                        st.error(f"Replace failed: {_friendly_error(exc)}")

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
                    admin_note = st.text_area(
                        "Admin note",
                        key=f"active_admin_note_{row_id}",
                        help="Optional operator context stored for audit history.",
                    )
                    replacement_email = st.text_input(
                        "Replacement email",
                        key=f"active_replace_email_{row_id}",
                        help="Creates a new active row and deactivates the current one.",
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
                        st.error(f"Replace failed: {_friendly_error(exc)}")

                if unsubscribe_clicked:
                    try:
                        mark_unsubscribed(supabase, row_id)
                        st.success("Subscription deactivated.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Unsubscribe failed: {_friendly_error(exc)}")

    with digests_tab:
        st.subheader("Weekly Digests")
        today = date.today()
        start_date = st.date_input(
            "Start Date",
            value=today - timedelta(days=7),
            key="player_updates_digest_start",
            help="Custom date windows are supported (not Monday–Sunday only).",
        )
        end_date = st.date_input(
            "End Date",
            value=today,
            key="player_updates_digest_end",
            help="Must be on or after Start Date.",
        )

        if end_date < start_date:
            st.error("End Date must be on or after Start Date.")
            return

        active_rows = list_active_subscriptions(supabase, club_id, limit=500)
        labels, label_to_pid = _digest_player_options(ctx, active_rows)

        st.markdown("#### Generate for all subscribed players")
        st.caption("This is the default digest generation flow.")
        if st.button("Generate for All Subscribed Players", disabled=not active_rows, use_container_width=True):
            try:
                result = generate_digests_for_active_subscriptions(
                    ctx,
                    start_date=start_date,
                    end_date=end_date,
                )
                st.success(
                    "Bulk generation complete: "
                    f"active={result['active_subscriptions']} · "
                    f"saved={result['saved']} · "
                    f"failed={result['failed']}"
                )
            except Exception as exc:
                st.error(f"Bulk digest generation failed: {_friendly_error(exc)}")

        preview_digest = st.session_state.get("player_updates_digest_preview")
        if isinstance(preview_digest, dict) and preview_digest:
            _render_digest_preview(preview_digest)

        st.divider()
        with st.expander("Single player options", expanded=False):
            st.caption("Use this only when you want to preview or regenerate one player manually.")
            selected_label = st.selectbox("Player", options=labels, index=0 if labels else None)
            selected_pid = label_to_pid.get(selected_label) if selected_label else None

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Generate Selected Digest", disabled=selected_pid is None):
                    try:
                        digest = compute_player_weekly_digest(
                            ctx,
                            player_id=int(selected_pid),
                            start_date=start_date,
                            end_date=end_date,
                        )
                        save_digest(
                            supabase,
                            club_id=club_id,
                            player_id=int(selected_pid),
                            week_start=start_date,
                            week_end=end_date,
                            generated_json=digest,
                            final_json=digest,
                        )
                        st.session_state["player_updates_digest_preview"] = digest
                        st.success("Digest generated and saved.")
                    except Exception as exc:
                        st.error(f"Digest generation failed: {_friendly_error(exc)}")

            with c2:
                if st.button("Preview Selected Digest", disabled=selected_pid is None):
                    try:
                        digest = compute_player_weekly_digest(
                            ctx,
                            player_id=int(selected_pid),
                            start_date=start_date,
                            end_date=end_date,
                        )
                        st.session_state["player_updates_digest_preview"] = digest
                        st.success("Digest preview loaded.")
                    except Exception as exc:
                        st.error(f"Digest preview failed: {_friendly_error(exc)}")

        st.divider()
        st.markdown("#### Saved digests in selected range")
        try:
            saved_rows = list_digests_for_range(
                supabase,
                club_id,
                week_start_from=start_date,
                week_start_to=end_date,
                player_id=None,
            )
        except Exception as exc:
            st.warning(f"Unable to load saved digests: {exc}")
            saved_rows = []

        if saved_rows:
            rows = []
            for row in saved_rows:
                rows.append(
                    {
                        "player_id": row.get("player_id"),
                        "player_name": _player_name(ctx, row.get("player_id")),
                        "week_start": row.get("week_start"),
                        "week_end": row.get("week_end"),
                        "updated_at": row.get("updated_at"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No saved digests for selected range.")

    with queue_tab:
        st.subheader("Send Queue")
        st.caption("Choose from digests that have already been generated, then queue and send them.")
        st.markdown("#### Mail Config Status")
        smtp_status = get_smtp_config_status()
        st.write(
            {
                "from_email": smtp_status.get("from_email") or "",
                "from_name": smtp_status.get("from_name") or "",
                "reply_to": smtp_status.get("reply_to") or "",
                "reply_to_configured": bool(smtp_status.get("reply_to_configured")),
                "use_tls": bool(smtp_status.get("use_tls")),
                "missing_required_smtp_keys": smtp_status.get("missing") or [],
            }
        )
        if smtp_status.get("port_error"):
            st.warning(f"SMTP config warning: {smtp_status['port_error']}")
        elif smtp_status.get("missing"):
            st.warning(
                "SMTP config missing required keys: "
                + ", ".join(str(key) for key in (smtp_status.get("missing") or []))
            )
        else:
            st.caption("SMTP config appears complete.")

        active_rows = list_active_subscriptions(supabase, club_id, limit=2000)
        active_by_player: dict[int, dict] = {}
        for row in active_rows:
            try:
                active_by_player[int(row.get("player_id"))] = row
            except Exception:
                continue

        try:
            saved_digest_rows = list_recent_digests(supabase, club_id, limit=500)
        except Exception as exc:
            st.warning(f"Unable to load generated digests: {exc}")
            saved_digest_rows = []

        try:
            outbox_rows = list_outbox_rows(supabase, club_id, limit=1000)
        except Exception as exc:
            st.warning(f"Unable to load outbox rows: {exc}")
            outbox_rows = []

        outbox_by_digest_key: dict[tuple[int, str, str], dict] = {}
        for row in outbox_rows:
            key = _digest_lookup_key(row)
            if key is not None and key not in outbox_by_digest_key:
                outbox_by_digest_key[key] = row

        digest_display_rows: list[dict] = []
        digest_options: dict[str, dict] = {}

        for row in saved_digest_rows:
            key = _digest_lookup_key(row)
            if key is None:
                continue
            pid = key[0]
            outbox = outbox_by_digest_key.get(key)
            active_subscription = active_by_player.get(pid)
            queue_status = str((outbox or {}).get("send_status") or "not_queued")
            digest_display_rows.append(
                {
                    "player_id": pid,
                    "player_name": _player_name(ctx, pid),
                    "week_start": row.get("week_start"),
                    "week_end": row.get("week_end"),
                    "queue_status": queue_status,
                    "email": (active_subscription or {}).get("email"),
                    "updated_at": row.get("updated_at"),
                }
            )
            if active_subscription is None:
                continue
            label = _saved_digest_label(ctx, row, queue_status=queue_status)
            if label not in digest_options:
                digest_options[label] = row

        if digest_display_rows:
            st.dataframe(pd.DataFrame(digest_display_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No generated digests available yet. Generate digests first in Weekly Digests.")

        selected_digest_labels = st.multiselect(
            "Generated digests to queue",
            options=list(digest_options.keys()),
            help="Only players with an active verified subscriber are available for queueing.",
        )
        selected_digest_rows = [digest_options[label] for label in selected_digest_labels if label in digest_options]

        q1, q2, q3 = st.columns(3)
        with q1:
            if st.button("Queue Selected Digests", disabled=not selected_digest_rows):
                try:
                    result = queue_saved_digest_rows(ctx, digest_rows=selected_digest_rows)
                    st.success(
                        "Queue complete: "
                        f"queued={result['queued']} · "
                        f"already_exists={result['already_exists']} · "
                        f"no_active_subscription={result['no_active_subscription']} · "
                        f"failed={result['failed']}"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Queue failed: {_friendly_error(exc)}")

        with q2:
            if st.button("Send Pending"):
                try:
                    result = send_pending_player_update_emails(ctx, limit=500)
                    st.success(
                        f"Attempted: {result['attempted']} · Sent: {result['sent']} · "
                        f"Skipped: {result['skipped']} · Errors: {result['errors']}"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Send pending failed: {_friendly_error(exc)}")

        with q3:
            if st.button("Retry Errored Rows", help="Reset all errored outbox rows back to pending."):
                try:
                    result = reset_outbox_rows_to_pending(
                        supabase,
                        club_id=club_id,
                        only_status="error",
                    )
                    st.success(
                        "Retry reset complete: "
                        f"matched={result['matched']} · "
                        f"reset_to_pending={result['reset_to_pending']} · "
                        f"failed={result['failed']}"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Retry errored rows failed: {_friendly_error(exc)}")

        if selected_digest_rows:
            preview_label = selected_digest_labels[0]
            preview_row = digest_options.get(preview_label)
            preview_payload = _saved_digest_payload(preview_row or {})
            if preview_payload:
                _render_digest_preview(preview_payload)

        st.divider()
        st.markdown("#### Outbox")
        if outbox_rows:
            display_rows = []
            for row in outbox_rows:
                display_rows.append(
                    {
                        "id": row.get("id"),
                        "player_id": row.get("player_id"),
                        "player_name": _player_name(ctx, row.get("player_id")),
                        "week_start": row.get("week_start"),
                        "week_end": row.get("week_end"),
                        "send_status": row.get("send_status"),
                        "sent_at": row.get("sent_at"),
                        "error_text": row.get("error_text"),
                        "created_at": row.get("created_at"),
                    }
                )
            st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No outbox rows.")
