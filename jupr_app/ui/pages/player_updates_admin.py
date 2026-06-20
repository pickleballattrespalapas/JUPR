from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from jupr_app.config import SMTPConfig, get_email_mode, get_env_or_default, get_public_base_url
from jupr_app.domain.notifications.player_profile_update_repo import (
    REQUEST_STATUS_UNSUBSCRIBED,
    approve_request,
    bulk_delete_pending_outbox_rows,
    delete_pending_outbox_row,
    list_active_subscriptions,
    list_subscriptions_by_status,
    list_digests_for_range,
    list_outbox_rows,
    list_pending_requests,
    mark_unsubscribed,
    reject_request,
    reset_outbox_rows_to_pending,
    replace_verified_subscriber,
)
from jupr_app.domain.notifications.player_update_sender import (
    generate_and_queue_digest_for_player,
    generate_and_queue_digests_for_active_subscriptions,
    send_pending_player_update_emails,
)
from jupr_app.domain.recaps.player_weekly_digest import compute_player_weekly_digest
from jupr_app.domain.admin.roles import normalize_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.ui.components.player_digest_layout import render_player_digest
from jupr_app.ui.components.player_picker import render_player_picker
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


def _log_subscription_action(
    supabase,
    *,
    club_id: str,
    actor_email: str,
    actor_role: str,
    action_type: str,
    row_id: str,
    before_json: dict | None = None,
    after_json: dict | None = None,
    note: str | None = None,
) -> str | None:
    result = write_admin_activity_log(
        supabase,
        build_activity_payload(
            club_id=club_id,
            actor_email=actor_email,
            actor_role=actor_role,
            action_type=action_type,
            entity_type="subscription",
            entity_id=row_id,
            before_json=before_json,
            after_json=after_json,
            note=note,
            source_page="player_updates_admin",
        ),
    )
    return result.warning


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


def _inactive_table_rows(ctx, rows: list[dict]) -> list[dict]:
    display_rows: list[dict] = []
    for row in rows:
        pid = row.get("player_id")
        display_rows.append(
            {
                "player_id": pid,
                "player_name": _player_name(ctx, pid),
                "email": row.get("email"),
                "request_status": row.get("request_status"),
                "unsubscribed_at": row.get("unsubscribed_at"),
                "verified_at": row.get("verified_at"),
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
    render_player_digest(digest)

    with st.expander("Debug payload", expanded=False):
        st.json(digest)

    with st.expander("Raw chart points", expanded=False):
        points = (digest.get("chart") or {}).get("points") or []
        if points:
            st.dataframe(pd.DataFrame(points), use_container_width=True, hide_index=True)
        else:
            st.caption("No chart points available in selected date range.")






def _ui_env_or_secret(name: str, default: str = "") -> str:
    value = get_env_or_default(name).strip()
    if value:
        return value
    try:
        secret_val = st.secrets.get(name, default)
    except Exception:
        return str(default).strip()
    if secret_val is None:
        return str(default).strip()
    return str(secret_val).strip()


def _ui_env_or_secret_bool(name: str, default: bool = False) -> bool:
    value = _ui_env_or_secret(name).lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on"}


def _resolve_smtp_config_from_ui() -> SMTPConfig | None:
    host = _ui_env_or_secret("SMTP_HOST")
    port_raw = _ui_env_or_secret("SMTP_PORT")
    username = _ui_env_or_secret("SMTP_USERNAME")
    password = _ui_env_or_secret("SMTP_PASSWORD")
    from_email = _ui_env_or_secret("SMTP_FROM_EMAIL")
    if not all([host, port_raw, username, password, from_email]):
        return None

    try:
        port = int(port_raw)
    except Exception:
        return None

    return SMTPConfig(
        host=host,
        port=port,
        username=username,
        password=password,
        from_email=from_email,
        from_name=_ui_env_or_secret("SMTP_FROM_NAME", "JUPR Notifications"),
        reply_to=_ui_env_or_secret("SMTP_REPLY_TO", "joe@juprleagues.com"),
        use_tls=_ui_env_or_secret_bool("SMTP_USE_TLS", default=True),
    )

def _resolve_public_base_url() -> str:
    base = str(st.session_state.get("base_url", "") or "").strip().rstrip("/")
    return base or get_public_base_url()

def _friendly_error(exc: Exception) -> str:
    text = str(exc or "").strip()
    return text or "Unknown error."


def _outbox_display_rows(ctx, rows: list[dict]) -> list[dict]:
    display_rows: list[dict] = []
    for row in rows:
        display_rows.append(
            {
                "player_id": row.get("player_id"),
                "player_name": _player_name(ctx, row.get("player_id")),
                "week_start": row.get("week_start"),
                "week_end": row.get("week_end"),
                "send_status": row.get("send_status"),
                "email": row.get("email"),
                "sent_at": row.get("sent_at"),
                "error_text": row.get("error_text"),
                "created_at": row.get("created_at"),
            }
        )
    return display_rows


def _normalized_text_filter(value: str) -> str:
    return str(value or "").strip().lower()


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
    admin_role = normalize_role(str(st.session_state.get("admin_role", "") or ""))

    pending_tab, active_tab, digests_tab, queue_tab = st.tabs(
        [
            "Pending Requests",
            "Active Profiles",
            "Player Digests",
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
                        before_row = dict(row)
                        approve_request(supabase, row_id, verified_by=actor, admin_note=admin_note)
                        log_warning = _log_subscription_action(
                            supabase,
                            club_id=club_id,
                            actor_email=actor,
                            actor_role=admin_role,
                            action_type="subscription_approve",
                            row_id=row_id,
                            before_json=before_row,
                            after_json={"request_status": "active", "verified_by": actor},
                            note=admin_note,
                        )
                        st.success("Request approved.")
                        if log_warning:
                            st.warning(log_warning)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Approve failed: {_friendly_error(exc)}")

                if reject_clicked:
                    try:
                        before_row = dict(row)
                        reject_request(supabase, row_id, admin_note=admin_note, verified_by=actor)
                        log_warning = _log_subscription_action(
                            supabase,
                            club_id=club_id,
                            actor_email=actor,
                            actor_role=admin_role,
                            action_type="subscription_reject",
                            row_id=row_id,
                            before_json=before_row,
                            after_json={"request_status": "rejected", "verified_by": actor},
                            note=admin_note,
                        )
                        st.success("Request rejected.")
                        if log_warning:
                            st.warning(log_warning)
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
        unsubscribed_rows = list_subscriptions_by_status(
            supabase,
            club_id,
            statuses=[REQUEST_STATUS_UNSUBSCRIBED],
            limit=200,
        )

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
                        before_row = dict(row)
                        mark_unsubscribed(supabase, row_id)
                        log_warning = _log_subscription_action(
                            supabase,
                            club_id=club_id,
                            actor_email=actor,
                            actor_role=admin_role,
                            action_type="subscription_unsubscribe",
                            row_id=row_id,
                            before_json=before_row,
                            after_json={"request_status": "unsubscribed"},
                            note=admin_note,
                        )
                        st.success("Subscription deactivated.")
                        if log_warning:
                            st.warning(log_warning)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Unsubscribe failed: {_friendly_error(exc)}")

        st.markdown("#### Recently Unsubscribed")
        if unsubscribed_rows:
            st.dataframe(pd.DataFrame(_inactive_table_rows(ctx, unsubscribed_rows)), use_container_width=True)
        else:
            st.caption("No unsubscribed subscriptions yet.")

    with digests_tab:
        st.subheader("Player Digests")
        today = date.today()
        start_date = st.date_input(
            "Digest Start Date",
            value=today - timedelta(days=7),
            key="player_updates_digest_start",
            help="Custom date windows are supported (not Monday–Sunday only).",
        )
        end_date = st.date_input(
            "Digest End Date",
            value=today,
            key="player_updates_digest_end",
            help="Must be on or after Digest Start Date.",
        )

        if end_date < start_date:
            st.error("Digest End Date must be on or after Digest Start Date.")
            return

        active_rows = list_active_subscriptions(supabase, club_id, limit=500)
        st.markdown("#### Generate for all subscribed players")
        st.caption("Generating digests here also queues them automatically for the Send Queue page.")
        only_players_with_matches = st.checkbox(
            "Only generate for players with matches during selected dates",
            value=True,
            key="player_updates_only_players_with_matches",
            help="Recommended. This prevents creating queued emails for players who did not play during this date window.",
        )
        bulk_button_text = (
            "Generate + Queue for Active Players With Matches"
            if only_players_with_matches
            else "Generate + Queue for All Subscribed Players"
        )
        if st.button(bulk_button_text, disabled=not active_rows, use_container_width=True):
            try:
                result = generate_and_queue_digests_for_active_subscriptions(
                    ctx,
                    start_date=start_date,
                    end_date=end_date,
                    only_players_with_matches=only_players_with_matches,
                )
                if only_players_with_matches:
                    st.success(
                        "Bulk generation complete: "
                        f"active={result['active_subscriptions']} · "
                        f"players with matches={result['players_with_matches']} · "
                        f"eligible={result['eligible_subscriptions']} · "
                        f"saved={result['saved']} · "
                        f"queued={result['queued']} · "
                        f"skipped no matches={result['skipped_no_matches']} · "
                        f"failed={result['failed']}"
                    )
                else:
                    st.success(
                        "Bulk generation complete: "
                        f"active={result['active_subscriptions']} · "
                        f"eligible={result['eligible_subscriptions']} · "
                        f"saved={result['saved']} · "
                        f"queued={result['queued']} · "
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
            active_player_ids = []
            for row in active_rows:
                try:
                    active_player_ids.append(int(row.get("player_id")))
                except Exception:
                    continue
            active_player_ids = sorted(set(active_player_ids))
            active_df = getattr(ctx, "df_players_all", pd.DataFrame())
            active_df = active_df.copy() if isinstance(active_df, pd.DataFrame) else pd.DataFrame()
            if not active_df.empty and "id" in active_df.columns:
                active_df = active_df[active_df["id"].astype(int).isin(active_player_ids)].copy()
            selected_pid = render_player_picker(
                active_df,
                label="Search player",
                key="player_updates_digest_single_player",
                include_inactive=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Generate + Queue Selected Digest", disabled=selected_pid is None):
                    try:
                        result = generate_and_queue_digest_for_player(
                            ctx,
                            player_id=int(selected_pid),
                            start_date=start_date,
                            end_date=end_date,
                        )
                        st.session_state["player_updates_digest_preview"] = result.get("digest") or {}
                        st.success(
                            "Digest generated: "
                            f"saved={result['saved']} · queued={result['queued']}"
                        )
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
        st.caption("Digests generated on Player Digests are queued automatically. Just hit send here.")
        try:
            current_email_mode = get_email_mode()
        except Exception as exc:
            current_email_mode = f"invalid ({_friendly_error(exc)})"
        st.info(f"Email mode: {current_email_mode}")

        try:
            outbox_rows = list_outbox_rows(supabase, club_id, limit=1000)
        except Exception as exc:
            st.warning(f"Unable to load outbox rows: {exc}")
            outbox_rows = []

        pending_rows = [row for row in outbox_rows if str(row.get("send_status") or "") == "pending"]
        sent_rows = [row for row in outbox_rows if str(row.get("send_status") or "") == "sent"]
        skipped_rows = [row for row in outbox_rows if str(row.get("send_status") or "") == "skipped"]
        error_rows = [row for row in outbox_rows if str(row.get("send_status") or "") == "error"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pending", len(pending_rows))
        m2.metric("Sent", len(sent_rows))
        m3.metric("Skipped", len(skipped_rows))
        m4.metric("Errors", len(error_rows))

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Send Pending", use_container_width=True):
                try:
                    result = send_pending_player_update_emails(
                        ctx,
                        limit=500,
                        public_base_url=_resolve_public_base_url(),
                        smtp_config=_resolve_smtp_config_from_ui(),
                    )
                    st.success(
                        f"Attempted: {result['attempted']} · Sent: {result['sent']} · "
                        f"Skipped: {result['skipped']} · Errors: {result['errors']} · "
                        f"Mode: {result.get('email_mode', 'unknown')}"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Send pending failed: {_friendly_error(exc)}")

        with c2:
            if st.button("Retry Errored Rows", help="Reset all errored outbox rows back to pending.", use_container_width=True):
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

        st.divider()
        st.markdown("#### Pending to send")
        if pending_rows:
            st.dataframe(pd.DataFrame(_outbox_display_rows(ctx, pending_rows)), use_container_width=True, hide_index=True)
            st.markdown("#### Bulk remove pending queued digests")
            st.caption(
                "This only removes pending queued emails. Saved digest previews and sent email history are not affected."
            )

            bulk_rows: list[dict] = []
            for row in pending_rows:
                outbox_id = str(row.get("id") or "").strip()
                if not outbox_id:
                    continue
                bulk_rows.append(
                    {
                        "outbox_id": outbox_id,
                        "player_name": _player_name(ctx, row.get("player_id")),
                        "player_id": row.get("player_id"),
                        "email": row.get("email"),
                        "week_start": row.get("week_start"),
                        "week_end": row.get("week_end"),
                        "created_at": row.get("created_at"),
                    }
                )

            fs1, fs2, fs3 = st.columns(3)
            with fs1:
                filter_week_start = st.text_input(
                    "Filter week_start",
                    key="bulk_delete_filter_week_start",
                    placeholder="YYYY-MM-DD",
                ).strip()
            with fs2:
                filter_week_end = st.text_input(
                    "Filter week_end",
                    key="bulk_delete_filter_week_end",
                    placeholder="YYYY-MM-DD",
                ).strip()
            with fs3:
                filter_text = _normalized_text_filter(
                    st.text_input(
                        "Filter player/email",
                        key="bulk_delete_filter_text",
                        placeholder="Name or email",
                    )
                )

            filtered_rows: list[dict] = []
            for row in bulk_rows:
                row_week_start = str(row.get("week_start") or "").strip()
                row_week_end = str(row.get("week_end") or "").strip()
                row_name = _normalized_text_filter(row.get("player_name") or "")
                row_email = _normalized_text_filter(row.get("email") or "")
                if filter_week_start and row_week_start != filter_week_start:
                    continue
                if filter_week_end and row_week_end != filter_week_end:
                    continue
                if filter_text and filter_text not in row_name and filter_text not in row_email:
                    continue
                filtered_rows.append(row)

            selection_key = "player_updates_bulk_selected_outbox_ids"
            selected_ids = set(st.session_state.get(selection_key, []))
            visible_ids = [str(row.get("outbox_id") or "").strip() for row in filtered_rows if str(row.get("outbox_id") or "").strip()]

            b1, b2 = st.columns(2)
            with b1:
                if st.button("Select all pending", use_container_width=True, disabled=not visible_ids):
                    selected_ids.update(visible_ids)
                    st.session_state[selection_key] = sorted(selected_ids)
                    st.rerun()
            with b2:
                if st.button("Clear selection", use_container_width=True, disabled=not selected_ids):
                    st.session_state[selection_key] = []
                    st.rerun()

            if filtered_rows:
                editor_rows = []
                for row in filtered_rows:
                    outbox_id = str(row.get("outbox_id") or "").strip()
                    editor_rows.append(
                        {
                            "selected": outbox_id in selected_ids,
                            "player_name": row.get("player_name"),
                            "player_id": row.get("player_id"),
                            "email": row.get("email"),
                            "week_start": row.get("week_start"),
                            "week_end": row.get("week_end"),
                            "created_at": row.get("created_at"),
                            "_outbox_id": outbox_id,
                        }
                    )
                editor_df = pd.DataFrame(editor_rows)
                edited_df = st.data_editor(
                    editor_df,
                    use_container_width=True,
                    hide_index=True,
                    disabled=["player_name", "player_id", "email", "week_start", "week_end", "created_at", "_outbox_id"],
                    column_order=["selected", "player_name", "player_id", "email", "week_start", "week_end", "created_at"],
                    key="bulk_delete_pending_editor",
                )
                edited_selected_ids = {
                    str(row.get("_outbox_id") or "").strip()
                    for _, row in edited_df.iterrows()
                    if bool(row.get("selected")) and str(row.get("_outbox_id") or "").strip()
                }
                selected_ids = (selected_ids - set(visible_ids)) | edited_selected_ids
                st.session_state[selection_key] = sorted(selected_ids)
            else:
                st.caption("No pending rows match the current filters.")

            selected_count = len(selected_ids)
            st.caption(f"Selected pending rows: {selected_count}")
            confirm_bulk_delete = st.checkbox(
                "I understand this will remove the selected pending digests from the send queue. Sent history will not be deleted.",
                key="confirm_bulk_delete_pending",
            )
            if st.button(
                "Delete selected pending digests",
                use_container_width=True,
                disabled=(selected_count == 0 or not confirm_bulk_delete),
            ):
                try:
                    result = bulk_delete_pending_outbox_rows(
                        supabase,
                        club_id=club_id,
                        outbox_ids=sorted(selected_ids),
                    )
                    st.session_state[selection_key] = []
                    st.success(f"Deleted {result['deleted']} pending queued digests.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Bulk delete failed: {_friendly_error(exc)}")

            st.markdown("#### Remove one pending queued digest")
            st.caption(
                "This only removes pending queued emails. Saved digest previews and sent email history are not affected."
            )
            for row in pending_rows:
                outbox_id = str(row.get("id") or "").strip()
                player_name = _player_name(ctx, row.get("player_id")) or f"Player #{row.get('player_id')}"
                with st.expander(
                    f"{player_name} · {row.get('email') or 'no email'} · "
                    f"{row.get('week_start')} → {row.get('week_end')} · {row.get('created_at') or 'n/a'}"
                ):
                    with st.form(f"delete_pending_outbox_{outbox_id}"):
                        confirm_delete = st.checkbox(
                            "I understand this will remove this pending digest from the send queue.",
                            key=f"confirm_delete_outbox_{outbox_id}",
                        )
                        delete_clicked = st.form_submit_button("Delete queued digest")
                    if delete_clicked:
                        try:
                            if not confirm_delete:
                                raise ValueError("Please confirm removal before deleting.")
                            delete_pending_outbox_row(supabase, club_id=club_id, outbox_id=outbox_id)
                            st.success("Queued digest removed.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Delete failed: {_friendly_error(exc)}")
        else:
            st.caption("No pending emails right now.")

        with st.expander("Outbox history", expanded=False):
            if outbox_rows:
                st.dataframe(pd.DataFrame(_outbox_display_rows(ctx, outbox_rows)), use_container_width=True, hide_index=True)
            else:
                st.caption("No outbox rows.")
