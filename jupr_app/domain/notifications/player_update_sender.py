from __future__ import annotations

from datetime import date
from typing import Any

import streamlit as st

from jupr_app.domain.notifications.player_profile_update_repo import (
    DEFAULT_PREFERENCES,
    REQUEST_STATUS_ACTIVE,
    SEND_STATUS_ERROR,
    SEND_STATUS_SENT,
    SEND_STATUS_SKIPPED,
    create_outbox_row,
    list_active_subscriptions,
    list_outbox_rows,
    save_digest,
    update_outbox_status,
)
from jupr_app.domain.notifications.player_update_charts import render_player_digest_chart_png
from jupr_app.domain.notifications.player_update_email_template import (
    build_player_update_email_html,
    build_player_update_email_subject,
    build_player_update_email_text,
)
from jupr_app.domain.notifications.smtp_mailer import send_email_with_inline_chart
from jupr_app.domain.recaps.player_weekly_digest import compute_player_weekly_digest


def _safe_subscription(supabase, subscription_id: str) -> dict | None:
    try:
        resp = (
            supabase.table("player_profile_update_subscriptions")
            .select("*")
            .eq("id", str(subscription_id))
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        return rows[0] if rows else None
    except Exception:
        return None


def _safe_digest_for_week(supabase, club_id: str, player_id: int, week_start: date) -> dict | None:
    try:
        resp = (
            supabase.table("player_weekly_profile_digests")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("player_id", int(player_id))
            .eq("week_start", week_start.isoformat())
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        return rows[0] if rows else None
    except Exception:
        return None


def _merge_links_for_send(*, digest: dict[str, Any], player_id: int, subscription_id: str) -> dict[str, Any]:
    links = dict((digest or {}).get("links") or {})
    links["player_profile"] = f"/?page=players&public=1&pid={int(player_id)}"
    links["unsubscribe"] = (
        f"/?page=players&public=1&pid={int(player_id)}&unsubscribe=1&sid={subscription_id}"
    )
    merged = dict(digest or {})
    merged["links"] = links
    return merged


def _is_send_only_if_changed_and_unchanged(subscription: dict, digest: dict) -> bool:
    prefs = subscription.get("preferences_json") or dict(DEFAULT_PREFERENCES)
    send_only_if_changed = bool((prefs or {}).get("send_only_if_changed", True))
    if not send_only_if_changed:
        return False

    summary = digest.get("summary") or {}
    matches_played = int(summary.get("matches_played") or 0)
    badges = digest.get("badges_earned") or []
    trophies = digest.get("trophies_earned") or []
    return matches_played == 0 and len(badges) == 0 and len(trophies) == 0


def queue_digest_outbox_rows_for_range(ctx, *, start_date: date, end_date: date) -> dict[str, int]:
    supabase = ctx.supabase
    club_id = str(ctx.club_id)

    active_rows = list_active_subscriptions(supabase, club_id, limit=2000)
    queued = 0
    already_exists = 0
    failed = 0

    for sub in active_rows:
        try:
            player_id = int(sub.get("player_id"))
            digest = compute_player_weekly_digest(ctx, player_id=player_id, start_date=start_date, end_date=end_date)
            save_digest(
                supabase,
                club_id=club_id,
                player_id=player_id,
                week_start=start_date,
                week_end=end_date,
                generated_json=digest,
                final_json=digest,
            )
            create_outbox_row(
                supabase,
                subscription_id=str(sub.get("id") or ""),
                club_id=club_id,
                player_id=player_id,
                week_start=start_date,
                week_end=end_date,
                email=str(sub.get("email") or ""),
            )
            queued += 1
        except Exception as exc:
            if "duplicate key" in str(exc).lower() or "unique" in str(exc).lower():
                already_exists += 1
            else:
                failed += 1

    return {
        "queued": queued,
        "already_exists": already_exists,
        "failed": failed,
    }


def send_pending_player_update_emails(ctx, *, limit: int = 100) -> dict[str, int]:
    supabase = ctx.supabase
    club_id = str(ctx.club_id)
    pending_rows = list_outbox_rows(supabase, club_id, status="pending", limit=max(1, int(limit)))

    sent = 0
    skipped = 0
    errors = 0

    for outbox in pending_rows:
        outbox_id = str(outbox.get("id") or "")
        subscription = _safe_subscription(supabase, str(outbox.get("subscription_id") or ""))
        try:
            if not subscription:
                raise ValueError("Subscription not found")
            if str(subscription.get("request_status") or "") != REQUEST_STATUS_ACTIVE:
                update_outbox_status(
                    supabase,
                    outbox_id,
                    send_status=SEND_STATUS_SKIPPED,
                    error_text="Subscription is no longer active.",
                )
                skipped += 1
                continue

            week_start = date.fromisoformat(str(outbox.get("week_start")))
            week_end = date.fromisoformat(str(outbox.get("week_end")))
            player_id = int(outbox.get("player_id"))

            digest_row = _safe_digest_for_week(supabase, club_id, player_id, week_start)
            digest = (digest_row or {}).get("final_json") or (digest_row or {}).get("generated_json") or {}
            if not digest:
                digest = compute_player_weekly_digest(ctx, player_id=player_id, start_date=week_start, end_date=week_end)
                save_digest(
                    supabase,
                    club_id=club_id,
                    player_id=player_id,
                    week_start=week_start,
                    week_end=week_end,
                    generated_json=digest,
                    final_json=digest,
                )

            digest = _merge_links_for_send(
                digest=digest,
                player_id=player_id,
                subscription_id=str(subscription.get("id") or ""),
            )

            if _is_send_only_if_changed_and_unchanged(subscription, digest):
                update_outbox_status(
                    supabase,
                    outbox_id,
                    send_status=SEND_STATUS_SKIPPED,
                    error_text="Skipped because send_only_if_changed is enabled and no changes were detected.",
                )
                skipped += 1
                continue

            chart_cid = "player-digest-chart"
            chart_png = render_player_digest_chart_png(digest)
            subject = build_player_update_email_subject(digest)
            html_body = build_player_update_email_html(digest, chart_cid if chart_png else None)
            text_body = build_player_update_email_text(digest)

            provider_message_id = send_email_with_inline_chart(
                to_email=str(outbox.get("email") or ""),
                subject=subject,
                html_body=html_body,
                text_body=text_body,
                chart_png_bytes=chart_png,
                chart_cid=chart_cid if chart_png else None,
            )

            update_outbox_status(
                supabase,
                outbox_id,
                send_status=SEND_STATUS_SENT,
                provider_message_id=provider_message_id,
                error_text=None,
            )
            (
                supabase.table("player_profile_update_subscriptions")
                .update({"last_digest_week_start": week_start.isoformat()})
                .eq("id", str(subscription.get("id") or ""))
                .execute()
            )
            sent += 1
        except Exception as exc:
            update_outbox_status(
                supabase,
                outbox_id,
                send_status=SEND_STATUS_ERROR,
                error_text=str(exc),
            )
            errors += 1

    return {
        "attempted": len(pending_rows),
        "sent": sent,
        "skipped": skipped,
        "errors": errors,
    }


def send_test_player_update_email(
    ctx,
    *,
    start_date: date,
    end_date: date,
    player_id: int | None = None,
    to_email: str | None = None,
) -> dict[str, str]:
    supabase = ctx.supabase
    club_id = str(ctx.club_id)
    admin_email = str(
        to_email
        or getattr(ctx, "admin_email", "")
        or st.session_state.get("admin_email", "")
        or getattr(ctx, "user_email", "")
        or st.session_state.get("user_email", "")
    ).strip()
    if not admin_email:
        raise ValueError("No admin email available for test send.")

    selected_player_id = player_id
    selected_subscription_id = "test-subscription"

    if selected_player_id is None:
        active = list_active_subscriptions(supabase, club_id, limit=1)
        if not active:
            raise ValueError("No active subscriptions available for test send.")
        selected_player_id = int(active[0].get("player_id"))
        selected_subscription_id = str(active[0].get("id") or selected_subscription_id)

    digest = compute_player_weekly_digest(
        ctx,
        player_id=int(selected_player_id),
        start_date=start_date,
        end_date=end_date,
    )
    digest = _merge_links_for_send(
        digest=digest,
        player_id=int(selected_player_id),
        subscription_id=selected_subscription_id,
    )

    chart_cid = "player-digest-chart"
    chart_png = render_player_digest_chart_png(digest)
    subject = f"[TEST] {build_player_update_email_subject(digest)}"
    html_body = build_player_update_email_html(digest, chart_cid if chart_png else None)
    text_body = build_player_update_email_text(digest)

    provider_message_id = send_email_with_inline_chart(
        to_email=admin_email,
        subject=subject,
        html_body=html_body,
        text_body=text_body,
        chart_png_bytes=chart_png,
        chart_cid=chart_cid if chart_png else None,
    )
    return {"to_email": admin_email, "provider_message_id": provider_message_id}
