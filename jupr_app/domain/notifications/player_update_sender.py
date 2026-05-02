from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

from jupr_app.domain.notifications.player_profile_update_repo import (
    DEFAULT_PREFERENCES,
    REQUEST_STATUS_ACTIVE,
    SEND_STATUS_ERROR,
    SEND_STATUS_SENT,
    SEND_STATUS_SKIPPED,
    create_outbox_row,
    ensure_unsubscribe_token,
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
from jupr_app.config import SMTPConfig, get_public_base_url
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


def _safe_digest_for_week(
    supabase,
    club_id: str,
    player_id: int,
    week_start: date,
    week_end: date,
) -> dict | None:
    try:
        resp = (
            supabase.table("player_weekly_profile_digests")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("player_id", int(player_id))
            .eq("week_start", week_start.isoformat())
            .eq("week_end", week_end.isoformat())
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        return rows[0] if rows else None
    except Exception:
        return None


def _normalize_public_base_url(public_base_url: str | None = None) -> str:
    return str(public_base_url or get_public_base_url()).strip().rstrip("/")


def _build_public_players_url(params: dict[str, str], *, public_base_url: str | None = None) -> str:
    query = {"page": "players", "public": "1"}
    for key, value in (params or {}).items():
        query[str(key)] = str(value)
    return f"{_normalize_public_base_url(public_base_url)}/?{urlencode(query)}"


def _merge_links_for_send(
    *,
    digest: dict[str, Any],
    player_id: int,
    subscription_id: str,
    unsubscribe_token: str | None = None,
    public_base_url: str | None = None,
) -> dict[str, Any]:
    links = dict((digest or {}).get("links") or {})
    links["player_profile"] = _build_public_players_url({"pid": str(int(player_id))}, public_base_url=public_base_url)
    unsubscribe_params = {"page": "email_preferences"}
    if str(unsubscribe_token or "").strip():
        unsubscribe_params["token"] = str(unsubscribe_token).strip()
    else:
        unsubscribe_params["sid"] = str(subscription_id)
    links["unsubscribe"] = f"{_normalize_public_base_url(public_base_url)}/?{urlencode(unsubscribe_params)}"
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


def _coerce_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        raise ValueError("Date value is required")
    return date.fromisoformat(text[:10])


def _find_active_subscription_for_player(active_rows: list[dict[str, Any]], player_id: int) -> dict[str, Any] | None:
    for row in active_rows:
        try:
            if int(row.get("player_id")) == int(player_id):
                return row
        except Exception:
            continue
    return None


def _coerce_match_day(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc).date()
    except Exception:
        try:
            return date.fromisoformat(text[:10])
        except Exception:
            return None


def _coerce_optional_player_id(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except Exception:
        return None


def get_player_ids_with_matches_in_range(ctx, *, start_date: date, end_date: date) -> set[int]:
    supabase = ctx.supabase
    club_id = str(ctx.club_id)
    query_end = end_date + timedelta(days=1)
    rows = (
        supabase.table("matches")
        .select("date,t1_p1,t1_p2,t2_p1,t2_p2")
        .eq("club_id", club_id)
        .gte("date", start_date.isoformat())
        .lte("date", query_end.isoformat())
        .execute()
    ).data or []

    player_ids: set[int] = set()
    for row in rows:
        match_day = _coerce_match_day((row or {}).get("date"))
        if match_day is None or match_day < start_date or match_day > end_date:
            continue
        for key in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
            pid = _coerce_optional_player_id((row or {}).get(key))
            if pid is not None:
                player_ids.add(pid)
    return player_ids


def _save_digest_for_subscription(ctx, *, subscription: dict[str, Any], start_date: date, end_date: date) -> dict[str, Any]:
    supabase = ctx.supabase
    club_id = str(ctx.club_id)
    player_id = int(subscription.get("player_id"))
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
    return digest


def _queue_outbox_for_subscription(
    ctx,
    *,
    subscription: dict[str, Any],
    start_date: date,
    end_date: date,
) -> str:
    supabase = ctx.supabase
    club_id = str(ctx.club_id)
    player_id = int(subscription.get("player_id"))
    create_outbox_row(
        supabase,
        subscription_id=str(subscription.get("id") or ""),
        club_id=club_id,
        player_id=player_id,
        week_start=start_date,
        week_end=end_date,
        email=str(subscription.get("email") or ""),
    )
    return "queued"


def generate_digests_for_active_subscriptions(ctx, *, start_date: date, end_date: date) -> dict[str, int]:
    supabase = ctx.supabase
    club_id = str(ctx.club_id)

    active_rows = list_active_subscriptions(supabase, club_id, limit=2000)
    saved = 0
    failed = 0

    for sub in active_rows:
        try:
            _save_digest_for_subscription(ctx, subscription=sub, start_date=start_date, end_date=end_date)
            saved += 1
        except Exception:
            failed += 1

    return {
        "active_subscriptions": len(active_rows),
        "saved": saved,
        "failed": failed,
    }


def generate_and_queue_digests_for_active_subscriptions(
    ctx,
    *,
    start_date: date,
    end_date: date,
    only_players_with_matches: bool = False,
) -> dict[str, int]:
    supabase = ctx.supabase
    club_id = str(ctx.club_id)

    active_rows = list_active_subscriptions(supabase, club_id, limit=2000)
    player_ids_with_matches = (
        get_player_ids_with_matches_in_range(ctx, start_date=start_date, end_date=end_date)
        if only_players_with_matches
        else set()
    )
    eligible_rows = (
        [
            row
            for row in active_rows
            if _coerce_optional_player_id(row.get("player_id")) in player_ids_with_matches
        ]
        if only_players_with_matches
        else list(active_rows)
    )
    saved = 0
    queued = 0
    failed = 0

    for sub in eligible_rows:
        try:
            _save_digest_for_subscription(ctx, subscription=sub, start_date=start_date, end_date=end_date)
            saved += 1
            _queue_outbox_for_subscription(ctx, subscription=sub, start_date=start_date, end_date=end_date)
            queued += 1
        except Exception:
            failed += 1

    return {
        "active_subscriptions": len(active_rows),
        "players_with_matches": len(player_ids_with_matches),
        "eligible_subscriptions": len(eligible_rows),
        "saved": saved,
        "queued": queued,
        "skipped_no_matches": max(0, len(active_rows) - len(eligible_rows)),
        "failed": failed,
    }


def generate_and_queue_digest_for_player(
    ctx,
    *,
    player_id: int,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    supabase = ctx.supabase
    club_id = str(ctx.club_id)
    active_rows = list_active_subscriptions(supabase, club_id, limit=2000)
    subscription = _find_active_subscription_for_player(active_rows, int(player_id))
    if subscription is None:
        raise ValueError("No active verified subscriber exists for that player.")

    digest = _save_digest_for_subscription(ctx, subscription=subscription, start_date=start_date, end_date=end_date)
    _queue_outbox_for_subscription(ctx, subscription=subscription, start_date=start_date, end_date=end_date)

    return {
        "player_id": int(player_id),
        "digest": digest,
        "saved": 1,
        "queued": 1,
    }


def queue_digest_outbox_rows_for_range(ctx, *, start_date: date, end_date: date) -> dict[str, int]:
    supabase = ctx.supabase
    club_id = str(ctx.club_id)

    active_rows = list_active_subscriptions(supabase, club_id, limit=2000)
    queued = 0
    failed = 0

    for sub in active_rows:
        try:
            _save_digest_for_subscription(ctx, subscription=sub, start_date=start_date, end_date=end_date)
            _queue_outbox_for_subscription(ctx, subscription=sub, start_date=start_date, end_date=end_date)
            queued += 1
        except Exception:
            failed += 1

    return {
        "queued": queued,
        "failed": failed,
    }


def queue_saved_digest_rows(ctx, *, digest_rows: list[dict[str, Any]]) -> dict[str, int]:
    supabase = ctx.supabase
    club_id = str(ctx.club_id)

    active_rows = list_active_subscriptions(supabase, club_id, limit=2000)
    active_by_player: dict[int, dict[str, Any]] = {}
    for row in active_rows:
        try:
            active_by_player[int(row.get("player_id"))] = row
        except Exception:
            continue

    queued = 0
    no_active_subscription = 0
    failed = 0

    for digest_row in digest_rows:
        try:
            player_id = int(digest_row.get("player_id"))
            subscription = active_by_player.get(player_id)
            if not subscription:
                no_active_subscription += 1
                continue

            create_outbox_row(
                supabase,
                subscription_id=str(subscription.get("id") or ""),
                club_id=club_id,
                player_id=player_id,
                week_start=_coerce_date(digest_row.get("week_start")),
                week_end=_coerce_date(digest_row.get("week_end")),
                email=str(subscription.get("email") or ""),
            )
            queued += 1
        except Exception:
            failed += 1

    return {
        "queued": queued,
        "no_active_subscription": no_active_subscription,
        "failed": failed,
    }


def send_pending_player_update_emails(
    ctx,
    *,
    limit: int = 100,
    public_base_url: str | None = None,
    smtp_config: SMTPConfig | None = None,
) -> dict[str, int]:
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

            digest_row = _safe_digest_for_week(supabase, club_id, player_id, week_start, week_end)
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
                unsubscribe_token=ensure_unsubscribe_token(
                    supabase,
                    str(subscription.get("id") or ""),
                ),
                public_base_url=public_base_url,
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
                unsubscribe_url=str(((digest.get("links") or {}).get("unsubscribe")) or "").strip() or None,
                smtp_config=smtp_config,
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
    public_base_url: str | None = None,
    smtp_config: SMTPConfig | None = None,
) -> dict[str, str]:
    supabase = ctx.supabase
    club_id = str(ctx.club_id)
    admin_email = str(
        to_email
        or getattr(ctx, "admin_email", "")
        or getattr(ctx, "user_email", "")
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
        unsubscribe_token=ensure_unsubscribe_token(supabase, selected_subscription_id),
        public_base_url=public_base_url,
    )

    chart_cid = "player-digest-chart"
    chart_png = render_player_digest_chart_png(digest)
    subject = f"[TEST] {build_player_update_email_subject(digest)}"
    html_body = build_player_update_email_html(digest, chart_cid if chart_png else None)
    text_body = build_player_update_email_text(digest)
    unsubscribe_url = str(((digest.get("links") or {}).get("unsubscribe")) or "").strip() or None

    provider_message_id = send_email_with_inline_chart(
        to_email=admin_email,
        subject=subject,
        html_body=html_body,
        text_body=text_body,
        chart_png_bytes=chart_png,
        chart_cid=chart_cid if chart_png else None,
        unsubscribe_url=unsubscribe_url,
        smtp_config=smtp_config,
    )
    return {"to_email": admin_email, "provider_message_id": provider_message_id}
