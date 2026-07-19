from __future__ import annotations

from datetime import date, datetime, timezone
import os
from types import SimpleNamespace
from typing import Any

from jupr_app.config import (
    EMAIL_MODE_DRY_RUN,
    EMAIL_MODE_STAGING_REDIRECT,
    SMTPConfig,
    get_email_mode,
    get_next_web_base_url,
)
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.notifications.player_profile_update_repo import (
    REQUEST_STATUS_ACTIVE,
    SEND_STATUS_ERROR,
    SEND_STATUS_SENT,
    SEND_STATUS_SKIPPED,
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
from jupr_app.domain.notifications.player_update_sender import (
    _is_send_only_if_changed_and_unchanged,
    _merge_links_for_send,
    _safe_digest_for_week,
    _safe_subscription,
    generate_and_queue_digests_for_active_subscriptions,
)
from jupr_app.domain.notifications.smtp_mailer import get_smtp_config_status, send_email_with_inline_chart
from jupr_app.domain.recaps.player_weekly_digest import compute_player_weekly_digest

TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
CONFIRM_SEND_PLAYER_UPDATES = "SEND PLAYER UPDATES"


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def is_admin_player_updates_enabled() -> bool:
    return _truthy_env("JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES")


def is_auto_player_updates_enabled() -> bool:
    return _truthy_env("JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS")


def is_api_audit_log_required() -> bool:
    return _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG")


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _coerce_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        raise ValueError("Date value is required.")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc).date()
    except Exception:
        return date.fromisoformat(text[:10])


def _safe_match_date(value: Any) -> date | None:
    try:
        return _coerce_date(value)
    except Exception:
        return None


def _build_ctx(supabase: Any, *, club_id: str) -> SimpleNamespace:
    rows = _safe_rows(supabase.table("players").select("id,name").eq("club_id", str(club_id)).execute())
    id_to_name: dict[int, str] = {}
    for row in rows:
        try:
            pid = int(row.get("id"))
        except Exception:
            continue
        name = str(row.get("name") or "").strip()
        if name:
            id_to_name[pid] = name
    return SimpleNamespace(supabase=supabase, club_id=str(club_id), id_to_name=id_to_name)


def build_admin_player_updates_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    if not is_admin_player_updates_enabled():
        return {
            "enabled": False,
            "status": "guarded_off",
            "send_range_endpoint": None,
            "auto_send_enabled": is_auto_player_updates_enabled(),
            "email_mode": get_email_mode(),
            "smtp_status": get_smtp_config_status(),
            "warnings": ["Next Player Updates Admin is disabled. Enable JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES on FastAPI for the closed-club pilot."],
        }
    active_count = None
    if supabase is not None:
        try:
            active_count = len(list_active_subscriptions(supabase, str(club_id), limit=2000))
        except Exception:
            active_count = None
    warnings: list[str] = []
    if not is_auto_player_updates_enabled():
        warnings.append("Automatic post-batch player update email sending is disabled. Set JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS=1 after email mode is verified.")
    smtp_status = get_smtp_config_status()
    if get_email_mode() != EMAIL_MODE_DRY_RUN and not bool(smtp_status.get("configured")):
        warnings.append("SMTP is not fully configured; live or staging-redirect sends will fail until SMTP_* secrets are set.")
    return {
        "enabled": True,
        "status": "ready_for_player_update_range_reports",
        "send_range_endpoint": "/admin/clubs/{club_id}/player-updates/send-range",
        "auto_send_enabled": is_auto_player_updates_enabled(),
        "email_mode": get_email_mode(),
        "smtp_status": smtp_status,
        "active_subscription_count": active_count,
        "warnings": warnings,
    }


def send_pending_player_update_emails_for_range(
    ctx: Any,
    *,
    start_date: date,
    end_date: date,
    limit: int = 1000,
    public_base_url: str | None = None,
    smtp_config: SMTPConfig | None = None,
) -> dict[str, int | str]:
    supabase = ctx.supabase
    club_id = str(ctx.club_id)
    pending_rows = list_outbox_rows(supabase, club_id, status="pending", limit=max(1, int(limit)))
    start_iso = start_date.isoformat()
    end_iso = end_date.isoformat()
    pending_rows = [
        row
        for row in pending_rows
        if str(row.get("week_start") or "") == start_iso and str(row.get("week_end") or "") == end_iso
    ]

    email_mode = get_email_mode()
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

            player_id = int(outbox.get("player_id"))
            digest_row = _safe_digest_for_week(supabase, club_id, player_id, start_date, end_date)
            digest = (digest_row or {}).get("final_json") or (digest_row or {}).get("generated_json") or {}
            if not digest:
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

            digest = _merge_links_for_send(
                digest=digest,
                player_id=player_id,
                subscription_id=str(subscription.get("id") or ""),
                unsubscribe_token=ensure_unsubscribe_token(supabase, str(subscription.get("id") or "")),
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
            unsubscribe_url = str(((digest.get("links") or {}).get("unsubscribe")) or "").strip() or None

            original_to_email = str(outbox.get("email") or "").strip()
            effective_to_email = original_to_email
            if email_mode == EMAIL_MODE_STAGING_REDIRECT:
                from jupr_app.config import get_env_or_default

                redirect_to = get_env_or_default("JUPR_STAGING_EMAIL_REDIRECT_TO").strip()
                if not redirect_to:
                    raise ValueError("JUPR_STAGING_EMAIL_REDIRECT_TO is required when JUPR_EMAIL_MODE=staging_redirect.")
                effective_to_email = redirect_to
                subject = f"[STAGING→{original_to_email}] {subject}"

            if email_mode == EMAIL_MODE_DRY_RUN:
                provider_message_id = "dry_run"
            else:
                provider_message_id = send_email_with_inline_chart(
                    to_email=effective_to_email,
                    subject=subject,
                    html_body=html_body,
                    text_body=text_body,
                    chart_png_bytes=chart_png,
                    chart_cid=chart_cid if chart_png else None,
                    unsubscribe_url=unsubscribe_url,
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
                .update({"last_digest_week_start": start_date.isoformat()})
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
        "email_mode": email_mode,
    }


def run_admin_player_update_range(
    supabase: Any,
    *,
    club_id: str,
    start_date: Any,
    end_date: Any,
    only_players_with_matches: bool = True,
    send_now: bool = True,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_player_updates_admin_range",
) -> dict[str, Any]:
    if not is_admin_player_updates_enabled():
        raise PermissionError("Next Player Updates Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_SEND_PLAYER_UPDATES:
        raise ValueError(f"Type {CONFIRM_SEND_PLAYER_UPDATES} to send player update emails.")
    start = _coerce_date(start_date)
    end = _coerce_date(end_date)
    if end < start:
        raise ValueError("End date must be on or after start date.")
    if (end - start).days > 45:
        raise ValueError("Player update report ranges are capped at 45 days per send.")

    ctx = _build_ctx(supabase, club_id=str(club_id))
    generation_result = generate_and_queue_digests_for_active_subscriptions(
        ctx,
        start_date=start,
        end_date=end,
        only_players_with_matches=bool(only_players_with_matches),
    )
    send_result: dict[str, Any] = {"mode": "skipped", "attempted": 0, "sent": 0, "skipped": 0, "errors": 0, "email_mode": get_email_mode()}
    if send_now:
        send_result = {
            "mode": "sent",
            **send_pending_player_update_emails_for_range(
                ctx,
                start_date=start,
                end_date=end,
                limit=2000,
                public_base_url=get_next_web_base_url(),
            ),
        }

    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="send_player_update_range_admin",
        entity_type="player_updates",
        entity_id=f"{start.isoformat()}:{end.isoformat()}",
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "only_players_with_matches": bool(only_players_with_matches),
            "send_now": bool(send_now),
            "generation_result": generation_result,
            "send_result": send_result,
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
        "mode": "player_update_range_send",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "generation_result": generation_result,
        "send_result": send_result,
        "warnings": warnings,
    }


def auto_send_player_updates_for_match_payloads(
    supabase: Any,
    *,
    club_id: str,
    match_payloads: list[dict[str, Any]],
    source: str = "auto_player_updates_after_batch",
) -> dict[str, Any]:
    if not is_auto_player_updates_enabled():
        return {"mode": "disabled", "reason": "JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS is not enabled."}
    dates = sorted({day for day in (_safe_match_date(row.get("date")) for row in (match_payloads or [])) if day is not None})
    if not dates:
        return {"mode": "skipped", "reason": "No match dates available for player update sending."}
    ctx = _build_ctx(supabase, club_id=str(club_id))
    totals = {"attempted": 0, "sent": 0, "skipped": 0, "errors": 0}
    windows: list[dict[str, Any]] = []
    for day in dates:
        result = send_pending_player_update_emails_for_range(
            ctx,
            start_date=day,
            end_date=day,
            limit=2000,
            public_base_url=get_next_web_base_url(),
        )
        windows.append({"start_date": day.isoformat(), "end_date": day.isoformat(), **result})
        for key in totals:
            totals[key] += int(result.get(key, 0) or 0)
    return {
        "mode": "auto_sent",
        "source": source,
        "windows": windows,
        **totals,
        "email_mode": get_email_mode(),
    }
