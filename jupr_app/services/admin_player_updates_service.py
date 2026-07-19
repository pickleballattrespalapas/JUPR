from __future__ import annotations

from datetime import date, datetime, timezone
import os
from types import SimpleNamespace
from typing import Any

from jupr_app.config import (
    EMAIL_MODE_DRY_RUN,
    EMAIL_MODE_LIVE,
    SMTPConfig,
    get_email_mode,
    get_next_web_base_url,
)
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.notifications.player_update_sender import (
    generate_and_queue_digests_for_active_subscriptions,
    send_pending_player_update_emails,
)
from jupr_app.domain.notifications.smtp_mailer import get_smtp_config_status

TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
CONFIRM_SEND_PLAYER_UPDATES = "SEND PLAYER UPDATES"


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def is_admin_player_updates_enabled() -> bool:
    return _truthy_env("JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES")


def is_auto_player_updates_enabled() -> bool:
    return _truthy_env("JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS")


def is_next_live_player_update_email_enabled() -> bool:
    return _truthy_env("JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL")


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
    del supabase, club_id
    smtp_configured = bool(get_smtp_config_status().get("ok"))
    if not is_admin_player_updates_enabled():
        return {
            "enabled": False,
            "status": "guarded_off",
            "send_range_endpoint": None,
            "auto_send_enabled": is_auto_player_updates_enabled(),
            "email_mode": get_email_mode(),
            "smtp_configured": smtp_configured,
            "warnings": ["Next Player Updates Admin is disabled. Enable JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES on FastAPI for the closed-club pilot."],
        }
    warnings: list[str] = []
    if not is_auto_player_updates_enabled():
        warnings.append("Automatic post-batch player update email sending is disabled. Set JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS=1 after email mode is verified.")
    if get_email_mode() != EMAIL_MODE_DRY_RUN and not smtp_configured:
        warnings.append("SMTP is not fully configured; live or staging-redirect sends will fail until SMTP_* secrets are set.")
    if get_email_mode() == EMAIL_MODE_LIVE and not is_next_live_player_update_email_enabled():
        warnings.append("Next Player Updates live delivery is blocked. Use dry_run/staging_redirect or deliberately enable JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL=1.")
    return {
        "enabled": True,
        "status": "ready_for_player_update_range_reports",
        "send_range_endpoint": "/admin/clubs/{club_id}/player-updates/send-range",
        "workspace_endpoint": "/admin/clubs/{club_id}/player-updates/workspace",
        "auto_send_enabled": is_auto_player_updates_enabled(),
        "email_mode": get_email_mode(),
        "smtp_configured": smtp_configured,
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
    outbox_items: list[dict[str, Any]] | None = None,
    actor_email: str = "",
) -> dict[str, int | str]:
    return send_pending_player_update_emails(
        ctx,
        limit=limit,
        public_base_url=public_base_url,
        smtp_config=smtp_config,
        start_date=start_date,
        end_date=end_date,
        outbox_items=outbox_items,
        actor_email=actor_email,
        enforce_next_live_gate=True,
    )


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

    if is_api_audit_log_required():
        intent_payload = build_activity_payload(
            club_id=str(club_id),
            actor_email=str(actor_email or ""),
            actor_role=str(actor_role or ""),
            action_type="send_player_update_range_admin_intent",
            entity_type="player_updates",
            entity_id=f"{start.isoformat()}:{end.isoformat()}",
            after_json={
                "source_client": "fastapi/nextjs",
                "phase": "intent",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "only_players_with_matches": bool(only_players_with_matches),
                "send_now": bool(send_now),
            },
            source_page=source,
            flagged_for_review=True,
        )
        intent_write = write_admin_activity_log(supabase, intent_payload)
        if not intent_write.ok:
            raise RuntimeError("Required audit intent could not be persisted; nothing was queued or sent.")

    try:
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
                    actor_email=str(actor_email or ""),
                ),
            }
    except Exception as exc:
        failure_payload = build_activity_payload(
            club_id=str(club_id),
            actor_email=str(actor_email or ""),
            actor_role=str(actor_role or ""),
            action_type="send_player_update_range_admin_failed",
            entity_type="player_updates",
            entity_id=f"{start.isoformat()}:{end.isoformat()}",
            after_json={
                "source_client": "fastapi/nextjs",
                "phase": "failed",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "send_now": bool(send_now),
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            },
            source_page=source,
            flagged_for_review=True,
        )
        failure_write = write_admin_activity_log(supabase, failure_payload)
        if not failure_write.ok and is_api_audit_log_required():
            raise RuntimeError(
                "Player update processing is uncertain and its required failure audit also failed. Reload before retrying."
            ) from exc
        raise

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
        raise RuntimeError(
            "Player updates may have been queued or sent, but the required completion audit failed. Reload before retrying."
        )

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
