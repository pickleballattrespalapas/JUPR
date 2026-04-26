from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

REQUEST_STATUS_PENDING = "pending_admin_review"
REQUEST_STATUS_ACTIVE = "active"
REQUEST_STATUS_REJECTED = "rejected"
REQUEST_STATUS_UNSUBSCRIBED = "unsubscribed"

SEND_STATUS_PENDING = "pending"
SEND_STATUS_SENT = "sent"
SEND_STATUS_SKIPPED = "skipped"
SEND_STATUS_ERROR = "error"

DEFAULT_PREFERENCES = {
    "frequency": "weekly",
    "send_only_if_changed": True,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_data(resp: Any) -> list[dict[str, Any]]:
    try:
        return list(resp.data or [])
    except Exception:
        return []


def _safe_first(resp: Any) -> dict[str, Any] | None:
    rows = _safe_data(resp)
    return rows[0] if rows else None


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError("player_id must be an integer-like value")


def _is_unique_violation(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return "duplicate key" in text or "unique" in text


def _coerce_date(value: date | datetime | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc).date()
    except Exception:
        try:
            return date.fromisoformat(text[:10])
        except Exception:
            return None


def _week_window_for_day(day: date) -> tuple[date, date]:
    week_start = day - timedelta(days=day.weekday())
    week_end = week_start + timedelta(days=6)
    return week_start, week_end


def _digest_window_for_match_day(day: date) -> tuple[date, date]:
    return day, day


def normalize_email(email: str) -> str:
    return str(email or "").strip().lower()


def _require_nonempty(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def create_public_request(
    supabase,
    *,
    club_id: str,
    player_id: int,
    email: str,
    request_note: str | None = None,
    preferences_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    club_id = _require_nonempty(club_id, "club_id")
    email_raw = _require_nonempty(email, "email")
    normalized = normalize_email(email_raw)
    player_id_int = _safe_int(player_id)

    existing = get_open_or_active_subscription(supabase, club_id, player_id_int)
    if existing is not None:
        existing_status = str(existing.get("request_status") or "").strip().lower()
        if existing_status == REQUEST_STATUS_ACTIVE:
            raise ValueError("This player already has an active verified subscriber.")
        raise ValueError("A verified updates request is already pending for this player.")

    payload = {
        "club_id": club_id,
        "player_id": player_id_int,
        "email": email_raw,
        "email_normalized": normalized,
        "request_status": REQUEST_STATUS_PENDING,
        "request_note": str(request_note or "").strip() or None,
        "preferences_json": preferences_json or dict(DEFAULT_PREFERENCES),
    }
    try:
        resp = (
            supabase.table("player_profile_update_subscriptions")
            .insert(payload)
            .execute()
        )
    except Exception as exc:
        if _is_unique_violation(exc):
            raise ValueError("A verified request or active subscriber already exists for this player.") from exc
        raise
    row = _safe_first(resp)
    if row is None:
        raise RuntimeError("Failed to create subscription request")
    return row


def get_open_or_active_subscription(supabase, club_id: str, player_id: int) -> dict[str, Any] | None:
    club_id = _require_nonempty(club_id, "club_id")
    player_id_int = _safe_int(player_id)
    resp = (
        supabase.table("player_profile_update_subscriptions")
        .select("*")
        .eq("club_id", club_id)
        .eq("player_id", player_id_int)
        .in_("request_status", [REQUEST_STATUS_PENDING, REQUEST_STATUS_ACTIVE])
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return _safe_first(resp)


def list_pending_requests(
    supabase,
    club_id: str,
    *,
    limit: int = 200,
    offset: int = 0,
) -> list[dict[str, Any]]:
    club_id = _require_nonempty(club_id, "club_id")
    upper = max(0, int(limit) - 1)
    start = max(0, int(offset))
    end = start + upper
    resp = (
        supabase.table("player_profile_update_subscriptions")
        .select("*")
        .eq("club_id", club_id)
        .eq("request_status", REQUEST_STATUS_PENDING)
        .order("created_at", desc=True)
        .range(start, end)
        .execute()
    )
    return _safe_data(resp)


def list_active_subscriptions(
    supabase,
    club_id: str,
    *,
    limit: int = 200,
    offset: int = 0,
) -> list[dict[str, Any]]:
    club_id = _require_nonempty(club_id, "club_id")
    upper = max(0, int(limit) - 1)
    start = max(0, int(offset))
    end = start + upper
    resp = (
        supabase.table("player_profile_update_subscriptions")
        .select("*")
        .eq("club_id", club_id)
        .eq("request_status", REQUEST_STATUS_ACTIVE)
        .order("verified_at", desc=True)
        .range(start, end)
        .execute()
    )
    return _safe_data(resp)


def reject_request(
    supabase,
    subscription_id: str,
    admin_note: str | None,
    verified_by: str,
) -> dict[str, Any]:
    subscription_id = _require_nonempty(subscription_id, "subscription_id")
    verified_by = _require_nonempty(verified_by, "verified_by")

    row = _safe_first(
        supabase.table("player_profile_update_subscriptions")
        .select("*")
        .eq("id", subscription_id)
        .limit(1)
        .execute()
    )
    if row is None:
        raise ValueError("Subscription not found")
    if row.get("request_status") != REQUEST_STATUS_PENDING:
        raise ValueError("Only pending requests can be rejected")

    payload = {
        "request_status": REQUEST_STATUS_REJECTED,
        "admin_note": str(admin_note or "").strip() or None,
        "verified_by": verified_by,
        "verified_at": _now_iso(),
    }
    updated = _safe_first(
        supabase.table("player_profile_update_subscriptions")
        .update(payload)
        .eq("id", subscription_id)
        .eq("request_status", REQUEST_STATUS_PENDING)
        .execute()
    )
    if updated is None:
        raise RuntimeError("Request could not be rejected")
    return updated


def approve_request(
    supabase,
    subscription_id: str,
    verified_by: str,
    admin_note: str | None = None,
) -> dict[str, Any]:
    subscription_id = _require_nonempty(subscription_id, "subscription_id")
    verified_by = _require_nonempty(verified_by, "verified_by")

    row = _safe_first(
        supabase.table("player_profile_update_subscriptions")
        .select("*")
        .eq("id", subscription_id)
        .limit(1)
        .execute()
    )
    if row is None:
        raise ValueError("Subscription not found")
    if row.get("request_status") != REQUEST_STATUS_PENDING:
        raise ValueError("approve_request only activates pending requests")

    conflict = get_open_or_active_subscription(supabase, str(row.get("club_id") or ""), int(row.get("player_id")))
    if conflict is not None and str(conflict.get("id") or "") != subscription_id:
        conflict_status = str(conflict.get("request_status") or "").strip().lower()
        if conflict_status == REQUEST_STATUS_ACTIVE:
            raise ValueError("Cannot approve: this player already has an active verified subscriber.")
        raise ValueError("Cannot approve: another pending request exists for this player.")

    payload = {
        "request_status": REQUEST_STATUS_ACTIVE,
        "admin_note": str(admin_note or "").strip() or None,
        "verified_by": verified_by,
        "verified_at": _now_iso(),
        "unsubscribed_at": None,
    }
    try:
        updated = _safe_first(
            supabase.table("player_profile_update_subscriptions")
            .update(payload)
            .eq("id", subscription_id)
            .eq("request_status", REQUEST_STATUS_PENDING)
            .execute()
        )
    except Exception as exc:
        if _is_unique_violation(exc):
            raise ValueError("Cannot approve: this player already has a pending or active verified subscriber.") from exc
        raise
    if updated is None:
        raise RuntimeError("Request could not be approved")
    return updated


def replace_verified_subscriber(
    supabase,
    old_subscription_id: str,
    new_email: str,
    new_request_note: str | None,
    verified_by: str,
    admin_note: str | None = None,
) -> dict[str, Any]:
    old_subscription_id = _require_nonempty(old_subscription_id, "old_subscription_id")
    verified_by = _require_nonempty(verified_by, "verified_by")
    new_email_raw = _require_nonempty(new_email, "new_email")

    current = _safe_first(
        supabase.table("player_profile_update_subscriptions")
        .select("*")
        .eq("id", old_subscription_id)
        .limit(1)
        .execute()
    )
    if current is None:
        raise ValueError("Old subscription not found")
    if current.get("request_status") != REQUEST_STATUS_ACTIVE:
        raise ValueError("Only active subscriptions can be replaced")

    now_iso = _now_iso()
    unsub_payload = {
        "request_status": REQUEST_STATUS_UNSUBSCRIBED,
        "unsubscribed_at": now_iso,
        "admin_note": str(admin_note or "").strip() or current.get("admin_note"),
    }
    unsubscribed = _safe_first(
        supabase.table("player_profile_update_subscriptions")
        .update(unsub_payload)
        .eq("id", old_subscription_id)
        .eq("request_status", REQUEST_STATUS_ACTIVE)
        .execute()
    )
    if unsubscribed is None:
        raise RuntimeError("Unable to unsubscribe the prior active subscription")

    new_row_payload = {
        "club_id": current.get("club_id"),
        "player_id": current.get("player_id"),
        "email": new_email_raw,
        "email_normalized": normalize_email(new_email_raw),
        "request_status": REQUEST_STATUS_ACTIVE,
        "request_note": str(new_request_note or "").strip() or None,
        "admin_note": str(admin_note or "").strip() or None,
        "verified_by": verified_by,
        "verified_at": now_iso,
        "preferences_json": current.get("preferences_json") or dict(DEFAULT_PREFERENCES),
    }
    try:
        inserted = _safe_first(
            supabase.table("player_profile_update_subscriptions")
            .insert(new_row_payload)
            .execute()
        )
    except Exception as exc:
        if _is_unique_violation(exc):
            raise ValueError("Cannot replace subscriber because another pending/active row exists for this player.") from exc
        raise
    if inserted is None:
        raise RuntimeError("Failed to create replacement active subscription")
    return inserted


def mark_unsubscribed(supabase, subscription_id: str) -> dict[str, Any]:
    subscription_id = _require_nonempty(subscription_id, "subscription_id")

    payload = {
        "request_status": REQUEST_STATUS_UNSUBSCRIBED,
        "unsubscribed_at": _now_iso(),
    }
    updated = _safe_first(
        supabase.table("player_profile_update_subscriptions")
        .update(payload)
        .eq("id", subscription_id)
        .in_("request_status", [REQUEST_STATUS_PENDING, REQUEST_STATUS_ACTIVE])
        .execute()
    )
    if updated is None:
        raise RuntimeError("Subscription could not be marked unsubscribed")
    return updated


def save_digest(
    supabase,
    *,
    club_id: str,
    player_id: int,
    week_start: date,
    week_end: date,
    generated_json: dict[str, Any] | None = None,
    final_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    club_id = _require_nonempty(club_id, "club_id")
    player_id_int = _safe_int(player_id)

    payload = {
        "club_id": club_id,
        "player_id": player_id_int,
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "generated_json": generated_json or {},
        "final_json": final_json or {},
    }
    upserted = _safe_first(
        supabase.table("player_weekly_profile_digests")
        .upsert(payload, on_conflict="club_id,player_id,week_start,week_end")
        .execute()
    )
    if upserted is None:
        raise RuntimeError("Digest could not be saved")
    return upserted


def list_digests_for_range(
    supabase,
    club_id: str,
    *,
    week_start_from: date,
    week_start_to: date,
    player_id: int | None = None,
) -> list[dict[str, Any]]:
    club_id = _require_nonempty(club_id, "club_id")

    query = (
        supabase.table("player_weekly_profile_digests")
        .select("*")
        .eq("club_id", club_id)
        .gte("week_start", week_start_from.isoformat())
        .lte("week_start", week_start_to.isoformat())
        .order("week_start", desc=True)
    )
    if player_id is not None:
        query = query.eq("player_id", _safe_int(player_id))
    return _safe_data(query.execute())


def list_recent_digests(
    supabase,
    club_id: str,
    *,
    limit: int = 200,
    offset: int = 0,
    player_id: int | None = None,
) -> list[dict[str, Any]]:
    club_id = _require_nonempty(club_id, "club_id")
    upper = max(0, int(limit) - 1)
    start = max(0, int(offset))
    end = start + upper

    query = (
        supabase.table("player_weekly_profile_digests")
        .select("*")
        .eq("club_id", club_id)
        .order("updated_at", desc=True)
        .range(start, end)
    )
    if player_id is not None:
        query = query.eq("player_id", _safe_int(player_id))
    return _safe_data(query.execute())


def create_outbox_row(
    supabase,
    *,
    subscription_id: str,
    club_id: str,
    player_id: int,
    week_start: date,
    week_end: date,
    email: str,
) -> dict[str, Any]:
    subscription_id = _require_nonempty(subscription_id, "subscription_id")
    club_id = _require_nonempty(club_id, "club_id")
    email_raw = _require_nonempty(email, "email")

    payload = {
        "subscription_id": subscription_id,
        "club_id": club_id,
        "player_id": _safe_int(player_id),
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "email": email_raw,
        "send_status": SEND_STATUS_PENDING,
    }
    try:
        row = _safe_first(
            supabase.table("player_profile_update_outbox")
            .insert(payload)
            .execute()
        )
    except Exception as exc:
        if _is_unique_violation(exc):
            raise ValueError("An outbox row already exists for this subscriber and date window.") from exc
        raise
    if row is None:
        raise RuntimeError("Outbox row could not be created")
    return row


def queue_player_updates_for_affected_subscribers(
    supabase,
    *,
    club_id: str,
    affected_player_ids: list[int] | set[int] | tuple[int, ...],
    match_dates: list[date | datetime | str] | tuple[date | datetime | str, ...] | None = None,
) -> dict[str, int]:
    club_id = _require_nonempty(club_id, "club_id")
    unique_player_ids: list[int] = sorted({_safe_int(pid) for pid in (affected_player_ids or [])})
    summary = {
        "affected_players": len(unique_player_ids),
        "active_subscriptions": 0,
        "week_windows": 0,
        "queued": 0,
        "already_queued": 0,
        "no_active_subscription": 0,
        "failed": 0,
    }
    if not unique_player_ids:
        return summary

    digest_windows: set[tuple[date, date]] = set()
    for raw_date in match_dates or []:
        parsed = _coerce_date(raw_date)
        if parsed is None:
            continue
        digest_windows.add(_digest_window_for_match_day(parsed))
    if not digest_windows:
        digest_windows.add(_digest_window_for_match_day(datetime.now(timezone.utc).date()))

    summary["week_windows"] = len(digest_windows)

    active_subs = list_active_subscriptions(supabase, club_id, limit=max(500, len(unique_player_ids) * 5))
    active_by_player: dict[int, dict[str, Any]] = {}
    for row in active_subs:
        pid_raw = row.get("player_id")
        try:
            pid = _safe_int(pid_raw)
        except Exception:
            continue
        if pid not in unique_player_ids or pid in active_by_player:
            continue
        active_by_player[pid] = row
    summary["active_subscriptions"] = len(active_by_player)
    summary["no_active_subscription"] = max(0, len(unique_player_ids) - len(active_by_player))

    for pid in unique_player_ids:
        subscription = active_by_player.get(pid)
        if subscription is None:
            continue
        subscription_id = str(subscription.get("id") or "").strip()
        email = str(subscription.get("email") or "").strip()
        if not subscription_id or not email:
            summary["failed"] += len(digest_windows)
            continue
        for week_start, week_end in sorted(digest_windows):
            try:
                create_outbox_row(
                    supabase,
                    subscription_id=subscription_id,
                    club_id=club_id,
                    player_id=pid,
                    week_start=week_start,
                    week_end=week_end,
                    email=email,
                )
                summary["queued"] += 1
            except Exception as exc:
                if isinstance(exc, ValueError) and "already exists" in str(exc).lower():
                    summary["already_queued"] += 1
                    continue
                summary["failed"] += 1
    return summary


def delete_pending_outbox_row(
    supabase,
    club_id: str,
    outbox_id: str,
) -> dict[str, Any]:
    club_id = _require_nonempty(club_id, "club_id")
    outbox_id = _require_nonempty(outbox_id, "outbox_id")

    deleted = _safe_data(
        supabase.table("player_profile_update_outbox")
        .delete()
        .eq("club_id", club_id)
        .eq("id", outbox_id)
        .eq("send_status", SEND_STATUS_PENDING)
        .execute()
    )
    if not deleted:
        raise ValueError("Only pending queued digests can be deleted.")
    return deleted[0]


def bulk_delete_pending_outbox_rows(
    supabase,
    *,
    club_id: str,
    outbox_ids: list[str],
) -> dict[str, int]:
    club_id = _require_nonempty(club_id, "club_id")
    normalized_ids = []
    for raw in outbox_ids or []:
        outbox_id = str(raw or "").strip()
        if outbox_id:
            normalized_ids.append(outbox_id)
    if not normalized_ids:
        raise ValueError("At least one outbox_id is required")

    unique_ids = list(dict.fromkeys(normalized_ids))
    requested = len(unique_ids)

    matched_rows = _safe_data(
        supabase.table("player_profile_update_outbox")
        .select("id")
        .eq("club_id", club_id)
        .eq("send_status", SEND_STATUS_PENDING)
        .in_("id", unique_ids)
        .execute()
    )
    matched_ids = [str(row.get("id") or "").strip() for row in matched_rows if str(row.get("id") or "").strip()]
    matched_pending = len(matched_ids)
    if not matched_ids:
        return {
            "requested": requested,
            "matched_pending": 0,
            "deleted": 0,
            "skipped": requested,
        }

    deleted_rows = _safe_data(
        supabase.table("player_profile_update_outbox")
        .delete()
        .eq("club_id", club_id)
        .eq("send_status", SEND_STATUS_PENDING)
        .in_("id", matched_ids)
        .execute()
    )
    deleted = len(deleted_rows)
    skipped = max(0, requested - deleted)
    return {
        "requested": requested,
        "matched_pending": matched_pending,
        "deleted": deleted,
        "skipped": skipped,
    }


def list_outbox_rows(
    supabase,
    club_id: str,
    *,
    status: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> list[dict[str, Any]]:
    club_id = _require_nonempty(club_id, "club_id")

    upper = max(0, int(limit) - 1)
    start = max(0, int(offset))
    end = start + upper
    query = (
        supabase.table("player_profile_update_outbox")
        .select("*")
        .eq("club_id", club_id)
        .order("created_at", desc=True)
        .range(start, end)
    )
    if status:
        normalized_status = str(status).strip().lower()
        if normalized_status not in {
            SEND_STATUS_PENDING,
            SEND_STATUS_SENT,
            SEND_STATUS_SKIPPED,
            SEND_STATUS_ERROR,
        }:
            raise ValueError("Invalid outbox status")
        query = query.eq("send_status", normalized_status)
    return _safe_data(query.execute())


def update_outbox_status(
    supabase,
    outbox_id: str,
    *,
    send_status: str,
    provider_message_id: str | None = None,
    error_text: str | None = None,
    sent_at: datetime | None = None,
) -> dict[str, Any]:
    outbox_id = _require_nonempty(outbox_id, "outbox_id")
    normalized_status = str(send_status or "").strip().lower()
    if normalized_status not in {SEND_STATUS_PENDING, SEND_STATUS_SENT, SEND_STATUS_SKIPPED, SEND_STATUS_ERROR}:
        raise ValueError("Invalid send_status")

    payload: dict[str, Any] = {
        "send_status": normalized_status,
        "provider_message_id": str(provider_message_id or "").strip() or None,
        "error_text": str(error_text or "").strip() or None,
    }
    if sent_at is not None:
        payload["sent_at"] = sent_at.astimezone(timezone.utc).isoformat()
    elif normalized_status == SEND_STATUS_SENT:
        payload["sent_at"] = _now_iso()

    updated = _safe_first(
        supabase.table("player_profile_update_outbox")
        .update(payload)
        .eq("id", outbox_id)
        .execute()
    )
    if updated is None:
        raise RuntimeError("Outbox row could not be updated")
    return updated


def reset_outbox_rows_to_pending(
    supabase,
    *,
    club_id: str,
    week_start_from: date | None = None,
    week_start_to: date | None = None,
    only_status: str = SEND_STATUS_ERROR,
) -> dict[str, int]:
    club_id = _require_nonempty(club_id, "club_id")
    normalized_status = str(only_status or "").strip().lower()
    if normalized_status not in {SEND_STATUS_PENDING, SEND_STATUS_SENT, SEND_STATUS_SKIPPED, SEND_STATUS_ERROR}:
        raise ValueError("Invalid only_status")

    query = (
        supabase.table("player_profile_update_outbox")
        .select("id")
        .eq("club_id", club_id)
        .eq("send_status", normalized_status)
    )
    if week_start_from is not None:
        query = query.gte("week_start", week_start_from.isoformat())
    if week_start_to is not None:
        query = query.lte("week_start", week_start_to.isoformat())

    rows = _safe_data(query.execute())
    if not rows:
        return {"matched": 0, "reset_to_pending": 0, "failed": 0}

    row_ids = [str(row.get("id") or "").strip() for row in rows if str(row.get("id") or "").strip()]
    if not row_ids:
        return {"matched": len(rows), "reset_to_pending": 0, "failed": 0}

    payload = {
        "send_status": SEND_STATUS_PENDING,
        "error_text": None,
        "provider_message_id": None,
        "sent_at": None,
    }
    updated_rows = _safe_data(
        supabase.table("player_profile_update_outbox")
        .update(payload)
        .eq("club_id", club_id)
        .in_("id", row_ids)
        .execute()
    )
    reset_count = len(updated_rows)
    failed_count = max(0, len(row_ids) - reset_count)
    return {"matched": len(row_ids), "reset_to_pending": reset_count, "failed": failed_count}
