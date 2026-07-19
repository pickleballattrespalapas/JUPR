from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import re
import secrets
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

REQUEST_STATUS_PENDING = "pending_admin_review"
REQUEST_STATUS_ACTIVE = "active"
REQUEST_STATUS_REJECTED = "rejected"
REQUEST_STATUS_UNSUBSCRIBED = "unsubscribed"

SEND_STATUS_PENDING = "pending"
SEND_STATUS_SENDING = "sending"
SEND_STATUS_SENT = "sent"
SEND_STATUS_SKIPPED = "skipped"
SEND_STATUS_ERROR = "error"

DEFAULT_PREFERENCES = {
    "frequency": "weekly",
    "send_only_if_changed": True,
}

PUBLIC_UNSUBSCRIBE_SCOPES = {"player_updates", "global"}
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
OUTBOX_SEND_LEASE = timedelta(minutes=30)


class StaleCommunicationsStateError(ValueError):
    """Raised when an operator acts on a row that changed after it was loaded."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_email_address(value: Any, *, field_name: str = "email") -> str:
    email = _require_nonempty(value, field_name)
    if not _EMAIL_RE.match(email):
        raise ValueError(f"{field_name} must be a valid email address")
    return email


def _utc_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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


def _safe_version(value: Any) -> int:
    try:
        version = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected_row_version must be an integer") from exc
    if version < 1:
        raise ValueError("expected_row_version must be positive")
    return version


def _is_unique_violation(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return "duplicate key" in text or "unique" in text


def _is_missing_column(exc: Exception, column_name: str) -> bool:
    text = str(exc or "").lower()
    return "column" in text and str(column_name or "").lower() in text and "does not exist" in text


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


def _new_unsubscribe_token() -> str:
    return secrets.token_urlsafe(24)


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


def get_subscription(
    supabase,
    *,
    club_id: str,
    subscription_id: str,
) -> dict[str, Any] | None:
    club_id = _require_nonempty(club_id, "club_id")
    subscription_id = _require_nonempty(subscription_id, "subscription_id")
    return _safe_first(
        supabase.table("player_profile_update_subscriptions")
        .select("*")
        .eq("club_id", club_id)
        .eq("id", subscription_id)
        .limit(1)
        .execute()
    )


def replace_verified_subscriber_atomic(
    supabase,
    *,
    club_id: str,
    old_subscription_id: str,
    new_email: str,
    new_request_note: str | None,
    verified_by: str,
    admin_note: str | None,
    expected_row_version: int,
    operation_key: str,
) -> dict[str, Any]:
    """Replace one active subscriber in a Postgres transaction.

    The RPC is service-role-only. ``operation_key`` makes a network retry return
    the already-created replacement instead of replacing it a second time.
    """

    club_id = _require_nonempty(club_id, "club_id")
    old_subscription_id = _require_nonempty(old_subscription_id, "old_subscription_id")
    verified_by = _require_nonempty(verified_by, "verified_by")
    email_raw = validate_email_address(new_email, field_name="new_email")
    operation_key = _require_nonempty(operation_key, "operation_key")
    expected_version = _safe_version(expected_row_version)
    try:
        response = supabase.rpc(
            "replace_verified_update_subscription",
            {
                "p_club_id": club_id,
                "p_old_subscription_id": old_subscription_id,
                "p_new_email": email_raw,
                "p_new_email_normalized": normalize_email(email_raw),
                "p_new_request_note": str(new_request_note or "").strip() or None,
                "p_verified_by": verified_by,
                "p_admin_note": str(admin_note or "").strip() or None,
                "p_expected_row_version": expected_version,
                "p_operation_key": operation_key,
            },
        ).execute()
    except Exception as exc:
        if "stale" in str(exc or "").lower():
            raise StaleCommunicationsStateError("Subscription changed. Reload before replacing it.") from exc
        raise
    data = getattr(response, "data", None)
    if isinstance(data, list):
        data = data[0] if data else None
    if not isinstance(data, dict):
        raise RuntimeError("Atomic replacement did not return the replacement subscription")
    return dict(data)


def mark_unsubscribed_guarded(
    supabase,
    *,
    club_id: str,
    subscription_id: str,
    expected_row_version: int,
) -> dict[str, Any]:
    club_id = _require_nonempty(club_id, "club_id")
    subscription_id = _require_nonempty(subscription_id, "subscription_id")
    expected_version = _safe_version(expected_row_version)
    updated = _safe_first(
        supabase.table("player_profile_update_subscriptions")
        .update({"request_status": REQUEST_STATUS_UNSUBSCRIBED, "unsubscribed_at": _now_iso()})
        .eq("club_id", club_id)
        .eq("id", subscription_id)
        .eq("request_status", REQUEST_STATUS_ACTIVE)
        .eq("row_version", expected_version)
        .execute()
    )
    if updated is None:
        current = get_subscription(supabase, club_id=club_id, subscription_id=subscription_id)
        if current is None:
            raise ValueError("Subscription not found")
        raise StaleCommunicationsStateError("Subscription changed. Reload before deactivating it.")
    return updated


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


def list_subscriptions_by_status(
    supabase,
    club_id: str,
    *,
    statuses: list[str] | tuple[str, ...],
    limit: int = 200,
    offset: int = 0,
) -> list[dict[str, Any]]:
    club_id = _require_nonempty(club_id, "club_id")
    normalized_statuses = [str(status or "").strip().lower() for status in statuses or [] if str(status or "").strip()]
    if not normalized_statuses:
        return []
    upper = max(0, int(limit) - 1)
    start = max(0, int(offset))
    end = start + upper
    resp = (
        supabase.table("player_profile_update_subscriptions")
        .select("*")
        .eq("club_id", club_id)
        .in_("request_status", normalized_statuses)
        .order("updated_at", desc=True)
        .range(start, end)
        .execute()
    )
    return _safe_data(resp)


def get_subscription_for_unsubscribe(
    supabase,
    *,
    unsubscribe_token: str | None = None,
    subscription_id: str | None = None,
) -> dict[str, Any] | None:
    token = str(unsubscribe_token or "").strip()
    sid = str(subscription_id or "").strip()

    if token:
        try:
            row = _safe_first(
                supabase.table("player_profile_update_subscriptions")
                .select("*")
                .eq("unsubscribe_token", token)
                .limit(1)
                .execute()
            )
            if row is not None:
                return row
        except Exception as exc:
            if not _is_missing_column(exc, "unsubscribe_token"):
                raise

    if sid:
        return _safe_first(
            supabase.table("player_profile_update_subscriptions")
            .select("*")
            .eq("id", sid)
            .limit(1)
            .execute()
        )
    return None


def ensure_unsubscribe_token(supabase, subscription_id: str) -> str | None:
    sid = _require_nonempty(subscription_id, "subscription_id")
    try:
        existing = _safe_first(
            supabase.table("player_profile_update_subscriptions")
            .select("id,unsubscribe_token")
            .eq("id", sid)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        if _is_missing_column(exc, "unsubscribe_token"):
            return None
        raise

    if existing is None:
        return None
    current = str(existing.get("unsubscribe_token") or "").strip()
    if current:
        return current

    for _ in range(3):
        token = _new_unsubscribe_token()
        try:
            updated = _safe_first(
                supabase.table("player_profile_update_subscriptions")
                .update({"unsubscribe_token": token})
                .eq("id", sid)
                .execute()
            )
            applied = str((updated or {}).get("unsubscribe_token") or "").strip()
            if applied:
                return applied
        except Exception as exc:
            if _is_missing_column(exc, "unsubscribe_token"):
                return None
            if _is_unique_violation(exc):
                continue
            raise
    return None


def unsubscribe_via_public_link(
    supabase,
    *,
    unsubscribe_token: str | None = None,
    subscription_id: str | None = None,
    preference_scope: str = "player_updates",
) -> tuple[dict[str, Any], bool, str]:
    scope = str(preference_scope or "").strip().lower() or "player_updates"
    if scope not in PUBLIC_UNSUBSCRIBE_SCOPES:
        raise ValueError("Unsupported email preference scope.")
    row = get_subscription_for_unsubscribe(
        supabase,
        unsubscribe_token=unsubscribe_token,
        subscription_id=subscription_id,
    )
    if row is None:
        raise ValueError("Subscription not found for this unsubscribe link.")

    status = str(row.get("request_status") or "").strip().lower()
    current_preferences = row.get("preferences_json") if isinstance(row.get("preferences_json"), dict) else {}
    preferences = dict(current_preferences)
    already_global = (
        preferences.get("optional_emails_enabled") is False
        or str(preferences.get("unsubscribe_scope") or "").strip().lower() == "global"
    )
    effective_scope = "global" if scope == "global" or already_global else "player_updates"
    preferences["player_updates_enabled"] = False
    preferences["unsubscribe_scope"] = effective_scope
    if effective_scope == "global":
        preferences["optional_emails_enabled"] = False

    changed = status != REQUEST_STATUS_UNSUBSCRIBED or preferences != current_preferences
    if not changed:
        return row, False, effective_scope

    payload = {
        "request_status": REQUEST_STATUS_UNSUBSCRIBED,
        "unsubscribed_at": row.get("unsubscribed_at") or _now_iso(),
        "preferences_json": preferences,
    }
    updated = _safe_first(
        supabase.table("player_profile_update_subscriptions")
        .update(payload)
        .eq("id", str(row.get("id") or ""))
        .execute()
    )
    if updated is None:
        raise RuntimeError("Subscription could not be marked unsubscribed")
    return updated, True, effective_scope


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
    operation_key: str | None = None,
    digest_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    subscription_id = _require_nonempty(subscription_id, "subscription_id")
    club_id = _require_nonempty(club_id, "club_id")
    email_raw = _require_nonempty(email, "email")

    clean_operation_key = str(operation_key or "").strip() or None
    exact_query = (
        supabase.table("player_profile_update_outbox")
        .select("*")
        .eq("club_id", club_id)
        .eq("subscription_id", subscription_id)
        .eq("week_start", week_start.isoformat())
        .eq("week_end", week_end.isoformat())
    )
    if clean_operation_key:
        existing = _safe_first(exact_query.eq("queue_operation_key", clean_operation_key).limit(1).execute())
        if existing is not None:
            return existing
    else:
        # Legacy match side effects do not supply a request key. Preserve their
        # historical one-row-per-window behavior while explicit admin queue
        # operations remain repeatable with distinct keys.
        if _safe_first(exact_query.limit(1).execute()) is not None:
            raise ValueError("An outbox row already exists for this subscriber and date window.")
        clean_operation_key = str(
            uuid5(
                NAMESPACE_URL,
                f"jupr:auto-player-update:{club_id}:{subscription_id}:{week_start.isoformat()}:{week_end.isoformat()}",
            )
        )

    payload = {
        "subscription_id": subscription_id,
        "club_id": club_id,
        "player_id": _safe_int(player_id),
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "email": email_raw,
        "send_status": SEND_STATUS_PENDING,
        "queue_operation_key": clean_operation_key,
        "digest_snapshot_json": dict(digest_snapshot or {}),
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


def delete_pending_outbox_rows_guarded(
    supabase,
    *,
    club_id: str,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    club_id = _require_nonempty(club_id, "club_id")
    normalized: list[tuple[str, int]] = []
    seen: set[str] = set()
    for item in items or []:
        outbox_id = _require_nonempty((item or {}).get("id"), "outbox id")
        if outbox_id in seen:
            continue
        seen.add(outbox_id)
        normalized.append((outbox_id, _safe_version((item or {}).get("expected_row_version"))))
    if not normalized:
        raise ValueError("At least one outbox row is required")

    deleted_rows: list[dict[str, Any]] = []
    stale_ids: list[str] = []
    for outbox_id, expected_version in normalized:
        deleted = _safe_first(
            supabase.table("player_profile_update_outbox")
            .delete()
            .eq("club_id", club_id)
            .eq("id", outbox_id)
            .eq("send_status", SEND_STATUS_PENDING)
            .eq("row_version", expected_version)
            .execute()
        )
        if deleted is None:
            stale_ids.append(outbox_id)
        else:
            deleted_rows.append(deleted)
    return {
        "requested": len(normalized),
        "deleted": len(deleted_rows),
        "stale": len(stale_ids),
        "stale_ids": stale_ids,
        "deleted_rows": deleted_rows,
    }


def list_outbox_rows(
    supabase,
    club_id: str,
    *,
    status: str | None = None,
    limit: int = 200,
    offset: int = 0,
    week_start: date | None = None,
    week_end: date | None = None,
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
            SEND_STATUS_SENDING,
            SEND_STATUS_SENT,
            SEND_STATUS_SKIPPED,
            SEND_STATUS_ERROR,
        }:
            raise ValueError("Invalid outbox status")
        query = query.eq("send_status", normalized_status)
    if week_start is not None:
        query = query.eq("week_start", week_start.isoformat())
    if week_end is not None:
        query = query.eq("week_end", week_end.isoformat())
    return _safe_data(query.execute())


def get_outbox_row(supabase, *, club_id: str, outbox_id: str) -> dict[str, Any] | None:
    club_id = _require_nonempty(club_id, "club_id")
    outbox_id = _require_nonempty(outbox_id, "outbox_id")
    return _safe_first(
        supabase.table("player_profile_update_outbox")
        .select("*")
        .eq("club_id", club_id)
        .eq("id", outbox_id)
        .limit(1)
        .execute()
    )


def claim_outbox_row_for_send(
    supabase,
    *,
    club_id: str,
    outbox_id: str,
    expected_row_version: int,
    actor_email: str,
    delivery_mode: str | None = None,
) -> dict[str, Any]:
    expected_version = _safe_version(expected_row_version)
    current = get_outbox_row(supabase, club_id=club_id, outbox_id=outbox_id)
    if current is None:
        raise ValueError("Outbox row not found")
    if str(current.get("send_status") or "") != SEND_STATUS_PENDING:
        raise StaleCommunicationsStateError("Outbox row is no longer pending. Reload the queue.")
    if _safe_version(current.get("row_version") or 1) != expected_version:
        raise StaleCommunicationsStateError("Outbox row changed. Reload the queue before sending.")
    payload = {
        "send_status": SEND_STATUS_SENDING,
        "attempt_count": int(current.get("attempt_count") or 0) + 1,
        "last_attempt_at": _now_iso(),
        "last_attempt_by": str(actor_email or "").strip() or None,
        "delivery_attempt_id": str(uuid4()),
        "delivery_mode": str(delivery_mode or "").strip() or None,
        "error_text": None,
    }
    updated = _safe_first(
        supabase.table("player_profile_update_outbox")
        .update(payload)
        .eq("club_id", str(club_id))
        .eq("id", str(outbox_id))
        .eq("send_status", SEND_STATUS_PENDING)
        .eq("row_version", expected_version)
        .execute()
    )
    if updated is None:
        raise StaleCommunicationsStateError("Outbox row changed. Reload the queue before sending.")
    return updated


def update_outbox_status(
    supabase,
    outbox_id: str,
    *,
    send_status: str,
    provider_message_id: str | None = None,
    error_text: str | None = None,
    sent_at: datetime | None = None,
    club_id: str | None = None,
    expected_row_version: int | None = None,
    expected_status: str | None = None,
    delivery_mode: str | None = None,
) -> dict[str, Any]:
    outbox_id = _require_nonempty(outbox_id, "outbox_id")
    normalized_status = str(send_status or "").strip().lower()
    if normalized_status not in {SEND_STATUS_PENDING, SEND_STATUS_SENDING, SEND_STATUS_SENT, SEND_STATUS_SKIPPED, SEND_STATUS_ERROR}:
        raise ValueError("Invalid send_status")

    payload: dict[str, Any] = {
        "send_status": normalized_status,
        "provider_message_id": str(provider_message_id or "").strip() or None,
        "error_text": str(error_text or "").strip() or None,
    }
    if delivery_mode is not None:
        payload["delivery_mode"] = str(delivery_mode or "").strip() or None
    if sent_at is not None:
        payload["sent_at"] = sent_at.astimezone(timezone.utc).isoformat()
    elif normalized_status == SEND_STATUS_SENT:
        payload["sent_at"] = _now_iso()

    query = supabase.table("player_profile_update_outbox").update(payload).eq("id", outbox_id)
    if club_id is not None:
        query = query.eq("club_id", _require_nonempty(club_id, "club_id"))
    if expected_row_version is not None:
        query = query.eq("row_version", _safe_version(expected_row_version))
    if expected_status is not None:
        query = query.eq("send_status", str(expected_status or "").strip().lower())
    updated = _safe_first(query.execute())
    if updated is None:
        if expected_row_version is not None or expected_status is not None:
            raise StaleCommunicationsStateError("Outbox row changed before delivery status could be finalized.")
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
    if normalized_status not in {SEND_STATUS_PENDING, SEND_STATUS_SENDING, SEND_STATUS_SENT, SEND_STATUS_SKIPPED, SEND_STATUS_ERROR}:
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


def retry_outbox_rows_guarded(
    supabase,
    *,
    club_id: str,
    items: list[dict[str, Any]],
    allow_uncertain: bool = False,
) -> dict[str, Any]:
    club_id = _require_nonempty(club_id, "club_id")
    reset_rows: list[dict[str, Any]] = []
    stale_ids: list[str] = []
    seen: set[str] = set()
    eligible: list[tuple[str, int, str]] = []
    for item in items or []:
        outbox_id = _require_nonempty((item or {}).get("id"), "outbox id")
        if outbox_id in seen:
            continue
        seen.add(outbox_id)
        expected_version = _safe_version((item or {}).get("expected_row_version"))
        current = get_outbox_row(supabase, club_id=club_id, outbox_id=outbox_id)
        if current is None:
            stale_ids.append(outbox_id)
            continue
        current_status = str(current.get("send_status") or "")
        if current_status not in {SEND_STATUS_ERROR, SEND_STATUS_SENDING}:
            stale_ids.append(outbox_id)
            continue
        if current_status == SEND_STATUS_SENDING:
            if not allow_uncertain:
                raise ValueError("Type RETRY UNCERTAIN EMAILS before resetting any sending row.")
            last_attempt_at = _utc_datetime(current.get("last_attempt_at"))
            if last_attempt_at is None:
                raise ValueError(
                    "A selected sending row has no claim timestamp. Reconcile it manually before retrying."
                )
            retry_after = last_attempt_at + OUTBOX_SEND_LEASE
            if datetime.now(timezone.utc) < retry_after:
                raise ValueError(
                    f"Outbox row {outbox_id} is still inside its 30-minute send lease. "
                    "Wait for the in-flight request to finish before retrying."
                )
        eligible.append((outbox_id, expected_version, current_status))

    if not seen:
        raise ValueError("At least one outbox row is required")

    # Perform no mutations until every selected row has passed the in-flight
    # lease check. That prevents a mixed selection from being partially reset.
    for outbox_id, expected_version, current_status in eligible:
        updated = _safe_first(
            supabase.table("player_profile_update_outbox")
            .update(
                {
                    "send_status": SEND_STATUS_PENDING,
                    "error_text": None,
                    "provider_message_id": None,
                    "sent_at": None,
                }
            )
            .eq("club_id", club_id)
            .eq("id", outbox_id)
            .eq("send_status", current_status)
            .eq("row_version", expected_version)
            .execute()
        )
        if updated is None:
            stale_ids.append(outbox_id)
        else:
            reset_rows.append(updated)
    return {
        "requested": len(seen),
        "reset_to_pending": len(reset_rows),
        "stale": len(stale_ids),
        "stale_ids": stale_ids,
        "rows": reset_rows,
    }


def claim_communications_admin_operation(
    supabase,
    *,
    club_id: str,
    operation_key: str,
    operation_type: str,
    request_json: dict[str, Any],
) -> dict[str, Any]:
    """Create or validate a request-level idempotency record.

    Reusing a key with changed scope is rejected before digest recomputation or
    queue writes. A completed retry returns the stored result verbatim.
    """

    club_id = _require_nonempty(club_id, "club_id")
    operation_key = _require_nonempty(operation_key, "operation_key")
    operation_type = _require_nonempty(operation_type, "operation_type")
    normalized_request = dict(request_json or {})

    def _existing() -> dict[str, Any] | None:
        return _safe_first(
            supabase.table("communications_admin_operations")
            .select("*")
            .eq("operation_key", operation_key)
            .limit(1)
            .execute()
        )

    existing = _existing()
    if existing is None:
        try:
            existing = _safe_first(
                supabase.table("communications_admin_operations")
                .insert(
                    {
                        "operation_key": operation_key,
                        "club_id": club_id,
                        "operation_type": operation_type,
                        "request_json": normalized_request,
                        "status": "started",
                    }
                )
                .execute()
            )
        except Exception as exc:
            if not _is_unique_violation(exc):
                raise
            existing = _existing()
    if existing is None:
        raise RuntimeError("Communications operation could not be claimed")
    return validate_communications_admin_operation(
        existing,
        club_id=club_id,
        operation_type=operation_type,
        request_json=normalized_request,
    )


def validate_communications_admin_operation(
    operation: dict[str, Any],
    *,
    club_id: str,
    operation_type: str,
    request_json: dict[str, Any],
) -> dict[str, Any]:
    if (
        str(operation.get("club_id") or "") != _require_nonempty(club_id, "club_id")
        or str(operation.get("operation_type") or "") != _require_nonempty(operation_type, "operation_type")
        or dict(operation.get("request_json") or {}) != dict(request_json or {})
    ):
        raise ValueError("operation_key was already used for a different communications request")
    return operation


def get_communications_admin_operation(supabase, *, operation_key: str) -> dict[str, Any] | None:
    return _safe_first(
        supabase.table("communications_admin_operations")
        .select("*")
        .eq("operation_key", _require_nonempty(operation_key, "operation_key"))
        .limit(1)
        .execute()
    )


def complete_communications_admin_operation(
    supabase,
    *,
    club_id: str,
    operation_key: str,
    result_json: dict[str, Any],
) -> dict[str, Any]:
    updated = _safe_first(
        supabase.table("communications_admin_operations")
        .update(
            {
                "status": "completed",
                "result_json": dict(result_json or {}),
                "completed_at": _now_iso(),
                "updated_at": _now_iso(),
            }
        )
        .eq("operation_key", _require_nonempty(operation_key, "operation_key"))
        .eq("club_id", _require_nonempty(club_id, "club_id"))
        .execute()
    )
    if updated is None:
        raise RuntimeError("Communications operation result could not be persisted")
    return updated
