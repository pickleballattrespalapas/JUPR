from __future__ import annotations

from typing import Any

from jupr_app.domain.notifications.player_profile_update_repo import (
    REQUEST_STATUS_ACTIVE,
    REQUEST_STATUS_PENDING,
    REQUEST_STATUS_REJECTED,
    REQUEST_STATUS_UNSUBSCRIBED,
    get_subscription_for_unsubscribe,
    unsubscribe_via_public_link,
)

SUPPORTED_EMAIL_PREFERENCE_SCOPES = {"player_updates", "global"}


def _clean_text(value: Any, *, limit: int = 240) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _mask_email(value: Any) -> str:
    email = _clean_text(value, limit=320)
    if "@" not in email:
        return ""
    local, domain = email.split("@", 1)
    if len(local) <= 2:
        masked_local = local[:1] + "*"
    else:
        masked_local = local[:1] + "*" * min(6, max(1, len(local) - 2)) + local[-1:]
    return f"{masked_local}@{domain}"


def _subscription_payload(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "id": _clean_text(row.get("id"), limit=120),
        "club_id": _clean_text(row.get("club_id"), limit=120),
        "player_id": row.get("player_id"),
        "email_masked": _mask_email(row.get("email")),
        "request_status": _clean_text(row.get("request_status"), limit=80),
        "preferences_json": row.get("preferences_json") if isinstance(row.get("preferences_json"), dict) else {},
        "verified_at": row.get("verified_at"),
        "unsubscribed_at": row.get("unsubscribed_at"),
    }


def _normalize_identifiers(*, token: str | None = None, ut: str | None = None, sid: str | None = None, subscription_id: str | None = None) -> tuple[str, str]:
    clean_token = _clean_text(token or ut, limit=500)
    clean_sid = _clean_text(sid or subscription_id, limit=160)
    return clean_token, clean_sid


def build_public_email_preferences(
    supabase: Any,
    *,
    token: str | None = None,
    ut: str | None = None,
    sid: str | None = None,
    subscription_id: str | None = None,
) -> dict[str, Any]:
    clean_token, clean_sid = _normalize_identifiers(token=token, ut=ut, sid=sid, subscription_id=subscription_id)
    if clean_sid and not clean_token:
        raise ValueError(
            "Legacy subscription-id preference links are no longer accepted. Use the tokenized link from a recent player update email."
        )
    if not clean_token and not clean_sid:
        return {
            "ok": True,
            "mode": "email_preferences",
            "found": False,
            "subscription": None,
            "status_options": [REQUEST_STATUS_PENDING, REQUEST_STATUS_ACTIVE, REQUEST_STATUS_REJECTED, REQUEST_STATUS_UNSUBSCRIBED],
            "scope_options": sorted(SUPPORTED_EMAIL_PREFERENCE_SCOPES),
            "message": "Use the unsubscribe or preference link from a player update email.",
        }
    row = get_subscription_for_unsubscribe(
        supabase,
        unsubscribe_token=clean_token or None,
    )
    return {
        "ok": True,
        "mode": "email_preferences",
        "found": bool(row),
        "subscription": _subscription_payload(row),
        "status_options": [REQUEST_STATUS_PENDING, REQUEST_STATUS_ACTIVE, REQUEST_STATUS_REJECTED, REQUEST_STATUS_UNSUBSCRIBED],
        "scope_options": sorted(SUPPORTED_EMAIL_PREFERENCE_SCOPES),
        "message": "Subscription found." if row else "Subscription not found for this link.",
    }


def apply_public_email_unsubscribe(
    supabase: Any,
    *,
    token: str | None = None,
    ut: str | None = None,
    sid: str | None = None,
    subscription_id: str | None = None,
    scope: str = "player_updates",
) -> dict[str, Any]:
    clean_token, clean_sid = _normalize_identifiers(token=token, ut=ut, sid=sid, subscription_id=subscription_id)
    if clean_sid and not clean_token:
        raise ValueError(
            "Legacy subscription-id preference links are no longer accepted. Use the tokenized link from a recent player update email."
        )
    if not clean_token:
        raise ValueError("An unsubscribe token is required.")
    clean_scope = _clean_text(scope, limit=80).lower() or "player_updates"
    if clean_scope not in SUPPORTED_EMAIL_PREFERENCE_SCOPES:
        raise ValueError("Unsupported email preference scope.")
    row, changed, effective_scope = unsubscribe_via_public_link(
        supabase,
        unsubscribe_token=clean_token or None,
        preference_scope=clean_scope,
    )
    return {
        "ok": True,
        "mode": "email_unsubscribe",
        "scope": clean_scope,
        "effective_scope": effective_scope,
        "changed": changed,
        "already_unsubscribed": not changed,
        "subscription": _subscription_payload(row),
        "message": (
            "You were already unsubscribed; no additional change was needed."
            if not changed
            else "You are unsubscribed from player update emails."
            if effective_scope == "player_updates"
            else "You are unsubscribed from all optional JUPR email categories managed by this preference system."
        ),
    }
