from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import os
import re
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

REQUEST_TYPES = {"data_correction", "profile_privacy", "general_support"}
REQUEST_STATUS_NEW = "new"
SUPPORT_RATE_WINDOW = timedelta(hours=1)
SUPPORT_DEDUPE_WINDOW = timedelta(hours=24)
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


class SupportIntakeRateLimitError(RuntimeError):
    """Raised when a public requester exceeds the durable intake limit."""


def _safe_text(value: Any, *, limit: int = 1000) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _clean_email(value: Any) -> str:
    email = _safe_text(value, limit=240).lower()
    if not email or not EMAIL_PATTERN.fullmatch(email):
        raise ValueError("A valid email is required.")
    return email


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
        return int(numeric) if numeric.is_integer() else None
    except Exception:
        return None


def _parse_iso(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_evidence_url(value: Any) -> str | None:
    raw = _safe_text(value, limit=600)
    if not raw:
        return None
    if any(ord(character) < 32 for character in raw):
        raise ValueError("Evidence links cannot contain control characters.")
    parsed = urlparse(raw)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Evidence links must use a complete http or https URL.")
    if parsed.username or parsed.password:
        raise ValueError("Evidence links cannot include embedded credentials.")
    return raw


def _normalized_fingerprint_part(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _request_fingerprint(*, club_id: str, request_type: str, payload: dict[str, Any]) -> str:
    parts = [
        club_id,
        request_type,
        payload.get("requester_email"),
        payload.get("player_id"),
        payload.get("player_name"),
        payload.get("match_id"),
        payload.get("tournament_id"),
        payload.get("subject"),
        payload.get("description"),
        payload.get("requested_action"),
        payload.get("evidence_url"),
    ]
    canonical = "\x1f".join(_normalized_fingerprint_part(part) for part in parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _rate_limit_per_hour() -> int:
    try:
        configured = int(os.getenv("JUPR_PUBLIC_SUPPORT_RATE_LIMIT_PER_HOUR", "5"))
    except ValueError:
        configured = 5
    return max(1, min(configured, 20))


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _safe_first(resp: Any) -> dict[str, Any] | None:
    rows = _safe_rows(resp)
    return rows[0] if rows else None


def _public_request_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "request_type": _safe_text(row.get("request_type"), limit=60),
        "status": _safe_text(row.get("status") or REQUEST_STATUS_NEW, limit=40),
        "created_at": row.get("created_at"),
    }


def _validate_player_scope(supabase: Any, *, club_id: str, player_id: int | None) -> int | None:
    if player_id is None:
        return None
    player = _safe_first(
        supabase.table("players")
        .select("id,club_id,name")
        .eq("club_id", str(club_id))
        .eq("id", int(player_id))
        .limit(1)
        .execute()
    )
    if not player:
        raise ValueError("The selected player does not belong to this club.")
    return int(player.get("id"))


def _recent_request_rows(supabase: Any, *, club_id: str, requester_email: str) -> list[dict[str, Any]]:
    try:
        return _safe_rows(
            supabase.table("public_support_requests")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("requester_email", requester_email)
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Support request anti-abuse checks are unavailable.") from exc


def create_public_support_intake_request(
    supabase: Any,
    *,
    club_id: str,
    club_slug: str,
    payload: dict[str, Any],
    source: str = "next_public_support_intake",
) -> dict[str, Any]:
    """Create a public data-correction/profile-privacy/support intake row.

    This intentionally creates a staff-review queue item only. It does not mutate
    player, match, rating, badge, registration, tournament, or privacy-display data.
    """

    if _safe_text(payload.get("website"), limit=200):
        return {
            "ok": True,
            "mode": "support_intake",
            "accepted": True,
            "message": "Request received.",
        }

    request_type = _safe_text(payload.get("request_type") or "general_support", limit=60).lower()
    if request_type not in REQUEST_TYPES:
        raise ValueError("Unsupported request type.")

    requester_name = _safe_text(payload.get("requester_name"), limit=160)
    requester_email = _clean_email(payload.get("requester_email"))
    if not requester_name:
        raise ValueError("Your name is required.")
    if not _safe_bool(payload.get("consent_to_contact")):
        raise ValueError("Consent to contact is required so staff can follow up on the request.")

    subject = _safe_text(payload.get("subject"), limit=240)
    description = _safe_text(payload.get("description"), limit=2400)
    requested_action = _safe_text(payload.get("requested_action"), limit=1200)
    if not subject:
        raise ValueError("A short subject is required.")
    if not description:
        raise ValueError("Request details are required.")

    raw_player_id = payload.get("player_id")
    player_id = _safe_int(raw_player_id)
    if raw_player_id not in (None, "") and player_id is None:
        raise ValueError("Player ID must be a number.")
    player_id = _validate_player_scope(supabase, club_id=str(club_id), player_id=player_id)
    evidence_url = _safe_evidence_url(payload.get("evidence_url"))

    normalized_for_fingerprint = {
        **payload,
        "requester_email": requester_email,
        "player_id": player_id,
        "evidence_url": evidence_url,
    }
    fingerprint = _request_fingerprint(
        club_id=str(club_id),
        request_type=request_type,
        payload=normalized_for_fingerprint,
    )
    now_dt = datetime.now(timezone.utc)
    dedupe_key = f"{fingerprint}:{now_dt.strftime('%Y%m%d')}"
    recent_rows = _recent_request_rows(
        supabase,
        club_id=str(club_id),
        requester_email=requester_email,
    )
    for existing in recent_rows:
        created_at = _parse_iso(existing.get("created_at"))
        if not created_at or now_dt - created_at > SUPPORT_DEDUPE_WINDOW:
            continue
        existing_fingerprint = _safe_text(existing.get("request_fingerprint"), limit=80)
        if existing_fingerprint and existing_fingerprint == fingerprint:
            return {
                "ok": True,
                "mode": "support_intake",
                "accepted": True,
                "deduplicated": True,
                "request": _public_request_payload(existing),
                "message": "This request was already received and remains in the staff review queue.",
            }
    rate_window_rows = [
        row
        for row in recent_rows
        if (created_at := _parse_iso(row.get("created_at"))) is not None
        and now_dt - created_at <= SUPPORT_RATE_WINDOW
    ]
    if len(rate_window_rows) >= _rate_limit_per_hour():
        raise SupportIntakeRateLimitError("Too many support requests were submitted. Please try again later.")

    now = now_dt.isoformat()
    row_id = f"req_{uuid4().hex[:20]}"
    insert_payload = {
        "id": row_id,
        "club_id": str(club_id),
        "club_slug": _safe_text(club_slug, limit=120),
        "request_type": request_type,
        "status": REQUEST_STATUS_NEW,
        "requester_name": requester_name,
        "requester_email": requester_email,
        "player_name": _safe_text(payload.get("player_name"), limit=160) or None,
        "player_id": player_id,
        "match_id": _safe_text(payload.get("match_id"), limit=120) or None,
        "tournament_id": _safe_text(payload.get("tournament_id"), limit=120) or None,
        "subject": subject,
        "description": description,
        "requested_action": requested_action or None,
        "evidence_url": evidence_url,
        "consent_to_contact": True,
        "source": _safe_text(payload.get("source") or source, limit=120),
        "request_fingerprint": fingerprint,
        "request_dedupe_key": dedupe_key,
        "identity_status": "pending" if request_type == "profile_privacy" else "not_required",
        "fulfillment_status": "pending" if request_type == "profile_privacy" else "not_required",
        "created_at": now,
        "updated_at": now,
    }

    try:
        created = _safe_first(supabase.table("public_support_requests").insert(insert_payload).execute())
    except Exception as exc:
        # The unique daily key closes the concurrent exact-retry race. Return the
        # already-created queue item without leaking database error details.
        try:
            created = _safe_first(
                supabase.table("public_support_requests")
                .select("*")
                .eq("club_id", str(club_id))
                .eq("request_dedupe_key", dedupe_key)
                .limit(1)
                .execute()
            )
        except Exception as lookup_exc:
            raise RuntimeError("Support request could not be created.") from lookup_exc
        if created:
            return {
                "ok": True,
                "mode": "support_intake",
                "accepted": True,
                "deduplicated": True,
                "request": _public_request_payload(created),
                "message": "This request was already received and remains in the staff review queue.",
            }
        raise RuntimeError("Support request could not be created.") from exc
    if not created:
        raise RuntimeError("Support request could not be created.")

    return {
        "ok": True,
        "mode": "support_intake",
        "accepted": True,
        "deduplicated": False,
        "request": _public_request_payload(created),
        "message": "Request received. Staff will review it before any data changes are made.",
    }
