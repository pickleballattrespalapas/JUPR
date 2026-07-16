from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

REQUEST_TYPES = {"data_correction", "profile_privacy", "general_support"}
REQUEST_STATUS_NEW = "new"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_text(value: Any, *, limit: int = 1000) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _clean_email(value: Any) -> str:
    return _safe_text(value, limit=240).lower()


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


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
    if not requester_email or "@" not in requester_email:
        raise ValueError("A valid email is required.")
    if not _safe_bool(payload.get("consent_to_contact")):
        raise ValueError("Consent to contact is required so staff can follow up on the request.")

    subject = _safe_text(payload.get("subject"), limit=240)
    description = _safe_text(payload.get("description"), limit=2400)
    requested_action = _safe_text(payload.get("requested_action"), limit=1200)
    if not subject:
        raise ValueError("A short subject is required.")
    if not description:
        raise ValueError("Request details are required.")

    now = _now_iso()
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
        "player_id": _safe_int(payload.get("player_id")),
        "match_id": _safe_text(payload.get("match_id"), limit=120) or None,
        "tournament_id": _safe_text(payload.get("tournament_id"), limit=120) or None,
        "subject": subject,
        "description": description,
        "requested_action": requested_action or None,
        "evidence_url": _safe_text(payload.get("evidence_url"), limit=600) or None,
        "consent_to_contact": True,
        "source": _safe_text(payload.get("source") or source, limit=120),
        "created_at": now,
        "updated_at": now,
    }

    created = _safe_first(supabase.table("public_support_requests").insert(insert_payload).execute())
    if not created:
        raise RuntimeError("Support request could not be created.")

    return {
        "ok": True,
        "mode": "support_intake",
        "accepted": True,
        "request": _public_request_payload(created),
        "message": "Request received. Staff will review it before any data changes are made.",
    }
