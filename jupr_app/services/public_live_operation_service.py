from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4


OPERATION_TABLE = "public_live_operations"
IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,159}$")
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


class PublicLiveConflictError(RuntimeError):
    """A stale version or reused idempotency key was rejected."""


class PublicLiveRateLimitError(RuntimeError):
    """A durable public-write rate limit was reached."""


class PublicLiveRecoveryRequiredError(RuntimeError):
    """A write outcome must be reconciled instead of blindly repeated."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_edit_token(token: str) -> str:
    return hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()


def edit_token_matches(token: str, expected_hash: str) -> bool:
    candidate = hash_edit_token(token)
    expected = str(expected_hash or "").strip().lower()
    return bool(SHA256_RE.fullmatch(expected)) and hmac.compare_digest(candidate, expected)


def normalize_idempotency_key(value: Any) -> str:
    key = str(value or "").strip()
    if not IDEMPOTENCY_RE.fullmatch(key):
        raise ValueError(
            "idempotency_key must be 8-160 characters using letters, numbers, dot, underscore, colon, or hyphen."
        )
    return key


def normalize_requester_hash(value: Any) -> str:
    requester_hash = str(value or "").strip().lower()
    if not SHA256_RE.fullmatch(requester_hash):
        raise RuntimeError("Public live requester identity could not be safely rate-limited.")
    return requester_hash


def operation_key(*, club_id: str, action: str, idempotency_key: str) -> str:
    key = normalize_idempotency_key(idempotency_key)
    return canonical_fingerprint(
        {
            "contract": "jupr:public-live:v1",
            "club_id": str(club_id or "").strip(),
            "action": str(action or "").strip(),
            "idempotency_key": key,
        }
    )


def _request_evidence(action: str, request_payload: dict[str, Any]) -> dict[str, Any]:
    """Retain recovery metadata without turning the durable ledger into a PII archive."""

    payload = dict(request_payload or {})
    if action == "create":
        names = list(payload.get("participant_names") or [])
        links = dict(payload.get("participant_player_ids") or {})
        return {
            "event_type": payload.get("event_type"),
            "live_mode": payload.get("live_mode"),
            "total_rounds": payload.get("total_rounds"),
            "court_sizes": payload.get("court_sizes") or [],
            "skill_levels": payload.get("skill_levels") or [],
            "participant_count": len(names),
            "linked_player_count": len(links),
            "host_supplied": bool(payload.get("host_name")),
        }
    if action == "substitute":
        return {
            "scope": payload.get("scope"),
            "round_number": payload.get("round_number"),
            "match_id": payload.get("match_id"),
            "original_participant_id": payload.get("original_participant_id"),
            "linked_substitute": payload.get("substitute_player_id") is not None,
        }
    return payload


def _safe_rows(response: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (response.data or [])]
    except Exception:
        return []


def _first(response: Any) -> dict[str, Any] | None:
    rows = _safe_rows(response)
    return rows[0] if rows else None


def _error_text(exc: Exception) -> str:
    parts = [str(exc)]
    for name in ("message", "details", "hint", "code"):
        value = getattr(exc, name, None)
        if value:
            parts.append(str(value))
    return " | ".join(parts).lower()


def ensure_public_live_operation_schema(supabase: Any) -> None:
    try:
        supabase.table(OPERATION_TABLE).select(
            "operation_key,club_id,session_key,action,idempotency_key,request_fingerprint,requester_hash,expected_version,status,executor_token,lease_expires_at"
        ).limit(1).execute()
    except Exception as exc:
        raise RuntimeError(
            "Public JUPR Live durability is unavailable. Apply the order-25 migration before enabling public writes."
        ) from exc


def get_public_live_operation(
    supabase: Any,
    *,
    club_id: str,
    action: str,
    idempotency_key: str,
) -> dict[str, Any] | None:
    key = operation_key(club_id=club_id, action=action, idempotency_key=idempotency_key)
    try:
        return _first(
            supabase.table(OPERATION_TABLE)
            .select("*")
            .eq("club_id", str(club_id))
            .eq("operation_key", key)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Unable to read the public JUPR Live recovery ledger.") from exc


def _limit_for(action: str) -> int:
    if action == "create":
        name, fallback, ceiling = "JUPR_PUBLIC_LIVE_CREATE_LIMIT_PER_HOUR", 8, 30
    else:
        name, fallback, ceiling = "JUPR_PUBLIC_LIVE_MUTATION_LIMIT_PER_HOUR", 120, 500
    try:
        configured = int(os.getenv(name, str(fallback)))
    except ValueError:
        configured = fallback
    return max(1, min(configured, ceiling))


def enforce_public_live_rate_limit(
    supabase: Any,
    *,
    club_id: str,
    requester_hash: str,
    action: str,
) -> None:
    safe_hash = normalize_requester_hash(requester_hash)
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    limit = _limit_for(action)
    try:
        rows = _safe_rows(
            supabase.table(OPERATION_TABLE)
            .select("operation_key,created_at")
            .eq("club_id", str(club_id))
            .eq("requester_hash", safe_hash)
            .eq("action", str(action))
            .gte("created_at", cutoff)
            .limit(limit)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Public JUPR Live anti-abuse checks are unavailable; no write was attempted.") from exc
    if len(rows) >= limit:
        raise PublicLiveRateLimitError("Too many JUPR Live requests were submitted. Please try again later.")


def begin_public_live_operation(
    supabase: Any,
    *,
    club_id: str,
    session_key: str | None,
    action: str,
    idempotency_key: str,
    requester_hash: str,
    expected_version: int | None,
    request_payload: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Persist one write intent; return `(operation, already_exists)`."""

    ensure_public_live_operation_schema(supabase)
    clean_idempotency = normalize_idempotency_key(idempotency_key)
    clean_requester = normalize_requester_hash(requester_hash)
    key = operation_key(club_id=club_id, action=action, idempotency_key=clean_idempotency)
    fingerprint = canonical_fingerprint(
        {
            "club_id": str(club_id),
            "session_key": str(session_key or ""),
            "action": str(action),
            "expected_version": expected_version,
            "payload": request_payload,
        }
    )
    existing = get_public_live_operation(
        supabase,
        club_id=str(club_id),
        action=str(action),
        idempotency_key=clean_idempotency,
    )
    if existing is not None:
        if str(existing.get("request_fingerprint") or "") != fingerprint:
            raise PublicLiveConflictError(
                "This idempotency key was already used for a different JUPR Live request."
            )
        return existing, True

    enforce_public_live_rate_limit(
        supabase,
        club_id=str(club_id),
        requester_hash=clean_requester,
        action=str(action),
    )
    now = utc_now_iso()
    payload = {
        "operation_key": key,
        "club_id": str(club_id),
        "session_key": str(session_key) if session_key else None,
        "action": str(action),
        "idempotency_key": clean_idempotency,
        "request_fingerprint": fingerprint,
        "requester_hash": clean_requester,
        "expected_version": int(expected_version) if expected_version is not None else None,
        "status": "intent",
        "request_json": _request_evidence(str(action), request_payload),
        "result_json": {},
        "created_at": now,
        "updated_at": now,
    }
    try:
        inserted = _first(supabase.table(OPERATION_TABLE).insert(payload).execute())
    except Exception as exc:
        raced = get_public_live_operation(
            supabase,
            club_id=str(club_id),
            action=str(action),
            idempotency_key=clean_idempotency,
        )
        if raced is not None and str(raced.get("request_fingerprint") or "") == fingerprint:
            return raced, True
        if "public live rate limit exceeded" in _error_text(exc):
            raise PublicLiveRateLimitError(
                "Too many JUPR Live requests were submitted. Please try again later."
            ) from exc
        raise RuntimeError("Public JUPR Live could not persist write intent; no session write was attempted.") from exc
    if inserted is None:
        raise RuntimeError("Public JUPR Live could not persist write intent; no session write was attempted.")
    return inserted, False


def update_public_live_operation(
    supabase: Any,
    *,
    club_id: str,
    operation_key_value: str,
    status: str,
    result: dict[str, Any] | None = None,
    error_text: str | None = None,
) -> dict[str, Any]:
    patch: dict[str, Any] = {
        "status": str(status),
        "error_text": str(error_text or "")[:1000] or None,
        "updated_at": utc_now_iso(),
    }
    if result is not None:
        patch["result_json"] = result
    if status in {"completed", "rejected", "recovery_required"}:
        patch["executor_token"] = None
        patch["lease_expires_at"] = None
    if status == "completed":
        patch["completed_at"] = utc_now_iso()
    try:
        updated = _first(
            supabase.table(OPERATION_TABLE)
            .update(patch)
            .eq("club_id", str(club_id))
            .eq("operation_key", str(operation_key_value))
            .execute()
        )
    except Exception as exc:
        raise PublicLiveRecoveryRequiredError(
            "JUPR Live may have changed, but its recovery record could not be updated. Reload before retrying."
        ) from exc
    if updated is None:
        raise PublicLiveRecoveryRequiredError(
            "JUPR Live may have changed, but its recovery record could not be updated. Reload before retrying."
        )
    return updated


def claim_public_live_completion_executor(
    supabase: Any,
    *,
    club_id: str,
    operation_key_value: str,
) -> str:
    """Atomically lease one executor for a multi-table completion attempt."""

    executor_token = str(uuid4())
    try:
        claimed = _first(
            supabase.rpc(
                "claim_public_live_completion_executor",
                {
                    "p_club_id": str(club_id),
                    "p_operation_key": str(operation_key_value),
                    "p_executor_token": executor_token,
                    "p_lease_seconds": 300,
                },
            ).execute()
        )
    except Exception as exc:
        raise PublicLiveRecoveryRequiredError(
            "Completion executor locking is unavailable. No Club Social submit was attempted."
        ) from exc
    if claimed is None:
        raise PublicLiveRecoveryRequiredError(
            "This completion is already being reconciled by another request. Wait for it to finish, then retry the same preserved operation if needed."
        )
    return executor_token


def completed_operation_result(operation: dict[str, Any]) -> dict[str, Any] | None:
    if str(operation.get("status") or "") != "completed":
        return None
    result = operation.get("result_json")
    if not isinstance(result, dict):
        raise PublicLiveRecoveryRequiredError(
            "The completed JUPR Live operation has no readable result. Reload the session before retrying."
        )
    return {**result, "idempotent_replay": True}
