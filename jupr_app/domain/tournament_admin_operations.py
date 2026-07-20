from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_tournament_admin_fingerprint(value: Any) -> str:
    """Return the canonical fingerprint used by Tournament Admin retries."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_tournament_admin_operation_request(
    *,
    club_id: str,
    surface: str,
    action: str,
    entity_type: str,
    entity_id: str,
    lock_scope: str | None = None,
    expected_state: str,
    payload: dict[str, Any],
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Build one deterministic, server-owned mutation identity.

    Confirmation text deliberately does not participate. When supplied, the
    browser-retained idempotency UUID does participate so an exact retry maps to
    one request while UUID reuse with changed state or payload is rejected.
    """

    request = {
        "club_id": str(club_id or "").strip(),
        "surface": str(surface or "").strip(),
        "action": str(action or "").strip(),
        "entity_type": str(entity_type or "").strip(),
        "entity_id": str(entity_id or "").strip(),
        "lock_scope": str(lock_scope or entity_id or "").strip(),
        "expected_state": str(expected_state or "").strip(),
        "payload": dict(payload or {}),
    }
    clean_idempotency_key = str(idempotency_key or "").strip()
    if clean_idempotency_key:
        request["idempotency_key"] = clean_idempotency_key
    fingerprint = stable_tournament_admin_fingerprint(request)
    operation_key = stable_tournament_admin_fingerprint(
        {"contract": "jupr:tournament-admin:v1", "request_fingerprint": fingerprint}
    )
    return {
        **request,
        "request_fingerprint": fingerprint,
        "operation_key": operation_key,
    }
