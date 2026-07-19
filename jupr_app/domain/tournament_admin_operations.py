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
) -> dict[str, Any]:
    """Build one deterministic, server-owned mutation identity.

    Confirmation text and browser-generated identifiers deliberately do not
    participate. A retry of the same reviewed state and mutation therefore
    resolves to the same operation, while a changed payload or state cannot be
    replayed under the old key.
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
    fingerprint = stable_tournament_admin_fingerprint(request)
    operation_key = stable_tournament_admin_fingerprint(
        {"contract": "jupr:tournament-admin:v1", "request_fingerprint": fingerprint}
    )
    return {
        **request,
        "request_fingerprint": fingerprint,
        "operation_key": operation_key,
    }
