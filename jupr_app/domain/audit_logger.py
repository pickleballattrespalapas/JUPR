from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from jupr_app.data.sb_write import sb_insert


def log_event(
    *,
    supabase: Any,
    club_id: str,
    actor: str,
    action_type: str,
    payload: Mapping[str, Any] | None = None,
) -> None:
    """Best-effort structured audit logging.

    Audit failures must never block the primary write operation.
    """
    try:
        sb_insert(
            supabase,
            "admin_audit_events",
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "club_id": str(club_id or "").strip(),
                "actor": str(actor or "system"),
                "action_type": str(action_type or "unknown"),
                "payload_json": dict(payload or {}),
            },
        )
    except Exception:
        return


__all__ = ["log_event"]
