from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.challenge_ladder import TIER_ORDER, ladder_bucket_challenge, normalize_tier_id
from jupr_app.services.public_challenge_ladder_service import build_public_challenge_ladder

TRUTHY = {"1", "true", "yes", "y", "on"}
FINAL_STATUSES = {"CANCELLED", "FORFEITED", "COMPLETED"}
CONFIRM = "SAVE LADDER"


def is_admin_challenge_ladder_enabled() -> bool:
    return os.getenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "").strip().lower() in TRUTHY


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _first(resp: Any) -> dict[str, Any] | None:
    rows = _safe_rows(resp)
    return rows[0] if rows else None


def _clean(value: Any, *, limit: int = 240) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _player_names(supabase: Any, *, club_id: str) -> dict[int, str]:
    try:
        rows = _safe_rows(supabase.table("players").select("id,name").eq("club_id", str(club_id)).execute())
    except Exception:
        rows = []
    names: dict[int, str] = {}
    for row in rows:
        pid = _safe_int(row.get("id"))
        if pid is not None:
            names[int(pid)] = _clean(row.get("name"), limit=160) or f"Player {pid}"
    return names


def _challenge_row(row: dict[str, Any], names: dict[int, str]) -> dict[str, Any]:
    challenger = _safe_int(row.get("challenger_id"))
    defender = _safe_int(row.get("defender_id"))
    return {
        "id": _safe_int(row.get("id")),
        "tier_id": normalize_tier_id(str(row.get("tier_id") or "")),
        "status": str(row.get("status") or ""),
        "bucket": ladder_bucket_challenge(row),
        "challenger_id": challenger,
        "challenger_name": names.get(int(challenger), f"Player {challenger}") if challenger is not None else "—",
        "defender_id": defender,
        "defender_name": names.get(int(defender), f"Player {defender}") if defender is not None else "—",
        "created_at": row.get("created_at"),
        "accept_by": row.get("accept_by"),
        "accepted_at": row.get("accepted_at"),
        "play_by": row.get("play_by"),
        "completed_at": row.get("completed_at"),
        "winner_id": _safe_int(row.get("winner_id")),
    }


def build_admin_challenge_ladder_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    if not is_admin_challenge_ladder_enabled():
        return {"enabled": False, "status": "guarded_off", "warnings": ["Enable JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER to use Challenge Ladder Admin in Next."]}
    summary = {"active_player_count": 0, "active_challenge_count": 0, "tier_count": len(TIER_ORDER)}
    if supabase is not None:
        try:
            summary = build_public_challenge_ladder(supabase, club_id=str(club_id)).get("summary", summary)
        except Exception:
            pass
    return {"enabled": True, "status": "ready_for_challenge_ladder_admin", "summary": summary, "warnings": []}


def get_admin_challenge_ladder_dashboard(supabase: Any, *, club_id: str) -> dict[str, Any]:
    if not is_admin_challenge_ladder_enabled():
        raise PermissionError("Next Challenge Ladder Admin is disabled.")
    public_payload = build_public_challenge_ladder(supabase, club_id=str(club_id))
    names = _player_names(supabase, club_id=str(club_id))
    try:
        challenge_rows = _safe_rows(supabase.table("ladder_challenges").select("*").eq("club_id", str(club_id)).order("created_at", desc=True).limit(500).execute())
    except Exception:
        challenge_rows = []
    bucket_counts: dict[str, int] = {}
    for row in challenge_rows:
        bucket = ladder_bucket_challenge(row)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    challenges = [_challenge_row(row, names) for row in challenge_rows[:100]]
    try:
        settings = _safe_rows(supabase.table("ladder_settings").select("*").eq("club_id", str(club_id)).limit(1).execute())
    except Exception:
        settings = []
    return {"ok": True, "mode": "challenge_ladder_admin_dashboard", **public_payload, "bucket_counts": bucket_counts, "challenges": challenges, "settings_row": settings[0] if settings else {}}


def update_admin_challenge_ladder_challenge(
    supabase: Any,
    *,
    club_id: str,
    challenge_id: int,
    status: str,
    admin_note: str | None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_challenge_ladder_admin",
) -> dict[str, Any]:
    if not is_admin_challenge_ladder_enabled():
        raise PermissionError("Next Challenge Ladder Admin is disabled.")
    if _clean(confirmation_text, limit=80).upper() != CONFIRM:
        raise ValueError(f"Type {CONFIRM} to update the challenge ladder item.")
    safe_id = _safe_int(challenge_id)
    if safe_id is None:
        raise ValueError("challenge_id is required")
    before = _first(supabase.table("ladder_challenges").select("*").eq("club_id", str(club_id)).eq("id", int(safe_id)).limit(1).execute())
    if before is None:
        raise ValueError("challenge not found")
    clean_status = _clean(status, limit=40).upper()
    allowed = {"PENDING_ACCEPTANCE", "ACCEPTED_SCHEDULING", "IN_PROGRESS", "AWAITING_VERIFICATION", "OVERDUE_PLAY", "CANCELLED", "FORFEITED", "COMPLETED"}
    if clean_status not in allowed:
        raise ValueError("unsupported challenge status")
    patch: dict[str, Any] = {"status": clean_status, "updated_at": datetime.now(timezone.utc).isoformat()}
    if clean_status in FINAL_STATUSES and not before.get("completed_at"):
        patch["completed_at"] = datetime.now(timezone.utc).isoformat()
    if admin_note is not None:
        patch["admin_note"] = _clean(admin_note, limit=1000) or None
    updated = _first(supabase.table("ladder_challenges").update(patch).eq("club_id", str(club_id)).eq("id", int(safe_id)).execute()) or {**before, **patch}
    names = _player_names(supabase, club_id=str(club_id))
    audit = build_activity_payload(
        club_id=str(club_id), actor_email=actor_email, actor_role=actor_role, action_type="update_challenge_ladder_admin", entity_type="ladder_challenge", entity_id=str(safe_id),
        before_json={"status": before.get("status")}, after_json={"source_client": "fastapi/nextjs", "status": updated.get("status")}, source_page=source, flagged_for_review=True,
    )
    write = write_admin_activity_log(supabase, audit)
    if not write.ok and os.getenv("JUPR_REQUIRE_API_AUDIT_LOG", "").strip().lower() in TRUTHY:
        raise RuntimeError("audit log write required but unavailable")
    return {"ok": True, "mode": "challenge_ladder_admin_update", "challenge": _challenge_row(updated, names), "warnings": [write.warning] if write.warning else []}
