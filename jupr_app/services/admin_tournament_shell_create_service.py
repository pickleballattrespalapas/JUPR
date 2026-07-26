from __future__ import annotations

from datetime import date
import os
from typing import Any
from uuid import UUID

from jupr_app.domain.admin_activity_log import (
    build_activity_payload,
    write_admin_activity_log,
)
from jupr_app.domain.event_tags import derive_default_date_tags, normalize_event_tags
from jupr_app.domain.tournament_admin_operations import (
    stable_tournament_admin_fingerprint,
)
from jupr_app.domain.tournament_registration_repo import get_tournament_record


TRUTHY = {"1", "true", "yes", "y", "on"}
CONFIRM_CREATE = "CREATE TOURNAMENT"
DEFAULT_TEAM_COUNT = 4


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY


def _assert_enabled() -> None:
    if not _truthy_env("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS"):
        raise PermissionError("Next Tournament Setup is disabled.")


def _clean_text(value: Any, *, limit: int) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _clean_uuid(value: Any, *, field: str) -> str:
    text = _clean_text(value, limit=64)
    try:
        parsed = UUID(text)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{field} must be a valid UUID.") from exc
    if str(parsed) != text.lower():
        raise ValueError(f"{field} must use canonical UUID format.")
    return str(parsed)


def _clean_date(value: Any, *, field: str) -> str | None:
    text = _clean_text(value, limit=40)
    if not text:
        return None
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid date in YYYY-MM-DD format.") from exc


def _create_payload(
    *,
    club_id: str,
    tournament_id: str,
    name: str,
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    clean_club_id = _clean_text(club_id, limit=120)
    clean_tournament_id = _clean_uuid(tournament_id, field="tournament_id")
    clean_name = _clean_text(name, limit=180)
    clean_start = _clean_date(start_date, field="start_date")
    clean_end = _clean_date(end_date, field="end_date")
    if not clean_club_id:
        raise ValueError("club_id is required.")
    if not clean_name:
        raise ValueError("Tournament name is required.")
    if clean_start and clean_end and clean_end < clean_start:
        raise ValueError("Tournament end date cannot be before its start date.")
    return {
        "id": clean_tournament_id,
        "club_id": clean_club_id,
        "name": clean_name,
        "status": "DRAFT",
        "team_count": DEFAULT_TEAM_COUNT,
        "start_date": clean_start,
        "end_date": clean_end,
        "event_tags": normalize_event_tags(
            {
                "skill_levels": [],
                "date_tags": derive_default_date_tags(
                    start_date=clean_start,
                    end_date=clean_end,
                ),
            }
        ),
    }


def _public_tournament(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "club_id": row.get("club_id"),
        "name": row.get("name"),
        "status": str(row.get("status") or "DRAFT").upper(),
        "start_date": row.get("start_date"),
        "end_date": row.get("end_date"),
        "event_tags": normalize_event_tags(row.get("event_tags")),
    }


def tournament_shell_absent_state_fingerprint(
    *,
    club_id: str,
    tournament_id: str,
) -> str:
    clean_tournament_id = _clean_uuid(tournament_id, field="tournament_id")
    return stable_tournament_admin_fingerprint(
        {
            "club_id": _clean_text(club_id, limit=120),
            "tournament_id": clean_tournament_id,
            "tournament": None,
        }
    )


def get_tournament_shell_creation_state_fingerprint(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
) -> str:
    clean_tournament_id = _clean_uuid(tournament_id, field="tournament_id")
    row = get_tournament_record(supabase, clean_tournament_id)
    return stable_tournament_admin_fingerprint(
        {
            "club_id": _clean_text(club_id, limit=120),
            "tournament_id": clean_tournament_id,
            "tournament": _public_tournament(row) if row else None,
        }
    )


def create_admin_tournament_shell(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    name: str,
    start_date: str | None,
    end_date: str | None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_tournament_setup_create_shell",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create one DRAFT shell; registration configuration stays a later step."""

    _assert_enabled()
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_CREATE:
        raise ValueError(f"Type {CONFIRM_CREATE} to create a tournament shell.")
    payload = _create_payload(
        club_id=club_id,
        tournament_id=tournament_id,
        name=name,
        start_date=start_date,
        end_date=end_date,
    )
    if get_tournament_record(supabase, payload["id"]):
        raise ValueError("A tournament already exists with this tournament_id.")
    if dry_run:
        return {
            "ok": True,
            "mode": "tournament_setup_shell_create_preflight",
            "dry_run": True,
            "write_count": 0,
            "tournament": _public_tournament(payload),
        }

    response = supabase.table("tournaments").insert(payload).execute()
    rows = [dict(row) for row in (getattr(response, "data", None) or [])]
    created = rows[0] if rows else get_tournament_record(supabase, payload["id"])
    if not created:
        raise RuntimeError(
            "Tournament shell insert returned without authoritative readback."
        )
    tournament = _public_tournament(created)
    audit = write_admin_activity_log(
        supabase,
        build_activity_payload(
            club_id=str(payload["club_id"]),
            actor_email=str(actor_email or ""),
            actor_role=str(actor_role or ""),
            action_type="tournament_setup_shell_create",
            entity_type="tournament",
            entity_id=str(payload["id"]),
            before_json=None,
            after_json={
                "source_client": "fastapi/nextjs",
                "source_page": source,
                "tournament": tournament,
            },
            source_page=source,
            flagged_for_review=True,
        ),
    )
    if not audit.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")
    warnings = [audit.warning] if audit.warning else []
    return {
        "ok": True,
        "mode": "tournament_setup_shell_create",
        "tournament": tournament,
        "warnings": warnings,
    }


def reconcile_admin_tournament_shell_creation(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    name: str,
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any] | None:
    """Recover only when the exact deterministic shell exists."""

    expected = _create_payload(
        club_id=club_id,
        tournament_id=tournament_id,
        name=name,
        start_date=start_date,
        end_date=end_date,
    )
    row = get_tournament_record(supabase, expected["id"])
    if not row:
        return None
    actual = _public_tournament(row)
    expected_public = _public_tournament(expected)
    if actual != expected_public:
        return None
    return {
        "ok": True,
        "mode": "tournament_setup_shell_create",
        "tournament": actual,
        "warnings": [],
    }
