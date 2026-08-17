from __future__ import annotations

import json
import os
from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.event_tags import normalize_event_tags
from jupr_app.services.admin_league_manager_service import (
    get_admin_league_manager_detail,
    is_admin_league_manager_enabled,
)

CONFIRM_CREATE_LEAGUE = "CREATE LEAGUE"
CONFIRM_DUPLICATE_LEAGUE = "DUPLICATE LEAGUE"
TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def _safe_rows(response: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (response.data or [])]
    except Exception:
        return []


def _clean_text(value: Any, *, limit: int) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _normalize_league_type(value: Any) -> str:
    clean = str(value or "Individual").strip().casefold()
    if clean not in {"individual", "team"}:
        raise ValueError("league_type must be Individual or Team.")
    return "Team" if clean == "team" else "Individual"


def _normalize_match_format(value: Any) -> str:
    clean = str(value or "doubles").strip().casefold()
    if clean not in {"doubles", "singles"}:
        raise ValueError("match_format must be doubles or singles.")
    return clean


def _bounded_int(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(float(value))
    except Exception as exc:
        raise ValueError(f"{field} must be a whole number.") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}.")
    return parsed


def _existing_league_names(supabase: Any, *, club_id: str) -> set[str]:
    rows = _safe_rows(
        supabase.table("leagues_metadata")
        .select("league_name")
        .eq("club_id", str(club_id))
        .execute()
    )
    return {
        _clean_text(row.get("league_name"), limit=120).casefold()
        for row in rows
        if _clean_text(row.get("league_name"), limit=120)
    }


def _is_unique_violation(exc: Exception) -> bool:
    code = str(getattr(exc, "code", "") or "")
    return code == "23505" or "duplicate key" in str(exc).lower()


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value in (None, ""):
        return {}
    try:
        parsed = json.loads(str(value))
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _find_league_row(supabase: Any, *, club_id: str, league_name: str) -> dict[str, Any] | None:
    normalized_name = _clean_text(league_name, limit=120).casefold()
    rows = _safe_rows(
        supabase.table("leagues_metadata")
        .select("*")
        .eq("club_id", str(club_id))
        .execute()
    )
    return next(
        (
            row
            for row in rows
            if _clean_text(row.get("league_name"), limit=120).casefold() == normalized_name
        ),
        None,
    )


def _cleanup_unaudited_draft(supabase: Any, *, club_id: str, league_name: str) -> None:
    """Best-effort compensation if a newly inserted draft cannot be audited."""

    try:
        (
            supabase.table("leagues_metadata")
            .delete()
            .eq("club_id", str(club_id))
            .eq("league_name", str(league_name))
            .eq("status", "draft")
            .execute()
        )
    except Exception:
        pass


def create_admin_league_manager_draft(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    description: str | None,
    min_games: Any,
    k_factor: Any,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    match_format: str = "doubles",
    league_type: str = "Individual",
    source: str = "next_league_manager_create",
) -> dict[str, Any]:
    if not is_admin_league_manager_enabled():
        raise PermissionError("Next League Manager is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_CREATE_LEAGUE:
        raise ValueError(f"Type {CONFIRM_CREATE_LEAGUE} to create the league draft.")

    clean_club_id = _clean_text(club_id, limit=120)
    clean_name = _clean_text(league_name, limit=120)
    if not clean_club_id:
        raise ValueError("club_id is required")
    if not clean_name:
        raise ValueError("league_name is required")

    clean_description = _clean_text(description, limit=2000)
    clean_match_format = _normalize_match_format(match_format)
    clean_league_type = _normalize_league_type(league_type)
    if clean_league_type == "Team" and clean_match_format == "singles":
        raise ValueError("Team leagues must use doubles; Team + Singles is not supported.")
    clean_min_games = _bounded_int(min_games, field="min_games", minimum=0, maximum=1000)
    clean_k_factor = _bounded_int(k_factor, field="k_factor", minimum=1, maximum=128)
    if clean_name.casefold() in _existing_league_names(supabase, club_id=clean_club_id):
        raise ValueError("A league with that name already exists for this club.")

    insert_payload = {
        "club_id": clean_club_id,
        "league_name": clean_name,
        "description": clean_description,
        "league_type": clean_league_type,
        "match_format": clean_match_format,
        "min_games": clean_min_games,
        "k_factor": clean_k_factor,
        "is_active": False,
        "status": "draft",
        "event_tags": normalize_event_tags({"skill_levels": [], "date_tags": []}),
    }
    try:
        inserted = _safe_rows(supabase.table("leagues_metadata").insert(insert_payload).execute())
    except Exception as exc:
        if _is_unique_violation(exc):
            raise ValueError("A league with that name already exists for this club.") from exc
        raise
    created = inserted[0] if inserted else insert_payload

    audit_payload = build_activity_payload(
        club_id=clean_club_id,
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="create_league_manager_draft_admin",
        entity_type="leagues_metadata",
        entity_id=clean_name,
        before_json={},
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "league": created,
        },
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")

    detail = get_admin_league_manager_detail(
        supabase,
        club_id=clean_club_id,
        league_name=clean_name,
    )
    return {
        "ok": True,
        "mode": "league_manager_draft_create",
        "created": True,
        "league": detail.get("league"),
        "detail": detail,
        "warnings": warnings,
    }


def duplicate_admin_league_manager_draft(
    supabase: Any,
    *,
    club_id: str,
    source_league_name: str,
    target_league_name: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_league_manager_duplicate",
) -> dict[str, Any]:
    if not is_admin_league_manager_enabled():
        raise PermissionError("Next League Manager is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_DUPLICATE_LEAGUE:
        raise ValueError(f"Type {CONFIRM_DUPLICATE_LEAGUE} to duplicate the league as a draft.")

    clean_club_id = _clean_text(club_id, limit=120)
    clean_source_name = _clean_text(source_league_name, limit=120)
    clean_target_name = _clean_text(target_league_name, limit=120)
    if not clean_club_id:
        raise ValueError("club_id is required")
    if not clean_source_name:
        raise ValueError("source_league_name is required")
    if not clean_target_name:
        raise ValueError("target_league_name is required")

    source_league = _find_league_row(
        supabase,
        club_id=clean_club_id,
        league_name=clean_source_name,
    )
    if source_league is None:
        raise ValueError("source league not found")
    if clean_target_name.casefold() in _existing_league_names(supabase, club_id=clean_club_id):
        raise ValueError("A league with that name already exists for this club.")

    copied_league_type = _normalize_league_type(source_league.get("league_type"))
    copied_match_format = _normalize_match_format(source_league.get("match_format"))
    if copied_league_type == "Team" and copied_match_format == "singles":
        raise ValueError("Team + Singles is not supported. Team leagues must use Doubles.")

    insert_payload = {
        "club_id": clean_club_id,
        "league_name": clean_target_name,
        "description": _clean_text(source_league.get("description"), limit=2000),
        "league_type": copied_league_type,
        "match_format": copied_match_format,
        "min_games": _bounded_int(
            source_league.get("min_games") if source_league.get("min_games") is not None else 0,
            field="min_games",
            minimum=0,
            maximum=1000,
        ),
        "k_factor": _bounded_int(
            source_league.get("k_factor") if source_league.get("k_factor") is not None else 32,
            field="k_factor",
            minimum=1,
            maximum=128,
        ),
        "is_active": False,
        "status": "draft",
        "schedule_config": _json_object(source_league.get("schedule_config")),
        "court_board_defaults": _json_object(source_league.get("court_board_defaults")),
        "rules_config": _json_object(source_league.get("rules_config")),
        "awards_config": _json_object(source_league.get("awards_config")),
        "event_tags": normalize_event_tags(source_league.get("event_tags")),
    }
    try:
        inserted = _safe_rows(supabase.table("leagues_metadata").insert(insert_payload).execute())
    except Exception as exc:
        if _is_unique_violation(exc):
            raise ValueError("A league with that name already exists for this club.") from exc
        raise
    created = inserted[0] if inserted else insert_payload

    audit_payload = build_activity_payload(
        club_id=clean_club_id,
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="duplicate_league_manager_draft_admin",
        entity_type="leagues_metadata",
        entity_id=clean_target_name,
        before_json={
            "source_league_name": clean_source_name,
            "source_league": source_league,
        },
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "source_league_name": clean_source_name,
            "league": created,
            "roster_copied": False,
        },
        source_page=source,
        flagged_for_review=True,
    )
    try:
        audit_write = write_admin_activity_log(supabase, audit_payload)
        warnings: list[str] = []
        if audit_write.warning:
            warnings.append(audit_write.warning)
        if not audit_write.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
            raise RuntimeError("audit log write required but unavailable")
    except Exception:
        _cleanup_unaudited_draft(
            supabase,
            club_id=clean_club_id,
            league_name=clean_target_name,
        )
        raise

    detail = get_admin_league_manager_detail(
        supabase,
        club_id=clean_club_id,
        league_name=clean_target_name,
    )
    return {
        "ok": True,
        "mode": "league_manager_draft_duplicate",
        "created": True,
        "league_name": clean_target_name,
        "source_league_name": clean_source_name,
        "roster_copied": False,
        "league": detail.get("league"),
        "detail": detail,
        "warnings": warnings,
    }
