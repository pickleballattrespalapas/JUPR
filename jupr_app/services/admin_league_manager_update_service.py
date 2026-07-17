from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import os

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_league_manager_service import (
    get_admin_league_manager_detail,
    is_admin_league_manager_enabled,
)

CONFIRM_SAVE_LEAGUE = "SAVE LEAGUE"
TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
ALLOWED_STATUSES = {"draft", "active", "paused", "ended", "archived"}


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _clean_text(value: Any, *, limit: int = 200) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _safe_int(value: Any, *, field: str, minimum: int | None = None, maximum: int | None = None) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(float(value))
    except Exception as exc:
        raise ValueError(f"{field} must be a whole number.") from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{field} must be at least {minimum}.")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{field} must be at most {maximum}.")
    return parsed


def _json_object(value: Any, *, field: str) -> dict[str, Any] | None:
    if value in (None, ""):
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object.")
    return dict(value)


def _fetch_league_meta(supabase: Any, *, club_id: str, league_name: str) -> dict[str, Any] | None:
    rows = _safe_rows(
        supabase.table("leagues_metadata")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("league_name", str(league_name))
        .limit(1)
        .execute()
    )
    return rows[0] if rows else None


def _normalize_patch(patch: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if "description" in patch and patch.get("description") is not None:
        normalized["description"] = _clean_text(patch.get("description"), limit=2000)
    if "status" in patch and patch.get("status") not in (None, ""):
        status = _clean_text(patch.get("status"), limit=40).lower()
        if status not in ALLOWED_STATUSES:
            raise ValueError("status must be one of draft, active, paused, ended, or archived.")
        normalized["status"] = status
        normalized["is_active"] = status == "active"
        if status == "ended":
            normalized.setdefault("ended_at", _now_iso())
        elif status == "active":
            normalized["ended_at"] = None
            normalized["ended_by"] = None

    if "k_factor" in patch:
        value = _safe_int(patch.get("k_factor"), field="k_factor", minimum=1, maximum=128)
        if value is not None:
            normalized["k_factor"] = value
    if "min_games" in patch:
        value = _safe_int(patch.get("min_games"), field="min_games", minimum=0, maximum=1000)
        if value is not None:
            normalized["min_games"] = value

    for field in ("schedule_config", "court_board_defaults", "rules_config", "awards_config", "event_tags"):
        if field in patch:
            obj = _json_object(patch.get(field), field=field)
            if obj is not None:
                normalized[field] = obj

    if not normalized:
        raise ValueError("No league settings were provided.")
    normalized["updated_at"] = _now_iso()
    return normalized


def update_admin_league_manager_settings(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    patch: dict[str, Any],
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_league_manager_settings_update",
) -> dict[str, Any]:
    if not is_admin_league_manager_enabled():
        raise PermissionError("Next League Manager is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_SAVE_LEAGUE:
        raise ValueError(f"Type {CONFIRM_SAVE_LEAGUE} to save league settings.")

    clean_league = _clean_text(league_name, limit=120)
    if not clean_league:
        raise ValueError("league_name is required")
    normalized = _normalize_patch(dict(patch or {}))
    before = _fetch_league_meta(supabase, club_id=str(club_id), league_name=clean_league)
    if before is None:
        raise ValueError("league not found")

    updated = _safe_rows(
        supabase.table("leagues_metadata")
        .update(normalized)
        .eq("club_id", str(club_id))
        .eq("league_name", clean_league)
        .execute()
    )
    after = updated[0] if updated else {**before, **normalized}

    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="update_league_manager_settings_admin",
        entity_type="leagues_metadata",
        entity_id=clean_league,
        before_json=before or {},
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "league_name": clean_league,
            "created": False,
            "patch": normalized,
            "league": after,
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

    detail = get_admin_league_manager_detail(supabase, club_id=str(club_id), league_name=clean_league)
    return {
        "ok": True,
        "mode": "league_manager_settings_update",
        "league": detail.get("league"),
        "detail": detail,
        "created": False,
        "warnings": warnings,
    }
