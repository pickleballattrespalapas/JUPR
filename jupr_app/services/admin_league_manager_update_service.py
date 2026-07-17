from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.leagues import normalize_league_status
from jupr_app.services.admin_league_manager_service import (
    get_admin_league_manager_detail,
    is_admin_league_manager_enabled,
)

CONFIRM_SAVE_LEAGUE = "SAVE LEAGUE"
TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
ACTIVE_SAFE_FIELDS = {"description"}
CLOSED_STATUSES = {"ended", "archived"}


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
        raise ValueError("Use the guarded league lifecycle action to change status.")

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


def _validate_edit_policy(*, status: str, normalized: dict[str, Any]) -> None:
    requested_fields = set(normalized) - {"updated_at"}
    if status == "draft":
        return
    if status in {"active", "paused"}:
        blocked = sorted(requested_fields - ACTIVE_SAFE_FIELDS)
        if blocked:
            raise ValueError(
                f"Only description can be edited while a league is {status}; "
                f"blocked fields: {', '.join(blocked)}."
            )
        return
    if status in CLOSED_STATUSES:
        raise ValueError(f"League settings are read-only after a league is {status}.")
    raise ValueError(f"League settings cannot be edited while status is {status}.")


def _rollback_settings_update(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    before: dict[str, Any],
    changed_fields: set[str],
    expected_updated_at: str,
) -> None:
    """Best-effort compensation when staging requires an audit row."""

    rollback = {field: before.get(field) for field in changed_fields}
    try:
        (
            supabase.table("leagues_metadata")
            .update(rollback)
            .eq("club_id", str(club_id))
            .eq("league_name", str(league_name))
            .eq("updated_at", str(expected_updated_at))
            .execute()
        )
    except Exception:
        pass


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
    before = _fetch_league_meta(supabase, club_id=str(club_id), league_name=clean_league)
    if before is None:
        raise ValueError("league not found")
    normalized = _normalize_patch(dict(patch or {}))
    league_status = normalize_league_status(before)
    _validate_edit_policy(status=league_status, normalized=normalized)

    update_query = (
        supabase.table("leagues_metadata")
        .update(normalized)
        .eq("club_id", str(club_id))
        .eq("league_name", clean_league)
    )
    raw_status = before.get("status")
    if raw_status not in (None, ""):
        update_query = update_query.eq("status", str(raw_status))
    raw_updated_at = before.get("updated_at")
    if raw_updated_at not in (None, ""):
        update_query = update_query.eq("updated_at", str(raw_updated_at))
    updated = _safe_rows(update_query.execute())
    if not updated:
        raise ValueError("League settings changed before this save completed; reload and try again.")
    after = updated[0]

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
            "edit_policy_status": league_status,
            "patch": normalized,
            "league": after,
        },
        source_page=source,
        flagged_for_review=True,
    )
    try:
        audit_write = write_admin_activity_log(supabase, audit_payload)
    except Exception:
        if _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
            _rollback_settings_update(
                supabase,
                club_id=str(club_id),
                league_name=clean_league,
                before=before,
                changed_fields=set(normalized),
                expected_updated_at=str(normalized["updated_at"]),
            )
        raise
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        _rollback_settings_update(
            supabase,
            club_id=str(club_id),
            league_name=clean_league,
            before=before,
            changed_fields=set(normalized),
            expected_updated_at=str(normalized["updated_at"]),
        )
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
