from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pandas as pd

from jupr_app.data.load import load_data
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.gamification.badge_audit import build_badge_audit_report
from jupr_app.domain.gamification.badge_debug import build_badge_debug_report
from jupr_app.domain.gamification.badge_registry import badge_schema_by_id, registry
from jupr_app.domain.gamification.badge_state import (
    ALLOWED_BADGE_STATES,
    can_transition_badge_state,
    normalize_badge_state,
)
from jupr_app.domain.gamification.recompute import run_badge_recompute

TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
CONFIRM_RECOMPUTE = "RECOMPUTE BADGES"
CONFIRM_REVOKE = "REVOKE BADGE"
CONFIRM_BADGE_STATE = "UPDATE BADGE STATE"


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def is_admin_badge_diagnostics_enabled() -> bool:
    return _truthy_env("JUPR_ENABLE_NEXT_ADMIN_BADGE_DIAGNOSTICS")


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _clean_text(value: Any, *, limit: int = 240) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return [_json_safe(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return _json_safe(value.to_dict())
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _load_ctx(supabase: Any, *, club_id: str, match_limit: int = 5000) -> SimpleNamespace:
    (
        df_players_all,
        df_players_active,
        df_leagues,
        df_matches,
        df_meta,
        df_badges,
        df_player_badges,
        name_to_id,
        id_to_name,
        schema_degraded,
        schema_degraded_reason,
    ) = load_data(supabase, str(club_id), match_limit=int(match_limit))
    return SimpleNamespace(
        supabase=supabase,
        club_id=str(club_id),
        df_players_all=df_players_all,
        df_players_active=df_players_active,
        df_leagues=df_leagues,
        df_matches=df_matches,
        df_meta=df_meta,
        df_badges=df_badges,
        df_player_badges=df_player_badges,
        name_to_id=name_to_id,
        id_to_name=id_to_name,
        schema_degraded=schema_degraded,
        schema_degraded_reason=schema_degraded_reason,
        public_mode=False,
        admin_logged_in=True,
    )


def build_admin_badge_diagnostics_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    if not is_admin_badge_diagnostics_enabled():
        return {
            "enabled": False,
            "status": "guarded_off",
            "options_endpoint": None,
            "warnings": ["Next Badge Diagnostics is disabled. Enable JUPR_ENABLE_NEXT_ADMIN_BADGE_DIAGNOSTICS on FastAPI."],
        }
    badge_count = 0
    player_badge_count = 0
    if supabase is not None:
        try:
            badge_count = len(_safe_rows(supabase.table("badges").select("badge_id").execute()))
        except Exception:
            badge_count = len(registry())
        try:
            player_badge_count = len(_safe_rows(supabase.table("player_badges").select("id").eq("club_id", str(club_id)).execute()))
        except Exception:
            player_badge_count = 0
    return {
        "enabled": True,
        "status": "ready_for_badge_diagnostics_and_repair",
        "options_endpoint": "/admin/clubs/{club_id}/badges/options",
        "debug_endpoint": "/admin/clubs/{club_id}/badges/debug",
        "audit_endpoint": "/admin/clubs/{club_id}/badges/audit",
        "recompute_endpoint": "/admin/clubs/{club_id}/badges/recompute",
        "revoke_endpoint": "/admin/clubs/{club_id}/badges/revoke",
        "state_endpoint": "/admin/clubs/{club_id}/badges/{badge_id}/state",
        "confirmation_text": {
            "recompute": CONFIRM_RECOMPUTE,
            "revoke": CONFIRM_REVOKE,
            "state": CONFIRM_BADGE_STATE,
        },
        "badge_count": badge_count,
        "player_badge_count": player_badge_count,
        "warnings": [],
    }


def list_admin_badge_diagnostic_options(supabase: Any, *, club_id: str) -> dict[str, Any]:
    if not is_admin_badge_diagnostics_enabled():
        raise PermissionError("Next Badge Diagnostics is disabled.")
    try:
        player_rows = _safe_rows(
            supabase.table("players")
            .select("id,name,rating,wins,losses,matches_played,active")
            .eq("club_id", str(club_id))
            .order("name", desc=False)
            .execute()
        )
    except Exception:
        player_rows = []
    try:
        badge_definition_rows = _safe_rows(
            supabase.table("badges")
            .select("badge_id,name,state,state_changed_at,state_change_reason")
            .order("name", desc=False)
            .execute()
        )
    except Exception:
        badge_definition_rows = []
    players = [
        {
            "id": _safe_int(row.get("id")),
            "name": _clean_text(row.get("name"), limit=160) or f"Player {row.get('id')}",
            "rating": row.get("rating"),
            "wins": row.get("wins"),
            "losses": row.get("losses"),
            "matches_played": row.get("matches_played"),
            "active": row.get("active"),
        }
        for row in player_rows
        if _safe_int(row.get("id")) is not None
    ]
    schema = badge_schema_by_id()
    definitions_by_id = {
        _clean_text(row.get("badge_id"), limit=120): row
        for row in badge_definition_rows
        if _clean_text(row.get("badge_id"), limit=120)
    }
    badge_ids = sorted(set(registry().keys()) | set(schema.keys()) | set(definitions_by_id))
    badges = []
    for badge_id in badge_ids:
        spec = registry().get(badge_id)
        badge_schema = schema.get(badge_id)
        definition = definitions_by_id.get(badge_id, {})
        badges.append(
            {
                "badge_id": badge_id,
                "name": _clean_text(definition.get("name"), limit=160) or getattr(spec, "name", None) or getattr(badge_schema, "name", None) or badge_id.replace("_", " ").title(),
                "status": getattr(badge_schema, "status", "live") if badge_schema is not None else "live",
                "state": normalize_badge_state(definition.get("state")),
                "state_changed_at": definition.get("state_changed_at"),
                "state_change_reason": _clean_text(definition.get("state_change_reason"), limit=500) or None,
                "definition_found": bool(definition),
                "scope": getattr(badge_schema, "scope", None) if badge_schema is not None else None,
                "award_timing": getattr(badge_schema, "award_timing", None) if badge_schema is not None else None,
            }
        )
    return {"ok": True, "mode": "badge_diagnostic_options", "players": players, "badges": badges, "player_count": len(players), "badge_count": len(badges)}


def update_admin_badge_definition_state(
    supabase: Any,
    *,
    club_id: str,
    badge_id: str,
    expected_state: str,
    target_state: str,
    reason: str,
    force: bool,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_badge_definition_state",
) -> dict[str, Any]:
    if not is_admin_badge_diagnostics_enabled():
        raise PermissionError("Next Badge Diagnostics is disabled.")
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_BADGE_STATE:
        raise ValueError(f"Type {CONFIRM_BADGE_STATE} to update badge state.")
    clean_badge_id = _clean_text(badge_id, limit=120)
    if not clean_badge_id:
        raise ValueError("badge_id is required.")
    clean_reason = _clean_text(reason, limit=500)
    if not clean_reason:
        raise ValueError("A badge state change reason is required.")
    normalized_expected = normalize_badge_state(expected_state)
    normalized_target = normalize_badge_state(target_state)
    if normalized_expected not in ALLOWED_BADGE_STATES:
        raise ValueError("A valid expected_state is required.")
    if normalized_target not in ALLOWED_BADGE_STATES:
        raise ValueError("A valid target_state is required.")

    before_rows = _safe_rows(
        supabase.table("badges")
        .select("badge_id,name,state,state_changed_at,state_change_reason")
        .eq("badge_id", clean_badge_id)
        .limit(1)
        .execute()
    )
    if not before_rows:
        raise ValueError("Badge definition not found.")
    before = before_rows[0]
    current_state = normalize_badge_state(before.get("state"))
    if current_state != normalized_expected:
        raise ValueError(
            f"Badge state changed from {normalized_expected} to {current_state}. Reload badge options before updating."
        )
    transition = can_transition_badge_state(current_state, normalized_target, force=bool(force))
    if not transition.allowed:
        raise ValueError(transition.reason or "Badge state transition is not allowed.")

    patch = {
        "state": normalized_target,
        "state_changed_at": datetime.now(timezone.utc).isoformat(),
        "state_change_reason": clean_reason,
    }
    updated_rows = _safe_rows(
        supabase.table("badges")
        .update(patch)
        .eq("badge_id", clean_badge_id)
        .eq("state", current_state)
        .execute()
    )
    if not updated_rows:
        raise ValueError("Badge state changed while it was being updated. Reload badge options and review it again.")
    updated = {**before, **updated_rows[0]}
    audit_write = write_admin_activity_log(
        supabase,
        build_activity_payload(
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="update_badge_definition_state",
            entity_type="badge_definition",
            entity_id=clean_badge_id,
            before_json={
                "badge_id": clean_badge_id,
                "name": _clean_text(before.get("name"), limit=160),
                "state": current_state,
                "state_changed_at": before.get("state_changed_at"),
                "state_change_reason": _clean_text(before.get("state_change_reason"), limit=500) or None,
            },
            after_json={
                "source_client": "fastapi/nextjs",
                "badge_id": clean_badge_id,
                "state": normalized_target,
                "state_changed_at": updated.get("state_changed_at"),
                "state_change_reason": clean_reason,
                "force": bool(force),
            },
            note=clean_reason,
            source_page=_clean_text(source, limit=120) or "next_badge_definition_state",
            flagged_for_review=True,
        ),
    )
    if not audit_write.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")
    return {
        "ok": True,
        "mode": "badge_definition_state_update",
        "badge": {
            "badge_id": clean_badge_id,
            "name": _clean_text(updated.get("name"), limit=160) or clean_badge_id,
            "state": normalized_target,
            "state_changed_at": updated.get("state_changed_at"),
            "state_change_reason": clean_reason,
        },
        "force": bool(force),
        "audit_warning": audit_write.warning,
    }


def build_admin_badge_debug(
    supabase: Any,
    *,
    club_id: str,
    player_id: int,
    badge_id: str,
    league_id: str | None = None,
    match_limit: int = 5000,
) -> dict[str, Any]:
    if not is_admin_badge_diagnostics_enabled():
        raise PermissionError("Next Badge Diagnostics is disabled.")
    ctx = _load_ctx(supabase, club_id=str(club_id), match_limit=int(match_limit))
    report = build_badge_debug_report(
        ctx,
        club_id=str(club_id),
        league_id=_clean_text(league_id, limit=120) or None,
        player_id=int(player_id),
        badge_id=_clean_text(badge_id, limit=120),
        limit_matches=int(match_limit),
    )
    return {"ok": True, "mode": "badge_debug", "report": _json_safe(report)}


def build_admin_badge_audit(
    supabase: Any,
    *,
    club_id: str,
    league_id: str | None = None,
    player_id: int | None = None,
    badge_id: str | None = None,
    context_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    include_non_live: bool = False,
    include_revoked: bool = False,
    match_limit: int = 5000,
) -> dict[str, Any]:
    if not is_admin_badge_diagnostics_enabled():
        raise PermissionError("Next Badge Diagnostics is disabled.")
    report = build_badge_audit_report(
        supabase,
        club_id=str(club_id),
        league_id=_clean_text(league_id, limit=120) or None,
        player_id=player_id,
        badge_id=_clean_text(badge_id, limit=120) or None,
        context_id=_clean_text(context_id, limit=240) or None,
        since=_clean_text(since, limit=40) or None,
        until=_clean_text(until, limit=40) or None,
        include_non_live=bool(include_non_live),
        include_revoked=bool(include_revoked),
        match_limit=int(match_limit),
    )
    return {"ok": True, "mode": "badge_audit", "report": _json_safe(report)}


def run_admin_badge_recompute(
    supabase: Any,
    *,
    club_id: str,
    mode: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    league_id: str | None = None,
    player_id: int | None = None,
    badge_id: str | None = None,
    context_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    include_non_live: bool = False,
    match_limit: int = 5000,
    revoke_reason: str | None = None,
    source: str = "next_badge_recompute",
) -> dict[str, Any]:
    if not is_admin_badge_diagnostics_enabled():
        raise PermissionError("Next Badge Diagnostics is disabled.")
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_RECOMPUTE:
        raise ValueError(f"Type {CONFIRM_RECOMPUTE} to run badge recompute.")
    clean_mode = _clean_text(mode or "dry-run", limit=40).lower()
    if clean_mode not in {"dry-run", "append-only", "strict"}:
        raise ValueError("mode must be dry-run, append-only, or strict")
    safe_player_id = _safe_int(player_id)
    summary = run_badge_recompute(
        supabase,
        club_id=str(club_id),
        mode=clean_mode,
        league_id=_clean_text(league_id, limit=120) or None,
        context_id=_clean_text(context_id, limit=240) or None,
        player_id=safe_player_id,
        badge_id=_clean_text(badge_id, limit=120) or None,
        since=_clean_text(since, limit=40) or None,
        until=_clean_text(until, limit=40) or None,
        created_by=str(actor_email or "admin"),
        revoke_reason=_clean_text(revoke_reason, limit=400) or None,
        allow_strict_global=False,
        match_limit=int(match_limit or 5000),
        include_non_live=bool(include_non_live),
    )
    log_result = write_admin_activity_log(
        supabase,
        build_activity_payload(
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="badge_recompute" if clean_mode != "dry-run" else "badge_recompute_dry_run",
            entity_type="badge_admin",
            entity_id=str(badge_id or player_id or league_id or context_id or "scoped_recompute"),
            before_json=None,
            after_json={"source_client": "fastapi/nextjs", "mode": clean_mode, "summary": _json_safe(summary)},
            note="Badge recompute from Next Badge Diagnostics",
            source_page=source,
            flagged_for_review=clean_mode != "dry-run",
        ),
    )
    if not log_result.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")
    return {"ok": True, "mode": "badge_recompute", "recompute_mode": clean_mode, "summary": _json_safe(summary), "audit_warning": log_result.warning}


def _select_badge_rows(
    supabase: Any,
    *,
    club_id: str,
    player_badge_id: str | None,
    player_id: int | None,
    badge_id: str | None,
    context_id: str | None,
) -> list[dict[str, Any]]:
    query = supabase.table("player_badges").select("*").eq("club_id", str(club_id))
    if player_badge_id:
        query = query.eq("id", str(player_badge_id))
    else:
        if player_id is None or not badge_id:
            raise ValueError("Provide player_badge_id, or player_id plus badge_id.")
        query = query.eq("player_id", int(player_id)).eq("badge_id", str(badge_id))
        if context_id:
            query = query.eq("context_id", str(context_id))
    return _safe_rows(query.execute())


def revoke_admin_player_badge(
    supabase: Any,
    *,
    club_id: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    player_badge_id: str | None = None,
    player_id: int | None = None,
    badge_id: str | None = None,
    context_id: str | None = None,
    revoke_reason: str | None = None,
    source: str = "next_badge_revoke",
) -> dict[str, Any]:
    if not is_admin_badge_diagnostics_enabled():
        raise PermissionError("Next Badge Diagnostics is disabled.")
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_REVOKE:
        raise ValueError(f"Type {CONFIRM_REVOKE} to revoke a badge row.")
    safe_player_id = _safe_int(player_id)
    rows = _select_badge_rows(
        supabase,
        club_id=str(club_id),
        player_badge_id=_clean_text(player_badge_id, limit=120) or None,
        player_id=safe_player_id,
        badge_id=_clean_text(badge_id, limit=120) or None,
        context_id=_clean_text(context_id, limit=240) or None,
    )
    if not rows:
        raise ValueError("No matching player_badges rows found to revoke.")
    now = datetime.now(timezone.utc).isoformat()
    reason = _clean_text(revoke_reason, limit=500) or "revoked from Next Badge Diagnostics"
    revoked_rows: list[dict[str, Any]] = []
    for row in rows:
        row_id = row.get("id")
        if row_id is None:
            continue
        patch = {"revoked_at": now, "revoked_by": str(actor_email or "admin"), "revoke_reason": reason}
        updated = _safe_rows(
            supabase.table("player_badges")
            .update(patch)
            .eq("club_id", str(club_id))
            .eq("id", row_id)
            .execute()
        )
        revoked_rows.extend(updated or [{**row, **patch}])
    log_result = write_admin_activity_log(
        supabase,
        build_activity_payload(
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="badge_revoke",
            entity_type="player_badge",
            entity_id=str(player_badge_id or badge_id or "badge_scope"),
            before_json={"rows": _json_safe(rows[:25]), "matched_count": len(rows)},
            after_json={"source_client": "fastapi/nextjs", "revoked_count": len(revoked_rows), "reason": reason},
            note="Badge row revocation from Next Badge Diagnostics",
            source_page=source,
            flagged_for_review=True,
        ),
    )
    if not log_result.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")
    return {"ok": True, "mode": "badge_revoke", "revoked_count": len(revoked_rows), "rows": _json_safe(revoked_rows), "audit_warning": log_result.warning}
