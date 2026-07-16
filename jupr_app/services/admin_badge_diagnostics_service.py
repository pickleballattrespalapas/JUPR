from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace
from typing import Any

import pandas as pd

from jupr_app.data.load import load_data
from jupr_app.domain.gamification.badge_audit import build_badge_audit_report
from jupr_app.domain.gamification.badge_debug import build_badge_debug_report
from jupr_app.domain.gamification.badge_registry import badge_schema_by_id, registry

TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}


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
        "status": "ready_for_badge_diagnostics_read_only",
        "options_endpoint": "/admin/clubs/{club_id}/badges/options",
        "debug_endpoint": "/admin/clubs/{club_id}/badges/debug",
        "audit_endpoint": "/admin/clubs/{club_id}/badges/audit",
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
    badge_ids = sorted(set(registry().keys()) | set(schema.keys()))
    badges = []
    for badge_id in badge_ids:
        spec = registry().get(badge_id)
        badge_schema = schema.get(badge_id)
        badges.append(
            {
                "badge_id": badge_id,
                "name": getattr(spec, "name", None) or getattr(badge_schema, "name", None) or badge_id.replace("_", " ").title(),
                "status": getattr(badge_schema, "status", "live") if badge_schema is not None else "live",
                "scope": getattr(badge_schema, "scope", None) if badge_schema is not None else None,
                "award_timing": getattr(badge_schema, "award_timing", None) if badge_schema is not None else None,
            }
        )
    return {"ok": True, "mode": "badge_diagnostic_options", "players": players, "badges": badges, "player_count": len(players), "badge_count": len(badges)}


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
