from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.leagues import compute_top_performer_awards_for_config, mint_top_performer_badges
from jupr_app.services.admin_league_manager_service import is_admin_league_manager_enabled

TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
CONFIRM_CLOSE_LEAGUE = "CLOSE LEAGUE"


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _first_row(resp: Any) -> dict[str, Any] | None:
    rows = _safe_rows(resp)
    return rows[0] if rows else None


def _clean_text(value: Any, *, limit: int = 240) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _json_value(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _fetch_table_rows(supabase: Any, table_name: str, *, club_id: str) -> list[dict[str, Any]]:
    try:
        return _safe_rows(supabase.table(table_name).select("*").eq("club_id", str(club_id)).execute())
    except Exception:
        return []


def _fetch_league_meta_row(supabase: Any, *, club_id: str, league_name: str) -> dict[str, Any] | None:
    clean_league = _clean_text(league_name, limit=120)
    try:
        return _first_row(
            supabase.table("leagues_metadata")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("league_name", clean_league)
            .limit(1)
            .execute()
        )
    except Exception:
        rows = _fetch_table_rows(supabase, "leagues_metadata", club_id=str(club_id))
        return next((row for row in rows if _clean_text(row.get("league_name"), limit=120) == clean_league), None)


def _id_to_name(players: list[dict[str, Any]]) -> dict[int, str]:
    result: dict[int, str] = {}
    for row in players:
        try:
            pid = int(row.get("id"))
        except Exception:
            continue
        name = _clean_text(row.get("name"), limit=160)
        if name:
            result[pid] = name
    return result


def _award_inputs(supabase: Any, *, club_id: str, league_name: str) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame, pd.DataFrame, dict[int, str]]:
    clean_league = _clean_text(league_name, limit=120)
    if not clean_league:
        raise ValueError("league_name is required")
    meta_row = _fetch_league_meta_row(supabase, club_id=str(club_id), league_name=clean_league)
    if not meta_row:
        raise ValueError("league not found")

    meta_rows = _fetch_table_rows(supabase, "leagues_metadata", club_id=str(club_id))
    league_rows = _fetch_table_rows(supabase, "league_ratings", club_id=str(club_id))
    player_rows = _fetch_table_rows(supabase, "players", club_id=str(club_id))
    id_to_name = _id_to_name(player_rows)

    df_meta = pd.DataFrame(meta_rows)
    df_leagues = pd.DataFrame(league_rows)
    awards_config = _json_value(meta_row.get("awards_config"), {}) or {}
    awards = compute_top_performer_awards_for_config(
        df_leagues,
        df_meta,
        id_to_name,
        clean_league,
        awards_config=awards_config if isinstance(awards_config, dict) else {},
    )
    return meta_row, awards, df_meta, df_leagues, id_to_name


def _league_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "league_name": _clean_text(row.get("league_name"), limit=120),
        "status": _clean_text(row.get("status"), limit=60) or ("active" if row.get("is_active") else "draft"),
        "is_active": bool(row.get("is_active", False)),
        "min_games": row.get("min_games"),
        "awards_config": _json_value(row.get("awards_config"), {}) or {},
        "end_awards": _json_value(row.get("end_awards"), {}) or {},
        "ended_at": row.get("ended_at"),
        "ended_by": row.get("ended_by"),
    }


def _candidate_payload(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, dict):
        return dict(candidate)
    return {
        "badge_id": getattr(candidate, "badge_id", None),
        "player_id": getattr(candidate, "player_id", None),
        "club_id": getattr(candidate, "club_id", None),
        "context_type": getattr(candidate, "context_type", None),
        "context_id": getattr(candidate, "context_id", None),
        "value_json": getattr(candidate, "value_json", None),
        "value_num": getattr(candidate, "value_num", None),
    }


def preview_admin_league_awards(supabase: Any, *, club_id: str, league_name: str) -> dict[str, Any]:
    if not is_admin_league_manager_enabled():
        raise PermissionError("Next League Manager is disabled.")
    meta_row, awards, _df_meta, _df_leagues, _id_map = _award_inputs(supabase, club_id=str(club_id), league_name=str(league_name))
    return {
        "ok": True,
        "mode": "league_awards_preview",
        "league": _league_payload(meta_row),
        "league_name": _clean_text(league_name, limit=120),
        "awards": awards,
        "award_count": len(awards),
        "warnings": [],
    }


def close_admin_league_and_award(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    award_badges: bool = True,
    source: str = "next_league_manager_awards_close",
) -> dict[str, Any]:
    if not is_admin_league_manager_enabled():
        raise PermissionError("Next League Manager is disabled.")
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_CLOSE_LEAGUE:
        raise ValueError(f"Type {CONFIRM_CLOSE_LEAGUE} to close this league and award top performers.")

    clean_league = _clean_text(league_name, limit=120)
    before, awards, _df_meta, _df_leagues, _id_map = _award_inputs(supabase, club_id=str(club_id), league_name=clean_league)
    ended_at = _now_iso()
    end_awards = {"top_performers": awards, "source": source, "generated_at": ended_at}
    update_payload = {
        "is_active": False,
        "status": "ended",
        "ended_at": ended_at,
        "ended_by": str(actor_email or ""),
        "end_awards": end_awards,
    }
    updated = _first_row(
        supabase.table("leagues_metadata")
        .update(update_payload)
        .eq("club_id", str(club_id))
        .eq("league_name", clean_league)
        .execute()
    ) or {**before, **update_payload}

    warnings: list[str] = []
    badge_candidates: list[dict[str, Any]] = []
    if award_badges and awards:
        try:
            created = mint_top_performer_badges(
                supabase,
                club_id=str(club_id),
                league_id=clean_league,
                awards=awards,
                ended_at=ended_at,
            )
            badge_candidates = [_candidate_payload(candidate) for candidate in (created or [])]
        except Exception as exc:  # noqa: BLE001 - badge tables can lag during staged migration
            warnings.append(f"Badge award write skipped: {exc}")

    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="close_league_award_top_performers_admin",
        entity_type="league",
        entity_id=clean_league,
        before_json={"league": _league_payload(before), "award_count": len(awards)},
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "league": _league_payload(updated),
            "award_count": len(awards),
            "badge_candidate_count": len(badge_candidates),
            "badge_write_attempted": bool(award_badges),
            "warnings": warnings,
        },
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")

    return {
        "ok": True,
        "mode": "league_awards_close",
        "league": _league_payload(updated),
        "league_name": clean_league,
        "awards": awards,
        "award_count": len(awards),
        "badge_candidates": badge_candidates,
        "badge_candidate_count": len(badge_candidates),
        "warnings": warnings,
    }
