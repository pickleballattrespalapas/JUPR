from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from typing import Any

from jupr_app.domain.leagues import normalize_league_status

TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
LEAGUE_MANAGER_EXTENDED_SELECT = "league_name,is_active,status,started_at,ended_at,ended_by,schedule_config,court_board_defaults,rules_config,awards_config,k_factor,min_games,event_tags"
LEAGUE_MANAGER_MINIMAL_SELECT = "league_name,is_active,k_factor,min_games"


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def is_admin_league_manager_enabled() -> bool:
    return _truthy_env("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER")


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _clean_text(value: Any, *, limit: int = 200) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _json_value(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        text = str(value).strip()[:10]
        return date.fromisoformat(text)
    except Exception:
        return None


def _schedule_preview(schedule_config: Any) -> list[dict[str, Any]]:
    cfg = _json_value(schedule_config, {}) or {}
    if not isinstance(cfg, dict):
        return []
    start_date = _parse_date(cfg.get("start_date"))
    if not start_date:
        return []
    weekday = _safe_int(cfg.get("weekday"))
    if weekday is None:
        return []
    end_date = _parse_date(cfg.get("end_date"))
    weeks = _safe_int(cfg.get("weeks"))
    time_start = _clean_text(cfg.get("time_start"), limit=20)
    time_end = _clean_text(cfg.get("time_end"), limit=20)
    blackout = {str(day) for day in (cfg.get("blackout_dates") or []) if day}
    first_date = start_date + timedelta(days=(int(weekday) - start_date.weekday()) % 7)
    dates: list[date] = []
    if weeks and weeks > 0:
        dates = [first_date + timedelta(weeks=idx) for idx in range(int(weeks))]
    elif end_date:
        current = first_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(weeks=1)
    preview: list[dict[str, Any]] = []
    for session, day in enumerate(dates, start=1):
        if day.isoformat() in blackout:
            continue
        preview.append({"session": session, "date": day.isoformat(), "start": time_start, "end": time_end})
    return preview


def _league_row_payload(row: dict[str, Any]) -> dict[str, Any]:
    league_name = _clean_text(row.get("league_name"), limit=120)
    return {
        "league_name": league_name,
        "status": normalize_league_status(row),
        "is_active": bool(row.get("is_active", False)),
        "started_at": row.get("started_at"),
        "ended_at": row.get("ended_at"),
        "ended_by": row.get("ended_by"),
        "k_factor": _safe_float(row.get("k_factor")),
        "min_games": _safe_int(row.get("min_games")),
        "schedule_config": _json_value(row.get("schedule_config"), {}) or {},
        "court_board_defaults": _json_value(row.get("court_board_defaults"), {}) or {},
        "rules_config": _json_value(row.get("rules_config"), {}) or {},
        "awards_config": _json_value(row.get("awards_config"), {}) or {},
        "event_tags": _json_value(row.get("event_tags"), {}) or {},
    }


def _fetch_league_rows(supabase: Any, *, club_id: str) -> list[dict[str, Any]]:
    try:
        rows = _safe_rows(
            supabase.table("leagues_metadata")
            .select(LEAGUE_MANAGER_EXTENDED_SELECT)
            .eq("club_id", str(club_id))
            .execute()
        )
    except Exception:
        rows = _safe_rows(
            supabase.table("leagues_metadata")
            .select(LEAGUE_MANAGER_MINIMAL_SELECT)
            .eq("club_id", str(club_id))
            .execute()
        )
    leagues = [_league_row_payload(row) for row in rows if _clean_text(row.get("league_name"), limit=120)]
    return sorted(leagues, key=lambda row: str(row.get("league_name") or "").lower())


def _fetch_player_names(supabase: Any, *, club_id: str) -> dict[int, str]:
    try:
        rows = _safe_rows(supabase.table("players").select("id,name").eq("club_id", str(club_id)).execute())
    except Exception:
        rows = []
    names: dict[int, str] = {}
    for row in rows:
        pid = _safe_int(row.get("id"))
        name = _clean_text(row.get("name"), limit=160)
        if pid is not None and name:
            names[int(pid)] = name
    return names


def _league_standings(supabase: Any, *, club_id: str, league_name: str, limit: int = 50) -> list[dict[str, Any]]:
    player_names = _fetch_player_names(supabase, club_id=str(club_id))
    try:
        rows = _safe_rows(
            supabase.table("league_ratings")
            .select("id,player_id,league_name,rating,starting_rating,wins,losses,matches_played,is_active,inactive_at")
            .eq("club_id", str(club_id))
            .eq("league_name", str(league_name))
            .execute()
        )
    except Exception:
        rows = []
    standings: list[dict[str, Any]] = []
    for row in rows:
        pid = _safe_int(row.get("player_id"))
        if pid is None:
            continue
        rating = _safe_float(row.get("rating"))
        standings.append(
            {
                "player_id": int(pid),
                "player_name": player_names.get(int(pid), f"Player {int(pid)}"),
                "rating": rating,
                "rating_jupr": None if rating is None else rating / 400.0,
                "starting_rating": row.get("starting_rating"),
                "wins": _safe_int(row.get("wins")) or 0,
                "losses": _safe_int(row.get("losses")) or 0,
                "matches_played": _safe_int(row.get("matches_played")) or 0,
                "is_active": bool(row.get("is_active", True)),
                "inactive_at": row.get("inactive_at"),
            }
        )
    standings.sort(key=lambda row: (float(row.get("rating") or 0), int(row.get("wins") or 0)), reverse=True)
    for rank, row in enumerate(standings, start=1):
        row["rank"] = rank
    return standings[: max(1, min(int(limit), 200))]


def build_admin_league_manager_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    if not is_admin_league_manager_enabled():
        return {
            "enabled": False,
            "status": "guarded_off",
            "leagues_endpoint": None,
            "league_detail_endpoint": None,
            "warnings": ["Next League Manager is disabled. Enable JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER on FastAPI for a closed-club pilot."],
        }
    leagues: list[dict[str, Any]] = []
    if supabase is not None:
        try:
            leagues = _fetch_league_rows(supabase, club_id=str(club_id))
        except Exception:
            leagues = []
    return {
        "enabled": True,
        "status": "ready_for_league_manager_read_foundation",
        "leagues_endpoint": "/admin/clubs/{club_id}/league-manager/leagues",
        "league_detail_endpoint": "/admin/clubs/{club_id}/league-manager/leagues/{league_name}",
        "league_count": len(leagues),
        "active_count": len([league for league in leagues if league.get("status") == "active"]),
        "warnings": ["Read-only foundation only. League setup, movement, score submission, and end-of-league awards remain Streamlit-only in this slice."],
    }


def list_admin_league_manager_leagues(supabase: Any, *, club_id: str) -> dict[str, Any]:
    if not is_admin_league_manager_enabled():
        raise PermissionError("Next League Manager is disabled.")
    leagues = _fetch_league_rows(supabase, club_id=str(club_id))
    return {"ok": True, "mode": "league_manager_list", "leagues": leagues, "count": len(leagues)}


def get_admin_league_manager_detail(supabase: Any, *, club_id: str, league_name: str) -> dict[str, Any]:
    if not is_admin_league_manager_enabled():
        raise PermissionError("Next League Manager is disabled.")
    clean_league = _clean_text(league_name, limit=120)
    leagues = _fetch_league_rows(supabase, club_id=str(club_id))
    league = next((row for row in leagues if row.get("league_name") == clean_league), None)
    if league is None:
        raise ValueError("league not found")
    standings = _league_standings(supabase, club_id=str(club_id), league_name=clean_league)
    return {
        "ok": True,
        "mode": "league_manager_detail",
        "league": league,
        "schedule_preview": _schedule_preview(league.get("schedule_config")),
        "standings": standings,
        "standings_count": len(standings),
    }
