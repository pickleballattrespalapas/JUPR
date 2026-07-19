from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any

from jupr_app.domain.leagues import normalize_league_status

TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
LEAGUE_MANAGER_EXTENDED_SELECT = "league_name,league_type,description,min_weeks,is_active,status,started_at,ended_at,ended_by,schedule_config,court_board_defaults,rules_config,awards_config,k_factor,min_games,event_tags"
LEAGUE_MANAGER_MINIMAL_SELECT = "league_name,is_active,k_factor,min_games"
LEAGUE_LIFECYCLE_ACTIONS = {
    "draft": ["start"],
    "active": ["pause", "end"],
    "paused": ["resume", "end"],
    "ended": ["archive"],
    "archived": [],
}


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


def _ical_escape(value: Any) -> str:
    text = str(value or "").replace("\\", "\\\\").replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("\n", "\\n").replace(";", "\\;").replace(",", "\\,")


def _ical_token(value: Any, *, fallback: str) -> str:
    token = "".join(char for char in str(value or "") if char.isalnum() or char in {"/", "_", "+", "-"})
    return token or fallback


def _ical_timezone(value: Any) -> str:
    raw_timezone = str(value or "UTC").strip()
    timezone_name = _ical_token(raw_timezone, fallback="UTC")
    if timezone_name != raw_timezone or len(timezone_name) > 80:
        return "UTC"
    return timezone_name


def _ical_time(value: Any, *, fallback: str) -> str:
    try:
        hour_text, minute_text = str(value or "").strip().split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}{minute:02d}00"
    except Exception:
        pass
    return fallback


def league_schedule_ics_filename(league_name: Any) -> str:
    stem = "".join(char.lower() if char.isalnum() else "-" for char in str(league_name or ""))
    stem = "-".join(part for part in stem.split("-") if part)[:80]
    return f"{stem or 'league'}-schedule.ics"


def build_league_schedule_ics(schedule_config: Any, *, league_name: Any) -> str:
    preview = _schedule_preview(schedule_config)
    if not preview:
        return ""
    cfg = _json_value(schedule_config, {}) or {}
    timezone_name = _ical_timezone(cfg.get("timezone"))
    summary = _ical_escape(league_name or "League Session")
    uid_stem = league_schedule_ics_filename(league_name).removesuffix("-schedule.ics")
    generated_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "PRODID:-//JUPR//League Schedule//EN",
        f"X-WR-CALNAME:{summary}",
        f"X-WR-TIMEZONE:{timezone_name}",
    ]
    for row in preview:
        date_token = str(row.get("date") or "").replace("-", "")
        if len(date_token) != 8 or not date_token.isdigit():
            continue
        session = int(row.get("session") or 0)
        start_time = _ical_time(row.get("start"), fallback="180000")
        end_time = _ical_time(row.get("end"), fallback="200000")
        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:{uid_stem}-{session}-{date_token}@jupr",
                f"DTSTAMP:{generated_at}",
                f"DTSTART;TZID={timezone_name}:{date_token}T{start_time}",
                f"DTEND;TZID={timezone_name}:{date_token}T{end_time}",
                f"SUMMARY:{summary}",
                f"DESCRIPTION:JUPR league session {session}",
                "END:VEVENT",
            ]
        )
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


def build_admin_league_schedule_preview(schedule_config: Any, *, league_name: Any) -> dict[str, Any]:
    clean_league = _clean_text(league_name, limit=120) or "League"
    preview = _schedule_preview(schedule_config)
    return {
        "ok": True,
        "mode": "league_manager_schedule_preview",
        "league_name": clean_league,
        "schedule_config": _json_value(schedule_config, {}) or {},
        "schedule_preview": preview,
        "schedule_ics": build_league_schedule_ics(schedule_config, league_name=clean_league),
        "schedule_ics_filename": league_schedule_ics_filename(clean_league),
    }


def _league_row_payload(row: dict[str, Any]) -> dict[str, Any]:
    league_name = _clean_text(row.get("league_name"), limit=120)
    return {
        "league_name": league_name,
        "league_type": _clean_text(row.get("league_type"), limit=80) or "Standard",
        "description": _clean_text(row.get("description"), limit=2000),
        "min_weeks": _safe_int(row.get("min_weeks")),
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


def _fetch_player_rows(supabase: Any, *, club_id: str) -> list[dict[str, Any]]:
    try:
        rows = _safe_rows(
            supabase.table("players")
            .select("id,name,active,last_game_at")
            .eq("club_id", str(club_id))
            .order("name", desc=False)
            .execute()
        )
    except Exception:
        rows = _safe_rows(supabase.table("players").select("id,name").eq("club_id", str(club_id)).order("name", desc=False).execute())
    return rows


def _fetch_player_names(supabase: Any, *, club_id: str) -> dict[int, str]:
    rows = _fetch_player_rows(supabase, club_id=str(club_id))
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


def _league_roster(supabase: Any, *, club_id: str, league_name: str, standings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    standing_by_player = {int(row["player_id"]): row for row in standings if row.get("player_id") is not None}
    roster: list[dict[str, Any]] = []
    for player in _fetch_player_rows(supabase, club_id=str(club_id)):
        pid = _safe_int(player.get("id"))
        if pid is None:
            continue
        league_row = standing_by_player.get(int(pid))
        rating = _safe_float((league_row or {}).get("rating"))
        league_active = bool((league_row or {}).get("is_active", False)) and not bool((league_row or {}).get("inactive_at")) if league_row else False
        roster.append(
            {
                "player_id": int(pid),
                "player_name": _clean_text(player.get("name"), limit=160) or f"Player {int(pid)}",
                "in_league": league_row is not None and league_active,
                "league_name": str(league_name),
                "rating": rating,
                "rating_jupr": None if rating is None else rating / 400.0,
                "wins": _safe_int((league_row or {}).get("wins")) or 0,
                "losses": _safe_int((league_row or {}).get("losses")) or 0,
                "matches_played": _safe_int((league_row or {}).get("matches_played")) or 0,
                "player_active": bool(player.get("active", True)),
                "league_active": league_active,
                "last_game_at": player.get("last_game_at"),
            }
        )
    return sorted(roster, key=lambda row: (not bool(row.get("in_league")), str(row.get("player_name") or "").lower()))


def admin_league_manager_lifecycle_state_error(league: dict[str, Any]) -> str | None:
    status = normalize_league_status(league)
    expected_active = status == "active"
    if bool(league.get("is_active", False)) == expected_active:
        return None
    return (
        f"League lifecycle state is inconsistent: status {status} requires "
        f"is_active={str(expected_active).lower()}."
    )


def validate_admin_league_manager_lifecycle_state(league: dict[str, Any]) -> str:
    error = admin_league_manager_lifecycle_state_error(league)
    if error:
        raise ValueError(error)
    return normalize_league_status(league)


def build_admin_league_manager_validation(
    league: dict[str, Any],
    *,
    roster: list[dict[str, Any]],
    schedule_preview: list[dict[str, Any]],
) -> dict[str, Any]:
    """Describe server-authoritative edit/lifecycle capabilities and state warnings."""

    status = normalize_league_status(league)
    errors: list[str] = []
    warnings: list[str] = []
    lifecycle_error = admin_league_manager_lifecycle_state_error(league)
    if lifecycle_error:
        errors.append(lifecycle_error)

    schedule_config = _json_value(league.get("schedule_config"), {}) or {}
    if schedule_config and not schedule_preview:
        errors.append("The saved schedule configuration does not produce any playable sessions.")
    if status in {"active", "paused"} and not league.get("started_at"):
        warnings.append("This running league has no recorded start timestamp.")
    if status in {"ended", "archived"} and not league.get("ended_at"):
        warnings.append("This closed league has no recorded end timestamp.")

    active_roster_count = len([row for row in roster if row.get("in_league")])
    if status in {"active", "paused"} and active_roster_count < 4:
        warnings.append("Fewer than four active roster members are available for doubles play.")
    if status == "draft" and active_roster_count == 0:
        warnings.append("Add roster members before opening a League Live session.")

    if status == "draft":
        settings_mode = "full"
    elif status in {"active", "paused"}:
        settings_mode = "description_only"
    else:
        settings_mode = "read_only"

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "capabilities": {
            "settings_mode": settings_mode,
            "roster_mutable": status in {"draft", "active", "paused"},
            "lifecycle_actions": list(LEAGUE_LIFECYCLE_ACTIONS.get(status, [])),
            "printable": True,
        },
    }


def build_admin_league_manager_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    if not is_admin_league_manager_enabled():
        return {
            "enabled": False,
            "status": "guarded_off",
            "leagues_endpoint": None,
            "league_create_endpoint": None,
            "league_duplicate_endpoint": None,
            "league_lifecycle_endpoint": None,
            "league_detail_endpoint": None,
            "league_settings_update_endpoint": None,
            "league_schedule_preview_endpoint": None,
            "league_roster_update_endpoint": None,
            "league_printout_endpoint": None,
            "top_players_printable_endpoint": None,
            "league_live_sessions_endpoint": None,
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
        "status": "ready_for_league_manager_roster_and_live_pilot",
        "leagues_endpoint": "/admin/clubs/{club_id}/league-manager/leagues",
        "league_create_endpoint": "/admin/clubs/{club_id}/league-manager/leagues",
        "league_duplicate_endpoint": "/admin/clubs/{club_id}/league-manager/leagues/{league_name}/duplicate",
        "league_lifecycle_endpoint": "/admin/clubs/{club_id}/league-manager/leagues/{league_name}/lifecycle",
        "league_detail_endpoint": "/admin/clubs/{club_id}/league-manager/leagues/{league_name}",
        "league_settings_update_endpoint": "/admin/clubs/{club_id}/league-manager/leagues/{league_name}",
        "league_schedule_preview_endpoint": "/admin/clubs/{club_id}/league-manager/leagues/{league_name}/schedule/preview",
        "league_roster_update_endpoint": "/admin/clubs/{club_id}/league-manager/leagues/{league_name}/roster/{player_id}",
        "league_printout_endpoint": "/admin/clubs/{club_id}/league-manager/leagues/{league_name}/printout",
        "top_players_printable_endpoint": "/admin/clubs/{club_id}/league-manager/top-players-printable",
        "league_live_sessions_endpoint": "/admin/clubs/{club_id}/league-manager/live-sessions",
        "league_count": len(leagues),
        "active_count": len([league for league in leagues if league.get("status") == "active"]),
        "warnings": [
            "Settings, explicit lifecycle actions, roster membership, Python-authoritative leader exports, "
            "browser-print output, and persisted League Live sessions are enabled through guarded FastAPI. "
            "Corrections still route through Match Log/Replay History."
        ],
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
    roster = _league_roster(supabase, club_id=str(club_id), league_name=clean_league, standings=standings)
    schedule_config = league.get("schedule_config")
    schedule_preview = _schedule_preview(schedule_config)
    validation = build_admin_league_manager_validation(
        league,
        roster=roster,
        schedule_preview=schedule_preview,
    )
    return {
        "ok": True,
        "mode": "league_manager_detail",
        "league": league,
        "schedule_preview": schedule_preview,
        "schedule_ics": build_league_schedule_ics(schedule_config, league_name=clean_league),
        "schedule_ics_filename": league_schedule_ics_filename(clean_league),
        "standings": standings,
        "standings_count": len(standings),
        "roster": roster,
        "roster_count": len(roster),
        "league_roster_count": len([row for row in roster if row.get("in_league")]),
        "validation": validation,
        "capabilities": validation["capabilities"],
    }
