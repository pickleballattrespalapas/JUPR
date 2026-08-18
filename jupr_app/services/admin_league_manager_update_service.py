from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timezone
from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.event_tags import derive_default_date_tags, normalize_event_tags
from jupr_app.services.admin_league_manager_service import (
    get_admin_league_manager_detail,
    is_admin_league_manager_enabled,
    validate_admin_league_manager_lifecycle_state,
)

CONFIRM_SAVE_LEAGUE = "SAVE LEAGUE"
TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
ACTIVE_SAFE_FIELDS = {"description"}
CLOSED_STATUSES = {"ended", "archived"}
AWARD_CATEGORY_KEYS = ("highest_rating", "most_improved", "best_win_pct", "most_wins")
MAX_CONFIG_JSON_BYTES = 50_000
LEAGUE_FORMATS = {
    "ladder",
    "round_robin",
    "rotating_partner",
    "fixed_team",
    "flex_challenge",
}
SESSION_MODES = {"scheduled_rounds", "live_court_board", "self_scheduled"}
STANDINGS_TIEBREAKS = {
    "wins_then_point_differential",
    "wins_then_total_points",
    "points_then_point_differential",
}
CORRECTION_WINDOWS = {"until_next_round", "same_day", "seven_days"}
SCORE_SUBMISSION_POLICIES = {"admin_only", "captain_or_admin", "rostered_player_or_admin"}
PLAYOFF_FORMATS = {"none", "single_elimination", "double_elimination"}


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
        numeric = float(value)
    except Exception as exc:
        raise ValueError(f"{field} must be a whole number.") from exc
    if isinstance(value, bool) or not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{field} must be a whole number.")
    parsed = int(numeric)
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


def _validate_json_shape(value: Any, *, field: str, depth: int = 0) -> None:
    if depth > 6:
        raise ValueError(f"{field} must not be nested more than 6 levels.")
    if isinstance(value, dict):
        if len(value) > 200:
            raise ValueError(f"{field} must contain at most 200 object keys per level.")
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 120:
                raise ValueError(f"{field} contains an invalid object key.")
            _validate_json_shape(item, field=field, depth=depth + 1)
        return
    if isinstance(value, list):
        if len(value) > 100:
            raise ValueError(f"{field} must contain at most 100 list values per level.")
        for item in value:
            _validate_json_shape(item, field=field, depth=depth + 1)
        return
    if isinstance(value, str):
        if len(value) > 5000:
            raise ValueError(f"{field} contains text longer than 5000 characters.")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field} contains a non-finite number.")
    if value is not None and not isinstance(value, (bool, int, float)):
        raise ValueError(f"{field} must contain JSON-compatible values.")


def _bounded_config(value: dict[str, Any], *, field: str) -> dict[str, Any]:
    _validate_json_shape(value, field=field)
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain JSON-compatible values.") from exc
    if len(encoded) > MAX_CONFIG_JSON_BYTES:
        raise ValueError(f"{field} must be at most {MAX_CONFIG_JSON_BYTES} bytes.")
    return value


def _config_text(value: Any, *, field: str, limit: int) -> str:
    if value in (None, ""):
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text.")
    return _clean_text(value, limit=limit)


def _date_text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = date.fromisoformat(text)
    except Exception as exc:
        raise ValueError(f"{field} must use YYYY-MM-DD format.") from exc
    if parsed.isoformat() != text:
        raise ValueError(f"{field} must use YYYY-MM-DD format.")
    return text


def _time_text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.strptime(text, "%H:%M")
    except Exception as exc:
        raise ValueError(f"{field} must use 24-hour HH:MM format.") from exc
    if parsed.strftime("%H:%M") != text:
        raise ValueError(f"{field} must use 24-hour HH:MM format.")
    return text


def _string_list(value: Any, *, field: str, max_items: int, item_limit: int) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list.")
    if len(value) > max_items:
        raise ValueError(f"{field} must contain at most {max_items} values.")
    normalized: list[str] = []
    for item in value:
        text = _config_text(item, field=field, limit=item_limit)
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def normalize_admin_league_schedule_config(value: Any) -> dict[str, Any]:
    obj = _json_object(value, field="schedule_config") or {}
    normalized = dict(obj)
    if "start_date" in obj:
        normalized["start_date"] = _date_text(obj.get("start_date"), field="schedule_config.start_date")
    if "weeks" in obj:
        weeks = _safe_int(obj.get("weeks"), field="schedule_config.weeks", minimum=0, maximum=260)
        normalized["weeks"] = weeks or None
    if "end_date" in obj:
        normalized["end_date"] = _date_text(obj.get("end_date"), field="schedule_config.end_date")
    if "weekday" in obj:
        normalized["weekday"] = _safe_int(obj.get("weekday"), field="schedule_config.weekday", minimum=0, maximum=6)
    if "time_start" in obj:
        normalized["time_start"] = _time_text(obj.get("time_start"), field="schedule_config.time_start")
    if "time_end" in obj:
        normalized["time_end"] = _time_text(obj.get("time_end"), field="schedule_config.time_end")
    if "timezone" in obj:
        timezone_name = _config_text(obj.get("timezone"), field="schedule_config.timezone", limit=80) or "UTC"
        if any(not (char.isalnum() or char in {"/", "_", "+", "-"}) for char in timezone_name):
            raise ValueError("schedule_config.timezone contains unsupported characters.")
        normalized["timezone"] = timezone_name
    if "blackout_dates" in obj:
        blackout_dates = _string_list(
            obj.get("blackout_dates"),
            field="schedule_config.blackout_dates",
            max_items=100,
            item_limit=10,
        )
        normalized["blackout_dates"] = [
            _date_text(item, field="schedule_config.blackout_dates") for item in blackout_dates
        ]
    if "session_capacity" in obj:
        capacity = _safe_int(
            obj.get("session_capacity"),
            field="schedule_config.session_capacity",
            minimum=0,
            maximum=1000,
        )
        normalized["session_capacity"] = capacity or None

    start_date = normalized.get("start_date") or ""
    end_date = normalized.get("end_date") or ""
    if start_date and end_date and date.fromisoformat(str(end_date)) < date.fromisoformat(str(start_date)):
        raise ValueError("schedule_config.end_date cannot be before start_date.")
    start_time = normalized.get("time_start") or ""
    end_time = normalized.get("time_end") or ""
    if start_time and end_time and str(end_time) <= str(start_time):
        raise ValueError("schedule_config.time_end must be after time_start.")
    return _bounded_config(normalized, field="schedule_config")


def _normalize_court_defaults(value: Any) -> dict[str, Any]:
    obj = _json_object(value, field="court_board_defaults") or {}
    normalized = dict(obj)
    for key in ("total_courts", "max_used_courts"):
        if key in obj:
            normalized[key] = _safe_int(
                obj.get(key),
                field=f"court_board_defaults.{key}",
                minimum=0,
                maximum=100,
            ) or 0
    if "court_identifiers" in obj:
        normalized["court_identifiers"] = _string_list(
            obj.get("court_identifiers"),
            field="court_board_defaults.court_identifiers",
            max_items=100,
            item_limit=40,
        )
    if "players_per_court" in obj:
        players_per_court = _config_text(
            obj.get("players_per_court"),
            field="court_board_defaults.players_per_court",
            limit=8,
        ) or "4"
        if players_per_court not in {"4", "5", "6+"}:
            raise ValueError("court_board_defaults.players_per_court must be 4, 5, or 6+.")
        normalized["players_per_court"] = players_per_court
    if "rotation_mode" in obj:
        rotation_mode = _config_text(
            obj.get("rotation_mode"),
            field="court_board_defaults.rotation_mode",
            limit=20,
        ) or "fixed"
        if rotation_mode not in {"fixed", "queue"}:
            raise ValueError("court_board_defaults.rotation_mode must be fixed or queue.")
        normalized["rotation_mode"] = rotation_mode
    if "game_format_points" in obj:
        normalized["game_format_points"] = _safe_int(
            obj.get("game_format_points"),
            field="court_board_defaults.game_format_points",
            minimum=1,
            maximum=99,
        )
    if "game_format_time" in obj:
        normalized["game_format_time"] = _safe_int(
            obj.get("game_format_time"),
            field="court_board_defaults.game_format_time",
            minimum=1,
            maximum=240,
        )
    total_courts = int(normalized.get("total_courts") or 0)
    max_used_courts = int(normalized.get("max_used_courts") or 0)
    if total_courts and max_used_courts > total_courts:
        raise ValueError("court_board_defaults.max_used_courts cannot exceed total_courts.")
    return _bounded_config(normalized, field="court_board_defaults")


def _choice(value: Any, *, field: str, options: set[str], default: str) -> str:
    clean = _config_text(value, field=field, limit=80).casefold() or default
    if clean not in options:
        raise ValueError(f"{field} must be one of: {', '.join(sorted(options))}.")
    return clean


def _normalize_match_structure(value: Any) -> dict[str, Any]:
    structure = _json_object(value, field="rules_config.competition.match_structure") or {}
    kind = _choice(
        structure.get("kind"),
        field="rules_config.competition.match_structure.kind",
        options={"fixed_games", "best_of"},
        default="fixed_games",
    )
    games = _safe_int(
        structure.get("games"),
        field="rules_config.competition.match_structure.games",
        minimum=1,
        maximum=9,
    )
    if games is None:
        games = 1
    if kind == "best_of" and (games < 3 or games % 2 == 0):
        raise ValueError("rules_config.competition.match_structure.best_of must use an odd game count of at least 3.")
    return {
        "kind": kind,
        "games": int(games),
        # Every completed pickleball game remains an official league game.
        # Best-of controls when play stops, not whether a played game counts.
        "result_counting": "each_game",
        "completion": "clinch" if kind == "best_of" else "all_games",
    }


def _validate_format_operation(*, league_format: str, session_mode: str) -> None:
    if league_format == "ladder" and session_mode == "self_scheduled":
        raise ValueError("Ladder leagues need scheduled rounds or a live court board.")
    if league_format == "flex_challenge" and session_mode != "self_scheduled":
        raise ValueError("Flex challenge leagues use self-scheduled play.")


def _normalize_rules_config(value: Any) -> dict[str, Any]:
    obj = _json_object(value, field="rules_config") or {}
    normalized = dict(obj)
    if "overview" in obj:
        overview = _json_object(obj.get("overview"), field="rules_config.overview") or {}
        clean_overview = dict(overview)
        if "league_type" in overview:
            clean_overview["league_type"] = _config_text(
                overview.get("league_type"), field="rules_config.overview.league_type", limit=80
            )
        if "divisions" in overview:
            clean_overview["divisions"] = _string_list(
                overview.get("divisions"),
                field="rules_config.overview.divisions",
                max_items=20,
                item_limit=80,
            )
        if "summary" in overview:
            clean_overview["summary"] = _config_text(
                overview.get("summary"), field="rules_config.overview.summary", limit=2000
            )
        if "league_format" in overview:
            clean_overview["league_format"] = _choice(
                overview.get("league_format"),
                field="rules_config.overview.league_format",
                options=LEAGUE_FORMATS,
                default="ladder",
            )
        normalized["overview"] = clean_overview
    if "competition" in obj:
        competition = _json_object(obj.get("competition"), field="rules_config.competition") or {}
        clean_competition = dict(competition)
        limits = {
            "scoring_rules": 2000,
            "match_format": 2000,
            "tie_break_rules": 2000,
            "dispute_window": 200,
            "dispute_policy": 500,
        }
        for key, limit in limits.items():
            if key in competition:
                clean_competition[key] = _config_text(
                    competition.get(key), field=f"rules_config.competition.{key}", limit=limit
                )
        if "scoring_profile" in competition:
            clean_competition["scoring_profile"] = _choice(
                competition.get("scoring_profile"),
                field="rules_config.competition.scoring_profile",
                options={"standard_pickleball"},
                default="standard_pickleball",
            )
        if "match_structure" in competition:
            clean_competition["match_structure"] = _normalize_match_structure(
                competition.get("match_structure")
            )
        if "standings_tiebreak" in competition:
            clean_competition["standings_tiebreak"] = _choice(
                competition.get("standings_tiebreak"),
                field="rules_config.competition.standings_tiebreak",
                options=STANDINGS_TIEBREAKS,
                default="wins_then_point_differential",
            )
        if "correction_window" in competition:
            clean_competition["correction_window"] = _choice(
                competition.get("correction_window"),
                field="rules_config.competition.correction_window",
                options=CORRECTION_WINDOWS,
                default="until_next_round",
            )
        if "score_submission_policy" in competition:
            clean_competition["score_submission_policy"] = _choice(
                competition.get("score_submission_policy"),
                field="rules_config.competition.score_submission_policy",
                options=SCORE_SUBMISSION_POLICIES,
                default="admin_only",
            )
        if "playoff_format" in competition:
            clean_competition["playoff_format"] = _choice(
                competition.get("playoff_format"),
                field="rules_config.competition.playoff_format",
                options=PLAYOFF_FORMATS,
                default="none",
            )
        normalized["competition"] = clean_competition
    if "operation" in obj:
        operation = _json_object(obj.get("operation"), field="rules_config.operation") or {}
        clean_operation = dict(operation)
        if "session_mode" in operation:
            clean_operation["session_mode"] = _choice(
                operation.get("session_mode"),
                field="rules_config.operation.session_mode",
                options=SESSION_MODES,
                default="scheduled_rounds",
            )
        for key in ("move_up_count", "move_down_count"):
            if key in operation:
                clean_operation[key] = _safe_int(
                    operation.get(key),
                    field=f"rules_config.operation.{key}",
                    minimum=0,
                    maximum=20,
                ) or 0
        normalized["operation"] = clean_operation
    overview = normalized.get("overview") if isinstance(normalized.get("overview"), dict) else {}
    operation = normalized.get("operation") if isinstance(normalized.get("operation"), dict) else {}
    if "league_format" in overview and "session_mode" in operation:
        _validate_format_operation(
            league_format=str(overview["league_format"]),
            session_mode=str(operation["session_mode"]),
        )
    return _bounded_config(normalized, field="rules_config")


def _award_depth(value: Any, *, field: str) -> int:
    depth = _safe_int(value, field=field, minimum=1, maximum=3)
    if depth not in {1, 3}:
        raise ValueError(f"{field} must be 1 or 3.")
    return int(depth)


def _normalize_awards_config(value: Any) -> dict[str, Any]:
    obj = _json_object(value, field="awards_config") or {}
    normalized = dict(obj)
    if "default_min_games" in obj:
        normalized["default_min_games"] = _safe_int(
            obj.get("default_min_games"), field="awards_config.default_min_games", minimum=0, maximum=1000
        ) or 0
    if "default_depth" in obj:
        normalized["default_depth"] = _award_depth(obj.get("default_depth"), field="awards_config.default_depth")
    if "categories" in obj:
        categories = _json_object(obj.get("categories"), field="awards_config.categories") or {}
        clean_categories = dict(categories)
        for key in AWARD_CATEGORY_KEYS:
            if key not in categories:
                continue
            category = _json_object(categories.get(key), field=f"awards_config.categories.{key}") or {}
            clean_category = dict(category)
            if "enabled" in category:
                if not isinstance(category.get("enabled"), bool):
                    raise ValueError(f"awards_config.categories.{key}.enabled must be true or false.")
                clean_category["enabled"] = bool(category.get("enabled"))
            if "min_games" in category:
                clean_category["min_games"] = _safe_int(
                    category.get("min_games"),
                    field=f"awards_config.categories.{key}.min_games",
                    minimum=0,
                    maximum=1000,
                ) or 0
            if "depth" in category:
                clean_category["depth"] = _award_depth(
                    category.get("depth"), field=f"awards_config.categories.{key}.depth"
                )
            clean_categories[key] = clean_category
        normalized["categories"] = clean_categories
    return _bounded_config(normalized, field="awards_config")


def _normalize_event_tags(value: Any) -> dict[str, Any]:
    obj = _json_object(value, field="event_tags") or {}
    return _bounded_config(normalize_event_tags(obj), field="event_tags")


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

    config_normalizers = {
        "schedule_config": normalize_admin_league_schedule_config,
        "court_board_defaults": _normalize_court_defaults,
        "rules_config": _normalize_rules_config,
        "awards_config": _normalize_awards_config,
        "event_tags": _normalize_event_tags,
    }
    for field, normalizer in config_normalizers.items():
        if field in patch and patch.get(field) is not None:
            normalized[field] = normalizer(patch.get(field))

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
    if "schedule_config" in normalized and "event_tags" not in normalized:
        schedule_config = normalized.get("schedule_config") or {}
        event_tags = normalize_event_tags(before.get("event_tags"))
        event_tags["date_tags"] = derive_default_date_tags(
            start_date=schedule_config.get("start_date"),
            end_date=schedule_config.get("end_date") or schedule_config.get("start_date"),
        )
        normalized["event_tags"] = _bounded_config(event_tags, field="event_tags")
    league_status = validate_admin_league_manager_lifecycle_state(before)
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
