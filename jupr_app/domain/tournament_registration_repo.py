from __future__ import annotations

import json
import math
import re
import uuid
from datetime import date, datetime, timezone
from typing import Any

from jupr_app.domain.event_tags import derive_default_date_tags, normalize_event_tags
from jupr_app.domain.tournament_public_references import build_public_tournament_reference

from .tournament_registration_compiler import compile_tournament_registration_state, validate_selection_against_skill

REGISTRATION_STATUS_OPTIONS = ["draft", "open", "closed"]
EVENT_TYPE_OPTIONS = ["SINGLES", "GENDER_DOUBLES", "MIXED_DOUBLES"]
GENDER_RESTRICTION_OPTIONS = ["ANY", "MEN", "WOMEN", "MIXED"]
PARTNER_MODE_OPTIONS = ["NONE", "HAS_PARTNER", "NEEDS_PARTNER"]
ADMIN_REGISTRATION_STATUS_OPTIONS = ["confirmed", "waitlist", "cancelled"]
ADMIN_PAYMENT_STATUS_OPTIONS = ["unpaid", "paid", "refunded"]


class SelectionWriteConflict(RuntimeError):
    """Raised when a concurrent selection change makes an admin write unsafe."""


class StaleTournamentRegistrationSelectionError(SelectionWriteConflict):
    """Backward-compatible name for an admin selection write conflict."""


class StaleTournamentRegistrationAdminError(SelectionWriteConflict):
    """An admin registration write lost its compare-and-swap version."""


class TournamentRegistrationEditConflictError(ValueError):
    """Raised when a token-gated edit was based on a stale registration view."""


class TournamentRegistrationImportedDrawError(ValueError):
    """Raised when a registration is locked because its event reached a draw."""


class TournamentRegistrationRelationshipLockedError(ValueError):
    """Raised when an edit would invalidate an active partner relationship."""


ADMIN_SELECTION_UPDATE_RPC = "admin_update_tournament_registration_selection"
PUBLIC_REGISTRATION_EDIT_RPC = "server_update_public_tournament_registration_edit"
PUBLIC_REGISTRATION_COMMERCE_CREATE_RPC = (
    "server_create_public_tournament_registration_with_commerce"
)
PUBLIC_REGISTRATION_COMMERCE_EDIT_RPC = (
    "server_update_public_tournament_registration_with_commerce"
)
SELECTION_WRITE_CONFLICT_CODE = "SELECTION_WRITE_CONFLICT"
SELECTION_WRITE_CONFLICT_MARKER = "JUPR_SELECTION_WRITE_CONFLICT"
SELECTION_NOT_FOUND_CODE = "SELECTION_NOT_FOUND"
RELATION_SELECTION_NOT_FOUND_MARKER = "JUPR_RELATION_SELECTION_NOT_FOUND"
SELECTION_INVALID_TARGET_MARKER = "JUPR_SELECTION_INVALID_TARGET"
SELECTION_INVALID_PATCH_MARKER = "JUPR_SELECTION_INVALID_PATCH"
REGISTRATION_EDIT_CONFLICT_CODE = "REGISTRATION_EDIT_CONFLICT"
REGISTRATION_EDIT_CONFLICT_MARKER = "JUPR_REGISTRATION_EDIT_CONFLICT"
REGISTRATION_EDIT_IMPORTED_CODE = "REGISTRATION_IMPORTED_TO_DRAW"
REGISTRATION_EDIT_IMPORTED_MARKER = "JUPR_REGISTRATION_IMPORTED_TO_DRAW"
REGISTRATION_EDIT_RELATIONSHIP_CODE = "REGISTRATION_RELATIONSHIP_LOCKED"
REGISTRATION_EDIT_RELATIONSHIP_MARKER = "JUPR_REGISTRATION_RELATIONSHIP_LOCKED"
REGISTRATION_COMMERCE_FULFILLED_CANCEL_MARKER = (
    "JUPR_TOURNAMENT_COMMERCE_FULFILLED_REGISTRATION_CANCEL_LOCKED"
)


REGISTRATION_SCHEMA_CONTRACT_MIGRATIONS = [
    "migrations/20261010_tournament_builder_refactor.sql",
    "migrations/20261018_tournament_registration_schema_contract.sql",
    "migrations/20261019_tournament_registration_partner_links.sql",
]

CORE_REGISTRATION_SCHEMA_TABLES = [
    "tournament_registration_settings",
    "tournament_registration_days",
    "tournament_event_options",
    "tournament_registrations",
    "tournament_registration_selections",
]

PARTNER_LINK_SCHEMA_TABLES = [
    "tournament_registration_partner_requests",
    "tournament_registration_team_links",
    "tournament_registration_team_members",
]

FOUR_PLAYER_TEAM_SCHEMA_TABLES = [
    "tournament_four_player_teams",
    "tournament_four_player_team_members",
]

REGISTRATION_SCHEMA_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "tournament_registration_settings": (
        "id",
        "tournament_id",
        "builder_draft_json",
        "builder_draft_updated_at",
        "location_name",
        "timezone",
        "sponsors_json",
        "weather_policy_markdown",
    ),
    "tournament_registration_days": (
        "id",
        "tournament_id",
        "enabled",
        "court_count",
        "court_labels",
        "court_open_time",
        "court_close_time",
        "court_notes",
    ),
    "tournament_event_options": (
        "id",
        "tournament_id",
        "registration_day_id",
        "scheduled_day_ids",
        "event_family_label",
        "division_name",
        "event_format_default",
        "scoring_default",
        "event_format_override",
        "scoring_override",
        "skill_mode",
        "age_mode",
        "age_rules",
        "waitlist_enabled",
        "partner_board_enabled",
        "status",
        "enabled",
    ),
    "tournament_registrations": (
        "id",
        "tournament_id",
    ),
    "tournament_registrations.player_id": (
        "player_id",
    ),
    "tournament_registration_partner_requests": (
        "id",
        "tournament_id",
        "event_option_id",
        "requester_selection_id",
        "requester_registration_id",
        "requester_player_id",
        "target_selection_id",
        "target_registration_id",
        "target_player_id",
        "target_display_name_snapshot",
        "status",
        "source",
        "created_at",
        "updated_at",
        "responded_at",
        "created_by_registration_id",
        "created_by_user_id",
    ),
    "tournament_registration_team_links": (
        "id",
        "tournament_id",
        "event_option_id",
        "registration1_id",
        "registration2_id",
        "selection1_id",
        "selection2_id",
        "player1_id",
        "player2_id",
        "status",
        "accepted_request_id",
        "created_at",
        "updated_at",
        "created_by_user_id",
    ),
    "tournament_registration_team_members": (
        "id",
        "team_link_id",
        "tournament_id",
        "event_option_id",
        "selection_id",
        "registration_id",
        "player_id",
        "player_order",
        "status",
        "created_at",
    ),
}

PUBLIC_EVENT_STATUS_VISIBLE = {"open", "tentative", "confirmed", "closed"}
PUBLIC_EVENT_STATUS_SELECTABLE = {"open", "tentative", "confirmed"}


def _with_normalized_event_tags(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return row
    tags = normalize_event_tags(row.get("event_tags"))
    if not tags.get("date_tags"):
        tags = normalize_event_tags({
            "skill_levels": tags.get("skill_levels"),
            "date_tags": derive_default_date_tags(start_date=row.get("start_date"), end_date=row.get("end_date")),
        })
    return {**row, "event_tags": tags}

def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _safe_data(resp: Any) -> list[dict[str, Any]]:
    try:
        return list(resp.data or [])
    except Exception:
        return []


def _safe_first(resp: Any) -> dict[str, Any] | None:
    rows = _safe_data(resp)
    return rows[0] if rows else None


def _rpc_object(resp: Any) -> dict[str, Any] | None:
    """Return one JSON object from a PostgREST RPC response."""
    data = getattr(resp, "data", None)
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        return data[0]
    return None


def _database_error_contains(exc: Exception, marker: str) -> bool:
    """Match a stable marker across postgrest-py exception representations."""
    values: list[str] = [str(exc or "")]
    for attr in ("code", "message", "details", "hint"):
        value = getattr(exc, attr, None)
        if value is not None:
            values.append(str(value))
    for arg in getattr(exc, "args", ()):
        if isinstance(arg, dict):
            values.extend(str(value) for value in arg.values() if value is not None)
        elif arg is not None:
            values.append(str(arg))
    return str(marker).upper() in "\n".join(values).upper()


def _normalize_email(value: Any) -> str:
    return str(value or "").strip().lower()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or f"tournament-{uuid.uuid4().hex[:6]}"


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_nan_like(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float):
        return math.isnan(value)

    value_type = type(value)
    type_name = value_type.__name__
    type_module = value_type.__module__
    if type_module.startswith("pandas") and type_name in {"NAType", "NaTType"}:
        return True
    if type_module.startswith("pandas") and str(value) == "NaT":
        return True

    return False


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if _is_nan_like(value):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _is_missing_column_error(exc: Exception, column_name: str, table_name: str) -> bool:
    text = str(exc or "")
    return (
        "PGRST204" in text
        and f"'{column_name}'" in text
        and f"'{table_name}'" in text
    )


def _schema_contract_error_message(missing: list[str]) -> str:
    missing_text = "; ".join(sorted(set(missing)))
    migration_text = " then ".join(REGISTRATION_SCHEMA_CONTRACT_MIGRATIONS)
    return (
        "Tournament registration schema contract check failed. "
        f"Missing required columns: {missing_text}. "
        f"Run {migration_text}, then refresh the app."
    )


def _schema_check_table_name(contract_key: str) -> str:
    return contract_key.split(".", 1)[0]


def assert_registration_schema_contract(supabase, *, required_tables: list[str] | None = None) -> None:
    tables = required_tables or list(REGISTRATION_SCHEMA_REQUIRED_COLUMNS.keys())
    failures: list[str] = []
    for contract_key in tables:
        required_columns = REGISTRATION_SCHEMA_REQUIRED_COLUMNS.get(contract_key, ())
        if not required_columns:
            continue
        table_name = _schema_check_table_name(contract_key)
        select_expr = ", ".join(required_columns)
        try:
            supabase.table(table_name).select(select_expr).limit(1).execute()
        except Exception as exc:
            failures.append(f"{contract_key} ({select_expr}): {exc}")
    if failures:
        raise ValueError(_schema_contract_error_message(failures))


def is_day_enabled(day: dict[str, Any]) -> bool:
    return _coerce_bool((day or {}).get("enabled", True))


def public_event_option_visibility(event: dict[str, Any]) -> str:
    if not _coerce_bool((event or {}).get("enabled", True)):
        return "hidden"
    status = str((event or {}).get("status") or "draft").strip().lower()
    if status in PUBLIC_EVENT_STATUS_SELECTABLE:
        return "selectable"
    if status in PUBLIC_EVENT_STATUS_VISIBLE:
        return "visible_blocked"
    return "hidden"


def _insert_registration_days(supabase, days: list[dict[str, Any]]) -> None:
    if not days:
        return
    supabase.table("tournament_registration_days").insert(days).execute()


def _tables_available(supabase, required_tables: list[str]) -> tuple[bool, str | None]:
    failures: list[str] = []
    for table_name in required_tables:
        try:
            supabase.table(table_name).select("id").limit(1).execute()
        except Exception as exc:
            failures.append(f"{table_name}: {exc}")
    if failures:
        return False, "Registration tables unavailable: " + " | ".join(failures)
    return True, None


def registration_feature_available(supabase) -> tuple[bool, str | None]:
    available, detail = _tables_available(supabase, CORE_REGISTRATION_SCHEMA_TABLES)
    if not available:
        return False, detail

    try:
        assert_registration_schema_contract(supabase, required_tables=CORE_REGISTRATION_SCHEMA_TABLES)
    except ValueError as exc:
        return False, str(exc)
    return True, None


def partner_link_schema_available(supabase) -> tuple[bool, str | None]:
    available, detail = _tables_available(supabase, PARTNER_LINK_SCHEMA_TABLES)
    if not available:
        return False, detail

    try:
        assert_registration_schema_contract(
            supabase,
            required_tables=["tournament_registrations.player_id", *PARTNER_LINK_SCHEMA_TABLES],
        )
    except ValueError as exc:
        return False, str(exc)
    return True, None


def four_player_team_schema_available(supabase) -> tuple[bool, str | None]:
    return _tables_available(supabase, FOUR_PLAYER_TEAM_SCHEMA_TABLES)


def list_existing_tournaments(supabase, club_id: str, *, include_archived: bool = False) -> list[dict[str, Any]]:
    query = (
        supabase.table("tournaments")
        .select("*")
        .eq("club_id", str(club_id))
    )
    if not include_archived:
        query = query.neq("status", "ARCHIVED")
    resp = query.order("created_at", desc=True).execute()
    return [_with_normalized_event_tags(row) or row for row in _safe_data(resp)]


def _count_table_rows(supabase, table_name: str, tournament_id: str) -> int:
    resp = (
        supabase.table(table_name)
        .select("id", count="exact")
        .eq("tournament_id", str(tournament_id))
        .execute()
    )
    try:
        return int(resp.count or 0)
    except Exception:
        return len(_safe_data(resp))


def get_tournament_usage_summary(supabase, tournament_id: str) -> dict[str, int]:
    tournament_id = str(tournament_id)
    return {
        "registrations": _count_table_rows(supabase, "tournament_registrations", tournament_id),
        "registration_selections": _count_table_rows(supabase, "tournament_registration_selections", tournament_id),
        "event_draws": _count_table_rows(supabase, "tournament_event_draws", tournament_id),
        "teams": _count_table_rows(supabase, "tournament_teams", tournament_id),
        "games": _count_table_rows(supabase, "tournament_games", tournament_id),
        "podium": _count_table_rows(supabase, "tournament_podium", tournament_id),
    }


def tournament_can_be_deleted(supabase, tournament: dict[str, Any]) -> tuple[bool, dict[str, int], str | None]:
    status = str((tournament or {}).get("status") or "").upper()
    tournament_id = str((tournament or {}).get("id") or "").strip()
    summary = get_tournament_usage_summary(supabase, tournament_id) if tournament_id else {}
    if status != "DRAFT":
        return False, summary, "Delete Draft is only available when tournament status is DRAFT."
    if any(int(summary.get(key) or 0) > 0 for key in ["registrations", "registration_selections", "event_draws", "teams", "games", "podium"]):
        return False, summary, "This tournament has existing operational/history records. Archive it instead of deleting."
    return True, summary, None


def archive_tournament(supabase, tournament_id: str) -> None:
    supabase.table("tournaments").update({"status": "ARCHIVED"}).eq("id", str(tournament_id)).execute()


def unarchive_tournament(supabase, tournament_id: str) -> None:
    supabase.table("tournaments").update({"status": "DRAFT"}).eq("id", str(tournament_id)).execute()


def delete_unused_draft_tournament(supabase, tournament: dict[str, Any]) -> None:
    tournament_id = str((tournament or {}).get("id") or "").strip()
    if not tournament_id:
        raise ValueError("Tournament id is required.")

    can_delete, _, reason = tournament_can_be_deleted(supabase, tournament)
    if not can_delete:
        raise ValueError(reason or "Tournament cannot be deleted.")

    (
        supabase.table("tournament_registration_settings")
        .delete()
        .eq("tournament_id", tournament_id)
        .execute()
    )
    (
        supabase.table("tournament_event_options")
        .delete()
        .eq("tournament_id", tournament_id)
        .execute()
    )
    (
        supabase.table("tournament_registration_days")
        .delete()
        .eq("tournament_id", tournament_id)
        .execute()
    )
    supabase.table("tournaments").delete().eq("id", tournament_id).execute()


def get_tournament_record(supabase, tournament_id: str) -> dict[str, Any] | None:
    resp = (
        supabase.table("tournaments")
        .select("*")
        .eq("id", str(tournament_id))
        .limit(1)
        .execute()
    )
    return _with_normalized_event_tags(_safe_first(resp))


def get_registration_settings(supabase, tournament_id: str, *, tournament_name: str | None = None) -> dict[str, Any]:
    resp = (
        supabase.table("tournament_registration_settings")
        .select("*")
        .eq("tournament_id", str(tournament_id))
        .limit(1)
        .execute()
    )
    row = _safe_first(resp)
    if row:
        return row
    return {
        "id": _uid("regset"),
        "tournament_id": str(tournament_id),
        "registration_slug": _slugify(tournament_name or str(tournament_id)),
        "locale": "en",
        "registration_status": "draft",
        "registration_open_at": None,
        "registration_close_at": None,
        "waitlist_enabled": True,
        "partner_board_enabled": True,
        "rules_markdown": "",
        "refund_policy_markdown": "",
        "weather_policy_markdown": "",
        "sponsor_markdown": "",
        "location_name": "",
        "timezone": "America/Mazatlan",
        "sponsors_json": [],
    }


def upsert_registration_settings(supabase, payload: dict[str, Any]) -> dict[str, Any]:
    clean = {
        "id": str(payload.get("id") or _uid("regset")),
        "tournament_id": str(payload.get("tournament_id")),
        "registration_slug": _slugify(str(payload.get("registration_slug") or payload.get("tournament_id") or "tournament")),
        "locale": str(payload.get("locale") or "en"),
        "registration_status": str(payload.get("registration_status") or "draft").lower(),
        "registration_open_at": payload.get("registration_open_at") or None,
        "registration_close_at": payload.get("registration_close_at") or None,
        "waitlist_enabled": _coerce_bool(payload.get("waitlist_enabled", True)),
        "partner_board_enabled": _coerce_bool(payload.get("partner_board_enabled", True)),
        "rules_markdown": str(payload.get("rules_markdown") or ""),
        "refund_policy_markdown": str(payload.get("refund_policy_markdown") or ""),
        "weather_policy_markdown": str(payload.get("weather_policy_markdown") or ""),
        "sponsor_markdown": str(payload.get("sponsor_markdown") or ""),
        "location_name": str(payload.get("location_name") or "").strip(),
        "timezone": str(payload.get("timezone") or "America/Mazatlan").strip(),
        "sponsors_json": _json_safe_value(
            payload.get("sponsors_json")
            if isinstance(payload.get("sponsors_json"), list)
            else []
        ),
        "updated_at": _now_iso(),
    }
    resp = (
        supabase.table("tournament_registration_settings")
        .upsert(clean, on_conflict="tournament_id")
        .execute()
    )
    return _safe_first(resp) or clean


def build_builder_draft_payload(
    *,
    days: list[dict[str, Any]],
    event_families: list[dict[str, Any]],
    divisions: list[dict[str, Any]],
    basics: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
    saved_step: str | None = None,
) -> dict[str, Any]:
    payload = {
        "version": 2,
        "saved_at": _now_iso(),
        "saved_step": str(saved_step or "").strip() or None,
        "days": list(days or []),
        "event_families": list(event_families or []),
        "divisions": list(divisions or []),
        "basics": dict(basics or {}),
        "settings": dict(settings or {}),
    }
    return _json_safe_value(payload)


def get_builder_draft(supabase, tournament_id: str) -> dict[str, Any] | None:
    settings = get_registration_settings(supabase, str(tournament_id))
    raw_draft = settings.get("builder_draft_json")
    if isinstance(raw_draft, dict):
        return raw_draft
    return None


def save_builder_draft(
    supabase,
    *,
    tournament_id: str,
    days: list[dict[str, Any]],
    event_families: list[dict[str, Any]],
    divisions: list[dict[str, Any]],
    basics: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
    saved_step: str | None = None,
) -> dict[str, Any]:
    assert_registration_schema_contract(supabase, required_tables=["tournament_registration_settings"])
    current_settings = get_registration_settings(supabase, str(tournament_id))
    payload = build_builder_draft_payload(
        days=days,
        event_families=event_families,
        divisions=divisions,
        basics=basics,
        settings=dict(settings or {}),
        saved_step=saved_step,
    )
    update_payload = {
        "id": str(current_settings.get("id") or _uid("regset")),
        "tournament_id": str(tournament_id),
        "builder_draft_json": payload,
        "builder_draft_updated_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    (
        supabase.table("tournament_registration_settings")
        .upsert(update_payload, on_conflict="tournament_id")
        .execute()
    )
    return payload


def clear_builder_draft(supabase, tournament_id: str) -> None:
    try:
        (
            supabase.table("tournament_registration_settings")
            .update(
                {
                    "builder_draft_json": None,
                    "builder_draft_updated_at": None,
                    "updated_at": _now_iso(),
                }
            )
            .eq("tournament_id", str(tournament_id))
            .execute()
        )
    except Exception as exc:
        if _is_missing_column_error(exc, "builder_draft_json", "tournament_registration_settings"):
            return
        raise


def list_registration_days(supabase, tournament_id: str) -> list[dict[str, Any]]:
    resp = (
        supabase.table("tournament_registration_days")
        .select("*")
        .eq("tournament_id", str(tournament_id))
        .order("sort_order")
        .execute()
    )
    return _safe_data(resp)


def list_event_options(supabase, tournament_id: str) -> list[dict[str, Any]]:
    resp = (
        supabase.table("tournament_event_options")
        .select("*")
        .eq("tournament_id", str(tournament_id))
        .order("sort_order")
        .execute()
    )
    return _safe_data(resp)


def count_tournament_registrations(supabase, tournament_id: str) -> int:
    resp = (
        supabase.table("tournament_registrations")
        .select("id", count="exact")
        .eq("tournament_id", str(tournament_id))
        .execute()
    )
    try:
        return int(resp.count or 0)
    except Exception:
        rows = _safe_data(resp)
        return len(rows)


def list_registration_usage_by_event_option(supabase, tournament_id: str) -> dict[str, dict[str, int]]:
    tournament_id = str(tournament_id)
    usage: dict[str, dict[str, int]] = {}

    selection_rows = _safe_data(
        supabase.table("tournament_registration_selections")
        .select("event_option_id")
        .eq("tournament_id", tournament_id)
        .execute()
    )
    for row in selection_rows:
        event_option_id = str(row.get("event_option_id") or "").strip()
        if not event_option_id:
            continue
        bucket = usage.setdefault(event_option_id, {"registrations": 0, "event_draws": 0, "teams": 0, "games": 0})
        bucket["registrations"] += 1

    for table_name, key in [
        ("tournament_event_draws", "event_draws"),
        ("tournament_teams", "teams"),
        ("tournament_games", "games"),
    ]:
        try:
            rows = _safe_data(
                supabase.table(table_name)
                .select("event_option_id")
                .eq("tournament_id", tournament_id)
                .execute()
            )
        except Exception:
            rows = []
        for row in rows:
            event_option_id = str(row.get("event_option_id") or "").strip()
            if not event_option_id:
                continue
            bucket = usage.setdefault(event_option_id, {"registrations": 0, "event_draws": 0, "teams": 0, "games": 0})
            bucket[key] += 1

    return usage


def list_registration_usage_by_day(supabase, tournament_id: str) -> dict[str, dict[str, int]]:
    tournament_id = str(tournament_id)
    usage: dict[str, dict[str, int]] = {}

    selection_rows = _safe_data(
        supabase.table("tournament_registration_selections")
        .select("registration_day_id")
        .eq("tournament_id", tournament_id)
        .execute()
    )
    for row in selection_rows:
        day_id = str(row.get("registration_day_id") or "").strip()
        if not day_id:
            continue
        bucket = usage.setdefault(day_id, {"registrations": 0, "event_draws": 0, "teams": 0, "games": 0})
        bucket["registrations"] += 1

    for table_name, key in [
        ("tournament_event_draws", "event_draws"),
        ("tournament_teams", "teams"),
        ("tournament_games", "games"),
    ]:
        try:
            rows = _safe_data(
                supabase.table(table_name)
                .select("registration_day_id")
                .eq("tournament_id", tournament_id)
                .execute()
            )
        except Exception:
            rows = []
        for row in rows:
            day_id = str(row.get("registration_day_id") or "").strip()
            if not day_id:
                continue
            bucket = usage.setdefault(day_id, {"registrations": 0, "event_draws": 0, "teams": 0, "games": 0})
            bucket[key] += 1

    return usage


def _entity_usage_total(usage: dict[str, int] | None) -> int:
    if not usage:
        return 0
    return int(usage.get("registrations", 0) or 0) + int(usage.get("event_draws", 0) or 0) + int(usage.get("teams", 0) or 0) + int(usage.get("games", 0) or 0)


def _event_identity_key(event: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(event.get("event_family_label") or "").strip().lower(),
        str(event.get("division_name") or event.get("label") or "").strip().lower(),
        str(event.get("event_type") or "").strip().upper(),
        str(event.get("gender_restriction") or "").strip().upper(),
        str(event.get("skill_label") or "").strip().lower(),
        str(event.get("age_label") or "").strip().lower(),
    )


DAY_CONFIGURATION_WRITE_FIELDS = {
    "id",
    "tournament_id",
    "sort_order",
    "label",
    "event_date",
    "court_count",
    "court_labels",
    "court_open_time",
    "court_close_time",
    "court_notes",
    "enabled",
}
EVENT_CONFIGURATION_WRITE_FIELDS = {
    "id",
    "tournament_id",
    "registration_day_id",
    "scheduled_day_ids",
    "sort_order",
    "label",
    "event_type",
    "gender_restriction",
    "skill_label",
    "age_label",
    "partner_required",
    "capacity_teams",
    "public_partner_board",
    "price_usd",
    "event_family_label",
    "division_name",
    "event_format_default",
    "scoring_default",
    "event_format_override",
    "scoring_override",
    "skill_mode",
    "age_mode",
    "age_rules",
    "waitlist_enabled",
    "partner_board_enabled",
    "status",
    "enabled",
}


def _deterministic_configuration_id(
    *,
    tournament_id: str,
    kind: str,
    index: int,
    row: dict[str, Any],
) -> str:
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)
    return f"{kind}_{uuid.uuid5(uuid.NAMESPACE_URL, f'jupr:{tournament_id}:{kind}:{index}:{canonical}').hex[:12]}"


def normalize_registration_configuration_payload(
    *,
    tournament_id: str,
    days: list[dict[str, Any]],
    event_options: list[dict[str, Any]],
    published_events: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return deterministic, tournament-scoped rows safe for service-role writes."""

    tournament_id = str(tournament_id)
    raw_days = _json_safe_value(list(days or []))
    raw_events = _json_safe_value(list(event_options or []))
    normalized_days: list[dict[str, Any]] = []
    day_aliases: dict[str, str] = {}
    day_ids: set[str] = set()
    for index, raw in enumerate(raw_days, start=1):
        supplied_tournament_id = str(raw.get("tournament_id") or "").strip()
        if supplied_tournament_id and supplied_tournament_id != tournament_id:
            raise ValueError(f"Invalid day payload at row {index}: tournament_id mismatch.")
        row_id = str(raw.get("id") or "").strip() or _deterministic_configuration_id(
            tournament_id=tournament_id,
            kind="day",
            index=index,
            row=raw,
        )
        if row_id in day_ids:
            raise ValueError(f"Invalid day payload at row {index}: duplicate id '{row_id}'.")
        day_ids.add(row_id)
        day_aliases[f"day_{index}"] = row_id
        clean = {key: raw.get(key) for key in DAY_CONFIGURATION_WRITE_FIELDS if key in raw}
        clean.update(
            {
                "id": row_id,
                "tournament_id": tournament_id,
                "sort_order": raw.get("sort_order") or index,
                "label": str(raw.get("label") or f"Day {index}"),
                "enabled": _coerce_bool(raw.get("enabled", True)),
            }
        )
        event_date = raw.get("event_date") or raw.get("date") or raw.get("start_date")
        if event_date not in (None, ""):
            clean["event_date"] = event_date
        normalized_days.append(clean)

    published_ids_by_identity = {
        _event_identity_key(row): str(row.get("id"))
        for row in (published_events or [])
        if str(row.get("id") or "").strip()
    }
    normalized_events: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    for index, raw in enumerate(raw_events, start=1):
        supplied_tournament_id = str(raw.get("tournament_id") or "").strip()
        if supplied_tournament_id and supplied_tournament_id != tournament_id:
            raise ValueError(f"Invalid event payload at row {index}: tournament_id mismatch.")
        row_id = str(raw.get("id") or "").strip()
        if not row_id:
            row_id = published_ids_by_identity.get(_event_identity_key(raw)) or _deterministic_configuration_id(
                tournament_id=tournament_id,
                kind="event",
                index=index,
                row=raw,
            )
        if row_id in event_ids:
            raise ValueError(f"Invalid event payload at row {index}: duplicate id '{row_id}'.")
        event_ids.add(row_id)
        raw_scheduled = raw.get("scheduled_day_ids")
        scheduled_day_ids = [
            day_aliases.get(str(value or "").strip(), str(value or "").strip())
            for value in (raw_scheduled if isinstance(raw_scheduled, list) else [])
            if str(value or "").strip()
        ]
        registration_day_id = str(raw.get("registration_day_id") or "").strip()
        registration_day_id = day_aliases.get(registration_day_id, registration_day_id)
        if not scheduled_day_ids and registration_day_id:
            scheduled_day_ids = [registration_day_id]
        if not scheduled_day_ids and len(normalized_days) == 1:
            scheduled_day_ids = [str(normalized_days[0]["id"])]
        scheduled_day_ids = list(dict.fromkeys(scheduled_day_ids))
        invalid_scheduled = [day_id for day_id in scheduled_day_ids if day_id not in day_ids]
        if invalid_scheduled:
            raise ValueError(
                f"Invalid event payload at row {index}: scheduled day '{invalid_scheduled[0]}' is not present in day payload."
            )
        registration_day_id = scheduled_day_ids[0] if scheduled_day_ids else registration_day_id
        if registration_day_id not in day_ids:
            raise ValueError(
                f"Invalid event payload at row {index}: registration_day_id '{registration_day_id}' is not present in day payload."
            )
        clean = {key: raw.get(key) for key in EVENT_CONFIGURATION_WRITE_FIELDS if key in raw}
        event_type = str(raw.get("event_type") or "").strip().upper()
        clean.update(
            {
                "id": row_id,
                "tournament_id": tournament_id,
                "registration_day_id": registration_day_id,
                "scheduled_day_ids": scheduled_day_ids,
                "sort_order": raw.get("sort_order") or index,
                "label": str(raw.get("label") or raw.get("division_name") or raw.get("event_family_label") or f"Event {index}"),
                "event_type": event_type,
                "gender_restriction": str(raw.get("gender_restriction") or "ANY").strip().upper(),
                "partner_required": _coerce_bool(raw.get("partner_required", event_type != "SINGLES")),
                "public_partner_board": _coerce_bool(raw.get("public_partner_board", raw.get("partner_board_enabled", True))),
                "waitlist_enabled": _coerce_bool(raw.get("waitlist_enabled", True)),
                "partner_board_enabled": _coerce_bool(raw.get("partner_board_enabled", True)),
                "enabled": _coerce_bool(raw.get("enabled", True)),
                "status": str(raw.get("status") or "draft").strip().lower(),
            }
        )
        if isinstance(clean.get("age_rules"), (dict, list)):
            clean["age_rules"] = json.dumps(clean["age_rules"], sort_keys=True, separators=(",", ":"))
        normalized_events.append(clean)
    return normalized_days, normalized_events


def assert_registration_configuration_row_ownership(
    supabase,
    *,
    tournament_id: str,
    days: list[dict[str, Any]],
    event_options: list[dict[str, Any]],
) -> None:
    """Reject an ID already owned by another tournament before any write."""

    for table_name, rows in (
        ("tournament_registration_days", days),
        ("tournament_event_options", event_options),
    ):
        for row in rows:
            row_id = str(row.get("id") or "").strip()
            existing = _safe_first(
                supabase.table(table_name)
                .select("id,tournament_id")
                .eq("id", row_id)
                .limit(1)
                .execute()
            )
            if existing and str(existing.get("tournament_id") or "") != str(tournament_id):
                raise ValueError(
                    f"Configuration row '{row_id}' belongs to another tournament and cannot be published here."
                )


def analyze_registration_publish_impact(
    supabase,
    *,
    tournament_id: str,
    days: list[dict[str, Any]],
    event_options: list[dict[str, Any]],
) -> dict[str, Any]:
    tournament_id = str(tournament_id)
    published_days = list_registration_days(supabase, tournament_id)
    published_events = list_event_options(supabase, tournament_id)
    days, event_options = normalize_registration_configuration_payload(
        tournament_id=tournament_id,
        days=days,
        event_options=event_options,
        published_events=published_events,
    )
    assert_registration_configuration_row_ownership(
        supabase,
        tournament_id=tournament_id,
        days=days,
        event_options=event_options,
    )
    usage_by_event = list_registration_usage_by_event_option(supabase, tournament_id)
    usage_by_day = list_registration_usage_by_day(supabase, tournament_id)

    published_days_by_id = {str(row.get("id")): row for row in published_days if str(row.get("id") or "").strip()}
    published_events_by_id = {str(row.get("id")): row for row in published_events if str(row.get("id") or "").strip()}

    draft_days = list(days)
    draft_day_ids = {str(row.get("id")) for row in draft_days}
    draft_events = list(event_options)
    draft_event_ids = {str(row.get("id")) for row in draft_events}

    creates: list[str] = []
    updates: list[str] = []
    soft_closes: list[str] = []
    deletes: list[str] = []
    warnings: list[str] = []
    blocked: list[str] = []
    blocked_details: list[dict[str, Any]] = []

    def block_event(message: str, *, event_id: str, label: str, field: str) -> None:
        blocked.append(message)
        blocked_details.append(
            {
                "message": message,
                "entity_type": "division",
                "entity_id": event_id,
                "entity_label": label,
                "field": field,
                "step": "divisions",
                "resolution_options": ["KEEP_PUBLISHED_VALUE", "EDIT_DRAFT"],
            }
        )

    for day in draft_days:
        day_id = str(day.get("id"))
        existing = published_days_by_id.get(day_id)
        if existing is None:
            creates.append(f"Day '{day.get('label') or day_id}' will be created.")
            continue
        updates.append(f"Day '{existing.get('label') or day_id}' will be updated.")
        usage = usage_by_day.get(day_id, {})
        if _entity_usage_total(usage) and str(existing.get("label") or "") != str(day.get("label") or ""):
            warnings.append(f"Populated day '{existing.get('label') or day_id}' is being relabeled.")

    for event in draft_events:
        event_id = str(event.get("id"))
        existing = published_events_by_id.get(event_id)
        label = str(event.get("division_name") or event.get("label") or event_id)
        if existing is None:
            creates.append(f"Division '{label}' will be created.")
            continue
        updates.append(f"Division '{str(existing.get('division_name') or existing.get('label') or event_id)}' will be updated.")
        usage = usage_by_event.get(event_id, {})
        occupied = int(usage.get("registrations", 0) or 0)
        has_usage = _entity_usage_total(usage) > 0

        existing_day_id = str(existing.get("registration_day_id") or "")
        draft_day_id = str(event.get("registration_day_id") or "")
        if has_usage and existing_day_id != draft_day_id:
            block_event(
                f"Cannot move populated division '{label}' to a different primary day.",
                event_id=event_id,
                label=label,
                field="registration_day_id",
            )
        existing_schedule = list(existing.get("scheduled_day_ids") or ([existing_day_id] if existing_day_id else []))
        draft_schedule = list(event.get("scheduled_day_ids") or ([draft_day_id] if draft_day_id else []))
        if has_usage and existing_schedule != draft_schedule:
            block_event(
                f"Cannot change the multi-day schedule for populated division '{label}'.",
                event_id=event_id,
                label=label,
                field="scheduled_day_ids",
            )
        if has_usage and str(existing.get("event_type") or "") != str(event.get("event_type") or ""):
            block_event(
                f"Cannot change participant type for populated division '{label}'.",
                event_id=event_id,
                label=label,
                field="event_type",
            )
        if has_usage and str(existing.get("gender_restriction") or "") != str(event.get("gender_restriction") or ""):
            block_event(
                f"Cannot change gender restriction for populated division '{label}'.",
                event_id=event_id,
                label=label,
                field="gender_restriction",
            )

        rule_columns = ["skill_label", "skill_mode", "age_label", "age_mode", "age_rules"]
        if has_usage and any(str(existing.get(column) or "") != str(event.get(column) or "") for column in rule_columns):
            block_event(
                f"Cannot change skill/age rules for populated division '{label}' in ways that could invalidate registrants.",
                event_id=event_id,
                label=label,
                field="skill_age_rules",
            )

        existing_capacity = existing.get("capacity_teams")
        next_capacity = event.get("capacity_teams")
        try:
            if existing_capacity is not None and next_capacity is not None:
                old_cap = int(existing_capacity)
                new_cap = int(next_capacity)
                if new_cap < old_cap:
                    if new_cap < occupied:
                        block_event(
                            f"Cannot reduce capacity for '{label}' below occupied teams ({occupied}).",
                            event_id=event_id,
                            label=label,
                            field="capacity_teams",
                        )
                    else:
                        warnings.append(f"Capacity for '{label}' is being reduced from {old_cap} to {new_cap} with {occupied} occupied.")
        except Exception:
            pass

        if has_usage and str(existing.get("division_name") or existing.get("label") or "") != label:
            warnings.append(f"Populated division '{existing.get('division_name') or existing.get('label') or event_id}' is being relabeled.")

        if str(existing.get("enabled", True)).lower() in {"true", "1"} and not _coerce_bool(event.get("enabled", True)):
            warnings.append(f"Division '{label}' will be hidden from future registration.")

    for existing_event in published_events:
        event_id = str(existing_event.get("id") or "")
        if not event_id or event_id in draft_event_ids:
            continue
        usage = usage_by_event.get(event_id, {})
        event_label = str(existing_event.get("division_name") or existing_event.get("label") or event_id)
        if _entity_usage_total(usage):
            soft_closes.append(f"Division '{event_label}' is omitted from draft and will be archived/closed (history preserved).")
            warnings.append(f"Division '{event_label}' has usage and cannot be destructively deleted; it will be soft-closed.")
        else:
            deletes.append(f"Division '{event_label}' is empty and will be deleted.")

    for existing_day in published_days:
        day_id = str(existing_day.get("id") or "")
        if not day_id or day_id in draft_day_ids:
            continue
        usage = usage_by_day.get(day_id, {})
        day_label = str(existing_day.get("label") or day_id)
        if _entity_usage_total(usage):
            soft_closes.append(f"Day '{day_label}' is omitted from draft and will be disabled (history preserved).")
            warnings.append(f"Day '{day_label}' has usage and cannot be destructively deleted; it will be soft-closed.")
        else:
            deletes.append(f"Day '{day_label}' is empty and will be deleted.")

    return {
        "summary": {
            "creates": len(creates),
            "updates": len(updates),
            "soft_closes": len(soft_closes),
            "deletes": len(deletes),
            "warnings": len(warnings),
            "blocked": len(blocked),
        },
        "creates": creates,
        "updates": updates,
        "soft_closes": soft_closes,
        "deletes": deletes,
        "warnings": warnings,
        "blocked": blocked,
        "blocked_details": blocked_details,
        "draft_days": draft_days,
        "draft_event_options": draft_events,
        "published_days": published_days,
        "published_event_options": published_events,
    }


def _legacy_day_aliases(days: list[dict[str, Any]]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for idx, day in enumerate(days, start=1):
        day_id = str(day.get("id") or "").strip()
        if not day_id:
            continue

        aliases[f"day_{idx}"] = day_id

        sort_order = day.get("sort_order")
        try:
            if sort_order not in (None, ""):
                aliases[f"day_{int(sort_order)}"] = day_id
        except Exception:
            pass

    return aliases


def replace_registration_configuration(
    supabase,
    *,
    tournament_id: str,
    days: list[dict[str, Any]],
    event_options: list[dict[str, Any]],
    allow_replace_with_registrations: bool = False,
) -> None:
    days, event_options = normalize_registration_configuration_payload(
        tournament_id=str(tournament_id),
        days=days,
        event_options=event_options,
        published_events=list_event_options(supabase, str(tournament_id)),
    )
    assert_registration_configuration_row_ownership(
        supabase,
        tournament_id=str(tournament_id),
        days=days,
        event_options=event_options,
    )
    assert_registration_schema_contract(
        supabase,
        required_tables=["tournament_registration_settings", "tournament_registration_days", "tournament_event_options"],
    )
    registration_count = count_tournament_registrations(supabase, tournament_id)
    if registration_count and not allow_replace_with_registrations:
        raise ValueError(
            "This tournament already has registrations. Freeze the current registration form or create a new tournament before replacing days and events."
        )

    day_ids: set[str] = set()
    for idx, day in enumerate(days, start=1):
        day_id = str(day.get("id") or "").strip()
        day_tournament_id = str(day.get("tournament_id") or "").strip()
        if not day_id:
            raise ValueError(f"Invalid day payload at row {idx}: missing id.")
        if not day_tournament_id:
            raise ValueError(f"Invalid day payload at row {idx}: missing tournament_id.")
        if day_tournament_id != str(tournament_id):
            raise ValueError(f"Invalid day payload at row {idx}: tournament_id mismatch.")
        day_ids.add(day_id)

    legacy_day_aliases = _legacy_day_aliases(days)

    normalized_event_options: list[dict[str, Any]] = []
    for event in event_options:
        registration_day_id = str(event.get("registration_day_id") or "").strip()
        if registration_day_id not in day_ids and registration_day_id in legacy_day_aliases:
            event = {
                **event,
                "registration_day_id": legacy_day_aliases[registration_day_id],
            }
        normalized_event_options.append(event)

    event_options = normalized_event_options

    for idx, event in enumerate(event_options, start=1):
        event_id = str(event.get("id") or "").strip()
        event_tournament_id = str(event.get("tournament_id") or "").strip()
        registration_day_id = str(event.get("registration_day_id") or "").strip()
        if not event_id:
            raise ValueError(f"Invalid event payload at row {idx}: missing id.")
        if not event_tournament_id:
            raise ValueError(f"Invalid event payload at row {idx}: missing tournament_id.")
        if event_tournament_id != str(tournament_id):
            raise ValueError(f"Invalid event payload at row {idx}: tournament_id mismatch.")
        if not registration_day_id:
            raise ValueError(f"Invalid event payload at row {idx}: missing registration_day_id.")
        if registration_day_id not in day_ids:
            raise ValueError(
                f"Invalid event payload at row {idx}: registration_day_id '{registration_day_id}' is not present in day payload."
            )

    try:
        (
            supabase.table("tournament_event_options")
            .delete()
            .eq("tournament_id", str(tournament_id))
            .execute()
        )
        (
            supabase.table("tournament_registration_days")
            .delete()
            .eq("tournament_id", str(tournament_id))
            .execute()
        )

        _insert_registration_days(supabase, days)
        if event_options:
            (
                supabase.table("tournament_event_options")
                .insert(event_options)
                .execute()
            )
    except Exception as exc:
        raise ValueError(f"Failed to replace registration configuration for tournament {tournament_id}: {exc}") from exc


def publish_registration_configuration(
    supabase,
    *,
    tournament_id: str,
    days: list[dict[str, Any]],
    event_options: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Publish rules summary:
    - If tournament has zero registrations, publish can use full replace for simplicity.
    - If tournament has registrations, publish uses a guarded diff:
      preserve stable IDs, block destructive mutations to populated rows,
      and convert removed populated rows into soft-closed/disabled records.
    - Public registration continues to consume published day/event rows only.
    """
    days, event_options = normalize_registration_configuration_payload(
        tournament_id=str(tournament_id),
        days=days,
        event_options=event_options,
        published_events=list_event_options(supabase, str(tournament_id)),
    )
    assert_registration_configuration_row_ownership(
        supabase,
        tournament_id=str(tournament_id),
        days=days,
        event_options=event_options,
    )
    registration_count = count_tournament_registrations(supabase, tournament_id)
    if registration_count == 0:
        replace_registration_configuration(
            supabase,
            tournament_id=tournament_id,
            days=days,
            event_options=event_options,
            allow_replace_with_registrations=False,
        )
        return {"mode": "replace", "blocked": [], "warnings": []}

    impact = analyze_registration_publish_impact(
        supabase,
        tournament_id=tournament_id,
        days=days,
        event_options=event_options,
    )
    if impact["blocked"]:
        raise ValueError("Publish blocked due to destructive changes: " + " | ".join(impact["blocked"]))

    published_days = impact["published_days"]
    published_events = impact["published_event_options"]
    draft_days = impact["draft_days"]
    draft_events = impact["draft_event_options"]
    draft_day_ids = {str(row.get("id")) for row in draft_days}
    draft_event_ids = {str(row.get("id")) for row in draft_events}
    usage_by_day = list_registration_usage_by_day(supabase, tournament_id)
    usage_by_event = list_registration_usage_by_event_option(supabase, tournament_id)

    day_upserts = list(draft_days)
    event_upserts = list(draft_events)
    day_delete_ids: list[str] = []
    event_delete_ids: list[str] = []

    for row in published_events:
        row_id = str(row.get("id") or "")
        if not row_id or row_id in draft_event_ids:
            continue
        if _entity_usage_total(usage_by_event.get(row_id)):
            event_upserts.append(
                {
                    **row,
                    "enabled": False,
                    "status": "closed",
                }
            )
        else:
            event_delete_ids.append(row_id)

    for row in published_days:
        row_id = str(row.get("id") or "")
        if not row_id or row_id in draft_day_ids:
            continue
        if _entity_usage_total(usage_by_day.get(row_id)):
            day_upserts.append(
                {
                    **row,
                    "enabled": False,
                }
            )
        else:
            day_delete_ids.append(row_id)

    published_day_ids = {str(row.get("id") or "") for row in published_days}
    published_event_ids = {str(row.get("id") or "") for row in published_events}
    day_inserts = [row for row in day_upserts if str(row.get("id") or "") not in published_day_ids]
    day_updates = [row for row in day_upserts if str(row.get("id") or "") in published_day_ids]
    event_inserts = [row for row in event_upserts if str(row.get("id") or "") not in published_event_ids]
    event_updates = [row for row in event_upserts if str(row.get("id") or "") in published_event_ids]

    for row in day_updates:
        row_id = str(row.get("id") or "")
        patch = {key: value for key, value in row.items() if key not in {"id", "tournament_id", "created_at"}}
        updated = _safe_data(
            supabase.table("tournament_registration_days")
            .update(patch)
            .eq("tournament_id", str(tournament_id))
            .eq("id", row_id)
            .execute()
        )
        if not updated:
            raise SelectionWriteConflict(f"Tournament day '{row_id}' changed during publish.")
    if day_inserts:
        supabase.table("tournament_registration_days").insert(day_inserts).execute()

    for row in event_updates:
        row_id = str(row.get("id") or "")
        patch = {key: value for key, value in row.items() if key not in {"id", "tournament_id", "created_at"}}
        updated = _safe_data(
            supabase.table("tournament_event_options")
            .update(patch)
            .eq("tournament_id", str(tournament_id))
            .eq("id", row_id)
            .execute()
        )
        if not updated:
            raise SelectionWriteConflict(f"Tournament event '{row_id}' changed during publish.")
    if event_inserts:
        supabase.table("tournament_event_options").insert(event_inserts).execute()
    if event_delete_ids:
        supabase.table("tournament_event_options").delete().eq("tournament_id", str(tournament_id)).in_("id", event_delete_ids).execute()
    if day_delete_ids:
        supabase.table("tournament_registration_days").delete().eq("tournament_id", str(tournament_id)).in_("id", day_delete_ids).execute()

    return {"mode": "guarded", "blocked": impact["blocked"], "warnings": impact["warnings"], "summary": impact["summary"]}


def list_registrations(supabase, tournament_id: str) -> list[dict[str, Any]]:
    resp = (
        supabase.table("tournament_registrations")
        .select("*")
        .eq("tournament_id", str(tournament_id))
        .order("submitted_at", desc=True)
        .execute()
    )
    return _safe_data(resp)


def list_registration_admin_rows(supabase, tournament_id: str) -> list[dict[str, Any]]:
    tournament_id = str(tournament_id)
    registrations = list_registrations(supabase, tournament_id)
    selections = list_registration_selections(supabase, tournament_id)
    days = {str(row.get("id")): row for row in list_registration_days(supabase, tournament_id)}
    events = {str(row.get("id")): row for row in list_event_options(supabase, tournament_id)}

    rows: list[dict[str, Any]] = []
    for reg in registrations:
        reg_id = str(reg.get("id") or "")
        reg_selections = [row for row in selections if str(row.get("registration_id")) == reg_id]
        if not reg_selections:
            rows.append(
                {
                    "registration_id": reg_id,
                    "selection_id": None,
                    "registration": reg,
                    "selection": None,
                    "day": None,
                    "event": None,
                }
            )
            continue
        for selection in reg_selections:
            day = days.get(str(selection.get("registration_day_id")))
            event = events.get(str(selection.get("event_option_id")))
            rows.append(
                {
                    "registration_id": reg_id,
                    "selection_id": str(selection.get("id") or ""),
                    "registration": reg,
                    "selection": selection,
                    "day": day,
                    "event": event,
                }
            )
    return rows


def update_registration_admin_fields(
    supabase,
    *,
    tournament_id: str,
    registration_id: str,
    status: str,
    payment_status: str,
) -> dict[str, Any]:
    clean_status = str(status or "").strip().lower()
    clean_payment_status = str(payment_status or "").strip().lower()
    if clean_status not in ADMIN_REGISTRATION_STATUS_OPTIONS:
        raise ValueError(f"Invalid registration status: {status}")
    if clean_payment_status not in ADMIN_PAYMENT_STATUS_OPTIONS:
        raise ValueError(f"Invalid payment status: {payment_status}")

    payload = {
        "status": clean_status,
        "payment_status": clean_payment_status,
        "updated_at": _now_iso(),
    }
    resp = (
        supabase.table("tournament_registrations")
        .update(payload)
        .eq("tournament_id", str(tournament_id))
        .eq("id", str(registration_id))
        .execute()
    )
    updated = _safe_first(resp)
    if not updated:
        raise ValueError("Registration not found for this tournament.")
    return updated


def create_admin_registration(supabase, *, tournament_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return save_registration(supabase, tournament_id=str(tournament_id), payload=payload)


def update_admin_registration(
    supabase,
    *,
    tournament_id: str,
    registration_id: str,
    payload: dict[str, Any],
    expected_updated_at: str | None = None,
) -> dict[str, Any]:
    update_payload = {
        "first_name": str(payload.get("first_name") or "").strip() or None,
        "last_name": str(payload.get("last_name") or "").strip() or None,
        "display_name": str(payload.get("display_name") or "").strip() or None,
        "email": _normalize_email(payload.get("email")) or None,
        "phone": str(payload.get("phone") or "").strip() or None,
        "gender": str(payload.get("gender") or "").strip() or None,
        "age": payload.get("age"),
        "dupr_id": str(payload.get("dupr_id") or "").strip() or None,
        "doubles_skill": payload.get("doubles_skill"),
        "singles_skill": payload.get("singles_skill"),
        "status": str(payload.get("status") or "").strip().lower() or None,
        "payment_status": str(payload.get("payment_status") or "").strip().lower() or None,
        "notes": str(payload.get("notes") or "").strip() or None,
        "updated_at": _now_iso(),
    }
    clean_payload = {k: v for k, v in update_payload.items() if v is not None}
    if clean_payload.get("status") and clean_payload["status"] not in ADMIN_REGISTRATION_STATUS_OPTIONS:
        raise ValueError(f"Invalid registration status: {clean_payload['status']}")
    if clean_payload.get("payment_status") and clean_payload["payment_status"] not in ADMIN_PAYMENT_STATUS_OPTIONS:
        raise ValueError(f"Invalid payment status: {clean_payload['payment_status']}")
    if clean_payload.get("email"):
        existing = _get_registration_by_email(supabase, str(tournament_id), str(clean_payload["email"]))
        if existing and str(existing.get("id")) != str(registration_id):
            raise ValueError("Another registration already uses that email.")

    query = (
        supabase.table("tournament_registrations")
        .update(clean_payload)
        .eq("tournament_id", str(tournament_id))
        .eq("id", str(registration_id))
    )
    if expected_updated_at is not None:
        query = query.eq("updated_at", str(expected_updated_at))
    try:
        resp = query.execute()
    except Exception as exc:
        if _database_error_contains(
            exc, REGISTRATION_COMMERCE_FULFILLED_CANCEL_MARKER
        ):
            raise ValueError(
                "This registration has fulfilled extras. Resolve fulfillment "
                "before cancelling the registration."
            ) from exc
        raise
    updated = _safe_first(resp)
    if not updated:
        if expected_updated_at is not None:
            raise StaleTournamentRegistrationAdminError(
                "Registration changed after it was loaded. Refresh and try again."
            )
        raise ValueError("Registration not found for this tournament.")
    return updated


def update_admin_registration_selection(
    supabase,
    *,
    tournament_id: str,
    selection_id: str,
    payload: dict[str, Any],
    expected_updated_at: str | None = None,
) -> dict[str, Any]:
    clean_payload: dict[str, Any] = {}
    for field in ("registration_day_id", "event_option_id"):
        if field in payload:
            value = str(payload.get(field) or "").strip() or None
            if value is not None:
                clean_payload[field] = value
    if "partner_mode" in payload:
        partner_mode = str(payload.get("partner_mode") or "").strip().upper() or None
        if partner_mode is not None:
            clean_payload["partner_mode"] = partner_mode
    for field in ("partner_name", "partner_phone", "partner_dupr_id", "partner_note"):
        if field in payload:
            clean_payload[field] = str(payload.get(field) or "").strip() or None
    if "partner_email" in payload:
        clean_payload["partner_email"] = _normalize_email(payload.get("partner_email")) or None
    for field in ("partner_skill", "partner_age"):
        if field in payload:
            clean_payload[field] = payload.get(field)
    if "show_on_partner_board" in payload:
        clean_payload["show_on_partner_board"] = _coerce_bool(payload.get("show_on_partner_board"))
    if clean_payload.get("partner_mode") and clean_payload["partner_mode"] not in PARTNER_MODE_OPTIONS:
        raise ValueError(f"Invalid partner mode: {clean_payload['partner_mode']}")

    # Older Streamlit callers do not yet send the row version. Resolve it before
    # invoking the transactional RPC so those callers remain functional while
    # every mutation still uses database-side compare-and-swap semantics.
    rpc_expected_updated_at = expected_updated_at
    if rpc_expected_updated_at is None:
        current_resp = (
            supabase.table("tournament_registration_selections")
            .select("updated_at")
            .eq("tournament_id", str(tournament_id))
            .eq("id", str(selection_id))
            .limit(1)
            .execute()
        )
        current = _safe_first(current_resp)
        if not current:
            raise ValueError("Registration selection not found for this tournament.")
        rpc_expected_updated_at = str(current.get("updated_at") or "").strip()
        if not rpc_expected_updated_at:
            raise RuntimeError("Registration selection is missing its write version.")

    params = {
        "p_tournament_id": str(tournament_id),
        "p_selection_id": str(selection_id),
        "p_expected_updated_at": str(rpc_expected_updated_at),
        "p_patch": clean_payload,
    }
    try:
        resp = supabase.rpc(ADMIN_SELECTION_UPDATE_RPC, params).execute()
    except Exception as exc:
        if _database_error_contains(exc, SELECTION_WRITE_CONFLICT_MARKER) or _database_error_contains(
            exc, SELECTION_WRITE_CONFLICT_CODE
        ):
            raise StaleTournamentRegistrationSelectionError(
                "Registration selection changed after it was loaded. Refresh and try again."
            ) from exc
        if _database_error_contains(exc, SELECTION_NOT_FOUND_CODE) or _database_error_contains(
            exc, RELATION_SELECTION_NOT_FOUND_MARKER
        ):
            raise ValueError("Registration selection not found for this tournament.") from exc
        if _database_error_contains(exc, SELECTION_INVALID_TARGET_MARKER):
            raise ValueError("Registration selection target is invalid.") from exc
        if _database_error_contains(exc, SELECTION_INVALID_PATCH_MARKER):
            raise ValueError("Registration selection update is invalid.") from exc
        raise RuntimeError("Registration selection update failed.") from exc

    result = _rpc_object(resp)
    if result is None:
        raise RuntimeError("Registration selection update returned an invalid response.")
    code = str(result.get("code") or "").strip().upper()
    if code == SELECTION_WRITE_CONFLICT_CODE:
        raise StaleTournamentRegistrationSelectionError(
            "Registration selection changed after it was loaded. Refresh and try again."
        )
    if code == SELECTION_NOT_FOUND_CODE:
        raise ValueError("Registration selection not found for this tournament.")
    updated = result.get("selection")
    if result.get("ok") is not True or not isinstance(updated, dict):
        raise RuntimeError("Registration selection update returned an invalid response.")
    return updated


def registration_is_imported_to_draw(
    supabase,
    *,
    tournament_id: str,
    selection_id: str | None = None,
    registration_id: str | None = None,
) -> bool:
    if not selection_id and not registration_id:
        return False
    selections: list[dict[str, Any]] = []
    if selection_id:
        selection_resp = (
            supabase.table("tournament_registration_selections")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("id", str(selection_id))
            .limit(1)
            .execute()
        )
        selection = _safe_first(selection_resp)
        if selection:
            selections = [selection]
    if not selections and registration_id:
        selection_resp = (
            supabase.table("tournament_registration_selections")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("registration_id", str(registration_id))
            .execute()
        )
        selections = _safe_data(selection_resp)
    if not selections:
        return False
    for selection in selections:
        day_id = str(selection.get("registration_day_id") or "")
        event_option_id = str(selection.get("event_option_id") or "")
        if not day_id or not event_option_id:
            continue
        teams_resp = (
            supabase.table("tournament_teams")
            .select("id")
            .eq("tournament_id", str(tournament_id))
            .eq("registration_day_id", day_id)
            .eq("event_option_id", event_option_id)
            .eq("source", "REGISTRATION")
            .limit(1)
            .execute()
        )
        if _safe_data(teams_resp):
            return True
    return False


def registration_has_imported_draw_selection(
    supabase,
    *,
    tournament_id: str,
    registration_id: str,
) -> bool:
    """Return whether any event on a registration has been imported to a draw."""

    return registration_is_imported_to_draw(
        supabase,
        tournament_id=str(tournament_id),
        registration_id=str(registration_id),
    )


def cancel_registration(supabase, *, tournament_id: str, registration_id: str) -> dict[str, Any]:
    existing = _safe_first(
        supabase.table("tournament_registrations")
        .select("payment_status")
        .eq("tournament_id", str(tournament_id))
        .eq("id", str(registration_id))
        .limit(1)
        .execute()
    ) or {}
    return update_registration_admin_fields(
        supabase,
        tournament_id=str(tournament_id),
        registration_id=str(registration_id),
        status="cancelled",
        payment_status=str(existing.get("payment_status") or "unpaid").lower(),
    )


def delete_registration(supabase, *, tournament_id: str, registration_id: str) -> None:
    if registration_is_imported_to_draw(supabase, tournament_id=str(tournament_id), registration_id=str(registration_id)):
        raise ValueError("Registration is already imported into a draw. Remove the draw team first.")
    (
        supabase.table("tournament_registration_selections")
        .delete()
        .eq("tournament_id", str(tournament_id))
        .eq("registration_id", str(registration_id))
        .execute()
    )
    (
        supabase.table("tournament_registrations")
        .delete()
        .eq("tournament_id", str(tournament_id))
        .eq("id", str(registration_id))
        .execute()
    )


def list_registration_selections(supabase, tournament_id: str) -> list[dict[str, Any]]:
    resp = (
        supabase.table("tournament_registration_selections")
        .select("*")
        .eq("tournament_id", str(tournament_id))
        .order("created_at")
        .execute()
    )
    return _safe_data(resp)


def list_partner_requests(supabase, tournament_id: str) -> list[dict[str, Any]]:
    resp = (
        supabase.table("tournament_registration_partner_requests")
        .select("*")
        .eq("tournament_id", str(tournament_id))
        .execute()
    )
    return _safe_data(resp)


def list_partner_team_links(supabase, tournament_id: str) -> list[dict[str, Any]]:
    resp = (
        supabase.table("tournament_registration_team_links")
        .select("*")
        .eq("tournament_id", str(tournament_id))
        .execute()
    )
    return _safe_data(resp)


def list_partner_team_members(supabase, tournament_id: str) -> list[dict[str, Any]]:
    resp = (
        supabase.table("tournament_registration_team_members")
        .select("*")
        .eq("tournament_id", str(tournament_id))
        .execute()
    )
    return _safe_data(resp)


def list_four_player_teams(supabase, tournament_id: str) -> list[dict[str, Any]]:
    resp = (
        supabase.table("tournament_four_player_teams")
        .select("*")
        .eq("tournament_id", str(tournament_id))
        .order("created_at")
        .execute()
    )
    return _safe_data(resp)


def list_four_player_team_members(
    supabase, tournament_id: str
) -> list[dict[str, Any]]:
    resp = (
        supabase.table("tournament_four_player_team_members")
        .select("*")
        .eq("tournament_id", str(tournament_id))
        .order("slot")
        .execute()
    )
    return _safe_data(resp)


def _get_existing_registration_by_email(supabase, tournament_id: str, email: str) -> dict[str, Any] | None:
    return _get_registration_by_email(supabase, tournament_id, email)


def _get_registration_by_email(supabase, tournament_id: str, email: str) -> dict[str, Any] | None:
    clean_email = _normalize_email(email)
    if not clean_email:
        return None
    resp = (
        supabase.table("tournament_registrations")
        .select("*")
        .eq("tournament_id", str(tournament_id))
        .eq("email", clean_email)
        .limit(1)
        .execute()
    )
    return _safe_first(resp)


def get_registration_by_email(supabase, tournament_id: str, email: str) -> dict[str, Any] | None:
    return _get_registration_by_email(supabase, tournament_id, email)


def save_registration(
    supabase,
    *,
    tournament_id: str,
    payload: dict[str, Any],
    expected_registration_id: str | None = None,
    allow_existing_unselectable_event_ids: set[str] | None = None,
    expected_updated_at: str | None = None,
    expected_selection_versions: list[dict[str, Any]] | None = None,
    atomic_edit: bool = False,
    commerce_transaction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    email = _normalize_email(payload.get("email"))
    if not email:
        raise ValueError("Email is required.")

    display_name = str(payload.get("display_name") or "").strip()
    if not display_name:
        first_name = str(payload.get("first_name") or "").strip()
        last_name = str(payload.get("last_name") or "").strip()
        display_name = " ".join(part for part in [first_name, last_name] if part).strip()
    if not display_name:
        raise ValueError("Player name is required.")

    existing = _get_existing_registration_by_email(supabase, tournament_id, email)
    if expected_registration_id:
        expected = get_registration_by_id(supabase, tournament_id, str(expected_registration_id))
        if not expected:
            raise ValueError("Expected registration was not found for this tournament.")
        if _normalize_email(expected.get("email")) != email:
            raise ValueError("Registration email cannot be changed from this edit link.")
        if existing and str(existing.get("id")) != str(expected_registration_id):
            raise ValueError("Expected registration does not match the registered email.")
        registration_id = str(expected_registration_id)
    elif existing and commerce_transaction is None:
        raise ValueError("A registration already exists for this email. Please use the secure edit link flow.")
    else:
        registration_id = str(
            payload.get("_registration_id") or _uid("reg")
        )
    submitted_at = _now_iso()
    if "status" in payload and payload.get("status") not in (None, ""):
        registration_status = str(payload.get("status")).strip().lower()
        if registration_status not in ADMIN_REGISTRATION_STATUS_OPTIONS:
            raise ValueError(f"Invalid registration status: {payload.get('status')}")
    elif expected_registration_id:
        registration_status = str((expected or {}).get("status") or "confirmed").strip().lower()
    else:
        registration_status = "confirmed"

    reg_row = {
        "id": registration_id,
        "tournament_id": str(tournament_id),
        "submitted_at": submitted_at,
        "status": registration_status,
        "payment_status": str(payload.get("payment_status") or "unpaid").lower(),
        "first_name": str(payload.get("first_name") or "").strip() or None,
        "last_name": str(payload.get("last_name") or "").strip() or None,
        "display_name": display_name,
        "email": email,
        "phone": str(payload.get("phone") or "").strip() or None,
        "player_id": payload.get("player_id"),
        "dupr_id": str(payload.get("dupr_id") or "").strip() or None,
        "doubles_skill": payload.get("doubles_skill"),
        "singles_skill": payload.get("singles_skill"),
        "age": payload.get("age"),
        "age_bracket": str(payload.get("age_bracket") or "").strip() or None,
        "gender": str(payload.get("gender") or "").strip() or None,
        "notes": str(payload.get("notes") or "").strip() or None,
        "wants_partner_board_contact": _coerce_bool(payload.get("wants_partner_board_contact", False)),
        "updated_at": submitted_at,
    }

    days = list_registration_days(supabase, str(tournament_id))
    day_lookup = {str(row.get("id")): row for row in days if is_day_enabled(row)}
    event_lookup = {str(row.get("id")): row for row in list_event_options(supabase, str(tournament_id))}

    allowed_existing_event_ids = {str(value) for value in (allow_existing_unselectable_event_ids or set())}
    rows: list[dict[str, Any]] = []
    for index, selection in enumerate(payload.get("selections") or []):
        if not selection.get("event_option_id"):
            continue

        event_option_id = str(selection.get("event_option_id"))
        event = event_lookup.get(event_option_id)
        if not event:
            raise ValueError(f"Selected division {event_option_id} is no longer available.")
        if str(event.get("registration_day_id") or "") not in day_lookup and event_option_id not in allowed_existing_event_ids:
            raise ValueError("Selected division is not on an enabled registration day.")

        visibility = public_event_option_visibility(event)
        if visibility != "selectable" and event_option_id not in allowed_existing_event_ids:
            status_label = str(event.get("status") or "draft").lower()
            division_label = str(event.get("division_name") or event.get("label") or event_option_id)
            raise ValueError(
                f"{division_label} is not open for public registration (status={status_label}, enabled={bool(event.get('enabled', True))})."
            )

        partner_email = _normalize_email(selection.get("partner_email")) or None
        partner_payload = {
            "display_name": str(selection.get("partner_name") or "").strip() or None,
            "email": partner_email,
            "doubles_skill": selection.get("partner_skill"),
            "singles_skill": selection.get("partner_skill"),
            "age": selection.get("partner_age"),
        }
        if partner_payload.get("doubles_skill") in (None, "") and partner_email:
            partner_registration = _get_registration_by_email(supabase, tournament_id, partner_email)
            if partner_registration:
                partner_payload["doubles_skill"] = partner_registration.get("doubles_skill")
                partner_payload["singles_skill"] = partner_registration.get("singles_skill")
        partner_mode = str(selection.get("partner_mode") or "NONE").upper()
        partner_for_validation = None
        if partner_mode == "HAS_PARTNER":
            partner_for_validation = partner_payload

        eligible, message = validate_selection_against_skill(
            event=event,
            selection=selection,
            player=reg_row,
            partner=partner_for_validation,
            allow_missing_partner_for_preview=False,
        )
        if not eligible:
            division_label = str(event.get("division_name") or event.get("label") or event_option_id)
            raise ValueError(f"{division_label}: {message or 'Skill eligibility requirements were not met.'}")

        rows.append(
            {
                "id": str(selection.get("id") or _uid("sel")),
                "tournament_id": str(tournament_id),
                "registration_id": registration_id,
                "registration_day_id": str(selection.get("registration_day_id")),
                "event_option_id": event_option_id,
                "partner_mode": partner_mode,
                "partner_name": str(selection.get("partner_name") or "").strip() or None,
                "partner_email": partner_email,
                "partner_phone": str(selection.get("partner_phone") or "").strip() or None,
                "partner_dupr_id": str(selection.get("partner_dupr_id") or "").strip() or None,
                "partner_skill": selection.get("partner_skill"),
                "partner_age": selection.get("partner_age"),
                "partner_note": str(selection.get("partner_note") or "").strip() or None,
                "show_on_partner_board": _coerce_bool(selection.get("show_on_partner_board", False)),
                "sort_order": index,
                "created_at": submitted_at,
                "updated_at": submitted_at,
            }
        )

    if atomic_edit:
        if not expected_registration_id:
            raise ValueError("Atomic registration edits require an expected registration id.")
        if not str(expected_updated_at or "").strip():
            raise TournamentRegistrationEditConflictError(
                "Registration changed after it was loaded. Refresh the edit link and try again."
            )
        registration_patch = {
            key: reg_row.get(key)
            for key in (
                "first_name",
                "last_name",
                "display_name",
                "phone",
                "dupr_id",
                "doubles_skill",
                "singles_skill",
                "age",
                "age_bracket",
                "gender",
                "notes",
                "wants_partner_board_contact",
            )
        }
        params: dict[str, Any] = {
            "p_tournament_id": str(tournament_id),
            "p_registration_id": str(expected_registration_id),
            "p_expected_updated_at": str(expected_updated_at),
            "p_expected_selection_versions": list(expected_selection_versions or []),
            "p_registration_patch": registration_patch,
            "p_selections": rows,
        }
        rpc_name = PUBLIC_REGISTRATION_EDIT_RPC
        if commerce_transaction is not None:
            params = {
                "p_club_id": str(commerce_transaction["club_id"]),
                "p_tournament_id": str(tournament_id),
                "p_registration_id": str(expected_registration_id),
                "p_expected_registration_updated_at": str(expected_updated_at),
                "p_expected_selection_versions": list(
                    expected_selection_versions or []
                ),
                "p_registration_patch": registration_patch,
                "p_selections": rows,
                "p_expected_order_updated_at": commerce_transaction.get(
                    "expected_order_updated_at"
                ),
                "p_quote_snapshot": commerce_transaction["quote"],
                "p_operation_idempotency_key": commerce_transaction[
                    "operation_idempotency_key"
                ],
                "p_order_idempotency_key": commerce_transaction[
                    "order_idempotency_key"
                ],
                "p_request_fingerprint": commerce_transaction[
                    "request_fingerprint"
                ],
                "p_actor_label": commerce_transaction["actor_label"],
                "p_source": commerce_transaction.get("source")
                or "next_public_tournament_registration_edit",
            }
            rpc_name = PUBLIC_REGISTRATION_COMMERCE_EDIT_RPC
        try:
            resp = supabase.rpc(rpc_name, params).execute()
        except Exception as exc:
            if _database_error_contains(exc, REGISTRATION_EDIT_CONFLICT_MARKER):
                raise TournamentRegistrationEditConflictError(
                    "Registration changed after it was loaded. Refresh the edit link and try again."
                ) from exc
            if _database_error_contains(exc, REGISTRATION_EDIT_IMPORTED_MARKER):
                raise TournamentRegistrationImportedDrawError(
                    "This registration is already imported into a draw and can no longer be edited publicly."
                ) from exc
            if _database_error_contains(exc, REGISTRATION_EDIT_RELATIONSHIP_MARKER):
                raise TournamentRegistrationRelationshipLockedError(
                    "This registration has an active partner relationship. Tournament staff must review the change."
                ) from exc
            if any(
                _database_error_contains(exc, marker)
                for marker in (
                    "JUPR_TOURNAMENT_COMMERCE_ORDER_STALE",
                    "JUPR_TOURNAMENT_COMMERCE_QUOTE_STALE",
                    "JUPR_TOURNAMENT_COMMERCE_INVENTORY_UNAVAILABLE",
                    "JUPR_TOURNAMENT_COMMERCE_CLAIM_GIVEAWAY_FULL",
                    "JUPR_TOURNAMENT_COMMERCE_REGISTRANT_GIVEAWAY_FULL",
                    "JUPR_TOURNAMENT_COMMERCE_IDEMPOTENCY_CONFLICT",
                )
            ):
                raise TournamentRegistrationEditConflictError(
                    "Tournament extras changed while the registration was "
                    "being saved. Refresh and review the current total."
                ) from exc
            raise RuntimeError("Tournament registration edit failed without changing the registration.") from exc

        result = _rpc_object(resp)
        if result is None:
            raise RuntimeError("Tournament registration edit returned an invalid response.")
        code = str(result.get("code") or "").strip().upper()
        if code == REGISTRATION_EDIT_CONFLICT_CODE:
            raise TournamentRegistrationEditConflictError(
                "Registration changed after it was loaded. Refresh the edit link and try again."
            )
        if code == REGISTRATION_EDIT_IMPORTED_CODE:
            raise TournamentRegistrationImportedDrawError(
                "This registration is already imported into a draw and can no longer be edited publicly."
            )
        if code == REGISTRATION_EDIT_RELATIONSHIP_CODE:
            raise TournamentRegistrationRelationshipLockedError(
                "This registration has an active partner relationship. Tournament staff must review the change."
            )
        if code == "REGISTRATION_NOT_FOUND":
            raise ValueError("Expected registration was not found for this tournament.")
        if result.get("ok") is not True:
            raise RuntimeError("Tournament registration edit returned an invalid response.")
        return {
            "registration_id": str(result.get("registration_id") or registration_id),
            "submitted_at": (expected or {}).get("submitted_at") or submitted_at,
            "updated_at": result.get("updated_at"),
            "selection_count": int(result.get("selection_count") or 0),
            "commerce_order": result.get("commerce_order"),
            "idempotent_replay": bool(result.get("idempotent_replay")),
        }

    if commerce_transaction is not None:
        params = {
            "p_club_id": str(commerce_transaction["club_id"]),
            "p_tournament_id": str(tournament_id),
            "p_registration": reg_row,
            "p_selections": rows,
            "p_quote_snapshot": commerce_transaction["quote"],
            "p_operation_idempotency_key": commerce_transaction[
                "operation_idempotency_key"
            ],
            "p_order_idempotency_key": commerce_transaction[
                "order_idempotency_key"
            ],
            "p_request_fingerprint": commerce_transaction[
                "request_fingerprint"
            ],
            "p_actor_label": commerce_transaction["actor_label"],
            "p_source": commerce_transaction.get("source")
            or "next_public_tournament_registration",
        }
        try:
            response = supabase.rpc(
                PUBLIC_REGISTRATION_COMMERCE_CREATE_RPC, params
            ).execute()
        except Exception as exc:
            if _database_error_contains(
                exc, "JUPR_TOURNAMENT_COMMERCE_REGISTRATION_DUPLICATE"
            ):
                raise ValueError(
                    "A registration already exists for this email. Please use "
                    "the secure edit-link flow."
                ) from exc
            if any(
                _database_error_contains(exc, marker)
                for marker in (
                    "JUPR_TOURNAMENT_COMMERCE_ORDER_STALE",
                    "JUPR_TOURNAMENT_COMMERCE_QUOTE_STALE",
                    "JUPR_TOURNAMENT_COMMERCE_INVENTORY_UNAVAILABLE",
                    "JUPR_TOURNAMENT_COMMERCE_CLAIM_GIVEAWAY_FULL",
                    "JUPR_TOURNAMENT_COMMERCE_REGISTRANT_GIVEAWAY_FULL",
                    "JUPR_TOURNAMENT_COMMERCE_IDEMPOTENCY_CONFLICT",
                )
            ):
                raise TournamentRegistrationEditConflictError(
                    "Tournament extras changed while the registration was "
                    "being saved. Review the current total and try again."
                ) from exc
            raise RuntimeError(
                "Tournament registration and extras were not saved."
            ) from exc
        result = _rpc_object(response)
        if not result or result.get("ok") is not True:
            raise RuntimeError(
                "Tournament registration and extras returned no recovery "
                "evidence."
            )
        return {
            "registration_id": str(
                result.get("registration_id") or registration_id
            ),
            "submitted_at": result.get("submitted_at") or submitted_at,
            "updated_at": result.get("updated_at"),
            "selection_count": int(result.get("selection_count") or 0),
            "commerce_order": result.get("commerce_order"),
            "idempotent_replay": bool(result.get("idempotent_replay")),
        }

    (
        supabase.table("tournament_registrations")
        .upsert(reg_row, on_conflict="id")
        .execute()
    )

    (
        supabase.table("tournament_registration_selections")
        .delete()
        .eq("registration_id", registration_id)
        .execute()
    )

    if rows:
        (
            supabase.table("tournament_registration_selections")
            .insert(rows)
            .execute()
        )

    return {
        "registration_id": registration_id,
        "submitted_at": submitted_at,
        "selection_count": len(rows),
    }


def get_public_tournament_bundle(
    supabase,
    *,
    club_id: str,
    tournament_id: str | None = None,
    registration_slug: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    if registration_slug:
        settings_resp = (
            supabase.table("tournament_registration_settings")
            .select("*")
            .eq("registration_slug", str(registration_slug))
            .limit(1)
            .execute()
        )
        settings = _safe_first(settings_resp)
        if not settings:
            return None, None, [], []
        tournament_id = str(settings.get("tournament_id"))
    elif tournament_id:
        settings = get_registration_settings(supabase, str(tournament_id))
    else:
        return None, None, [], []

    tournament = get_tournament_record(supabase, str(tournament_id))
    if not tournament or str(tournament.get("club_id")) != str(club_id):
        return None, None, [], []
    if str(tournament.get("status") or "").upper() == "ARCHIVED":
        return None, None, [], []

    days = list_registration_days(supabase, str(tournament_id))
    event_options = list_event_options(supabase, str(tournament_id))
    return tournament, settings, days, event_options


def list_open_public_tournaments(supabase, club_id: str) -> list[dict[str, Any]]:
    settings_rows = _safe_data(
        supabase.table("tournament_registration_settings").select("*").eq("registration_status", "open").execute()
    )
    out: list[dict[str, Any]] = []
    for settings in settings_rows:
        tournament = get_tournament_record(supabase, str(settings.get("tournament_id")))
        if not tournament:
            continue
        if str(tournament.get("club_id")) != str(club_id):
            continue
        if str(tournament.get("status") or "").upper() == "ARCHIVED":
            continue
        out.append({"tournament": tournament, "settings": settings})
    out.sort(key=lambda row: str(row.get("tournament", {}).get("created_at") or ""), reverse=True)
    return out


def build_registration_state(supabase, tournament: dict[str, Any], settings: dict[str, Any], days: list[dict[str, Any]], event_options: list[dict[str, Any]]) -> dict[str, Any]:
    registrations = list_registrations(supabase, str(tournament.get("id")))
    selections = list_registration_selections(supabase, str(tournament.get("id")))
    partner_schema_ok, partner_schema_detail = partner_link_schema_available(supabase)
    if partner_schema_ok:
        partner_requests = list_partner_requests(supabase, str(tournament.get("id")))
        partner_links = list_partner_team_links(supabase, str(tournament.get("id")))
        team_members = list_partner_team_members(supabase, str(tournament.get("id")))
    else:
        partner_requests = []
        partner_links = []
        team_members = []
    has_four_player_event = any(
        str(event.get("competition_format") or "").upper() == "FOUR_PLAYER_TEAM"
        for event in event_options
    )
    four_player_schema_ok = True
    four_player_schema_detail = None
    if has_four_player_event:
        four_player_schema_ok, four_player_schema_detail = (
            four_player_team_schema_available(supabase)
        )
    if four_player_schema_ok:
        four_player_teams = (
            list_four_player_teams(supabase, str(tournament.get("id")))
            if has_four_player_event
            else []
        )
        four_player_team_members = (
            list_four_player_team_members(supabase, str(tournament.get("id")))
            if has_four_player_event
            else []
        )
    else:
        four_player_teams = []
        four_player_team_members = []
    state = compile_tournament_registration_state(
        tournament=tournament,
        settings=settings,
        days=days,
        event_options=event_options,
        registrations=registrations,
        selections=selections,
        partner_requests=partner_requests,
        partner_links=partner_links,
        team_members=team_members,
        four_player_teams=four_player_teams,
        four_player_team_members=four_player_team_members,
    )
    if not partner_schema_ok:
        state["partner_link_schema_available"] = False
        state["partner_link_schema_detail"] = partner_schema_detail
    else:
        state["partner_link_schema_available"] = True
    if has_four_player_event:
        state["four_player_team_schema_available"] = four_player_schema_ok
        if not four_player_schema_ok:
            state["four_player_team_schema_detail"] = four_player_schema_detail
    return state


def build_public_tournament_roster_state(
    supabase,
    tournament: dict[str, Any],
    settings: dict[str, Any],
    days: list[dict[str, Any]],
    event_options: list[dict[str, Any]],
) -> dict[str, Any]:
    state = build_registration_state(supabase, tournament, settings, days, event_options)
    event_lookup = {str(row.get("id")): row for row in (state.get("event_options") or [])}

    status_map = {
        "CONFIRMED": "Registered",
        "ADMIN_CONFIRMED": "Registered",
        "WAITLIST": "Waitlist",
        "REVIEW": "Review",
        "PARTNER_MISSING": "Review",
        "NEEDS_PARTNER": "Needs Partner",
        "PENDING_PARTNER_REQUEST": "Pending Partner Request",
        "LEGACY_PARTNER_UNRESOLVED": "Review",
    }

    email_pattern = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
    phone_pattern = re.compile(r"(?<!\w)(?:\+?\d[\s().-]*){7,}\d(?!\w)")

    def _public_text(value: Any, *, limit: int = 160) -> str:
        return str(value or "").replace("<", "").replace(">", "").strip()[:limit]

    def _public_display_name(value: Any) -> str:
        text = _public_text(value)
        if not text or email_pattern.search(text) or phone_pattern.search(text):
            return "Player"
        return text

    def _public_age_bracket(member: dict[str, Any]) -> str | None:
        explicit = _public_text(member.get("age_bracket"), limit=40)
        if explicit:
            return explicit
        try:
            age = int(member.get("age"))
        except (TypeError, ValueError):
            return None
        if age < 18:
            return "Under 18"
        if age >= 60:
            return "60+"
        lower = (age // 10) * 10
        return f"{lower}-{lower + 9}"

    def _public_note(value: Any) -> str:
        text = _public_text(value, limit=240)
        text = email_pattern.sub("[contact removed]", text)
        text = phone_pattern.sub("[contact removed]", text)
        return text

    def _public_skill(value: Any) -> str | int | float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return value
        text = _public_text(value, limit=24)
        return text or None

    def _public_status(value: Any) -> str:
        # Unknown and unresolved states fail closed to Review. A missing value
        # must never be interpreted as a confirmed registration in the browser.
        return status_map.get(str(value or "").upper(), "Review")

    def _public_entry_type(value: Any) -> str:
        return {
            "confirmed_team": "Team",
            "four_player_team": "Team",
            "four_player_team_setup_required": "Registration",
            "needs_partner": "Player",
            "pending_partner_request": "Pairing",
        }.get(str(value or "").strip().lower(), "Registration")

    def _public_member(member: dict[str, Any]) -> dict[str, Any]:
        return {
            "display_name": _public_display_name(member.get("display_name")),
            "skill": _public_skill(member.get("skill")),
            "age_bracket": _public_age_bracket(member),
        }

    registrations_by_event: list[dict[str, Any]] = []
    confirmed_teams: list[dict[str, Any]] = []
    pending_partner_requests: list[dict[str, Any]] = []
    unresolved_partner_entries: list[dict[str, Any]] = []
    players_needing_partners: list[dict[str, Any]] = []
    partner_board_entries: list[dict[str, Any]] = []
    unique_players: set[str] = set()
    public_board_enabled = _coerce_bool(
        (state.get("settings") or settings or {}).get("partner_board_enabled", False)
    )
    board_event_ids = {
        str(row.get("id") or "")
        for row in (state.get("event_options") or [])
        if str(row.get("id") or "")
        and _coerce_bool(row.get("enabled", True))
        and _coerce_bool(row.get("partner_board_enabled", row.get("public_partner_board", False)))
        and str(row.get("status") or "draft").strip().lower()
        in {"open", "tentative", "confirmed", "published", "active"}
    }
    board_selection_ids = {
        str(row.get("selection_id") or "")
        for row in (state.get("partner_board") or [])
        if public_board_enabled
        and str(row.get("selection_id") or "")
        and str(row.get("event_option_id") or "") in board_event_ids
    }
    contact_registration_ids = {
        str(row.get("id") or "")
        for row in (state.get("registrations") or [])
        if str(row.get("id") or "")
        and _coerce_bool(row.get("wants_partner_board_contact", False))
        and str(row.get("status") or "confirmed").strip().upper() not in {"CANCELLED", "WITHDRAWN"}
    }

    for roster in state.get("event_rosters", []):
        event_option_id = str(roster.get("event_option_id") or "")
        event_option = event_lookup.get(event_option_id) or {}
        event_rows: list[dict[str, Any]] = []

        for entry in roster.get("entries", []):
            source_members = [member or {} for member in (entry.get("members") or [])]
            members = [_public_member(member) for member in source_members]
            if not members:
                continue
            for index, member in enumerate(source_members):
                identity = next(
                    (
                        str(member.get(field) or "").strip()
                        for field in ("player_id", "registration_id", "selection_id")
                        if str(member.get(field) or "").strip()
                    ),
                    "",
                )
                if not identity:
                    identity = str(members[index].get("display_name") or "").strip().lower()
                if identity:
                    unique_players.add(identity)

            status = str(entry.get("status") or "").upper()
            selection_ids = sorted(
                str(value).strip()
                for value in (entry.get("source_selection_ids") or [])
                if str(value or "").strip()
            )
            registration_ids = sorted(
                str(value).strip()
                for value in (entry.get("source_registration_ids") or [])
                if str(value or "").strip()
            )
            entry_source = "|".join(selection_ids or registration_ids)
            public_entry_key = build_public_tournament_reference(
                tournament_id=str(tournament.get("id") or ""),
                namespace="roster-entry",
                source_id=entry_source,
            )
            event_row = {
                "public_entry_key": public_entry_key or None,
                "event_day_label": _public_text(roster.get("event_day_label"), limit=120),
                "event_family": _public_text(event_option.get("event_family_label") or roster.get("event_label") or "Event", limit=160),
                "division": _public_text(event_option.get("division_name") or roster.get("event_label") or "Division", limit=160),
                "event_label": _public_text(roster.get("event_label"), limit=160),
                "status": _public_status(status),
                "entry_type": _public_entry_type(entry.get("entry_type")),
                "members": members,
            }
            registrations_by_event.append(event_row)
            event_rows.append(event_row)
            if status in {"CONFIRMED", "ADMIN_CONFIRMED"}:
                confirmed_teams.append(event_row)
            elif status == "PENDING_PARTNER_REQUEST":
                pending_partner_requests.append(event_row)
            elif status not in {"NEEDS_PARTNER", "WAITLIST"}:
                unresolved_partner_entries.append(event_row)

            if status == "NEEDS_PARTNER":
                primary = members[0] if members else {}
                source_selection_id = selection_ids[0] if selection_ids else str((source_members[0] if source_members else {}).get("selection_id") or "").strip()
                source_registration_id = registration_ids[0] if registration_ids else str((source_members[0] if source_members else {}).get("registration_id") or "").strip()
                board_entry_key = build_public_tournament_reference(
                    tournament_id=str(tournament.get("id") or ""),
                    namespace="partner-board-selection",
                    source_id=source_selection_id,
                )
                needs_partner_row = {
                    "player_name": primary.get("display_name"),
                    "board_entry_key": board_entry_key or None,
                    "event_day_label": event_row["event_day_label"],
                    "event_family": event_row["event_family"],
                    "division": event_row["division"],
                    "event_label": event_row["event_label"],
                    "skill": primary.get("skill"),
                    "age_bracket": primary.get("age_bracket"),
                    "note": _public_note(entry.get("notes")),
                }
                players_needing_partners.append(needs_partner_row)
                if (
                    source_selection_id in board_selection_ids
                    and source_registration_id in contact_registration_ids
                ):
                    partner_board_entries.append(dict(needs_partner_row))

    return {
        "registrations_by_event": registrations_by_event,
        "confirmed_teams": confirmed_teams,
        "pending_partner_requests": pending_partner_requests,
        "unresolved_partner_entries": unresolved_partner_entries,
        "players_needing_partners": players_needing_partners,
        "partner_board_entries": partner_board_entries,
        "summary": {
            "total_registrations": int(state.get("summary", {}).get("total_registrations") or 0),
            "total_players": len(unique_players),
            "players_needing_partners": len(players_needing_partners),
            "partner_board_entries": len(partner_board_entries),
            "waitlist": int(state.get("summary", {}).get("waitlist_entries") or 0),
        },
    }


def build_public_urls(*, base_url: str, tournament_id: str, registration_slug: str | None = None) -> dict[str, str]:
    tournament_id = str(tournament_id)
    base_url = str(base_url or "").rstrip("/")
    reg_q = f"tournament={registration_slug}" if registration_slug else f"tournament_id={tournament_id}"
    board_q = reg_q
    return {
        "registration": f"{base_url}/?public=1&page=tournament_registration&{reg_q}",
        "roster": f"{base_url}/?public=1&page=tournament_roster&{reg_q}",
        "partner_board": f"{base_url}/?public=1&page=tournament_partner_board&{board_q}",
        "admin_manager": f"{base_url}/?page=tournament_manager&tournament_id={tournament_id}",
        "admin_operations": f"{base_url}/?page=tournaments&tournament_id={tournament_id}",
    }


def registration_is_open(settings: dict[str, Any]) -> tuple[bool, str | None]:
    status = str(settings.get("registration_status") or "draft").lower()
    if status != "open":
        return False, "Registration is not currently open."

    now = datetime.now(timezone.utc)
    open_at = settings.get("registration_open_at")
    close_at = settings.get("registration_close_at")

    if open_at:
        try:
            if datetime.fromisoformat(str(open_at).replace("Z", "+00:00")) > now:
                return False, "Registration has not opened yet."
        except Exception:
            pass
    if close_at:
        try:
            if datetime.fromisoformat(str(close_at).replace("Z", "+00:00")) < now:
                return False, "Registration is closed."
        except Exception:
            pass
    return True, None


def get_registration_by_id(supabase, tournament_id: str, registration_id: str) -> dict[str, Any] | None:
    resp = (
        supabase.table("tournament_registrations")
        .select("*")
        .eq("tournament_id", str(tournament_id))
        .eq("id", str(registration_id))
        .limit(1)
        .execute()
    )
    return _safe_first(resp)


def get_registration_confirmation_bundle(supabase, tournament_id: str, registration_id: str) -> dict[str, Any]:
    tournament_id = str(tournament_id)
    registration_id = str(registration_id)
    tournament = get_tournament_record(supabase, tournament_id) or {}
    settings = get_registration_settings(supabase, tournament_id) or {}
    registration = get_registration_by_id(supabase, tournament_id, registration_id)
    if not registration:
        return {
            "tournament": tournament,
            "settings": settings,
            "registration": None,
            "selections": [],
            "days": [],
            "event_options": [],
            "total_price_usd": 0,
        }
    days = list_registration_days(supabase, tournament_id)
    event_options = list_event_options(supabase, tournament_id)
    selections = _safe_data(
        supabase.table("tournament_registration_selections")
        .select("*")
        .eq("tournament_id", tournament_id)
        .eq("registration_id", registration_id)
        .order("sort_order")
        .execute()
    )
    event_lookup = {str(row.get("id")): row for row in event_options}
    total = 0.0
    for selection in selections:
        event = event_lookup.get(str(selection.get("event_option_id") or "")) or {}
        try:
            total += float(event.get("price_usd") or 0)
        except Exception:
            total += 0.0
    return {
        "tournament": tournament,
        "settings": settings,
        "registration": registration,
        "selections": selections,
        "days": days,
        "event_options": event_options,
        "total_price_usd": total,
    }
