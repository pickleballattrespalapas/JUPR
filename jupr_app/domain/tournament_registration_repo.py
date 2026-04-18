from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
import math
import re
import uuid

from jupr_app.domain.event_tags import derive_default_date_tags, normalize_event_tags

from .tournament_registration_compiler import compile_tournament_registration_state, validate_selection_against_skill

REGISTRATION_STATUS_OPTIONS = ["draft", "open", "closed"]
EVENT_TYPE_OPTIONS = ["SINGLES", "GENDER_DOUBLES", "MIXED_DOUBLES"]
GENDER_RESTRICTION_OPTIONS = ["ANY", "MEN", "WOMEN", "MIXED"]
PARTNER_MODE_OPTIONS = ["NONE", "HAS_PARTNER", "NEEDS_PARTNER"]
ADMIN_REGISTRATION_STATUS_OPTIONS = ["pending", "confirmed", "waitlist", "cancelled"]
ADMIN_PAYMENT_STATUS_OPTIONS = ["unpaid", "paid", "refunded"]

REGISTRATION_SCHEMA_CONTRACT_MIGRATIONS = [
    "migrations/20261010_tournament_builder_refactor.sql",
    "migrations/20261018_tournament_registration_schema_contract.sql",
]

REGISTRATION_SCHEMA_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "tournament_registration_settings": (
        "id",
        "tournament_id",
        "builder_draft_json",
        "builder_draft_updated_at",
    ),
    "tournament_registration_days": (
        "id",
        "tournament_id",
        "enabled",
    ),
    "tournament_event_options": (
        "id",
        "tournament_id",
        "registration_day_id",
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


def assert_registration_schema_contract(supabase, *, required_tables: list[str] | None = None) -> None:
    tables = required_tables or list(REGISTRATION_SCHEMA_REQUIRED_COLUMNS.keys())
    failures: list[str] = []
    for table_name in tables:
        required_columns = REGISTRATION_SCHEMA_REQUIRED_COLUMNS.get(table_name, ())
        if not required_columns:
            continue
        select_expr = ", ".join(required_columns)
        try:
            supabase.table(table_name).select(select_expr).limit(1).execute()
        except Exception as exc:
            failures.append(f"{table_name} ({select_expr}): {exc}")
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


def registration_feature_available(supabase) -> tuple[bool, str | None]:
    required_tables = [
        "tournament_registration_settings",
        "tournament_registration_days",
        "tournament_event_options",
        "tournament_registrations",
        "tournament_registration_selections",
    ]
    failures: list[str] = []
    for table_name in required_tables:
        try:
            supabase.table(table_name).select("id").limit(1).execute()
        except Exception as exc:
            failures.append(f"{table_name}: {exc}")
    if failures:
        return False, "Registration tables unavailable: " + " | ".join(failures)

    try:
        assert_registration_schema_contract(supabase)
    except ValueError as exc:
        return False, str(exc)
    return True, None


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
        "sponsor_markdown": "",
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
        "sponsor_markdown": str(payload.get("sponsor_markdown") or ""),
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
    saved_step: str | None = None,
) -> dict[str, Any]:
    payload = {
        "version": 1,
        "saved_at": _now_iso(),
        "saved_step": str(saved_step or "").strip() or None,
        "days": list(days or []),
        "event_families": list(event_families or []),
        "divisions": list(divisions or []),
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
    saved_step: str | None = None,
) -> dict[str, Any]:
    assert_registration_schema_contract(supabase, required_tables=["tournament_registration_settings"])
    settings = get_registration_settings(supabase, str(tournament_id))
    payload = build_builder_draft_payload(
        days=days,
        event_families=event_families,
        divisions=divisions,
        saved_step=saved_step,
    )
    update_payload = {
        "id": str(settings.get("id") or _uid("regset")),
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


def list_registrations(supabase, tournament_id: str) -> list[dict[str, Any]]:
    resp = (
        supabase.table("tournament_registrations")
        .select("*")
        .eq("tournament_id", str(tournament_id))
        .order("submitted_at", desc=True)
        .execute()
    )
    return _safe_data(resp)


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


def list_registration_selections(supabase, tournament_id: str) -> list[dict[str, Any]]:
    resp = (
        supabase.table("tournament_registration_selections")
        .select("*")
        .eq("tournament_id", str(tournament_id))
        .order("created_at")
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


def save_registration(supabase, *, tournament_id: str, payload: dict[str, Any]) -> dict[str, Any]:
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
    registration_id = str(existing.get("id")) if existing else _uid("reg")
    submitted_at = _now_iso()

    reg_row = {
        "id": registration_id,
        "tournament_id": str(tournament_id),
        "submitted_at": submitted_at,
        "status": str(payload.get("status") or "pending").lower(),
        "payment_status": str(payload.get("payment_status") or "unpaid").lower(),
        "first_name": str(payload.get("first_name") or "").strip() or None,
        "last_name": str(payload.get("last_name") or "").strip() or None,
        "display_name": display_name,
        "email": email,
        "phone": str(payload.get("phone") or "").strip() or None,
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

    rows: list[dict[str, Any]] = []
    for index, selection in enumerate(payload.get("selections") or []):
        if not selection.get("event_option_id"):
            continue

        event_option_id = str(selection.get("event_option_id"))
        event = event_lookup.get(event_option_id)
        if not event:
            raise ValueError(f"Selected division {event_option_id} is no longer available.")
        if str(event.get("registration_day_id") or "") not in day_lookup:
            raise ValueError("Selected division is not on an enabled registration day.")

        visibility = public_event_option_visibility(event)
        if visibility != "selectable":
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
            }
        )

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
    return compile_tournament_registration_state(
        tournament=tournament,
        settings=settings,
        days=days,
        event_options=event_options,
        registrations=registrations,
        selections=selections,
    )


def build_public_urls(*, base_url: str, tournament_id: str, registration_slug: str | None = None) -> dict[str, str]:
    tournament_id = str(tournament_id)
    base_url = str(base_url or "").rstrip("/")
    reg_q = f"tournament={registration_slug}" if registration_slug else f"tournament_id={tournament_id}"
    board_q = reg_q
    return {
        "registration": f"{base_url}/?public=1&page=tournament_registration&{reg_q}",
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
