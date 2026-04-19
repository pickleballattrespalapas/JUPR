from __future__ import annotations

import json
import math
import uuid
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from jupr_app.domain.event_tags import derive_default_date_tags, normalize_event_tags
from jupr_app.domain.tournament_registration_repo import (
    REGISTRATION_STATUS_OPTIONS,
    analyze_registration_publish_impact,
    build_public_urls,
    count_tournament_registrations,
    get_builder_draft,
    get_registration_settings,
    get_tournament_record,
    list_event_options,
    list_existing_tournaments,
    list_registration_days,
    publish_registration_configuration,
    registration_feature_available,
    save_builder_draft,
    upsert_registration_settings,
)
from jupr_app.ui.layout import page_shell

COMPETITION_FORMATS = ["ROUND_ROBIN", "SINGLE_ELIM", "DOUBLE_ELIM", "ROUND_ROBIN_PLUS_PLAYOFF"]
SCORING_OPTIONS = ["GAME_TO_11", "GAME_TO_15", "GAME_TO_21", "BEST_2_OF_3"]
AGE_MODES = ["ALL_AGES", "FIXED_AGE_BRACKET", "AUTO_AGE_SPLIT", "SPLIT_AGE"]
PARTICIPANT_TYPES = ["SINGLES", "GENDER_DOUBLES", "MIXED_DOUBLES"]
GENDER_RESTRICTIONS = ["ANY", "MEN", "WOMEN", "MIXED"]
DIVISION_STATUSES = ["draft", "open", "tentative", "confirmed", "closed"]
SELECTABLE_DIVISION_STATUSES = {"open", "tentative", "confirmed"}
VISIBLE_CLOSED_DIVISION_STATUSES = {"closed"}
SKILL_LABEL_OPTIONS = ["Open", "3.0", "3.5", "4.0", "4.5", "5.0", "5.5"]

STANDARD_EVENT_TEMPLATES = [
    {
        "event_family": "Men's Doubles",
        "participant_type": "GENDER_DOUBLES",
        "gender_restriction": "MEN",
        "default_format": "ROUND_ROBIN_PLUS_PLAYOFF",
        "default_scoring": "GAME_TO_15",
        "default_waitlist": True,
        "default_partner_board": True,
        "default_capacity_teams": 16,
        "default_price_usd": 0.0,
        "default_status": "open",
    },
    {
        "event_family": "Women's Doubles",
        "participant_type": "GENDER_DOUBLES",
        "gender_restriction": "WOMEN",
        "default_format": "ROUND_ROBIN_PLUS_PLAYOFF",
        "default_scoring": "GAME_TO_15",
        "default_waitlist": True,
        "default_partner_board": True,
        "default_capacity_teams": 16,
        "default_price_usd": 0.0,
        "default_status": "open",
    },
    {
        "event_family": "Mixed Doubles",
        "participant_type": "MIXED_DOUBLES",
        "gender_restriction": "MIXED",
        "default_format": "ROUND_ROBIN_PLUS_PLAYOFF",
        "default_scoring": "GAME_TO_15",
        "default_waitlist": True,
        "default_partner_board": True,
        "default_capacity_teams": 16,
        "default_price_usd": 0.0,
        "default_status": "open",
    },
    {
        "event_family": "Men's Singles",
        "participant_type": "SINGLES",
        "gender_restriction": "MEN",
        "default_format": "ROUND_ROBIN_PLUS_PLAYOFF",
        "default_scoring": "GAME_TO_15",
        "default_waitlist": True,
        "default_partner_board": False,
        "default_capacity_teams": 16,
        "default_price_usd": 0.0,
        "default_status": "open",
    },
    {
        "event_family": "Women's Singles",
        "participant_type": "SINGLES",
        "gender_restriction": "WOMEN",
        "default_format": "ROUND_ROBIN_PLUS_PLAYOFF",
        "default_scoring": "GAME_TO_15",
        "default_waitlist": True,
        "default_partner_board": False,
        "default_capacity_teams": 16,
        "default_price_usd": 0.0,
        "default_status": "open",
    },
]

AGE_MODE_HELP = {
    "ALL_AGES": "One combined division regardless of age.",
    "FIXED_AGE_BRACKET": "Set a fixed age label such as 50+ or 60+.",
    "AUTO_AGE_SPLIT": "Players register into one skill division. The system can later split into age groups if each group reaches the minimum number of teams.",
    "SPLIT_AGE": "Use a partner rule such as one player 50+ and one player under 50.",
}

DAYS_EDITOR_COLUMNS = ["event_date", "label", "enabled"]
EVENT_TEMPLATE_COLUMNS = [
    "event_family",
    "participant_type",
    "gender_restriction",
    "default_format",
    "default_scoring",
    "default_waitlist",
    "default_partner_board",
    "default_capacity_teams",
    "default_price_usd",
    "default_status",
]
DIVISION_EDITOR_COLUMNS = [
    "event_family",
    "division_name",
    "skill_label",
    "age_mode",
    "age_label",
    "age_ranges",
    "min_teams_per_age_group",
    "split_age_threshold",
    "assigned_day",
    "capacity_teams",
    "price_usd",
    "waitlist_enabled",
    "partner_board_enabled",
    "status",
    "division_format",
    "division_scoring",
    "notes",
    "generated_from_event",
]


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _parse_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    text = _safe_text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).date()
    except Exception:
        return None


def _fmt_dt(value: Any) -> str:
    text = _safe_text(value).replace("+00:00", "Z")
    if not text:
        return ""
    return text[:-1][:16] if text.endswith("Z") else text[:16]


def _parse_datetime_value(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    text = _safe_text(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except Exception:
        return None


def _parse_local_dt(value: Any) -> str | None:
    parsed = _parse_datetime_value(value)
    if parsed is not None:
        return parsed.isoformat()
    text = _safe_text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).isoformat()
    except Exception:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat()
        except Exception:
            return None


def _date_rows(start_date: Any, end_date: Any) -> list[dict[str, Any]]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if not start or not end or end < start:
        return []
    rows: list[dict[str, Any]] = []
    cursor = start
    idx = 1
    while cursor <= end:
        rows.append(
            {
                "event_date": cursor.isoformat(),
                "label": f"Day {idx} · {cursor.strftime('%a %b %d')}",
                "enabled": True,
            }
        )
        cursor += timedelta(days=1)
        idx += 1
    return rows


def _date_window_message(start_date: Any, end_date: Any) -> tuple[str, str]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if not start or not end:
        return (
            "warning",
            "Tournament dates are missing. Add start and end dates to auto-generate the default day schedule.",
        )
    if end < start:
        return ("error", "End date cannot be before start date. Fix dates before generating days.")
    day_count = (end - start).days + 1
    return ("info", f"Date window: {start.isoformat()} → {end.isoformat()} ({day_count} day{'s' if day_count != 1 else ''}).")


def _infer_date_window_from_days_rows(days_rows: list[dict[str, Any]] | None) -> tuple[date | None, date | None]:
    rows = days_rows or []
    if not rows:
        return None, None

    enabled_dates: list[date] = []
    all_dates: list[date] = []
    for row in rows:
        parsed_date = _parse_date((row or {}).get("event_date"))
        if not parsed_date:
            continue
        all_dates.append(parsed_date)
        if _coerce_bool((row or {}).get("enabled"), default=True):
            enabled_dates.append(parsed_date)

    preferred_dates = enabled_dates or all_dates
    if not preferred_dates:
        return None, None
    return min(preferred_dates), max(preferred_dates)


def _update_tournament_shell(supabase, tournament_id: str, *, name: str, start_date: date | None, end_date: date | None) -> None:
    payload = {
        "name": name.strip(),
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "event_tags": normalize_event_tags({
            "skill_levels": [],
            "date_tags": derive_default_date_tags(start_date=start_date, end_date=end_date),
        }),
    }
    try:
        supabase.table("tournaments").update(payload).eq("id", tournament_id).execute()
        return
    except Exception:
        pass
    supabase.table("tournaments").update({"name": name.strip()}).eq("id", tournament_id).execute()


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = _safe_text(value).lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "y", "on"}


def _coerce_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    text = _safe_text(value)
    if not text:
        return None
    try:
        number = float(text)
        if not math.isfinite(number):
            return None
        return int(number)
    except Exception:
        return None


def _coerce_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    text = _safe_text(value)
    if not text:
        return None
    try:
        number = float(text)
        if not math.isfinite(number):
            return None
        return number
    except Exception:
        return None


def _ensure_editor_columns(df: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame(columns=columns)
    out = df.copy()
    out = out.loc[:, ~pd.Index(out.columns).duplicated(keep="last")]
    for column in columns:
        if column not in out.columns:
            out[column] = pd.Series([None] * len(out), index=out.index)
    return out.loc[:, columns]


def _df_with_hidden_ids(rows: list[dict[str, Any]], id_key: str, ordered_columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=ordered_columns)
    df = pd.DataFrame(rows)
    if id_key in df.columns:
        df = df.set_index(id_key)
    return _ensure_editor_columns(df[ordered_columns], ordered_columns)


def _seed_days(days: list[dict[str, Any]], tournament: dict[str, Any]) -> pd.DataFrame:
    if days:
        rows = [
            {
                "id": str(row.get("id") or _uid("day")),
                "event_date": row.get("event_date"),
                "label": row.get("label") or "Day",
                "enabled": bool(row.get("enabled", True)),
            }
            for row in days
        ]
        return _df_with_hidden_ids(rows, "id", DAYS_EDITOR_COLUMNS)

    generated = _date_rows(tournament.get("start_date"), tournament.get("end_date"))
    if not generated:
        return pd.DataFrame(columns=DAYS_EDITOR_COLUMNS)
    rows = [{"id": _uid("day"), **row} for row in generated]
    return _df_with_hidden_ids(rows, "id", DAYS_EDITOR_COLUMNS)


def _sync_days_with_date_range(
    tournament_id: str,
    start_date: date | None,
    end_date: date | None,
    existing_days: list[dict[str, Any]],
    existing_event_options: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    generated_days = _date_rows(start_date, end_date)
    if not generated_days:
        return [], existing_event_options

    existing_by_date = {
        _safe_text(row.get("event_date")): row
        for row in existing_days
        if _safe_text(row.get("event_date"))
    }

    synced_days: list[dict[str, Any]] = []
    for sort_order, generated in enumerate(generated_days, start=1):
        existing = existing_by_date.get(_safe_text(generated.get("event_date")))
        synced_days.append(
            {
                "id": str(existing.get("id") if existing else _uid("day")),
                "tournament_id": tournament_id,
                "sort_order": sort_order,
                "label": _safe_text((existing or {}).get("label")) or _safe_text(generated.get("label")),
                "event_date": _safe_text(generated.get("event_date")) or None,
                "enabled": bool((existing or {}).get("enabled", True)),
            }
        )

    new_day_ids = [str(day.get("id")) for day in synced_days]
    old_day_id_to_new_day_id: dict[str, str] = {}
    for index, old_day in enumerate(existing_days):
        old_id = _safe_text(old_day.get("id"))
        if not old_id or not new_day_ids:
            continue
        old_date = _safe_text(old_day.get("event_date"))
        matched = next((day for day in synced_days if _safe_text(day.get("event_date")) == old_date), None)
        if matched:
            old_day_id_to_new_day_id[old_id] = str(matched.get("id"))
            continue
        old_day_id_to_new_day_id[old_id] = new_day_ids[min(index, len(new_day_ids) - 1)]

    fallback_day_id = new_day_ids[0] if new_day_ids else None
    synced_event_options: list[dict[str, Any]] = []
    for event in existing_event_options:
        existing_day_id = _safe_text(event.get("registration_day_id"))
        reassigned_day_id = old_day_id_to_new_day_id.get(existing_day_id) or fallback_day_id
        synced_event_options.append({**event, "registration_day_id": reassigned_day_id})

    return synced_days, synced_event_options


def _clear_tournament_manager_state(tournament_id: str) -> None:
    stale_keys = [
        f"tm_days_seed_{tournament_id}",
        f"tm_events_seed_{tournament_id}",
        f"tm_divisions_seed_{tournament_id}",
        f"tm_days_editor_{tournament_id}",
        f"tm_events_editor_{tournament_id}",
        f"tm_divisions_editor_{tournament_id}",
    ]
    for key in stale_keys:
        st.session_state.pop(key, None)


def _seed_event_templates(event_options: list[dict[str, Any]]) -> pd.DataFrame:
    grouped: dict[str, dict[str, Any]] = {}
    for row in event_options:
        family = _safe_text(row.get("event_family_label") or row.get("label") or "Event")
        if not family:
            continue
        grouped.setdefault(
            family,
            {
                "id": _uid("evt"),
                "event_family": family,
                "participant_type": row.get("event_type") or "SINGLES",
                "gender_restriction": row.get("gender_restriction") or "ANY",
                "default_format": row.get("event_format_default") or "ROUND_ROBIN_PLUS_PLAYOFF",
                "default_scoring": row.get("scoring_default") or "GAME_TO_15",
                "default_waitlist": bool(row.get("waitlist_enabled", True)),
                "default_partner_board": bool(row.get("partner_board_enabled", row.get("public_partner_board", True))),
                "default_capacity_teams": _coerce_int(row.get("capacity_teams")) or 16,
                "default_price_usd": _coerce_float(row.get("price_usd")) or 0.0,
                "default_status": _safe_text(row.get("status") or "open"),
            },
        )
    rows = list(grouped.values())
    return _df_with_hidden_ids(
        rows,
        "id",
        EVENT_TEMPLATE_COLUMNS,
    )


def _draft_rows_to_df(rows: list[dict[str, Any]], columns: list[str], id_key: str = "id") -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized_rows.append(
            {
                id_key: str((row or {}).get(id_key) or _uid(id_key)),
                **{column: (row or {}).get(column) for column in columns},
            }
        )
    return _df_with_hidden_ids(normalized_rows, id_key, columns)


def _empty_event_templates_df() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_TEMPLATE_COLUMNS)


def _standard_event_templates_df() -> pd.DataFrame:
    rows = [{"id": _uid("evt"), **row} for row in STANDARD_EVENT_TEMPLATES]
    return _df_with_hidden_ids(
        rows,
        "id",
        EVENT_TEMPLATE_COLUMNS,
    )


def _empty_divisions_df() -> pd.DataFrame:
    return pd.DataFrame(columns=DIVISION_EDITOR_COLUMNS)


def _event_family_name_exists(events_df: pd.DataFrame, event_family: str, exclude_id: str | None = None) -> bool:
    needle = _safe_text(event_family).lower()
    if not needle:
        return False
    for event_id, row in events_df.iterrows():
        if exclude_id is not None and str(event_id) == str(exclude_id):
            continue
        if _safe_text(row.get("event_family")).lower() == needle:
            return True
    return False


def _add_event_family_row(events_df: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
    out = _ensure_editor_columns(events_df, EVENT_TEMPLATE_COLUMNS).copy()
    row_id = _uid("evt")
    for column in EVENT_TEMPLATE_COLUMNS:
        if column not in out.columns:
            out[column] = None
        out.at[row_id, column] = payload.get(column)
    return _ensure_editor_columns(out, EVENT_TEMPLATE_COLUMNS)


def _update_event_family_row(events_df: pd.DataFrame, event_id: str, payload: dict[str, Any]) -> pd.DataFrame:
    out = _ensure_editor_columns(events_df, EVENT_TEMPLATE_COLUMNS).copy()
    if str(event_id) not in {str(idx) for idx in out.index.tolist()}:
        return out
    for column in EVENT_TEMPLATE_COLUMNS:
        if column not in out.columns:
            out[column] = None
        out.at[event_id, column] = payload.get(column)
    return _ensure_editor_columns(out, EVENT_TEMPLATE_COLUMNS)


def _delete_event_family_row(events_df: pd.DataFrame, event_id: str) -> pd.DataFrame:
    out = _ensure_editor_columns(events_df, EVENT_TEMPLATE_COLUMNS).copy()
    if str(event_id) in {str(idx) for idx in out.index.tolist()}:
        out = out.drop(index=event_id)
    return _ensure_editor_columns(out, EVENT_TEMPLATE_COLUMNS)


def _add_division_row(divisions_df: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
    out = _ensure_editor_columns(divisions_df, DIVISION_EDITOR_COLUMNS).copy()
    out.loc[_uid("div"), DIVISION_EDITOR_COLUMNS] = [payload.get(column) for column in DIVISION_EDITOR_COLUMNS]
    return _ensure_editor_columns(out, DIVISION_EDITOR_COLUMNS)


def _update_division_row(divisions_df: pd.DataFrame, division_id: str, payload: dict[str, Any]) -> pd.DataFrame:
    out = _ensure_editor_columns(divisions_df, DIVISION_EDITOR_COLUMNS).copy()
    if str(division_id) not in {str(idx) for idx in out.index.tolist()}:
        return out
    out.loc[division_id, DIVISION_EDITOR_COLUMNS] = [payload.get(column) for column in DIVISION_EDITOR_COLUMNS]
    return _ensure_editor_columns(out, DIVISION_EDITOR_COLUMNS)


def _delete_division_row(divisions_df: pd.DataFrame, division_id: str) -> pd.DataFrame:
    out = _ensure_editor_columns(divisions_df, DIVISION_EDITOR_COLUMNS).copy()
    if str(division_id) in {str(idx) for idx in out.index.tolist()}:
        out = out.drop(index=division_id)
    return _ensure_editor_columns(out, DIVISION_EDITOR_COLUMNS)


def _sanitize_divisions_for_event_families(divisions_df: pd.DataFrame, event_families: list[str]) -> pd.DataFrame:
    divisions_df = _ensure_editor_columns(divisions_df, DIVISION_EDITOR_COLUMNS)
    if divisions_df.empty:
        return divisions_df
    valid_families = [_safe_text(family) for family in event_families if _safe_text(family)]
    if not valid_families:
        return _empty_divisions_df()
    fallback = valid_families[0]
    out = divisions_df.copy()
    out["event_family"] = out["event_family"].apply(
        lambda value: value if _safe_text(value) in valid_families else fallback
    )
    return out


def _coerce_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_safe_text(item) for item in value if _safe_text(item)]
    text = _safe_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [_safe_text(item) for item in parsed if _safe_text(item)]
    except Exception:
        pass
    return [part.strip() for part in text.split(",") if part.strip()]


def _compose_division_name(event_name: str, skill_label: str, age_mode: str, age_label: str) -> str:
    parts = [_safe_text(event_name), _safe_text(skill_label)]
    if _safe_text(age_mode) == "FIXED_AGE_BRACKET" and _safe_text(age_label):
        parts.append(_safe_text(age_label))
    return " ".join(part for part in parts if part).strip()


def _age_variants_for_mode(age_mode: str, age_labels: list[str]) -> list[str]:
    if age_mode == "FIXED_AGE_BRACKET":
        return age_labels or ["All Ages"]
    if age_mode == "AUTO_AGE_SPLIT":
        return ["Auto Age Split"]
    if age_mode == "SPLIT_AGE":
        return ["Split Age"]
    return ["All Ages"]


def _division_generation_preview_count(*, selected_days: list[str], selected_skills: list[str], age_mode: str, age_labels: list[str]) -> int:
    age_variants = _age_variants_for_mode(age_mode, age_labels)
    return len(selected_days) * len(selected_skills) * len(age_variants)


def _generate_division_rows_for_event_batch(
    *,
    event_row: pd.Series,
    selected_days: list[str],
    selected_skills: list[str],
    age_mode: str,
    age_labels: list[str],
    batch_overrides: dict[str, Any],
    current_divisions_df: pd.DataFrame,
    generation_mode: str,
) -> tuple[pd.DataFrame, int, int]:
    out = _ensure_editor_columns(current_divisions_df, DIVISION_EDITOR_COLUMNS).copy()
    event_name = _safe_text(event_row.get("event_family"))
    days = [_safe_text(day) for day in selected_days if _safe_text(day)]
    skills = [_safe_text(skill) for skill in selected_skills if _safe_text(skill)] or ["Open"]
    age_mode = _safe_text(age_mode) or "ALL_AGES"
    age_labels = [_safe_text(label) for label in age_labels if _safe_text(label)]

    if not event_name or not days or not skills:
        return out, 0, 0

    age_variants = _age_variants_for_mode(age_mode, age_labels)

    generation_keys: list[tuple[str, str, str, str]] = []
    for day in days:
        for skill in skills:
            for age_label in age_variants:
                generation_keys.append((event_name.lower(), _safe_text(day).lower(), _safe_text(skill).lower(), _safe_text(age_label).lower()))

    if generation_mode == "replace_generated_set":
        keep_rows: list[pd.Series] = []
        for _, row in out.iterrows():
            existing_key = (
                _safe_text(row.get("event_family")).lower(),
                _safe_text(row.get("assigned_day")).lower(),
                _safe_text(row.get("skill_label")).lower(),
                _safe_text(row.get("age_label")).lower(),
            )
            generated = bool(row.get("generated_from_event", False))
            if generated and existing_key in set(generation_keys):
                continue
            keep_rows.append(row)
        if keep_rows:
            out = pd.DataFrame(keep_rows).reset_index(drop=True)
            out.index = [_uid("div") for _ in range(len(out))]
        else:
            out = _empty_divisions_df()

    existing_keys = {
        (
            _safe_text(row.get("event_family")).lower(),
            _safe_text(row.get("assigned_day")).lower(),
            _safe_text(row.get("skill_label")).lower(),
            _safe_text(row.get("age_label")).lower(),
        )
        for _, row in out.iterrows()
    }

    created = 0
    skipped = 0
    for day in days:
        for skill in skills:
            for age_label in age_variants:
                key = (event_name.lower(), _safe_text(day).lower(), _safe_text(skill).lower(), _safe_text(age_label).lower())
                if key in existing_keys:
                    skipped += 1
                    continue
                capacity_override = _coerce_int(batch_overrides.get("capacity_teams"))
                price_override = _coerce_float(batch_overrides.get("price_usd"))
                status_override = _safe_text(batch_overrides.get("status"))
                format_override = _safe_text(batch_overrides.get("division_format"))
                scoring_override = _safe_text(batch_overrides.get("division_scoring"))
                notes_default = _safe_text(batch_overrides.get("notes"))
                payload = {
                    "event_family": event_name,
                    "division_name": _compose_division_name(event_name, skill, age_mode, age_label),
                    "skill_label": _safe_text(skill) or "Open",
                    "age_mode": age_mode or "ALL_AGES",
                    "age_label": _safe_text(age_label) or "All Ages",
                    "age_ranges": "",
                    "min_teams_per_age_group": _coerce_int(event_row.get("min_teams_per_age_group")) if age_mode == "AUTO_AGE_SPLIT" else None,
                    "split_age_threshold": 50 if age_mode == "SPLIT_AGE" else None,
                    "assigned_day": _safe_text(day),
                    "capacity_teams": capacity_override if capacity_override is not None else (_coerce_int(event_row.get("default_capacity_teams")) or 16),
                    "price_usd": price_override if price_override is not None else (_coerce_float(event_row.get("default_price_usd")) or 0.0),
                    "waitlist_enabled": bool(batch_overrides.get("waitlist_enabled"))
                    if batch_overrides.get("waitlist_enabled") is not None
                    else bool(event_row.get("default_waitlist", True)),
                    "partner_board_enabled": bool(batch_overrides.get("partner_board_enabled"))
                    if batch_overrides.get("partner_board_enabled") is not None
                    else bool(event_row.get("default_partner_board", True)),
                    "status": status_override or _safe_text(event_row.get("default_status") or "open"),
                    "division_format": format_override,
                    "division_scoring": scoring_override,
                    "notes": notes_default,
                    "generated_from_event": True,
                }
                out = _add_division_row(out, payload)
                existing_keys.add(key)
                created += 1

    return _ensure_editor_columns(out, DIVISION_EDITOR_COLUMNS), created, skipped


def _parse_age_rules(raw_value: Any) -> dict[str, Any]:
    text = _safe_text(raw_value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {"notes": text}


def _seed_divisions(days_df: pd.DataFrame, event_templates_df: pd.DataFrame, event_options: list[dict[str, Any]]) -> pd.DataFrame:
    days_df = _ensure_editor_columns(days_df, DAYS_EDITOR_COLUMNS)
    event_templates_df = _ensure_editor_columns(event_templates_df, EVENT_TEMPLATE_COLUMNS)
    day_lookup = {str(idx): _safe_text(row.get("label") or idx) for idx, row in days_df.to_dict("index").items()}
    default_event_family = next(iter(event_templates_df["event_family"].tolist()), "") if not event_templates_df.empty else ""
    rows: list[dict[str, Any]] = []
    for row in event_options:
        age_rules = _parse_age_rules(row.get("age_rules"))
        rows.append(
            {
                "id": str(row.get("id") or _uid("div")),
                "event_family": _safe_text(row.get("event_family_label") or row.get("label") or default_event_family),
                "division_name": _safe_text(row.get("division_name") or row.get("label")),
                "skill_label": _safe_text(row.get("skill_label") or "Open"),
                "age_mode": _safe_text(row.get("age_mode") or "ALL_AGES"),
                "age_label": _safe_text(row.get("age_label") or "All Ages"),
                "age_ranges": _safe_text(age_rules.get("age_ranges") or age_rules.get("groups")),
                "min_teams_per_age_group": _coerce_int(age_rules.get("min_teams_per_age_group") or age_rules.get("min_teams")),
                "split_age_threshold": _coerce_int(age_rules.get("split_age_threshold") or age_rules.get("one_over") or age_rules.get("threshold")),
                "assigned_day": day_lookup.get(str(row.get("registration_day_id")), ""),
                "capacity_teams": row.get("capacity_teams"),
                "price_usd": row.get("price_usd"),
                "waitlist_enabled": bool(row.get("waitlist_enabled", True)),
                "partner_board_enabled": bool(row.get("partner_board_enabled", row.get("public_partner_board", True))),
                "status": _safe_text(row.get("status") or "draft").lower(),
                "division_format": _safe_text(row.get("event_format_override") or ""),
                "division_scoring": _safe_text(row.get("scoring_override") or ""),
                "notes": _safe_text(age_rules.get("notes")),
                "generated_from_event": False,
            }
        )

    return _df_with_hidden_ids(
        rows,
        "id",
        DIVISION_EDITOR_COLUMNS,
    )


def _encode_age_rules(row: pd.Series) -> str | None:
    mode = _safe_text(row.get("age_mode") or "ALL_AGES")
    payload: dict[str, Any] = {
        "mode": mode,
        "younger_player_controls_age": True,
        "higher_skill_player_controls_skill": True,
    }
    age_label = _safe_text(row.get("age_label"))
    if age_label:
        payload["age_label"] = age_label

    age_ranges = _safe_text(row.get("age_ranges"))
    if age_ranges:
        payload["age_ranges"] = age_ranges

    min_teams = _coerce_int(row.get("min_teams_per_age_group"))
    if min_teams is not None:
        payload["min_teams_per_age_group"] = min_teams

    split_threshold = _coerce_int(row.get("split_age_threshold"))
    if split_threshold is not None:
        payload["split_age_threshold"] = split_threshold
        payload["split_age_rule"] = {"one_player_over_or_equal": split_threshold, "one_player_under": split_threshold}

    notes = _safe_text(row.get("notes"))
    if notes:
        payload["notes"] = notes

    if mode == "ALL_AGES" and not notes and not age_ranges and split_threshold is None and min_teams is None:
        return None
    return json.dumps(payload, ensure_ascii=False)


def _display_division_name(row: pd.Series) -> str:
    name = _safe_text(row.get("division_name"))
    if name:
        return name
    family = _safe_text(row.get("event_family") or "Division")
    skill = _safe_text(row.get("skill_label"))
    age = _safe_text(row.get("age_label"))
    suffix = " ".join(part for part in [skill, age] if part and part.lower() not in {"open", "all ages"})
    return f"{family} {suffix}".strip() or family


def _display_optional(value: Any, fallback: str = "—") -> str:
    text = _safe_text(value)
    return text if text else fallback


def _display_labelized(value: Any, fallback: str = "—") -> str:
    text = _safe_text(value).replace("_", " ")
    text = " ".join(text.split())
    if not text:
        return fallback
    return text.title()


def _truncate_text(value: Any, max_len: int = 40) -> str:
    text = _safe_text(value)
    if not text:
        return "—"
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 1].rstrip()}…"


def _resolve_division_format(row: pd.Series, template_lookup: dict[str, pd.Series]) -> str:
    direct_value = _safe_text(row.get("division_format"))
    if direct_value:
        return _display_labelized(direct_value)
    event_family = _safe_text(row.get("event_family"))
    template_row = template_lookup.get(event_family)
    fallback_value = _safe_text(template_row.get("default_format")) if template_row is not None else ""
    return _display_labelized(fallback_value)


def _resolve_division_scoring(row: pd.Series, template_lookup: dict[str, pd.Series]) -> str:
    direct_value = _safe_text(row.get("division_scoring"))
    if direct_value:
        return _display_labelized(direct_value)
    event_family = _safe_text(row.get("event_family"))
    template_row = template_lookup.get(event_family)
    fallback_value = _safe_text(template_row.get("default_scoring")) if template_row is not None else ""
    return _display_labelized(fallback_value)


def _validate_builder(days_df: pd.DataFrame, event_templates_df: pd.DataFrame, divisions_df: pd.DataFrame) -> list[str]:
    days_df = _ensure_editor_columns(days_df, DAYS_EDITOR_COLUMNS)
    event_templates_df = _ensure_editor_columns(event_templates_df, EVENT_TEMPLATE_COLUMNS)
    divisions_df = _ensure_editor_columns(divisions_df, DIVISION_EDITOR_COLUMNS)

    errors: list[str] = []
    enabled_days = days_df[days_df["enabled"].fillna(False).astype(bool)] if not days_df.empty else pd.DataFrame(columns=DAYS_EDITOR_COLUMNS)
    if enabled_days.empty:
        errors.append("Enable at least one tournament day.")

    if event_templates_df.empty:
        errors.append("Create at least one event.")
    else:
        families = [_safe_text(v) for v in event_templates_df["event_family"].tolist()]
        blank_count = sum(1 for v in families if not v)
        if blank_count:
            errors.append("Every event needs a name.")
        if len({v.lower() for v in families if v}) != len([v for v in families if v]):
            errors.append("Event names must be unique.")

    if divisions_df.empty:
        errors.append("Create at least one division.")
    else:
        family_options = {v for v in event_templates_df["event_family"].tolist() if _safe_text(v)}
        day_labels = {v for v in enabled_days["label"].tolist() if _safe_text(v)}
        for idx, row in divisions_df.iterrows():
            event_family = _safe_text(row.get("event_family"))
            assigned_day = _safe_text(row.get("assigned_day"))
            division_name = _display_division_name(row)
            if event_family not in family_options:
                errors.append(f"Division '{division_name}' references an event that is not defined.")
            if assigned_day not in day_labels:
                errors.append(f"Division '{division_name}' must be assigned to an enabled day.")
            if _safe_text(row.get("age_mode")) == "AUTO_AGE_SPLIT" and _coerce_int(row.get("min_teams_per_age_group")) is None:
                errors.append(f"Division '{division_name}' uses Auto Age Split and needs a minimum teams per age group value.")
            if _safe_text(row.get("age_mode")) == "SPLIT_AGE" and _coerce_int(row.get("split_age_threshold")) is None:
                errors.append(f"Division '{division_name}' uses Split Age and needs a threshold such as 50.")
    return sorted(set(errors))


def _build_payloads(
    tournament_id: str,
    days_df: pd.DataFrame,
    event_templates_df: pd.DataFrame,
    divisions_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Canonical persisted registration contract (consumed by public registration):
    - days: id/tournament_id/sort_order/label/event_date/enabled
    - event options: day assignment, family/division labels, skill+age labels and modes,
      format/scoring defaults and overrides, capacity, partner/waitlist controls,
      and publication semantics via status + enabled.
    """
    days_df = _ensure_editor_columns(days_df, DAYS_EDITOR_COLUMNS)
    event_templates_df = _ensure_editor_columns(event_templates_df, EVENT_TEMPLATE_COLUMNS)
    divisions_df = _ensure_editor_columns(divisions_df, DIVISION_EDITOR_COLUMNS)

    day_payload: list[dict[str, Any]] = []
    day_label_to_id: dict[str, str] = {}
    for sort_order, (day_id, row) in enumerate(days_df.iterrows(), start=1):
        if not bool(row.get("enabled", True)):
            continue
        label = _safe_text(row.get("label") or f"Day {sort_order}")
        current_day_id = str(day_id or _uid("day"))
        day_label_to_id[label] = current_day_id
        day_payload.append(
            {
                "id": current_day_id,
                "tournament_id": str(tournament_id),
                "sort_order": sort_order,
                "label": label,
                "event_date": _safe_text(row.get("event_date")) or None,
                "enabled": True,
            }
        )

    template_lookup = {
        _safe_text(row.get("event_family")): row for _, row in event_templates_df.iterrows() if _safe_text(row.get("event_family"))
    }

    event_payload: list[dict[str, Any]] = []
    for sort_order, (division_id, row) in enumerate(divisions_df.iterrows(), start=1):
        event_family = _safe_text(row.get("event_family"))
        assigned_day = _safe_text(row.get("assigned_day"))
        if assigned_day not in day_label_to_id or event_family not in template_lookup:
            continue
        template = template_lookup[event_family]
        participant_type = _safe_text(template.get("participant_type") or "SINGLES")
        division_name = _display_division_name(row)
        event_payload.append(
            {
                "id": str(division_id or _uid("event")),
                "tournament_id": str(tournament_id),
                "registration_day_id": day_label_to_id[assigned_day],
                "sort_order": sort_order,
                "label": division_name,
                "event_type": participant_type,
                "gender_restriction": _safe_text(template.get("gender_restriction") or "ANY"),
                "skill_label": _safe_text(row.get("skill_label") or "Open"),
                "skill_mode": "OPEN" if _safe_text(row.get("skill_label") or "Open").lower() == "open" else "SKILL_BRACKET",
                "age_label": _safe_text(row.get("age_label") or "All Ages"),
                "partner_required": participant_type != "SINGLES",
                "capacity_teams": _coerce_int(row.get("capacity_teams")),
                "public_partner_board": bool(row.get("partner_board_enabled", template.get("default_partner_board", True))),
                "price_usd": _coerce_float(row.get("price_usd")),
                "event_family_label": event_family,
                "division_name": division_name,
                "event_format_default": _safe_text(template.get("default_format") or "ROUND_ROBIN_PLUS_PLAYOFF"),
                "scoring_default": _safe_text(template.get("default_scoring") or "GAME_TO_15"),
                "event_format_override": _safe_text(row.get("division_format")) or None,
                "scoring_override": _safe_text(row.get("division_scoring")) or None,
                "age_mode": _safe_text(row.get("age_mode") or "ALL_AGES"),
                "age_rules": _encode_age_rules(row),
                "waitlist_enabled": bool(row.get("waitlist_enabled", template.get("default_waitlist", True))),
                "partner_board_enabled": bool(row.get("partner_board_enabled", template.get("default_partner_board", True))),
                "status": _safe_text(row.get("status") or "draft").lower(),
                "enabled": True,
            }
        )

    return day_payload, event_payload


def _registration_readiness_summary(event_payload: list[dict[str, Any]]) -> dict[str, int]:
    total = len(event_payload)
    selectable = 0
    visible_closed = 0
    hidden = 0
    hidden_draft = 0
    for row in event_payload:
        status = _safe_text(row.get("status") or "draft").lower()
        enabled = bool(row.get("enabled", True))
        if not enabled or status in {"draft", "disabled"}:
            hidden += 1
            if status == "draft":
                hidden_draft += 1
        elif status in SELECTABLE_DIVISION_STATUSES:
            selectable += 1
        elif status in VISIBLE_CLOSED_DIVISION_STATUSES:
            visible_closed += 1
        else:
            hidden += 1
    return {
        "total": total,
        "selectable": selectable,
        "visible_closed": visible_closed,
        "hidden": hidden,
        "hidden_draft": hidden_draft,
    }


def _schedule_preview_rows(days_df: pd.DataFrame, divisions_df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    days_df = _ensure_editor_columns(days_df, DAYS_EDITOR_COLUMNS)
    divisions_df = _ensure_editor_columns(divisions_df, DIVISION_EDITOR_COLUMNS)
    grouped_rows: list[tuple[str, pd.DataFrame]] = []
    if days_df.empty or divisions_df.empty:
        return grouped_rows
    for _, day_row in days_df.iterrows():
        if not bool(day_row.get("enabled", True)):
            continue
        day_label = _safe_text(day_row.get("label"))
        day_divisions = divisions_df[divisions_df["assigned_day"] == day_label]
        if day_divisions.empty:
            continue
        preview = (
            day_divisions[[
                "event_family",
                "division_name",
                "skill_label",
                "age_mode",
                "age_label",
                "division_format",
                "division_scoring",
            ]]
            .rename(
                columns={
                    "event_family": "Event",
                    "division_name": "Division",
                    "skill_label": "Skill",
                    "age_mode": "Age Mode",
                    "age_label": "Age Label",
                    "division_format": "Format Override",
                    "division_scoring": "Scoring Override",
                }
            )
            .copy()
        )
        grouped_rows.append((day_label, preview))
    return grouped_rows


def _df_to_draft_rows(df: pd.DataFrame, columns: list[str], id_key: str = "id") -> list[dict[str, Any]]:
    out = _ensure_editor_columns(df, columns)
    rows: list[dict[str, Any]] = []
    for row_id, row in out.iterrows():
        rows.append(
            {
                id_key: str(row_id or _uid(id_key)),
                **{column: row.get(column) for column in columns},
            }
        )
    return rows


def _render_event_family_form(
    *,
    form_key: str,
    mode: str,
    defaults: dict[str, Any] | None,
    submit_label: str,
    disabled: bool,
) -> tuple[bool, bool, dict[str, Any]]:
    defaults = defaults or {}
    with st.form(form_key):
        col1, col2 = st.columns(2)
        with col1:
            event_family = st.text_input("Event name", value=_safe_text(defaults.get("event_family")), disabled=disabled)
            participant_type = st.selectbox(
                "Participant type",
                PARTICIPANT_TYPES,
                index=PARTICIPANT_TYPES.index(_safe_text(defaults.get("participant_type") or "SINGLES"))
                if _safe_text(defaults.get("participant_type") or "SINGLES") in PARTICIPANT_TYPES
                else 0,
                disabled=disabled,
            )
            gender_restriction = st.selectbox(
                "Gender restriction",
                GENDER_RESTRICTIONS,
                index=GENDER_RESTRICTIONS.index(_safe_text(defaults.get("gender_restriction") or "ANY"))
                if _safe_text(defaults.get("gender_restriction") or "ANY") in GENDER_RESTRICTIONS
                else 0,
                disabled=disabled,
            )
        with col2:
            default_format = st.selectbox(
                "Default format",
                COMPETITION_FORMATS,
                index=COMPETITION_FORMATS.index(_safe_text(defaults.get("default_format") or "ROUND_ROBIN_PLUS_PLAYOFF"))
                if _safe_text(defaults.get("default_format") or "ROUND_ROBIN_PLUS_PLAYOFF") in COMPETITION_FORMATS
                else 0,
                disabled=disabled,
            )
            default_scoring = st.selectbox(
                "Default scoring",
                SCORING_OPTIONS,
                index=SCORING_OPTIONS.index(_safe_text(defaults.get("default_scoring") or "GAME_TO_15"))
                if _safe_text(defaults.get("default_scoring") or "GAME_TO_15") in SCORING_OPTIONS
                else 0,
                disabled=disabled,
            )
            default_waitlist = st.checkbox("Default waitlist", value=bool(defaults.get("default_waitlist", True)), disabled=disabled)
            default_partner_board = st.checkbox(
                "Default partner board",
                value=bool(defaults.get("default_partner_board", True)),
                disabled=disabled,
            )
            default_capacity_teams = st.number_input(
                "Default capacity",
                min_value=1,
                step=1,
                value=int(_coerce_int(defaults.get("default_capacity_teams")) or 16),
                disabled=disabled,
            )
            default_price_usd = st.number_input(
                "Default price",
                min_value=0.0,
                step=1.0,
                value=float(_coerce_float(defaults.get("default_price_usd")) or 0.0),
                disabled=disabled,
            )
            default_status = st.selectbox(
                "Default status",
                DIVISION_STATUSES,
                index=DIVISION_STATUSES.index(_safe_text(defaults.get("default_status") or "open"))
                if _safe_text(defaults.get("default_status") or "open") in DIVISION_STATUSES
                else DIVISION_STATUSES.index("open"),
                disabled=disabled,
            )
        submit_col, cancel_col = st.columns(2)
        submitted = submit_col.form_submit_button(submit_label, type="primary", disabled=disabled)
        canceled = cancel_col.form_submit_button("Cancel")

    payload = {
        "event_family": _safe_text(event_family),
        "participant_type": participant_type,
        "gender_restriction": gender_restriction,
        "default_format": default_format,
        "default_scoring": default_scoring,
        "default_waitlist": bool(default_waitlist),
        "default_partner_board": bool(default_partner_board),
        "default_capacity_teams": _coerce_int(default_capacity_teams) or 16,
        "default_price_usd": _coerce_float(default_price_usd) or 0.0,
        "default_status": _safe_text(default_status) or "open",
    }
    return submitted, canceled, payload


def _render_event_generation_form(
    *,
    form_key: str,
    event_defaults: dict[str, Any],
    day_label_options: list[str],
    disabled: bool,
) -> tuple[bool, bool, dict[str, Any]]:
    with st.form(form_key):
        st.caption(f"Event Family: **{_safe_text(event_defaults.get('event_family') or 'Unnamed Event')}**")
        left_col, right_col = st.columns(2)
        with left_col:
            selected_days = st.multiselect("Selected days", options=day_label_options, disabled=disabled)
            selected_skill_levels = st.multiselect(
                "Selected skill levels",
                options=SKILL_LABEL_OPTIONS,
                default=["Open"],
                disabled=disabled,
            )
            age_mode = st.selectbox("Age mode", AGE_MODES, index=0, disabled=disabled)
            if age_mode == "FIXED_AGE_BRACKET":
                age_labels = st.multiselect(
                    "Age labels",
                    options=["18+", "35+", "50+", "60+", "70+"],
                    disabled=disabled,
                )
            else:
                age_labels = []
        with right_col:
            st.markdown("##### Batch defaults for this run")
            override_capacity_enabled = st.checkbox("Override capacity", value=False, disabled=disabled)
            capacity_teams = st.number_input(
                "Capacity",
                min_value=1,
                step=1,
                value=int(_coerce_int(event_defaults.get("default_capacity_teams")) or 16),
                disabled=disabled or not override_capacity_enabled,
            )
            override_price_enabled = st.checkbox("Override price", value=False, disabled=disabled)
            price_usd = st.number_input(
                "Price",
                min_value=0.0,
                step=1.0,
                value=float(_coerce_float(event_defaults.get("default_price_usd")) or 0.0),
                disabled=disabled or not override_price_enabled,
            )
            override_status_enabled = st.checkbox("Override status", value=False, disabled=disabled)
            status = st.selectbox(
                "Status",
                DIVISION_STATUSES,
                index=DIVISION_STATUSES.index(_safe_text(event_defaults.get("default_status") or "open"))
                if _safe_text(event_defaults.get("default_status") or "open") in DIVISION_STATUSES
                else DIVISION_STATUSES.index("open"),
                disabled=disabled or not override_status_enabled,
            )
            waitlist_enabled = st.checkbox("Waitlist enabled", value=bool(event_defaults.get("default_waitlist", True)), disabled=disabled)
            partner_board_enabled = st.checkbox(
                "Partner board enabled",
                value=bool(event_defaults.get("default_partner_board", True)),
                disabled=disabled,
            )
            format_override = st.selectbox("Format override", [""] + COMPETITION_FORMATS, index=0, disabled=disabled)
            scoring_override = st.selectbox("Scoring override", [""] + SCORING_OPTIONS, index=0, disabled=disabled)
            notes_default = st.text_area("Notes default", value="", height=100, disabled=disabled)

        generation_mode = st.radio(
            "Duplicate handling",
            options=["create_missing_only", "replace_generated_set"],
            format_func=lambda value: "Create missing only" if value == "create_missing_only" else "Replace generated set",
            index=0,
            horizontal=True,
            disabled=disabled,
        )
        preview_count = _division_generation_preview_count(
            selected_days=selected_days,
            selected_skills=selected_skill_levels or ["Open"],
            age_mode=age_mode,
            age_labels=age_labels,
        )
        st.caption(f"Preview: {preview_count} potential division(s).")
        submit_col, cancel_col = st.columns(2)
        submitted = submit_col.form_submit_button("Generate Divisions", type="primary", disabled=disabled)
        canceled = cancel_col.form_submit_button("Cancel")

    payload = {
        "selected_days": selected_days,
        "selected_skill_levels": selected_skill_levels or ["Open"],
        "age_mode": _safe_text(age_mode) or "ALL_AGES",
        "age_labels": age_labels,
        "generation_mode": generation_mode,
        "batch_overrides": {
            "capacity_teams": _coerce_int(capacity_teams) if override_capacity_enabled else None,
            "price_usd": _coerce_float(price_usd) if override_price_enabled else None,
            "status": _safe_text(status) if override_status_enabled else "",
            "waitlist_enabled": bool(waitlist_enabled),
            "partner_board_enabled": bool(partner_board_enabled),
            "division_format": _safe_text(format_override),
            "division_scoring": _safe_text(scoring_override),
            "notes": _safe_text(notes_default),
        },
    }
    return submitted, canceled, payload


def _render_division_form(
    *,
    form_key: str,
    mode: str,
    defaults: dict[str, Any] | None,
    event_family_options: list[str],
    day_label_options: list[str],
    submit_label: str,
    disabled: bool,
) -> tuple[bool, bool, dict[str, Any]]:
    defaults = defaults or {}
    with st.form(form_key):
        col1, col2 = st.columns(2)
        with col1:
            event_family = st.selectbox(
                "Event",
                event_family_options,
                index=event_family_options.index(_safe_text(defaults.get("event_family")))
                if _safe_text(defaults.get("event_family")) in event_family_options
                else 0,
                disabled=disabled,
            )
            division_name = st.text_input("Division name", value=_safe_text(defaults.get("division_name")), disabled=disabled)
            saved_skill_label = _safe_text(defaults.get("skill_label") or "Open")
            skill_label = st.selectbox(
                "Skill level",
                SKILL_LABEL_OPTIONS,
                index=SKILL_LABEL_OPTIONS.index(saved_skill_label) if saved_skill_label in SKILL_LABEL_OPTIONS else 0,
                disabled=disabled,
            )
            st.caption(
                "Skill level is a controlled division band. Open has no skill restriction. For rated divisions, doubles teams play to the higher-rated player: at least one player must be in-band and no player may be above the band."
            )
            age_mode = st.selectbox(
                "Age mode",
                AGE_MODES,
                index=AGE_MODES.index(_safe_text(defaults.get("age_mode") or "ALL_AGES"))
                if _safe_text(defaults.get("age_mode") or "ALL_AGES") in AGE_MODES
                else 0,
                disabled=disabled,
            )
            age_label = st.text_input("Age label", value=_safe_text(defaults.get("age_label") or "All Ages"), disabled=disabled)
            age_ranges = st.text_input("Custom age ranges", value=_safe_text(defaults.get("age_ranges")), disabled=disabled)
            min_teams_per_age_group = st.number_input(
                "Min teams per age group",
                min_value=1,
                step=1,
                value=int(_coerce_int(defaults.get("min_teams_per_age_group")) or 1),
                disabled=disabled,
            )
            split_age_threshold = st.number_input(
                "Split-age threshold",
                min_value=1,
                step=1,
                value=int(_coerce_int(defaults.get("split_age_threshold")) or 50),
                disabled=disabled,
            )
            assigned_day = st.selectbox(
                "Assigned day",
                day_label_options,
                index=day_label_options.index(_safe_text(defaults.get("assigned_day")))
                if _safe_text(defaults.get("assigned_day")) in day_label_options
                else 0,
                disabled=disabled,
            )
        with col2:
            capacity_default = _coerce_int(defaults.get("capacity_teams"))
            price_default = _coerce_float(defaults.get("price_usd"))
            capacity_teams = st.number_input(
                "Capacity teams",
                min_value=1,
                step=1,
                value=int(capacity_default or 16),
                disabled=disabled,
            )
            price_usd = st.number_input("Price USD", min_value=0.0, step=1.0, value=float(price_default or 0.0), disabled=disabled)
            waitlist_enabled = st.checkbox("Waitlist enabled", value=bool(defaults.get("waitlist_enabled", True)), disabled=disabled)
            partner_board_enabled = st.checkbox(
                "Partner board enabled",
                value=bool(defaults.get("partner_board_enabled", True)),
                disabled=disabled,
            )
            status = st.selectbox(
                "Status",
                DIVISION_STATUSES,
                index=DIVISION_STATUSES.index(_safe_text(defaults.get("status") or "open"))
                if _safe_text(defaults.get("status") or "open") in DIVISION_STATUSES
                else DIVISION_STATUSES.index("open"),
                disabled=disabled,
            )
            division_format = st.selectbox(
                "Format override",
                [""] + COMPETITION_FORMATS,
                index=([""] + COMPETITION_FORMATS).index(_safe_text(defaults.get("division_format")))
                if _safe_text(defaults.get("division_format")) in [""] + COMPETITION_FORMATS
                else 0,
                disabled=disabled,
            )
            division_scoring = st.selectbox(
                "Scoring override",
                [""] + SCORING_OPTIONS,
                index=([""] + SCORING_OPTIONS).index(_safe_text(defaults.get("division_scoring")))
                if _safe_text(defaults.get("division_scoring")) in [""] + SCORING_OPTIONS
                else 0,
                disabled=disabled,
            )
            notes = st.text_area("Notes", value=_safe_text(defaults.get("notes")), height=120, disabled=disabled)
        submit_col, cancel_col = st.columns(2)
        submitted = submit_col.form_submit_button(submit_label, type="primary", disabled=disabled)
        canceled = cancel_col.form_submit_button("Cancel")

    payload = {
        "event_family": _safe_text(event_family),
        "division_name": _safe_text(division_name),
        "skill_label": _safe_text(skill_label) or "Open",
        "age_mode": _safe_text(age_mode) or "ALL_AGES",
        "age_label": _safe_text(age_label) or "All Ages",
        "age_ranges": _safe_text(age_ranges),
        "min_teams_per_age_group": _coerce_int(min_teams_per_age_group),
        "split_age_threshold": _coerce_int(split_age_threshold),
        "assigned_day": _safe_text(assigned_day),
        "capacity_teams": _coerce_int(capacity_teams),
        "price_usd": _coerce_float(price_usd),
        "waitlist_enabled": bool(waitlist_enabled),
        "partner_board_enabled": bool(partner_board_enabled),
        "status": _safe_text(status) or "open",
        "division_format": _safe_text(division_format),
        "division_scoring": _safe_text(division_scoring),
        "notes": _safe_text(notes),
    }
    if payload["age_mode"] != "AUTO_AGE_SPLIT":
        payload["min_teams_per_age_group"] = None
    if payload["age_mode"] != "SPLIT_AGE":
        payload["split_age_threshold"] = None
    return submitted, canceled, payload


def render(ctx):
    page_shell(
        "🛠️ Tournament Manager",
        "Create the tournament structure the way you actually run it: dates → events → divisions → schedule → publish.",
        mode_label="Admin",
    )

    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    supabase = getattr(ctx, "supabase", None)
    club_id = getattr(ctx, "club_id", None)
    if supabase is None or club_id is None:
        st.error("Missing database context.")
        st.stop()

    available, detail = registration_feature_available(supabase)
    if not available:
        st.error("Tournament registration tables are not available yet.")
        if detail:
            st.caption(detail)
        st.stop()

    st.subheader("Select Tournament")
    show_archived = st.checkbox("Show archived", value=False, key="tournament_manager_show_archived")
    st.caption("Archived tournaments are hidden from default selectors and public registration.")
    tournaments = list_existing_tournaments(supabase, str(club_id), include_archived=show_archived)
    if not tournaments:
        st.info("Create a tournament shell on the Tournaments page first, or enable Show archived.")
        st.stop()

    st.caption("Choose which tournament shell to manage before editing metadata, days, events, and divisions.")
    requested_id = _safe_text(st.query_params.get("tournament_id"))
    labels = []
    for row in tournaments:
        status = _safe_text(row.get('status') or 'DRAFT')
        status_label = 'ARCHIVED' if status.upper() == 'ARCHIVED' else status
        labels.append(f"{row.get('name')} ({status_label})")
    default_index = 0
    if requested_id:
        for idx, row in enumerate(tournaments):
            if str(row.get("id")) == requested_id:
                default_index = idx
                break
    picked = st.selectbox("Tournament", labels, index=default_index)
    tournament = tournaments[labels.index(picked)]
    tournament_id = str(tournament.get("id"))
    st.query_params["tournament_id"] = tournament_id
    tournament = get_tournament_record(supabase, tournament_id) or tournament

    settings = get_registration_settings(supabase, tournament_id, tournament_name=_safe_text(tournament.get("name")))
    days = list_registration_days(supabase, tournament_id)
    event_options = list_event_options(supabase, tournament_id)
    persisted_builder_draft = get_builder_draft(supabase, tournament_id)
    registration_count = count_tournament_registrations(supabase, tournament_id)

    if st.session_state.pop(f"tm_refresh_{tournament_id}", False):
        tournament = get_tournament_record(supabase, tournament_id) or tournament
        settings = get_registration_settings(supabase, tournament_id, tournament_name=_safe_text(tournament.get("name")))
        days = list_registration_days(supabase, tournament_id)
        event_options = list_event_options(supabase, tournament_id)
        persisted_builder_draft = get_builder_draft(supabase, tournament_id)
        registration_count = count_tournament_registrations(supabase, tournament_id)

    draft_days_rows = []
    if isinstance(persisted_builder_draft, dict):
        draft_days_rows = persisted_builder_draft.get("days") or []
    inferred_start_date, inferred_end_date = _infer_date_window_from_days_rows(draft_days_rows or days)
    canonical_start_date = _parse_date(tournament.get("start_date"))
    canonical_end_date = _parse_date(tournament.get("end_date"))
    canonical_has_range = bool(_date_rows(canonical_start_date, canonical_end_date))
    inferred_has_range = inferred_start_date is not None and inferred_end_date is not None
    regenerate_start_date = canonical_start_date or inferred_start_date
    regenerate_end_date = canonical_end_date or inferred_end_date

    effective_start_date = canonical_start_date or inferred_start_date or date.today()
    effective_end_date = canonical_end_date or inferred_end_date or effective_start_date
    if effective_end_date < effective_start_date:
        effective_end_date = effective_start_date
    using_inferred_window = (
        (canonical_start_date is None or canonical_end_date is None)
        and inferred_start_date is not None
        and inferred_end_date is not None
    )

    st.info(
        "Recommended setup order: 1) save tournament info and dates, 2) review days, 3) define events and defaults, 4) review generated divisions, 5) publish links."
    )

    metrics = st.columns(4)
    metrics[0].metric("Registration status", _safe_text(settings.get("registration_status") or "draft"))
    metrics[1].metric("Days", len(days))
    metrics[2].metric("Divisions", len(event_options))
    metrics[3].metric("Registrations", registration_count)

    tabs = st.tabs([
        "1. Tournament Info",
        "2. Days",
        "3. Events",
        "4. Divisions",
        "5. Schedule Preview",
        "6. Publish & QA",
    ])

    with tabs[0]:
        st.subheader("Tournament Info")
        st.caption("Set the tournament shell, registration window, and published content. The date range is used to generate tournament days.")
        start_default = effective_start_date
        end_default = effective_end_date
        safe_end_default = end_default if end_default >= start_default else start_default
        reg_open_default = _parse_datetime_value(settings.get("registration_open_at"))
        reg_close_default = _parse_datetime_value(settings.get("registration_close_at"))
        default_open_dt = reg_open_default or datetime.combine(start_default, datetime.min.time())
        default_close_dt = reg_close_default or datetime.combine(safe_end_default, datetime.min.time())

        current_start_label = _safe_text(tournament.get("start_date")) or (
            inferred_start_date.isoformat() if inferred_start_date else "No start date"
        )
        current_end_label = _safe_text(tournament.get("end_date")) or (
            inferred_end_date.isoformat() if inferred_end_date else "No end date"
        )
        st.caption(
            f"Current tournament: **{_safe_text(tournament.get('name') or 'Untitled Tournament')}** · "
            f"{current_start_label} → {current_end_label}"
        )
        if using_inferred_window:
            st.info(
                "Using saved draft day range: "
                f"{inferred_start_date.isoformat()} → {inferred_end_date.isoformat()}. "
                "Save Tournament Info to write these dates to the tournament shell."
            )

        with st.form("edit_tournament"):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Tournament name", value=_safe_text(tournament.get("name")))
                start_date = st.date_input("Start date", value=start_default)
                end_date = st.date_input("End date", value=safe_end_default)
                slug = st.text_input("Registration slug", value=_safe_text(settings.get("registration_slug")))
                locale = st.selectbox(
                    "Locale",
                    ["en", "es", "bilingual"],
                    index=["en", "es", "bilingual"].index(_safe_text(settings.get("locale") or "en"))
                    if _safe_text(settings.get("locale") or "en") in ["en", "es", "bilingual"]
                    else 0,
                )
            with c2:
                status = st.selectbox(
                    "Registration status",
                    REGISTRATION_STATUS_OPTIONS,
                    index=REGISTRATION_STATUS_OPTIONS.index(_safe_text(settings.get("registration_status") or "draft"))
                    if _safe_text(settings.get("registration_status") or "draft") in REGISTRATION_STATUS_OPTIONS
                    else 0,
                )
                reg_open_enabled = st.checkbox(
                    "Set registration open time",
                    value=reg_open_default is not None,
                )
                reg_open = st.datetime_input(
                    "Registration opens",
                    value=default_open_dt,
                    disabled=not reg_open_enabled,
                )
                reg_close_enabled = st.checkbox(
                    "Set registration close time",
                    value=reg_close_default is not None,
                )
                reg_close_min = reg_open if reg_open_enabled else default_open_dt
                reg_close = st.datetime_input(
                    "Registration closes",
                    value=default_close_dt if default_close_dt >= reg_close_min else reg_close_min,
                    min_value=reg_close_min,
                    disabled=not reg_close_enabled,
                )
                sponsor = st.text_area("Sponsor / callout text", value=_safe_text(settings.get("sponsor_markdown")), height=90)
                refund = st.text_area("Refund policy", value=_safe_text(settings.get("refund_policy_markdown")), height=90)
            notes = st.text_area("Rules / registration notes", value=_safe_text(settings.get("rules_markdown")), height=140)

            submitted = st.form_submit_button("Save tournament info", use_container_width=True)

        if submitted:
            errors: list[str] = []
            if not _safe_text(name):
                errors.append("Tournament name cannot be blank.")
            if end_date and start_date and end_date < start_date:
                errors.append("End date cannot be before start date.")
            if reg_open_enabled and reg_close_enabled and reg_close < reg_open:
                errors.append("Registration close cannot be before registration open.")

            if errors:
                for error in errors:
                    st.error(error)
            else:
                original_start = _parse_date(tournament.get("start_date"))
                original_end = _parse_date(tournament.get("end_date"))
                dates_changed = original_start != start_date or original_end != end_date

                _update_tournament_shell(
                    supabase,
                    tournament_id,
                    name=name,
                    start_date=start_date,
                    end_date=end_date,
                )
                upsert_registration_settings(
                    supabase,
                    {
                        "id": settings.get("id"),
                        "tournament_id": tournament_id,
                        "registration_slug": slug or None,
                        "locale": locale,
                        "registration_status": status,
                        "registration_open_at": _parse_local_dt(reg_open) if reg_open_enabled else None,
                        "registration_close_at": _parse_local_dt(reg_close) if reg_close_enabled else None,
                        "sponsor_markdown": sponsor,
                        "refund_policy_markdown": refund,
                        "rules_markdown": notes,
                        "waitlist_enabled": True,
                        "partner_board_enabled": True,
                    },
                )
                if dates_changed:
                    synced_days, synced_events = _sync_days_with_date_range(
                        tournament_id,
                        start_date,
                        end_date,
                        days,
                        event_options,
                    )
                    try:
                        publish_registration_configuration(
                            supabase,
                            tournament_id=tournament_id,
                            days=synced_days,
                            event_options=synced_events,
                        )
                    except ValueError as exc:
                        st.error(str(exc))
                        st.stop()

                _clear_tournament_manager_state(tournament_id)
                st.session_state[f"tm_refresh_{tournament_id}"] = True

                st.success(
                    "Tournament info and days synchronized."
                    if dates_changed
                    else "Tournament info saved."
                )
                st.rerun()

        st.divider()
        st.caption("Admin-only actions are kept separate from the main save form.")

    days_seed_key = f"tm_days_seed_{tournament_id}"
    draft_loaded_key = f"tm_draft_loaded_{tournament_id}"
    draft_last_saved_notice_key = f"tm_draft_notice_{tournament_id}"
    draft_last_saved_step_key = f"tm_draft_saved_step_{tournament_id}"
    draft_last_saved_at_key = f"tm_draft_saved_at_{tournament_id}"
    if draft_loaded_key not in st.session_state:
        st.session_state[draft_loaded_key] = False
    if not st.session_state[draft_loaded_key]:
        if isinstance(persisted_builder_draft, dict):
            st.session_state[days_seed_key] = _draft_rows_to_df(
                persisted_builder_draft.get("days") or [],
                DAYS_EDITOR_COLUMNS,
            )
            st.session_state[f"tm_events_seed_{tournament_id}"] = _draft_rows_to_df(
                persisted_builder_draft.get("event_families") or [],
                EVENT_TEMPLATE_COLUMNS,
            )
            st.session_state[f"tm_divisions_seed_{tournament_id}"] = _draft_rows_to_df(
                persisted_builder_draft.get("divisions") or [],
                DIVISION_EDITOR_COLUMNS,
            )
            st.session_state[draft_last_saved_at_key] = _safe_text(persisted_builder_draft.get("saved_at"))
            st.session_state[draft_last_saved_step_key] = _safe_text(persisted_builder_draft.get("saved_step"))
        st.session_state[draft_loaded_key] = True

    if days_seed_key not in st.session_state:
        st.session_state[days_seed_key] = _seed_days(days, tournament)
    events_seed_key = f"tm_events_seed_{tournament_id}"
    if events_seed_key not in st.session_state:
        st.session_state[events_seed_key] = _seed_event_templates(event_options)
    event_form_mode_key = f"tm_event_form_mode_{tournament_id}"
    event_edit_id_key = f"tm_event_edit_id_{tournament_id}"
    division_form_mode_key = f"tm_division_form_mode_{tournament_id}"
    division_edit_id_key = f"tm_division_edit_id_{tournament_id}"
    generate_event_id_key = f"tm_generate_event_id_{tournament_id}"
    generation_result_key = f"tm_generation_result_{tournament_id}"
    load_templates_confirm_key = f"tm_load_templates_confirm_{tournament_id}"
    if event_form_mode_key not in st.session_state:
        st.session_state[event_form_mode_key] = None
    if event_edit_id_key not in st.session_state:
        st.session_state[event_edit_id_key] = None
    if division_form_mode_key not in st.session_state:
        st.session_state[division_form_mode_key] = None
    if division_edit_id_key not in st.session_state:
        st.session_state[division_edit_id_key] = None
    if generate_event_id_key not in st.session_state:
        st.session_state[generate_event_id_key] = None
    if generation_result_key not in st.session_state:
        st.session_state[generation_result_key] = ""
    if load_templates_confirm_key not in st.session_state:
        st.session_state[load_templates_confirm_key] = False

    if st.session_state.get(draft_last_saved_notice_key):
        st.success("Draft progress saved.")
        st.caption("This step is saved to your tournament draft.")
        st.caption("Final Save Builder Changes is still required before publishing or sharing links.")
        st.session_state[draft_last_saved_notice_key] = False

    if _safe_text(st.session_state.get(draft_last_saved_at_key)):
        saved_step = _safe_text(st.session_state.get(draft_last_saved_step_key))
        step_suffix = f" · Step: {saved_step}" if saved_step else ""
        st.caption(f"Draft last saved: {_safe_text(st.session_state.get(draft_last_saved_at_key))}{step_suffix}")
        st.caption("Draft progress is saved, but final builder changes have not yet been published.")

    with tabs[1]:
        st.subheader("Days")
        st.caption("Days are created from the tournament date range. Relabel them to match how you actually talk about the schedule, such as 'Mixed Doubles Day' or 'Championship Sunday'.")
        if not canonical_has_range and not inferred_has_range:
            st.warning("No tournament date range is saved yet. You can add days manually, or save dates in Tournament Info to auto-generate days.")
        days_df = st.data_editor(
            st.session_state[days_seed_key],
            hide_index=True,
            num_rows="dynamic",
            key=f"tm_days_editor_{tournament_id}",
            disabled=False,
            column_config={
                "event_date": st.column_config.TextColumn("Date (YYYY-MM-DD)"),
                "label": st.column_config.TextColumn("Day label"),
                "enabled": st.column_config.CheckboxColumn("Enabled"),
            },
            use_container_width=True,
        )
        days_df = _ensure_editor_columns(days_df, DAYS_EDITOR_COLUMNS)
        st.session_state[days_seed_key] = days_df.copy()
        if st.button("Regenerate days from tournament dates"):
            generated = _df_with_hidden_ids(
                [{"id": _uid("day"), **row} for row in _date_rows(regenerate_start_date, regenerate_end_date)],
                "id",
                DAYS_EDITOR_COLUMNS,
            )
            if generated.empty:
                st.warning("Cannot regenerate days yet. Save a start/end range in Tournament Info or create draft days with valid dates first.")
            else:
                st.session_state[days_seed_key] = generated
            st.rerun()
        if st.button("Save Days Draft", key=f"tm_save_days_draft_{tournament_id}"):
            saved_payload = save_builder_draft(
                supabase,
                tournament_id=tournament_id,
                days=_df_to_draft_rows(st.session_state[days_seed_key], DAYS_EDITOR_COLUMNS),
                event_families=_df_to_draft_rows(st.session_state[events_seed_key], EVENT_TEMPLATE_COLUMNS),
                divisions=_df_to_draft_rows(st.session_state.get(f"tm_divisions_seed_{tournament_id}"), DIVISION_EDITOR_COLUMNS),
                saved_step="Days",
            )
            st.session_state[draft_last_saved_notice_key] = True
            st.session_state[draft_last_saved_step_key] = _safe_text(saved_payload.get("saved_step"))
            st.session_state[draft_last_saved_at_key] = _safe_text(saved_payload.get("saved_at"))
            st.rerun()

    divisions_seed_key = f"tm_divisions_seed_{tournament_id}"
    days_df = _ensure_editor_columns(st.session_state[days_seed_key], DAYS_EDITOR_COLUMNS)
    events_df = _ensure_editor_columns(st.session_state[events_seed_key], EVENT_TEMPLATE_COLUMNS)
    if divisions_seed_key not in st.session_state:
        st.session_state[divisions_seed_key] = _ensure_editor_columns(
            _seed_divisions(days_df, events_df, event_options),
            DIVISION_EDITOR_COLUMNS,
        )
    day_label_options = [label for label in days_df[days_df["enabled"] == True]["label"].tolist() if _safe_text(label)] or [
        label for label in days_df["label"].tolist() if _safe_text(label)
    ]
    ordered_day_labels = list(dict.fromkeys(_safe_text(label) for label in day_label_options if _safe_text(label)))

    with tabs[2]:
        st.subheader("Events")
        st.caption("Define persistent event-family defaults in this step. Use Generate Divisions for one-off batch runs.")
        events_df = _ensure_editor_columns(st.session_state[events_seed_key], EVENT_TEMPLATE_COLUMNS)
        st.session_state[events_seed_key] = events_df.copy()

        action_col1, action_col2 = st.columns([1, 1])
        if action_col1.button("Create Event", type="primary"):
            st.session_state[event_form_mode_key] = "create"
            st.session_state[event_edit_id_key] = None
            st.rerun()
        if action_col2.button("Load standard events"):
            if not events_df.empty and not st.session_state.get(load_templates_confirm_key):
                st.session_state[load_templates_confirm_key] = True
                st.warning("Existing events found. Click again to append missing standard templates.")
            else:
                current = events_df.copy()
                for template in STANDARD_EVENT_TEMPLATES:
                    if not _event_family_name_exists(current, _safe_text(template.get("event_family"))):
                        current = _add_event_family_row(current, template)
                st.session_state[events_seed_key] = _ensure_editor_columns(current, EVENT_TEMPLATE_COLUMNS)
                st.session_state[load_templates_confirm_key] = False
                st.success("Standard events loaded.")
                st.rerun()

        event_mode = st.session_state.get(event_form_mode_key)
        event_edit_id = st.session_state.get(event_edit_id_key)
        if event_mode in {"create", "edit"}:
            defaults: dict[str, Any] | None = None
            if event_mode == "edit" and event_edit_id is not None and str(event_edit_id) in {str(idx) for idx in events_df.index.tolist()}:
                defaults = events_df.loc[event_edit_id].to_dict()
            st.markdown("#### Edit Event" if event_mode == "edit" else "#### Add Event")
            submitted, canceled, payload = _render_event_family_form(
                form_key=f"tm_event_family_form_{tournament_id}_{event_mode}",
                mode=event_mode,
                defaults=defaults,
                submit_label="Save Event" if event_mode == "edit" else "Add Event",
                disabled=False,
            )
            if canceled:
                st.session_state[event_form_mode_key] = None
                st.session_state[event_edit_id_key] = None
                st.rerun()
            if submitted:
                errors: list[str] = []
                if not payload["event_family"]:
                    errors.append("Event name is required.")
                if _event_family_name_exists(events_df, payload["event_family"], exclude_id=event_edit_id if event_mode == "edit" else None):
                    errors.append("Event name must be unique.")
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    if event_mode == "edit" and event_edit_id is not None:
                        st.session_state[events_seed_key] = _update_event_family_row(events_df, str(event_edit_id), payload)
                    else:
                        st.session_state[events_seed_key] = _add_event_family_row(events_df, payload)
                    st.session_state[event_form_mode_key] = None
                    st.session_state[event_edit_id_key] = None
                    st.rerun()

        events_df = _ensure_editor_columns(st.session_state[events_seed_key], EVENT_TEMPLATE_COLUMNS)
        if events_df.empty:
            st.info("No events yet. Click Create Event to start.")
        else:
            preview = events_df.reset_index(drop=True).rename(
                columns={
                    "event_family": "Event",
                    "participant_type": "Participant Type",
                    "gender_restriction": "Gender",
                    "default_format": "Default Format",
                    "default_scoring": "Default Scoring",
                    "default_waitlist": "Default Waitlist",
                    "default_partner_board": "Default Partner Board",
                    "default_capacity_teams": "Default Capacity",
                    "default_price_usd": "Default Price",
                    "default_status": "Default Status",
                }
            )
            st.dataframe(preview, hide_index=True, use_container_width=True)
            st.markdown("##### Manage Events")
            divisions_snapshot = _ensure_editor_columns(st.session_state.get(f"tm_divisions_seed_{tournament_id}"), DIVISION_EDITOR_COLUMNS)
            for event_id, row in events_df.iterrows():
                name = _safe_text(row.get("event_family") or "Unnamed Event")
                row_cols = st.columns([4, 1, 1, 1.2])
                row_cols[0].markdown(f"**{name}** · {_safe_text(row.get('participant_type'))} · {_safe_text(row.get('gender_restriction'))}")
                if row_cols[1].button("Edit", key=f"tm_evt_edit_{tournament_id}_{event_id}"):
                    st.session_state[event_form_mode_key] = "edit"
                    st.session_state[event_edit_id_key] = str(event_id)
                    st.rerun()
                if row_cols[2].button("Delete", key=f"tm_evt_del_{tournament_id}_{event_id}"):
                    has_dependent_divisions = not divisions_snapshot[
                        divisions_snapshot["event_family"].apply(lambda value: _safe_text(value).lower() == name.lower())
                    ].empty
                    if has_dependent_divisions:
                        st.error(f"Cannot delete '{name}' because divisions are attached. Delete those divisions first.")
                    else:
                        st.session_state[events_seed_key] = _delete_event_family_row(events_df, str(event_id))
                        st.success(f"Deleted event '{name}'.")
                        st.rerun()
                if row_cols[3].button("Generate Divisions", key=f"tm_evt_gen_{tournament_id}_{event_id}"):
                    st.session_state[generate_event_id_key] = str(event_id)
                    st.rerun()
        generate_event_id = st.session_state.get(generate_event_id_key)
        if generate_event_id is not None and str(generate_event_id) in {str(idx) for idx in events_df.index.tolist()}:
            generator_defaults = events_df.loc[generate_event_id].to_dict()
            st.markdown("#### Generate Divisions")
            submitted, canceled, generation_payload = _render_event_generation_form(
                form_key=f"tm_generate_form_{tournament_id}_{generate_event_id}",
                event_defaults=generator_defaults,
                day_label_options=ordered_day_labels,
                disabled=False,
            )
            if canceled:
                st.session_state[generate_event_id_key] = None
                st.rerun()
            if submitted:
                errors: list[str] = []
                selected_days = _coerce_list(generation_payload.get("selected_days"))
                selected_skills = _coerce_list(generation_payload.get("selected_skill_levels"))
                age_mode = _safe_text(generation_payload.get("age_mode") or "ALL_AGES")
                age_labels = _coerce_list(generation_payload.get("age_labels"))
                if not selected_days:
                    errors.append("Select at least one day for division generation.")
                if not selected_skills:
                    errors.append("Select at least one skill level for division generation.")
                if age_mode == "FIXED_AGE_BRACKET" and not age_labels:
                    errors.append("Fixed Age Brackets requires at least one age label.")
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    divisions_df_current = _ensure_editor_columns(st.session_state[divisions_seed_key], DIVISION_EDITOR_COLUMNS)
                    next_divisions, created_count, skipped_count = _generate_division_rows_for_event_batch(
                        event_row=events_df.loc[generate_event_id],
                        selected_days=selected_days,
                        selected_skills=selected_skills,
                        age_mode=age_mode,
                        age_labels=age_labels,
                        batch_overrides=generation_payload.get("batch_overrides") or {},
                        current_divisions_df=divisions_df_current,
                        generation_mode=_safe_text(generation_payload.get("generation_mode") or "create_missing_only"),
                    )
                    st.session_state[divisions_seed_key] = next_divisions
                    st.session_state[generation_result_key] = (
                        f"Generated {created_count} division(s). Skipped {skipped_count} duplicate(s)."
                    )
                    st.session_state[generate_event_id_key] = None
                    st.rerun()
        st.caption("Events define persistent defaults only. Divisions can still be edited one-off on the Divisions tab.")
        if _safe_text(st.session_state.get(generation_result_key)):
            st.success(_safe_text(st.session_state.get(generation_result_key)))
            st.session_state[generation_result_key] = ""
        if st.button("Save Events Draft", key=f"tm_save_events_draft_{tournament_id}"):
            saved_payload = save_builder_draft(
                supabase,
                tournament_id=tournament_id,
                days=_df_to_draft_rows(st.session_state[days_seed_key], DAYS_EDITOR_COLUMNS),
                event_families=_df_to_draft_rows(st.session_state[events_seed_key], EVENT_TEMPLATE_COLUMNS),
                divisions=_df_to_draft_rows(st.session_state.get(f"tm_divisions_seed_{tournament_id}"), DIVISION_EDITOR_COLUMNS),
                saved_step="Events",
            )
            st.session_state[draft_last_saved_notice_key] = True
            st.session_state[draft_last_saved_step_key] = _safe_text(saved_payload.get("saved_step"))
            st.session_state[draft_last_saved_at_key] = _safe_text(saved_payload.get("saved_at"))
            st.rerun()

    divisions_seed_key = f"tm_divisions_seed_{tournament_id}"
    if divisions_seed_key not in st.session_state:
        st.session_state[divisions_seed_key] = _ensure_editor_columns(
            _seed_divisions(days_df, events_df, event_options),
            DIVISION_EDITOR_COLUMNS,
        )
    divisions_seed_df = _sanitize_divisions_for_event_families(
        _ensure_editor_columns(st.session_state[divisions_seed_key], DIVISION_EDITOR_COLUMNS),
        [family for family in events_df["event_family"].tolist() if _safe_text(family)],
    )
    st.session_state[divisions_seed_key] = divisions_seed_df.copy()
    day_label_options = [label for label in days_df[days_df["enabled"] == True]["label"].tolist() if _safe_text(label)] or [
        label for label in days_df["label"].tolist() if _safe_text(label)
    ]
    ordered_day_labels = list(dict.fromkeys(_safe_text(label) for label in day_label_options if _safe_text(label)))
    event_family_options = [family for family in events_df["event_family"].tolist() if _safe_text(family)]
    with tabs[3]:
        st.subheader("Divisions")
        st.caption("Review, edit, and remove generated divisions. Manual one-off additions are optional.")
        create_disabled = not event_family_options or not day_label_options
        if st.button("Create Division", type="primary", disabled=create_disabled):
            st.session_state[division_form_mode_key] = "create"
            st.session_state[division_edit_id_key] = None
            st.rerun()
        if not event_family_options:
            st.info("Create an event first before adding divisions.")
        if event_family_options and not day_label_options:
            st.info("Enable at least one tournament day before adding divisions.")

        division_mode = st.session_state.get(division_form_mode_key)
        division_edit_id = st.session_state.get(division_edit_id_key)
        if division_mode in {"create", "edit"} and event_family_options and day_label_options:
            defaults: dict[str, Any] | None = None
            if division_mode == "edit" and division_edit_id is not None and str(division_edit_id) in {str(idx) for idx in divisions_seed_df.index.tolist()}:
                defaults = divisions_seed_df.loc[division_edit_id].to_dict()
            st.markdown("#### Edit Division" if division_mode == "edit" else "#### Add Division")
            submitted, canceled, payload = _render_division_form(
                form_key=f"tm_division_form_{tournament_id}_{division_mode}",
                mode=division_mode,
                defaults=defaults,
                event_family_options=event_family_options,
                day_label_options=day_label_options,
                submit_label="Save Division" if division_mode == "edit" else "Add Division",
                disabled=False,
            )
            if canceled:
                st.session_state[division_form_mode_key] = None
                st.session_state[division_edit_id_key] = None
                st.rerun()
            if submitted:
                errors: list[str] = []
                if not payload["event_family"]:
                    errors.append("Event family is required.")
                if not payload["division_name"]:
                    errors.append("Division name is required.")
                if not payload["assigned_day"]:
                    errors.append("Assigned day is required.")
                if payload["age_mode"] == "AUTO_AGE_SPLIT" and _coerce_int(payload.get("min_teams_per_age_group")) is None:
                    errors.append("Auto Age Split requires min teams per age group.")
                if payload["age_mode"] == "SPLIT_AGE" and _coerce_int(payload.get("split_age_threshold")) is None:
                    errors.append("Split Age requires split-age threshold.")
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    if division_mode == "edit" and division_edit_id is not None:
                        st.session_state[divisions_seed_key] = _update_division_row(divisions_seed_df, str(division_edit_id), payload)
                    else:
                        st.session_state[divisions_seed_key] = _add_division_row(divisions_seed_df, payload)
                    st.session_state[division_form_mode_key] = None
                    st.session_state[division_edit_id_key] = None
                    st.rerun()

        divisions_df = _ensure_editor_columns(st.session_state[divisions_seed_key], DIVISION_EDITOR_COLUMNS)
        if divisions_df.empty:
            st.info("No divisions yet. Click Create Division to start.")
        else:
            template_lookup = {
                _safe_text(template_row.get("event_family")): template_row
                for _, template_row in events_df.iterrows()
                if _safe_text(template_row.get("event_family"))
            }
            st.markdown("##### Existing Divisions")
            divisions_day_labels = {
                _safe_text(value) for value in divisions_df["assigned_day"].tolist() if _safe_text(value)
            }
            ordered_division_days = [day for day in ordered_day_labels if day in divisions_day_labels]
            remaining_days = list(
                dict.fromkeys(
                    _safe_text(value)
                    for value in divisions_df["assigned_day"].tolist()
                    if _safe_text(value) and _safe_text(value) not in ordered_division_days
                )
            )
            for day_index, day in enumerate(ordered_division_days + remaining_days):
                day_df = divisions_df[divisions_df["assigned_day"].apply(lambda value: _safe_text(value) == day)]
                if day_df.empty:
                    continue
                with st.expander(day, expanded=(day_index == 0)):
                    day_families = sorted({
                        _safe_text(value) for value in day_df["event_family"].tolist() if _safe_text(value)
                    })
                    for family in day_families:
                        family_df = day_df[day_df["event_family"].apply(lambda value: _safe_text(value) == family)]
                        if family_df.empty:
                            continue
                        st.markdown(f"**{family}**")
                        header_cols = st.columns([5.0, 1.0, 1.2, 1.0, 1.0, 1.0, 2.4])
                        header_cols[0].markdown("**Division**")
                        header_cols[1].markdown("**Skill**")
                        header_cols[2].markdown("**Age**")
                        header_cols[3].markdown("**Status**")
                        header_cols[4].markdown("**Capacity**")
                        header_cols[5].markdown("**Price**")
                        header_cols[6].markdown("**Actions**")
                        for division_id, row in family_df.iterrows():
                            division_name = _display_division_name(row)
                            row_cols = st.columns([5.0, 1.0, 1.2, 1.0, 1.0, 1.0, 2.4])
                            row_cols[0].markdown(f"**{_display_optional(division_name)}**")
                            row_cols[1].markdown(_display_optional(row.get("skill_label")))
                            row_cols[2].markdown(_display_optional(row.get("age_label")))
                            status_value = _safe_text(row.get("status") or "draft").upper()
                            row_cols[3].markdown(f"**`{status_value}`**")
                            row_cols[4].markdown(_display_optional(row.get("capacity_teams")))
                            price_value = row.get("price_usd")
                            row_cols[5].markdown(f"${float(price_value):.2f}" if _coerce_float(price_value) is not None else "—")
                            action_cols = row_cols[6].columns([1, 1], gap="small")
                            if action_cols[0].button("✏️ Edit", key=f"tm_div_edit_{tournament_id}_{division_id}"):
                                st.session_state[division_form_mode_key] = "edit"
                                st.session_state[division_edit_id_key] = str(division_id)
                                st.rerun()
                            if action_cols[1].button("🗑️ Delete", key=f"tm_div_del_{tournament_id}_{division_id}"):
                                st.session_state[divisions_seed_key] = _delete_division_row(divisions_df, str(division_id))
                                st.success(f"Deleted division '{division_name}'.")
                                st.rerun()
                            notes_text = _safe_text(row.get("notes"))
                            secondary_parts = [
                                f"Age Mode: {_display_labelized(row.get('age_mode'))}",
                                f"Format: {_resolve_division_format(row, template_lookup)}",
                                f"Scoring: {_resolve_division_scoring(row, template_lookup)}",
                            ]
                            if notes_text:
                                secondary_parts.append(f"Notes: {_truncate_text(notes_text)}")
                            st.caption(" · ".join(secondary_parts))
                            st.divider()
        for mode, help_text in AGE_MODE_HELP.items():
            st.caption(f"**{mode.replace('_', ' ').title()}** — {help_text}")
        if st.button("Save Divisions Draft", key=f"tm_save_divisions_draft_{tournament_id}"):
            saved_payload = save_builder_draft(
                supabase,
                tournament_id=tournament_id,
                days=_df_to_draft_rows(st.session_state[days_seed_key], DAYS_EDITOR_COLUMNS),
                event_families=_df_to_draft_rows(st.session_state[events_seed_key], EVENT_TEMPLATE_COLUMNS),
                divisions=_df_to_draft_rows(st.session_state[divisions_seed_key], DIVISION_EDITOR_COLUMNS),
                saved_step="Divisions",
            )
            st.session_state[draft_last_saved_notice_key] = True
            st.session_state[draft_last_saved_step_key] = _safe_text(saved_payload.get("saved_step"))
            st.session_state[draft_last_saved_at_key] = _safe_text(saved_payload.get("saved_at"))
            st.rerun()

    with tabs[4]:
        st.subheader("Schedule Preview")
        preview_groups = _schedule_preview_rows(days_df, divisions_df)
        if not preview_groups:
            st.info("Add divisions and assign them to days to build the schedule preview.")
        for day_label, preview_df in preview_groups:
            st.markdown(f"#### {day_label}")
            st.dataframe(preview_df, hide_index=True, use_container_width=True)

    with tabs[5]:
        st.subheader("Publish & QA")
        public_urls = build_public_urls(
            base_url=_safe_text(st.session_state.get("base_url")),
            tournament_id=tournament_id,
            registration_slug=settings.get("registration_slug"),
        )
        st.link_button("Public registration form", public_urls["registration"])
        st.link_button("Public partner board", public_urls["partner_board"])
        st.caption("Publish registration changes before sharing links when days, events, or divisions have changed.")

        validation_errors = _validate_builder(days_df, events_df, divisions_df)
        if validation_errors:
            for error in validation_errors:
                st.warning(error)

        days_payload, event_payload = _build_payloads(tournament_id, days_df, events_df, divisions_df)
        readiness = _registration_readiness_summary(event_payload)
        st.markdown("##### Registration readiness")
        readiness_cols = st.columns(4)
        readiness_cols[0].metric("Total divisions", readiness["total"])
        readiness_cols[1].metric("Publicly selectable", readiness["selectable"])
        readiness_cols[2].metric("Visible but closed", readiness["visible_closed"])
        readiness_cols[3].metric("Hidden from registration", readiness["hidden"])
        if readiness["hidden_draft"] > 0:
            st.warning(
                f"{readiness['hidden_draft']} draft division(s) are hidden and will not appear in public registration."
            )
        zero_selectable = readiness["total"] > 0 and readiness["selectable"] == 0
        if zero_selectable:
            st.error("This publish would leave the public registration form with 0 selectable divisions.")
        elif readiness["hidden"] > 0:
            st.warning("Some divisions are hidden from public registration (draft/disabled).")

        bulk_open_disabled = readiness["hidden_draft"] == 0
        if st.button("Set all draft divisions to open", disabled=bulk_open_disabled):
            divisions_next = _ensure_editor_columns(st.session_state[divisions_seed_key], DIVISION_EDITOR_COLUMNS).copy()
            if not divisions_next.empty:
                status_series = divisions_next["status"].apply(lambda value: _safe_text(value).lower())
                draft_mask = status_series == "draft"
                if draft_mask.any():
                    divisions_next.loc[draft_mask, "status"] = "open"
                    st.session_state[divisions_seed_key] = divisions_next
            st.rerun()

        publish_impact = analyze_registration_publish_impact(
            supabase,
            tournament_id=tournament_id,
            days=days_payload,
            event_options=event_payload,
        ) if days_payload and event_payload else None
        if publish_impact:
            summary = publish_impact.get("summary", {})
            metric_cols = st.columns(6)
            metric_cols[0].metric("Creates", int(summary.get("creates", 0)))
            metric_cols[1].metric("Updates", int(summary.get("updates", 0)))
            metric_cols[2].metric("Soft closes", int(summary.get("soft_closes", 0)))
            metric_cols[3].metric("Deletes", int(summary.get("deletes", 0)))
            metric_cols[4].metric("Warnings", int(summary.get("warnings", 0)))
            metric_cols[5].metric("Blocked", int(summary.get("blocked", 0)))

            for section_title, rows in [
                ("Safe creates", publish_impact.get("creates", [])),
                ("Safe updates", publish_impact.get("updates", [])),
                ("Safe close/archive actions", publish_impact.get("soft_closes", [])),
                ("Safe hard deletes (unused only)", publish_impact.get("deletes", [])),
            ]:
                if rows:
                    st.markdown(f"##### {section_title}")
                    for row in rows:
                        st.caption(f"• {row}")
            if publish_impact.get("warnings"):
                st.markdown("##### Warnings")
                for row in publish_impact.get("warnings", []):
                    st.warning(row)
            if publish_impact.get("blocked"):
                st.markdown("##### Blocked destructive changes")
                for row in publish_impact.get("blocked", []):
                    st.error(row)

        has_blocked_changes = bool(publish_impact and publish_impact.get("blocked"))
        publish_ack_required = zero_selectable
        publish_acknowledged = (
            st.checkbox(
                "I understand this publish will leave public registration with no selectable divisions",
                value=False,
            )
            if publish_ack_required
            else True
        )
        publish_disabled = has_blocked_changes or (publish_ack_required and not publish_acknowledged)
        if st.button("Publish registration changes", type="primary", disabled=publish_disabled):
            if validation_errors:
                st.error("Resolve the builder warnings before publishing.")
            elif not days_payload:
                st.error("Enable at least one tournament day.")
            elif not event_payload:
                st.error("Create at least one division before publishing.")
            elif publish_ack_required and not publish_acknowledged:
                st.error("Acknowledge the zero-selectable warning to continue publishing.")
            elif has_blocked_changes:
                st.error("Publish is blocked until destructive changes are resolved.")
            else:
                try:
                    publish_registration_configuration(
                        supabase,
                        tournament_id=tournament_id,
                        days=days_payload,
                        event_options=event_payload,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()
                saved_payload = save_builder_draft(
                    supabase,
                    tournament_id=tournament_id,
                    days=_df_to_draft_rows(days_df, DAYS_EDITOR_COLUMNS),
                    event_families=_df_to_draft_rows(events_df, EVENT_TEMPLATE_COLUMNS),
                    divisions=_df_to_draft_rows(divisions_df, DIVISION_EDITOR_COLUMNS),
                    saved_step="Publish & QA",
                )
                st.session_state[draft_last_saved_step_key] = _safe_text(saved_payload.get("saved_step"))
                st.session_state[draft_last_saved_at_key] = _safe_text(saved_payload.get("saved_at"))
                st.success("Registration changes published.")
                st.rerun()

        st.info("Registration review, issue triage, and workbook exports now live on 📋 Tournament Operations.")
