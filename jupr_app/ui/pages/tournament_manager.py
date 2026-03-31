from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from jupr_app.domain.event_tags import derive_default_date_tags, normalize_event_tags
from jupr_app.domain.tournament_registration_exports import build_registration_workbook
from jupr_app.domain.tournament_registration_repo import (
    REGISTRATION_STATUS_OPTIONS,
    build_public_urls,
    build_registration_state,
    count_tournament_registrations,
    get_registration_settings,
    get_tournament_record,
    list_event_options,
    list_existing_tournaments,
    list_registration_days,
    registration_feature_available,
    replace_registration_configuration,
    upsert_registration_settings,
)
from jupr_app.ui.layout import page_shell

COMPETITION_FORMATS = ["ROUND_ROBIN", "SINGLE_ELIM", "DOUBLE_ELIM", "ROUND_ROBIN_PLUS_PLAYOFF"]
SCORING_OPTIONS = ["GAME_TO_11", "GAME_TO_15", "GAME_TO_21", "BEST_2_OF_3"]
AGE_MODES = ["ALL_AGES", "FIXED_AGE_BRACKET", "AUTO_AGE_SPLIT", "SPLIT_AGE"]
PARTICIPANT_TYPES = ["SINGLES", "GENDER_DOUBLES", "MIXED_DOUBLES"]
GENDER_RESTRICTIONS = ["ANY", "MEN", "WOMEN", "MIXED"]
DIVISION_STATUSES = ["draft", "open", "tentative", "confirmed", "closed"]

STANDARD_EVENT_TEMPLATES = [
    {
        "event_family": "Men's Doubles",
        "participant_type": "GENDER_DOUBLES",
        "gender_restriction": "MEN",
        "default_format": "ROUND_ROBIN_PLUS_PLAYOFF",
        "default_scoring": "GAME_TO_15",
        "default_waitlist": True,
        "default_partner_board": True,
    },
    {
        "event_family": "Women's Doubles",
        "participant_type": "GENDER_DOUBLES",
        "gender_restriction": "WOMEN",
        "default_format": "ROUND_ROBIN_PLUS_PLAYOFF",
        "default_scoring": "GAME_TO_15",
        "default_waitlist": True,
        "default_partner_board": True,
    },
    {
        "event_family": "Mixed Doubles",
        "participant_type": "MIXED_DOUBLES",
        "gender_restriction": "MIXED",
        "default_format": "ROUND_ROBIN_PLUS_PLAYOFF",
        "default_scoring": "GAME_TO_15",
        "default_waitlist": True,
        "default_partner_board": True,
    },
    {
        "event_family": "Men's Singles",
        "participant_type": "SINGLES",
        "gender_restriction": "MEN",
        "default_format": "ROUND_ROBIN_PLUS_PLAYOFF",
        "default_scoring": "GAME_TO_15",
        "default_waitlist": True,
        "default_partner_board": False,
    },
    {
        "event_family": "Women's Singles",
        "participant_type": "SINGLES",
        "gender_restriction": "WOMEN",
        "default_format": "ROUND_ROBIN_PLUS_PLAYOFF",
        "default_scoring": "GAME_TO_15",
        "default_waitlist": True,
        "default_partner_board": False,
    },
]

AGE_MODE_HELP = {
    "ALL_AGES": "One combined division regardless of age.",
    "FIXED_AGE_BRACKET": "Set a fixed age label such as 50+ or 60+.",
    "AUTO_AGE_SPLIT": "Players register into one skill division. The system can later split into age groups if each group reaches the minimum number of teams.",
    "SPLIT_AGE": "Use a partner rule such as one player 50+ and one player under 50.",
}


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


def _parse_local_dt(value: str) -> str | None:
    text = _safe_text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).isoformat()
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
    text = _safe_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def _coerce_float(value: Any) -> float | None:
    text = _safe_text(value)
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _df_with_hidden_ids(rows: list[dict[str, Any]], id_key: str, ordered_columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=ordered_columns)
    df = pd.DataFrame(rows)
    if id_key in df.columns:
        df = df.set_index(id_key)
    return df[ordered_columns]


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
        return _df_with_hidden_ids(rows, "id", ["event_date", "label", "enabled"])

    generated = _date_rows(tournament.get("start_date"), tournament.get("end_date"))
    if not generated:
        return pd.DataFrame(columns=["event_date", "label", "enabled"])
    rows = [{"id": _uid("day"), **row} for row in generated]
    return _df_with_hidden_ids(rows, "id", ["event_date", "label", "enabled"])


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
            },
        )
    rows = list(grouped.values()) or [{"id": _uid("evt"), **row} for row in STANDARD_EVENT_TEMPLATES]
    return _df_with_hidden_ids(
        rows,
        "id",
        [
            "event_family",
            "participant_type",
            "gender_restriction",
            "default_format",
            "default_scoring",
            "default_waitlist",
            "default_partner_board",
        ],
    )


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
    day_lookup = {str(idx): _safe_text(row.get("label") or idx) for idx, row in days_df.to_dict("index").items()}
    default_event_family = next(iter(event_templates_df["event_family"].tolist()), "Men's Doubles") if not event_templates_df.empty else "Men's Doubles"
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
                "status": _safe_text(row.get("status") or "draft"),
                "division_format": _safe_text(row.get("event_format_override") or ""),
                "division_scoring": _safe_text(row.get("scoring_override") or ""),
                "notes": _safe_text(age_rules.get("notes")),
            }
        )

    if not rows:
        rows = [
            {
                "id": _uid("div"),
                "event_family": default_event_family,
                "division_name": f"{default_event_family} Open",
                "skill_label": "Open",
                "age_mode": "ALL_AGES",
                "age_label": "All Ages",
                "age_ranges": "",
                "min_teams_per_age_group": None,
                "split_age_threshold": None,
                "assigned_day": next(iter(day_lookup.values()), "Day 1"),
                "capacity_teams": 16,
                "price_usd": None,
                "waitlist_enabled": True,
                "partner_board_enabled": default_event_family not in {"Men's Singles", "Women's Singles"},
                "status": "draft",
                "division_format": "",
                "division_scoring": "",
                "notes": "",
            }
        ]

    return _df_with_hidden_ids(
        rows,
        "id",
        [
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
        ],
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


def _validate_builder(days_df: pd.DataFrame, event_templates_df: pd.DataFrame, divisions_df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    enabled_days = days_df[days_df["enabled"].fillna(False).astype(bool)] if not days_df.empty else pd.DataFrame()
    if enabled_days.empty:
        errors.append("Enable at least one tournament day.")

    if event_templates_df.empty:
        errors.append("Create at least one event family.")
    else:
        families = [_safe_text(v) for v in event_templates_df["event_family"].tolist()]
        blank_count = sum(1 for v in families if not v)
        if blank_count:
            errors.append("Every event family needs a name.")
        if len({v.lower() for v in families if v}) != len([v for v in families if v]):
            errors.append("Event family names must be unique.")

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
                errors.append(f"Division '{division_name}' references an event family that is not defined.")
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
                "status": _safe_text(row.get("status") or "draft"),
                "enabled": True,
            }
        )

    return day_payload, event_payload


def _schedule_preview_rows(days_df: pd.DataFrame, divisions_df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
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

    tournaments = list_existing_tournaments(supabase, str(club_id))
    if not tournaments:
        st.info("Create a tournament shell on the Tournaments page first.")
        st.stop()

    st.subheader("Select Tournament")
    st.caption("Choose which tournament shell to manage before editing metadata, days, events, and divisions.")
    requested_id = _safe_text(st.query_params.get("tournament_id"))
    labels = [f"{row.get('name')} ({row.get('status')})" for row in tournaments]
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
    registration_count = count_tournament_registrations(supabase, tournament_id)

    if st.session_state.pop(f"tm_refresh_{tournament_id}", False):
        tournament = get_tournament_record(supabase, tournament_id) or tournament
        settings = get_registration_settings(supabase, tournament_id, tournament_name=_safe_text(tournament.get("name")))
        days = list_registration_days(supabase, tournament_id)
        event_options = list_event_options(supabase, tournament_id)
        registration_count = count_tournament_registrations(supabase, tournament_id)

    structure_locked = bool(registration_count)
    if structure_locked:
        st.warning(
            "This tournament already has registrations. Structural changes to days, events, and divisions are locked to protect submitted data."
        )
    else:
        st.info(
            "Recommended setup order: 1) save tournament info and dates, 2) review days, 3) define event families and defaults, 4) add divisions and assign each division to one day, 5) publish links."
        )

    metrics = st.columns(4)
    metrics[0].metric("Registration status", _safe_text(settings.get("registration_status") or "draft"))
    metrics[1].metric("Days", len(days))
    metrics[2].metric("Divisions", len(event_options))
    metrics[3].metric("Registrations", registration_count)

    tabs = st.tabs([
        "1. Tournament Info",
        "2. Days",
        "3. Event Families",
        "4. Divisions",
        "5. Schedule Preview",
        "6. Publish & QA",
    ])

    with tabs[0]:
        st.subheader("Tournament Info")
        st.caption("Set the tournament shell, registration window, and published content. The date range is used to generate tournament days.")
        start_default = _parse_date(tournament.get("start_date")) or date.today()
        end_default = _parse_date(tournament.get("end_date")) or start_default
        safe_end_default = end_default if end_default >= start_default else start_default

        st.caption(
            f"Current tournament: **{_safe_text(tournament.get('name') or 'Untitled Tournament')}** · "
            f"{_safe_text(tournament.get('start_date') or 'No start date')} → {_safe_text(tournament.get('end_date') or 'No end date')}"
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
                reg_open = st.text_input("Registration opens (YYYY-MM-DDTHH:MM)", value=_fmt_dt(settings.get("registration_open_at")))
                reg_close = st.text_input("Registration closes (YYYY-MM-DDTHH:MM)", value=_fmt_dt(settings.get("registration_close_at")))
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
                persisted_tournament = get_tournament_record(supabase, tournament_id) or {}
                persisted_start = _parse_date(persisted_tournament.get("start_date"))
                persisted_end = _parse_date(persisted_tournament.get("end_date"))
                upsert_registration_settings(
                    supabase,
                    {
                        "id": settings.get("id"),
                        "tournament_id": tournament_id,
                        "registration_slug": slug or None,
                        "locale": locale,
                        "registration_status": status,
                        "registration_open_at": _parse_local_dt(reg_open),
                        "registration_close_at": _parse_local_dt(reg_close),
                        "sponsor_markdown": sponsor,
                        "refund_policy_markdown": refund,
                        "rules_markdown": notes,
                        "waitlist_enabled": True,
                        "partner_board_enabled": True,
                    },
                )
                if dates_changed and not structure_locked:
                    synced_days, synced_events = _sync_days_with_date_range(
                        tournament_id,
                        persisted_start,
                        persisted_end,
                        days,
                        event_options,
                    )
                    replace_registration_configuration(
                        supabase,
                        tournament_id=tournament_id,
                        days=synced_days,
                        event_options=synced_events,
                    )

                _clear_tournament_manager_state(tournament_id)
                st.session_state[f"tm_refresh_{tournament_id}"] = True

                st.success(
                    "Tournament info and days synchronized."
                    if dates_changed and not structure_locked
                    else "Tournament info saved."
                )
                st.rerun()

        st.divider()
        st.caption("Admin-only actions are kept separate from the main save form.")

    days_seed_key = f"tm_days_seed_{tournament_id}"
    if days_seed_key not in st.session_state:
        st.session_state[days_seed_key] = _seed_days(days, tournament)

    with tabs[1]:
        st.subheader("Days")
        st.caption("Days are created from the tournament date range. Relabel them to match how you actually talk about the schedule, such as 'Mixed Doubles Day' or 'Championship Sunday'.")
        if not _date_rows(tournament.get("start_date"), tournament.get("end_date")):
            st.warning("No tournament date range is saved yet. You can add days manually, or save dates in Tournament Info to auto-generate days.")
        if structure_locked:
            st.caption("Days are view-only because registrations already exist.")
        days_df = st.data_editor(
            st.session_state[days_seed_key],
            hide_index=True,
            num_rows="dynamic",
            key=f"tm_days_editor_{tournament_id}",
            disabled=structure_locked,
            column_config={
                "event_date": st.column_config.TextColumn("Date (YYYY-MM-DD)"),
                "label": st.column_config.TextColumn("Day label"),
                "enabled": st.column_config.CheckboxColumn("Enabled"),
            },
            use_container_width=True,
        )
        st.session_state[days_seed_key] = days_df.copy()
        if st.button("Regenerate days from tournament dates", disabled=structure_locked):
            generated = _seed_days([], tournament)
            if generated.empty:
                st.warning("Cannot regenerate days yet. Save both start and end dates first.")
            else:
                st.session_state[days_seed_key] = generated
            st.rerun()

    event_seed_df = _seed_event_templates(event_options)
    with tabs[2]:
        st.subheader("Event Families")
        st.caption("Create the event families first. These hold the default format, scoring, and partner settings that the divisions inherit.")
        if structure_locked:
            st.caption("Event families are view-only because registrations already exist.")
        events_df = st.data_editor(
            event_seed_df,
            hide_index=True,
            num_rows="dynamic",
            key=f"tm_events_editor_{tournament_id}",
            disabled=structure_locked,
            column_config={
                "event_family": st.column_config.TextColumn("Event family"),
                "participant_type": st.column_config.SelectboxColumn("Participant type", options=PARTICIPANT_TYPES),
                "gender_restriction": st.column_config.SelectboxColumn("Gender restriction", options=GENDER_RESTRICTIONS),
                "default_format": st.column_config.SelectboxColumn("Default format", options=COMPETITION_FORMATS),
                "default_scoring": st.column_config.SelectboxColumn("Default scoring", options=SCORING_OPTIONS),
                "default_waitlist": st.column_config.CheckboxColumn("Default waitlist"),
                "default_partner_board": st.column_config.CheckboxColumn("Default partner board"),
            },
            use_container_width=True,
        )
        if st.button("Reset to standard event families", disabled=structure_locked):
            st.session_state.pop(f"tm_events_editor_{tournament_id}", None)
            st.rerun()
        st.caption("Typical default: all doubles divisions can inherit Round Robin + Playoff and Games to 15, while individual divisions override only when needed.")

    divisions_seed_df = _seed_divisions(days_df, events_df, event_options)
    day_label_options = [label for label in days_df[days_df["enabled"] == True]["label"].tolist()] or days_df["label"].tolist() or ["Day 1"]
    event_family_options = [family for family in events_df["event_family"].tolist() if _safe_text(family)] or ["Men's Doubles"]
    with tabs[3]:
        st.subheader("Divisions")
        st.caption("Each row is a playable division. Assign each division to exactly one day. Multiple events and divisions can happen on the same day.")
        st.info(
            "Age rule defaults baked into the builder: for doubles age divisions the younger player determines eligibility, and if teammates have different skill levels the team plays up into the higher skill division."
        )
        if structure_locked:
            st.caption("Divisions are view-only because registrations already exist.")
        divisions_df = st.data_editor(
            divisions_seed_df,
            hide_index=True,
            num_rows="dynamic",
            key=f"tm_divisions_editor_{tournament_id}",
            disabled=structure_locked,
            column_config={
                "event_family": st.column_config.SelectboxColumn("Event family", options=event_family_options),
                "division_name": st.column_config.TextColumn("Division name"),
                "skill_label": st.column_config.TextColumn("Skill label"),
                "age_mode": st.column_config.SelectboxColumn("Age mode", options=AGE_MODES),
                "age_label": st.column_config.TextColumn("Age label"),
                "age_ranges": st.column_config.TextColumn("Custom age ranges"),
                "min_teams_per_age_group": st.column_config.NumberColumn("Min teams per age group", step=1, min_value=1),
                "split_age_threshold": st.column_config.NumberColumn("Split-age threshold", step=1, min_value=1),
                "assigned_day": st.column_config.SelectboxColumn("Assigned day", options=day_label_options),
                "capacity_teams": st.column_config.NumberColumn("Capacity", step=1, min_value=1),
                "price_usd": st.column_config.NumberColumn("Price USD", step=1),
                "waitlist_enabled": st.column_config.CheckboxColumn("Waitlist"),
                "partner_board_enabled": st.column_config.CheckboxColumn("Partner board"),
                "status": st.column_config.SelectboxColumn("Status", options=DIVISION_STATUSES),
                "division_format": st.column_config.SelectboxColumn("Format override", options=[""] + COMPETITION_FORMATS),
                "division_scoring": st.column_config.SelectboxColumn("Scoring override", options=[""] + SCORING_OPTIONS),
                "notes": st.column_config.TextColumn("Notes"),
            },
            use_container_width=True,
        )
        for mode, help_text in AGE_MODE_HELP.items():
            st.caption(f"**{mode.replace('_', ' ').title()}** — {help_text}")

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
        st.caption("Save the builder before sharing links if you changed days, events, or divisions.")

        validation_errors = _validate_builder(days_df, events_df, divisions_df)
        if validation_errors:
            for error in validation_errors:
                st.warning(error)

        days_payload, event_payload = _build_payloads(tournament_id, days_df, events_df, divisions_df)
        if st.button("Save builder changes", type="primary", disabled=structure_locked):
            if validation_errors:
                st.error("Resolve the builder warnings before saving.")
            elif not days_payload:
                st.error("Enable at least one tournament day.")
            elif not event_payload:
                st.error("Create at least one division before saving.")
            else:
                replace_registration_configuration(
                    supabase,
                    tournament_id=tournament_id,
                    days=days_payload,
                    event_options=event_payload,
                )
                st.success("Tournament schedule, events, and divisions saved.")
                st.rerun()

        state = build_registration_state(
            supabase,
            get_tournament_record(supabase, tournament_id) or tournament,
            settings,
            list_registration_days(supabase, tournament_id),
            list_event_options(supabase, tournament_id),
        )
        registrations = state.get("registrations", [])
        issues = state.get("issues", [])
        st.metric("Registrations received", len(registrations))
        if issues:
            st.caption(f"Current registration issues: {len(issues)}")
        if registrations:
            st.dataframe(pd.DataFrame(registrations), hide_index=True, use_container_width=True)
            st.download_button(
                "Download registration workbook",
                data=build_registration_workbook(tournament=tournament, state=state),
                file_name=f"{_safe_text(tournament.get('name') or 'tournament').replace(' ', '_')}_registration.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
