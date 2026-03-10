from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import streamlit as st

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
DIVISION_STATUSES = ["draft", "open", "closed"]


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _parse_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).date()
    except Exception:
        return None


def _fmt_dt(value: Any) -> str:
    if value in (None, "", "None"):
        return ""
    text = str(value).strip().replace("+00:00", "Z")
    return text[:-1][:16] if text.endswith("Z") else text[:16]


def _parse_local_dt(value: str) -> str | None:
    text = str(value or "").strip()
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
        rows.append({"event_date": cursor.isoformat(), "label": f"Day {idx} · {cursor.strftime('%a %b %d')}", "enabled": True})
        cursor += timedelta(days=1)
        idx += 1
    return rows


def _seed_days(days: list[dict[str, Any]], tournament: dict[str, Any]) -> pd.DataFrame:
    if days:
        rows = [{"id": str(d.get("id")), "event_date": d.get("event_date"), "label": d.get("label"), "enabled": bool(d.get("enabled", True))} for d in days]
        return pd.DataFrame(rows)
    generated = _date_rows(tournament.get("start_date"), tournament.get("end_date"))
    if not generated:
        generated = [{"event_date": None, "label": "Day 1", "enabled": True}]
    return pd.DataFrame([{"id": _uid("day"), **r} for r in generated])


def _seed_divisions(days: list[dict[str, Any]], event_options: list[dict[str, Any]]) -> pd.DataFrame:
    day_ids = [str(d.get("id")) for d in days] or ["day_1"]
    rows = []
    for row in event_options:
        rows.append(
            {
                "id": str(row.get("id") or _uid("event")),
                "event_family": row.get("event_family_label") or row.get("label") or "Event",
                "division_name": row.get("division_name") or row.get("label") or "Division",
                "participant_type": row.get("event_type") or "SINGLES",
                "gender_restriction": row.get("gender_restriction") or "ANY",
                "skill_mode": row.get("skill_mode") or "Open",
                "age_mode": row.get("age_mode") or "ALL_AGES",
                "age_label": row.get("age_label") or "All Ages",
                "capacity_teams": row.get("capacity_teams"),
                "price_usd": row.get("price_usd"),
                "waitlist_enabled": bool(row.get("waitlist_enabled", True)),
                "partner_board_enabled": bool(row.get("partner_board_enabled", row.get("public_partner_board", True))),
                "status": row.get("status") or "draft",
                "assigned_day_id": str(row.get("registration_day_id") or day_ids[0]),
                "event_format_default": row.get("event_format_default") or "ROUND_ROBIN",
                "scoring_default": row.get("scoring_default") or "GAME_TO_11",
                "event_format_override": row.get("event_format_override") or "",
                "scoring_override": row.get("scoring_override") or "",
                "age_rules": row.get("age_rules") or "",
            }
        )
    if not rows:
        rows = [
            {
                "id": _uid("event"),
                "event_family": "Men's Doubles",
                "division_name": "Men's Doubles Open",
                "participant_type": "GENDER_DOUBLES",
                "gender_restriction": "MEN",
                "skill_mode": "Open",
                "age_mode": "ALL_AGES",
                "age_label": "All Ages",
                "capacity_teams": 16,
                "price_usd": None,
                "waitlist_enabled": True,
                "partner_board_enabled": True,
                "status": "draft",
                "assigned_day_id": day_ids[0],
                "event_format_default": "ROUND_ROBIN_PLUS_PLAYOFF",
                "scoring_default": "GAME_TO_15",
                "event_format_override": "",
                "scoring_override": "",
                "age_rules": "",
            }
        ]
    return pd.DataFrame(rows)


def _build_payloads(tournament_id: str, days_df: pd.DataFrame, divisions_df: pd.DataFrame):
    day_payload = []
    day_ids: set[str] = set()
    for idx, row in days_df.fillna("").iterrows():
        if not bool(row.get("enabled", True)):
            continue
        day_id = str(row.get("id") or _uid("day"))
        day_ids.add(day_id)
        day_payload.append(
            {
                "id": day_id,
                "tournament_id": str(tournament_id),
                "sort_order": idx,
                "label": str(row.get("label") or f"Day {idx + 1}"),
                "event_date": str(row.get("event_date") or "").strip() or None,
                "enabled": True,
            }
        )

    events_payload = []
    for idx, row in divisions_df.fillna("").iterrows():
        day_id = str(row.get("assigned_day_id") or "").strip()
        if day_id not in day_ids:
            continue
        division_name = str(row.get("division_name") or "").strip()
        event_family = str(row.get("event_family") or "").strip()
        label = division_name or f"{event_family} Division"
        events_payload.append(
            {
                "id": str(row.get("id") or _uid("event")),
                "tournament_id": str(tournament_id),
                "registration_day_id": day_id,
                "sort_order": idx,
                "label": label,
                "event_type": str(row.get("participant_type") or "SINGLES"),
                "gender_restriction": str(row.get("gender_restriction") or "ANY"),
                "skill_label": str(row.get("skill_mode") or "Open"),
                "age_label": str(row.get("age_label") or "All Ages"),
                "partner_required": str(row.get("participant_type") or "SINGLES") != "SINGLES",
                "capacity_teams": int(float(row["capacity_teams"])) if str(row.get("capacity_teams") or "").strip() else None,
                "public_partner_board": bool(row.get("partner_board_enabled", True)),
                "price_usd": float(row["price_usd"]) if str(row.get("price_usd") or "").strip() else None,
                "event_family_label": event_family or label,
                "division_name": division_name or label,
                "event_format_default": str(row.get("event_format_default") or "ROUND_ROBIN"),
                "scoring_default": str(row.get("scoring_default") or "GAME_TO_11"),
                "event_format_override": str(row.get("event_format_override") or "").strip() or None,
                "scoring_override": str(row.get("scoring_override") or "").strip() or None,
                "age_mode": str(row.get("age_mode") or "ALL_AGES"),
                "age_rules": str(row.get("age_rules") or "").strip() or None,
                "waitlist_enabled": bool(row.get("waitlist_enabled", True)),
                "partner_board_enabled": bool(row.get("partner_board_enabled", True)),
                "status": str(row.get("status") or "draft"),
                "enabled": True,
            }
        )
    return day_payload, events_payload


def render(ctx):
    page_shell("🛠️ Tournament Manager", "Operator-friendly tournament setup and registration publishing.", mode_label="Admin")
    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    supabase = getattr(ctx, "supabase", None)
    club_id = getattr(ctx, "club_id", None)
    if supabase is None or club_id is None:
        st.error("Missing database context.")
        st.stop()

    ok, detail = registration_feature_available(supabase)
    if not ok:
        st.error("Tournament registration tables are missing.")
        if detail:
            st.caption(detail)
        st.stop()

    tournaments = list_existing_tournaments(supabase, str(club_id))
    if not tournaments:
        st.info("Create a tournament in Tournaments first.")
        st.stop()

    labels = [f"{t.get('name')} ({t.get('status')})" for t in tournaments]
    picked = st.selectbox("Tournament", labels)
    tournament = tournaments[labels.index(picked)]
    tournament_id = str(tournament.get("id"))
    settings = get_registration_settings(supabase, tournament_id, tournament_name=str(tournament.get("name") or ""))
    days = list_registration_days(supabase, tournament_id)
    event_options = list_event_options(supabase, tournament_id)
    reg_count = count_tournament_registrations(supabase, tournament_id)

    tabs = st.tabs(["Tournament Info", "Days", "Events", "Divisions", "Schedule Preview", "Publish / Registration Links"])

    with tabs[0]:
        st.subheader("Tournament Info")
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Tournament name", value=str(tournament.get("name") or ""))
            start_date = st.text_input("Start date (YYYY-MM-DD)", value=str(tournament.get("start_date") or ""))
            end_date = st.text_input("End date (YYYY-MM-DD)", value=str(tournament.get("end_date") or ""))
            slug = st.text_input("Registration slug", value=str(settings.get("registration_slug") or ""))
            locale = st.selectbox("Locale", ["en", "es", "bilingual"], index=["en", "es", "bilingual"].index(str(settings.get("locale") or "en")))
        with c2:
            status = st.selectbox("Registration status", REGISTRATION_STATUS_OPTIONS, index=REGISTRATION_STATUS_OPTIONS.index(str(settings.get("registration_status") or "draft")))
            reg_open = st.text_input("Registration opens (local datetime)", value=_fmt_dt(settings.get("registration_open_at")))
            reg_close = st.text_input("Registration closes (local datetime)", value=_fmt_dt(settings.get("registration_close_at")))
            sponsor = st.text_area("Sponsor text", value=str(settings.get("sponsor_markdown") or ""), height=80)
            refund = st.text_area("Refund policy", value=str(settings.get("refund_policy_markdown") or ""), height=80)
        notes = st.text_area("Notes / rules", value=str(settings.get("rules_markdown") or ""), height=120)
        if st.button("Save tournament info", type="primary"):
            supabase.table("tournaments").update({"name": name.strip(), "start_date": start_date or None, "end_date": end_date or None}).eq("id", tournament_id).execute()
            upsert_registration_settings(
                supabase,
                {
                    "id": settings.get("id"),
                    "tournament_id": tournament_id,
                    "registration_slug": slug,
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
            st.success("Tournament info saved.")
            st.rerun()

    days_df = st.data_editor(
        _seed_days(days, tournament),
        hide_index=True,
        num_rows="dynamic",
        disabled=bool(reg_count),
        key=f"days_editor_{tournament_id}",
        column_config={
            "id": st.column_config.TextColumn("Internal ID", disabled=True),
            "event_date": st.column_config.TextColumn("Date"),
            "label": st.column_config.TextColumn("Day label"),
            "enabled": st.column_config.CheckboxColumn("Enabled"),
        },
    )
    divisions_df = st.data_editor(
        _seed_divisions(days_df.to_dict("records"), event_options),
        hide_index=True,
        num_rows="dynamic",
        disabled=bool(reg_count),
        key=f"div_editor_{tournament_id}",
        column_config={
            "id": st.column_config.TextColumn("Internal ID", disabled=True),
            "event_family": st.column_config.TextColumn("Event"),
            "division_name": st.column_config.TextColumn("Division"),
            "participant_type": st.column_config.SelectboxColumn("Participant type", options=PARTICIPANT_TYPES),
            "gender_restriction": st.column_config.SelectboxColumn("Gender restriction", options=["ANY", "MEN", "WOMEN", "MIXED"]),
            "skill_mode": st.column_config.TextColumn("Skill mode / label"),
            "age_mode": st.column_config.SelectboxColumn("Age mode", options=AGE_MODES),
            "age_label": st.column_config.TextColumn("Age label"),
            "age_rules": st.column_config.TextColumn("Age split rules / thresholds"),
            "capacity_teams": st.column_config.NumberColumn("Capacity", step=1, min_value=1),
            "price_usd": st.column_config.NumberColumn("Price USD", step=1),
            "waitlist_enabled": st.column_config.CheckboxColumn("Waitlist"),
            "partner_board_enabled": st.column_config.CheckboxColumn("Partner board"),
            "status": st.column_config.SelectboxColumn("Status", options=DIVISION_STATUSES),
            "assigned_day_id": st.column_config.SelectboxColumn("Assigned day", options=[str(d.get("id")) for d in days_df.to_dict("records")]),
            "event_format_default": st.column_config.SelectboxColumn("Default format", options=COMPETITION_FORMATS),
            "scoring_default": st.column_config.SelectboxColumn("Default scoring", options=SCORING_OPTIONS),
            "event_format_override": st.column_config.SelectboxColumn("Division format override", options=[""] + COMPETITION_FORMATS),
            "scoring_override": st.column_config.SelectboxColumn("Division scoring override", options=[""] + SCORING_OPTIONS),
        },
    )

    with tabs[1]:
        st.subheader("Days")
        st.caption("Start/end date auto-generates days. You can relabel days or disable unused days.")
        if st.button("Regenerate days from tournament dates", disabled=bool(reg_count)):
            regenerated = pd.DataFrame([{"id": _uid("day"), **r} for r in _date_rows(tournament.get("start_date"), tournament.get("end_date"))])
            st.session_state[f"days_editor_{tournament_id}"] = regenerated
            st.rerun()

    with tabs[2]:
        st.subheader("Events")
        if divisions_df.empty:
            st.info("Add divisions to create events.")
        else:
            event_defaults = (
                divisions_df.groupby("event_family")[["event_format_default", "scoring_default", "participant_type"]]
                .agg(lambda x: list(x)[0])
                .reset_index()
                .rename(columns={"event_family": "Event", "event_format_default": "Default Format", "scoring_default": "Default Scoring", "participant_type": "Participant Type"})
            )
            st.dataframe(event_defaults, use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("Divisions")
        st.caption("Each division is assigned to exactly one day. Event defaults inherit; per-division overrides are optional.")

    with tabs[4]:
        st.subheader("Schedule Preview")
        day_lookup = {str(d.get("id")): d.get("label") for d in days_df.to_dict("records")}
        for day_id, grp in divisions_df.groupby("assigned_day_id"):
            st.markdown(f"**{day_lookup.get(str(day_id), day_id)}**")
            preview = grp[["event_family", "division_name", "participant_type", "event_format_default", "event_format_override", "scoring_default", "scoring_override"]].copy()
            st.dataframe(preview, use_container_width=True, hide_index=True)

    with tabs[5]:
        st.subheader("Publish / Registration Links")
        public_urls = build_public_urls(base_url=str(st.session_state.get("base_url") or ""), tournament_id=tournament_id, registration_slug=settings.get("registration_slug"))
        st.link_button("Public registration form", public_urls["registration"])
        st.link_button("Public partner board", public_urls["partner_board"])

        day_payload, event_payload = _build_payloads(tournament_id, days_df, divisions_df)
        if st.button("Save builder changes", type="primary", disabled=bool(reg_count)):
            if not day_payload:
                st.error("Enable at least one day.")
            elif not event_payload:
                st.error("Create at least one division.")
            else:
                replace_registration_configuration(supabase, tournament_id=tournament_id, days=day_payload, event_options=event_payload)
                st.success("Tournament setup saved.")
                st.rerun()

        state = build_registration_state(supabase, get_tournament_record(supabase, tournament_id) or tournament, settings, list_registration_days(supabase, tournament_id), list_event_options(supabase, tournament_id))
        regs = state.get("registrations", [])
        st.caption(f"Registrations: {len(regs)}")
        if regs:
            st.dataframe(pd.DataFrame(regs), use_container_width=True, hide_index=True)
            st.download_button("Download registration workbook", data=build_registration_workbook(tournament=tournament, state=state), file_name=f"{str(tournament.get('name') or 'tournament').replace(' ','_')}_registration.xlsx")
