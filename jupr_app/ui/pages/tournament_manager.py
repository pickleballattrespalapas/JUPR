from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from jupr_app.domain.tournament_registration_exports import build_registration_workbook
from jupr_app.domain.tournament_registration_repo import (
    EVENT_TYPE_OPTIONS,
    GENDER_RESTRICTION_OPTIONS,
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


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _coerce_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def _coerce_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _fmt_dt(value: Any) -> str:
    if value in (None, "", "None"):
        return ""
    text = str(value).strip().replace("+00:00", "Z")
    if text.endswith("Z"):
        text = text[:-1]
    return text[:16]


def _parse_local_dt(value: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).isoformat()
    except Exception:
        return None


def _days_editor_seed(days: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for row in days:
        rows.append(
            {
                "day_key": str(row.get("id")),
                "label": row.get("label"),
                "event_date": row.get("event_date"),
            }
        )
    if not rows:
        rows = [
            {"day_key": "day_1", "label": "Day 1", "event_date": None},
        ]
    return pd.DataFrame(rows)


def _events_editor_seed(days: list[dict[str, Any]], events: list[dict[str, Any]]) -> pd.DataFrame:
    day_lookup = {str(row.get("id") or row.get("day_key")): row.get("label") for row in days}
    rows = []
    for row in events:
        rows.append(
            {
                "event_key": str(row.get("id")),
                "day_key": str(row.get("registration_day_id")),
                "day_label": day_lookup.get(str(row.get("registration_day_id"))) or str(row.get("registration_day_id")),
                "label": row.get("label"),
                "event_type": str(row.get("event_type") or "SINGLES"),
                "gender_restriction": str(row.get("gender_restriction") or "ANY"),
                "skill_label": row.get("skill_label"),
                "age_label": row.get("age_label"),
                "partner_required": bool(row.get("partner_required")),
                "capacity_teams": row.get("capacity_teams"),
                "public_partner_board": bool(row.get("public_partner_board", True)),
                "price_usd": row.get("price_usd"),
            }
        )
    if not rows:
        default_day = str((days[0].get("id") or days[0].get("day_key"))) if days else "day_1"
        rows = [
            {
                "event_key": "event_mens_doubles",
                "day_key": default_day,
                "day_label": day_lookup.get(default_day, "Day 1"),
                "label": "Men's Doubles",
                "event_type": "GENDER_DOUBLES",
                "gender_restriction": "MEN",
                "skill_label": "Open",
                "age_label": "All Ages",
                "partner_required": True,
                "capacity_teams": 8,
                "public_partner_board": True,
                "price_usd": None,
            },
            {
                "event_key": "event_womens_doubles",
                "day_key": default_day,
                "day_label": day_lookup.get(default_day, "Day 1"),
                "label": "Women's Doubles",
                "event_type": "GENDER_DOUBLES",
                "gender_restriction": "WOMEN",
                "skill_label": "Open",
                "age_label": "All Ages",
                "partner_required": True,
                "capacity_teams": 8,
                "public_partner_board": True,
                "price_usd": None,
            },
            {
                "event_key": "event_mixed_doubles",
                "day_key": default_day,
                "day_label": day_lookup.get(default_day, "Day 1"),
                "label": "Mixed Doubles",
                "event_type": "MIXED_DOUBLES",
                "gender_restriction": "MIXED",
                "skill_label": "Open",
                "age_label": "All Ages",
                "partner_required": True,
                "capacity_teams": 8,
                "public_partner_board": True,
                "price_usd": None,
            },
            {
                "event_key": "event_mens_singles",
                "day_key": default_day,
                "day_label": day_lookup.get(default_day, "Day 1"),
                "label": "Men's Singles",
                "event_type": "SINGLES",
                "gender_restriction": "MEN",
                "skill_label": "Open",
                "age_label": "All Ages",
                "partner_required": False,
                "capacity_teams": 16,
                "public_partner_board": False,
                "price_usd": None,
            },
            {
                "event_key": "event_womens_singles",
                "day_key": default_day,
                "day_label": day_lookup.get(default_day, "Day 1"),
                "label": "Women's Singles",
                "event_type": "SINGLES",
                "gender_restriction": "WOMEN",
                "skill_label": "Open",
                "age_label": "All Ages",
                "partner_required": False,
                "capacity_teams": 16,
                "public_partner_board": False,
                "price_usd": None,
            },
        ]
    return pd.DataFrame(rows)


def _build_config_payloads(
    *,
    tournament_id: str,
    days_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    days_payload: list[dict[str, Any]] = []
    day_id_map: dict[str, str] = {}
    for idx, row in days_df.fillna("").iterrows():
        label = str(row.get("label") or "").strip()
        if not label:
            continue
        source_day_key = str(row.get("day_key") or f"day_{idx + 1}")
        day_id = source_day_key if source_day_key else _uid("day")
        day_id_map[source_day_key] = day_id
        days_payload.append(
            {
                "id": day_id,
                "tournament_id": str(tournament_id),
                "sort_order": len(days_payload) + 1,
                "label": label,
                "event_date": row.get("event_date") or None,
            }
        )

    events_payload: list[dict[str, Any]] = []
    for idx, row in events_df.fillna("").iterrows():
        label = str(row.get("label") or "").strip()
        if not label:
            continue
        raw_day_key = str(row.get("day_key") or "").strip()
        mapped_day_id = day_id_map.get(raw_day_key) or raw_day_key
        if not mapped_day_id:
            raise ValueError(f"Event '{label}' is missing a day assignment.")
        events_payload.append(
            {
                "id": str(row.get("event_key") or _uid("event")),
                "tournament_id": str(tournament_id),
                "registration_day_id": str(mapped_day_id),
                "sort_order": len(events_payload) + 1,
                "label": label,
                "event_type": str(row.get("event_type") or "SINGLES").upper(),
                "gender_restriction": str(row.get("gender_restriction") or "ANY").upper(),
                "skill_label": str(row.get("skill_label") or "").strip() or None,
                "age_label": str(row.get("age_label") or "").strip() or None,
                "partner_required": bool(row.get("partner_required")),
                "capacity_teams": _coerce_int(row.get("capacity_teams")),
                "public_partner_board": bool(row.get("public_partner_board", True)),
                "price_usd": _coerce_float(row.get("price_usd")),
            }
        )
    return days_payload, events_payload


def _render_metrics(state: dict[str, Any]) -> None:
    summary = state.get("summary", {})
    cols = st.columns(6)
    cols[0].metric("Registrations", summary.get("total_registrations", 0))
    cols[1].metric("Selections", summary.get("total_selections", 0))
    cols[2].metric("Confirmed", summary.get("confirmed_entries", 0))
    cols[3].metric("Needs partner", summary.get("needs_partner_entries", 0))
    cols[4].metric("Waitlist", summary.get("waitlist_entries", 0))
    cols[5].metric("Issues", summary.get("issue_count", 0))


def render(ctx):
    mode_label = "Public" if bool(getattr(ctx, "public_mode", False)) else "Admin"
    page_shell(
        "🏆 Tournament Manager",
        "Registration setup, partner board publishing, roster compilation, and export tools.",
        mode_label=mode_label,
    )

    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    supabase = getattr(ctx, "supabase", None)
    club_id = str(getattr(ctx, "club_id", ""))
    if supabase is None or not club_id:
        st.error("Missing database context.")
        st.stop()

    available, detail = registration_feature_available(supabase)
    if not available:
        st.error(
            "Tournament registration tables are not available yet. Apply the SQL migration in "
            "jupr/sql/20260309_tournament_registration.sql first."
        )
        if detail:
            st.caption(detail)
        st.stop()

    tournaments = list_existing_tournaments(supabase, club_id)
    if not tournaments:
        st.info("Create a tournament first on the Tournaments page.")
        st.stop()

    st.markdown("### Admin Setup Steps")
    st.markdown(
        """
1. **Pick tournament** (created on the **Tournaments** page).
2. **Configure registration settings** (slug, status, open/close dates, locale, waitlist).
3. **Define tournament days** (Day 1, Day 2, etc.).
4. **Define event options** (doubles/singles, gender, skill/age labels, partner rules, capacity, pricing).
5. **Publish registration links** for players and partner board.
6. **Monitor registrations and compiled rosters**. Once registrations exist, structural day/event edits are locked.
        """
    )
    st.caption("This page configures registration for the selected tournament. Tournament shell creation and live bracket operations happen on the Tournaments page.")

    preselected = str(st.query_params.get("tournament_id", "")).strip()
    labels = [f"{row.get('name')} ({row.get('status')})" for row in tournaments]
    default_index = 0
    if preselected:
        for idx, row in enumerate(tournaments):
            if str(row.get("id")) == preselected:
                default_index = idx
                break
    selected_label = st.selectbox("Select tournament", labels, index=default_index)
    selected_idx = labels.index(selected_label)
    tournament = tournaments[selected_idx]
    tournament_id = str(tournament.get("id"))
    st.query_params["tournament_id"] = tournament_id

    settings = get_registration_settings(supabase, tournament_id, tournament_name=str(tournament.get("name") or ""))
    days = list_registration_days(supabase, tournament_id)
    event_options = list_event_options(supabase, tournament_id)
    state = build_registration_state(supabase, tournament, settings, days, event_options)
    links = build_public_urls(
        base_url=str(st.session_state.get("base_url") or ""),
        tournament_id=tournament_id,
        registration_slug=settings.get("registration_slug"),
    )

    st.caption(f"Tournament ID: {tournament_id}")
    _render_metrics(state)

    with st.expander("Links"):
        st.text_input("Public registration", value=links["registration"], key=f"reg_link_{tournament_id}")
        st.text_input("Public partner board", value=links["partner_board"], key=f"board_link_{tournament_id}")
        st.text_input("Admin operations", value=links["admin_operations"], key=f"ops_link_{tournament_id}")

    tabs = st.tabs(["Step 2: Settings", "Step 3-4: Days & Events", "Step 5: Publish & Registrations", "Partner Board", "Step 6: Issues & Rosters"])

    with tabs[0]:
        st.subheader("Registration settings")
        c1, c2 = st.columns(2)
        with c1:
            registration_slug = st.text_input(
                "Public slug",
                value=str(settings.get("registration_slug") or ""),
                help="Used in public links if present. Keep it short and unique.",
            )
            registration_status = st.selectbox(
                "Registration status",
                REGISTRATION_STATUS_OPTIONS,
                index=REGISTRATION_STATUS_OPTIONS.index(str(settings.get("registration_status") or "draft")),
            )
            locale = st.selectbox(
                "Locale",
                ["en", "es", "bilingual"],
                index=["en", "es", "bilingual"].index(str(settings.get("locale") or "en"))
                if str(settings.get("locale") or "en") in ["en", "es", "bilingual"]
                else 0,
            )
            max_events_per_day = st.number_input(
                "Max events per day",
                min_value=1,
                max_value=4,
                step=1,
                value=int(settings.get("max_events_per_day") or 1),
            )
            waitlist_enabled = st.checkbox("Enable waitlist", value=bool(settings.get("waitlist_enabled", True)))
            partner_board_enabled = st.checkbox(
                "Enable public partner board",
                value=bool(settings.get("partner_board_enabled", True)),
            )
        with c2:
            open_at = st.text_input(
                "Registration opens at (ISO or YYYY-MM-DDTHH:MM)",
                value=_fmt_dt(settings.get("registration_open_at")),
            )
            close_at = st.text_input(
                "Registration closes at (ISO or YYYY-MM-DDTHH:MM)",
                value=_fmt_dt(settings.get("registration_close_at")),
            )
            sponsor_markdown = st.text_area(
                "Sponsor / callout text",
                value=str(settings.get("sponsor_markdown") or ""),
                height=100,
            )
        rules_markdown = st.text_area(
            "Rules / registration notes",
            value=str(settings.get("rules_markdown") or ""),
            height=180,
        )
        refund_policy_markdown = st.text_area(
            "Refund policy",
            value=str(settings.get("refund_policy_markdown") or ""),
            height=120,
        )

        if st.button("Save settings", type="primary"):
            try:
                new_settings = upsert_registration_settings(
                    supabase,
                    {
                        "id": settings.get("id"),
                        "tournament_id": tournament_id,
                        "registration_slug": registration_slug,
                        "registration_status": registration_status,
                        "locale": locale,
                        "registration_open_at": _parse_local_dt(open_at) or open_at or None,
                        "registration_close_at": _parse_local_dt(close_at) or close_at or None,
                        "waitlist_enabled": waitlist_enabled,
                        "partner_board_enabled": partner_board_enabled,
                        "max_events_per_day": max_events_per_day,
                        "rules_markdown": rules_markdown,
                        "refund_policy_markdown": refund_policy_markdown,
                        "sponsor_markdown": sponsor_markdown,
                    },
                )
                st.success("Registration settings saved.")
                settings = new_settings
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save registration settings: {exc}")

    with tabs[1]:
        st.subheader("Days and event options")
        st.caption("Once registrations are submitted, structural day/event changes are locked to preserve data integrity.")
        registration_count = count_tournament_registrations(supabase, tournament_id)
        if registration_count:
            st.warning(
                "This tournament already has registrations. Replacing the day/event configuration could invalidate submitted forms, so configuration editing is locked here."
            )
            st.caption("Clone the tournament or create a new one for major structure changes.")

        days_df = st.data_editor(
            _days_editor_seed(days),
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key=f"tm_days_editor_{tournament_id}",
            disabled=bool(registration_count),
            column_config={
                "day_key": st.column_config.TextColumn("Day ID / key"),
                "label": st.column_config.TextColumn("Day label"),
                "event_date": st.column_config.TextColumn("Date (YYYY-MM-DD)"),
            },
        )

        day_key_options = [str(value) for value in days_df.get("day_key", pd.Series(dtype=str)).fillna("").tolist() if str(value).strip()]
        events_df = st.data_editor(
            _events_editor_seed(days_df.to_dict("records"), event_options),
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key=f"tm_events_editor_{tournament_id}",
            disabled=bool(registration_count),
            column_config={
                "event_key": st.column_config.TextColumn("Event ID / key"),
                "day_key": st.column_config.SelectboxColumn("Day key", options=day_key_options or ["day_1"]),
                "day_label": st.column_config.TextColumn("Day label", disabled=True),
                "label": st.column_config.TextColumn("Event label"),
                "event_type": st.column_config.SelectboxColumn("Event format", options=EVENT_TYPE_OPTIONS),
                "gender_restriction": st.column_config.SelectboxColumn("Eligible gender", options=GENDER_RESTRICTION_OPTIONS),
                "skill_label": st.column_config.TextColumn("Skill label"),
                "age_label": st.column_config.TextColumn("Age label"),
                "partner_required": st.column_config.CheckboxColumn("Partner required at registration"),
                "capacity_teams": st.column_config.NumberColumn("Capacity", step=1, min_value=1),
                "public_partner_board": st.column_config.CheckboxColumn("Public partner board"),
                "price_usd": st.column_config.NumberColumn("Price USD", step=1),
            },
        )

        if st.button("Replace day/event configuration", disabled=bool(registration_count)):
            try:
                days_payload, events_payload = _build_config_payloads(
                    tournament_id=tournament_id,
                    days_df=days_df,
                    events_df=events_df,
                )
                if not days_payload:
                    st.error("Add at least one registration day.")
                elif not events_payload:
                    st.error("Add at least one event option.")
                else:
                    replace_registration_configuration(
                        supabase,
                        tournament_id=tournament_id,
                        days=days_payload,
                        event_options=events_payload,
                    )
                    st.success("Registration form structure saved.")
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not replace registration configuration: {exc}")

    with tabs[2]:
        st.subheader("Publish links and registrations")
        _render_metrics(state)
        regs = state.get("registrations", [])
        if not regs:
            st.info("No registrations yet.")
        else:
            regs_df = pd.DataFrame(regs)
            visible_cols = [
                col for col in [
                    "submitted_at",
                    "display_name",
                    "email",
                    "phone",
                    "dupr_id",
                    "doubles_skill",
                    "singles_skill",
                    "age",
                    "age_bracket",
                    "gender",
                    "payment_status",
                    "notes",
                ] if col in regs_df.columns
            ]
            st.dataframe(regs_df[visible_cols], use_container_width=True, hide_index=True)

            workbook_bytes = build_registration_workbook(tournament=tournament, state=state)
            st.download_button(
                "Download compiled registration workbook",
                data=workbook_bytes,
                file_name=f"{str(tournament.get('name') or 'tournament').strip().replace(' ', '_')}_registration.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with tabs[3]:
        st.subheader("Public partner board preview")
        board_rows = state.get("partner_board", [])
        if not board_rows:
            st.info("Nobody is publicly listed as needing a partner yet.")
        else:
            table_rows = []
            for row in board_rows:
                player = row.get("player") or {}
                table_rows.append(
                    {
                        "Day": row.get("event_day_label"),
                        "Event": row.get("event_label"),
                        "Player": player.get("display_name"),
                        "Email": player.get("email") if row.get("show_contact_email") else None,
                        "Skill": player.get("skill"),
                        "Age": player.get("age"),
                        "Note": row.get("note"),
                    }
                )
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    with tabs[4]:
        st.subheader("Issues")
        issues = state.get("issues", [])
        if issues:
            st.dataframe(pd.DataFrame(issues), use_container_width=True, hide_index=True)
        else:
            st.success("No current registration issues.")

        st.divider()
        st.subheader("Compiled rosters")
        for roster in state.get("event_rosters", []):
            with st.expander(f"{roster.get('event_day_label')} — {roster.get('event_label')}"):
                rows = []
                for entry in roster.get("entries", []):
                    members = entry.get("members") or []
                    rows.append(
                        {
                            "Status": entry.get("status"),
                            "Member 1": (members[0] or {}).get("display_name") if len(members) > 0 else None,
                            "Member 1 Email": (members[0] or {}).get("email") if len(members) > 0 else None,
                            "Member 2": (members[1] or {}).get("display_name") if len(members) > 1 else None,
                            "Member 2 Email": (members[1] or {}).get("email") if len(members) > 1 else None,
                            "Submitted": entry.get("submitted_at"),
                        }
                    )
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.caption("No entries yet.")
