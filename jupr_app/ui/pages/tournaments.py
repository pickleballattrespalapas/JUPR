from __future__ import annotations

from datetime import date, datetime
from io import StringIO
from typing import Any

import pandas as pd
import streamlit as st

from jupr_app.domain.event_tags import derive_default_date_tags, normalize_event_tags
from jupr_app.domain.match_processing import process_matches
from jupr_app.domain.tournament_match_payload import build_tournament_match_payload
from jupr_app.domain.tournaments import (
    SUPPORTED_TEAM_COUNTS,
    build_playoff_games,
    build_podium_payload,
    build_round_robin_games,
    compute_podium_from_playoffs,
    compute_podium_from_rr,
    compute_round_robin_standings,
    finalize_game,
    resolve_playoff_dependencies,
)
from jupr_app.domain.tournament_podium import award_tournament_trophies_from_podium, upsert_tournament_podium
from jupr_app.domain.tournament_registration_repo import (
    build_public_urls,
    build_registration_state,
    get_registration_settings,
    list_event_options as list_registration_event_options,
    list_registration_days,
    registration_feature_available,
    upsert_registration_settings,
)
from jupr_app.ui.layout import page_shell

LEGACY_DEFAULT_TEAM_COUNT = 4
TOURNAMENT_STATUS_OPTIONS = ["DRAFT", "REGISTRATION", "REGISTRATION_OPEN", "REGISTRATION_CLOSED"]
TOURNAMENT_LOCALE_OPTIONS = ["en", "es", "bilingual"]
IMPORT_SOURCE_OPTIONS = ["REGISTRATION", "MANUAL", "BULK_UPLOAD"]


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_name(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _parse_optional_date(value: Any) -> str | None:
    if isinstance(value, date):
        return value.isoformat()
    text = _safe_text(value)
    return text or None




def _tournament_date_window_text(tournament: dict[str, Any]) -> str | None:
    start = _safe_text(tournament.get("start_date"))
    end = _safe_text(tournament.get("end_date"))
    if not start or not end:
        return None
    return f"{start} → {end}"
def _resolve_player_id(name: str, name_to_id: dict[str, Any], id_to_name: dict[Any, str]) -> Any:
    if name in name_to_id:
        return name_to_id[name]
    normalized = _normalize_name(name)
    if not normalized:
        return None
    for known_name, player_id in name_to_id.items():
        if _normalize_name(known_name) == normalized:
            return player_id
    for player_id, known_name in id_to_name.items():
        if _normalize_name(known_name) == normalized:
            return player_id
    return None


def _find_event_roster(registration_bridge: dict | None, *, event_option_id: str, registration_day_id: str) -> dict | None:
    state = (registration_bridge or {}).get("state") or {}
    for roster in state.get("event_rosters") or []:
        if str(roster.get("event_option_id")) == str(event_option_id) and str(roster.get("event_day_id")) == str(registration_day_id):
            return roster
    return None


def _parse_bulk_upload(file, pasted_text: str) -> pd.DataFrame:
    if file is not None:
        name = str(getattr(file, "name", "")).lower()
        if name.endswith(".xlsx"):
            return pd.read_excel(file)
        return pd.read_csv(file)
    text = _safe_text(pasted_text)
    if not text:
        return pd.DataFrame()
    sep = "\t" if "\t" in text and text.count("\t") >= text.count(",") else ","
    return pd.read_csv(StringIO(text), sep=sep)


def _canonical_import_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=["Player 1", "Player 2", "Team Name", "Seed", "Notes"])
    aliases = {
        "player 1": "Player 1",
        "player1": "Player 1",
        "player one": "Player 1",
        "player 2": "Player 2",
        "player2": "Player 2",
        "player two": "Player 2",
        "team": "Team Name",
        "team name": "Team Name",
        "seed": "Seed",
        "notes": "Notes",
    }
    renamed = {}
    for col in raw_df.columns:
        key = " ".join(str(col).strip().lower().split())
        renamed[col] = aliases.get(key, str(col).strip())
    df = raw_df.rename(columns=renamed).copy()
    for col in ["Player 1", "Player 2", "Team Name", "Seed", "Notes"]:
        if col not in df.columns:
            df[col] = None
    return df[["Player 1", "Player 2", "Team Name", "Seed", "Notes"]]


def _insert_tournament_shell(supabase, payload: dict) -> None:
    try:
        supabase.table("tournaments").insert(payload).execute()
        return
    except Exception:
        pass

    fallback_payload = {
        "club_id": payload["club_id"],
        "name": payload["name"],
        "status": payload.get("status") or "DRAFT",
        "team_count": int(payload.get("team_count") or LEGACY_DEFAULT_TEAM_COUNT),
    }
    supabase.table("tournaments").insert(fallback_payload).execute()


def _list_draws(supabase, tournament_id: str) -> list[dict[str, Any]]:
    try:
        resp = (
            supabase.table("tournament_event_draws")
            .select("*")
            .eq("tournament_id", tournament_id)
            .order("created_at", desc=False)
            .execute()
        )
        return resp.data or []
    except Exception:
        return []


def _create_draw(supabase, payload: dict) -> None:
    supabase.table("tournament_event_draws").insert(payload).execute()


def _update_ops_field(supabase, tournament_id: str, draw_id: str | None, field: str, value: Any) -> None:
    if draw_id:
        try:
            supabase.table("tournament_event_draws").update({field: value}).eq("id", draw_id).execute()
            return
        except Exception:
            pass
    supabase.table("tournaments").update({field: value}).eq("id", tournament_id).execute()


def _get_draw_setting(draw: dict[str, Any] | None, tournament: dict[str, Any], field: str, default: Any) -> Any:
    if draw and draw.get(field) not in (None, "", "None"):
        return draw.get(field)
    if tournament.get(field) not in (None, "", "None"):
        return tournament.get(field)
    return default


def _effective_team_count(draw: dict[str, Any] | None, tournament: dict[str, Any], teams_by_number: dict[int, dict]) -> int:
    seeded = _get_draw_setting(draw, tournament, "team_count", None)
    try:
        value = int(seeded) if seeded is not None else None
    except Exception:
        value = None
    if value in SUPPORTED_TEAM_COUNTS:
        return value
    populated = max(teams_by_number.keys(), default=0)
    if populated in SUPPORTED_TEAM_COUNTS:
        return populated
    return LEGACY_DEFAULT_TEAM_COUNT


def _team_rows_for_editor(teams_by_number: dict[int, dict], id_to_name: dict[Any, str], slot_count: int) -> pd.DataFrame:
    max_slot = max(slot_count, max(teams_by_number.keys(), default=0))
    rows = []
    for slot in range(1, max_slot + 1):
        team = teams_by_number.get(slot) or {}
        rows.append(
            {
                "Team / Slot": slot,
                "Player 1": id_to_name.get(team.get("player1_id")),
                "Player 2": id_to_name.get(team.get("player2_id")),
                "Source": _safe_text(team.get("source") or "MANUAL"),
                "Notes": _safe_text(team.get("notes")),
            }
        )
    return pd.DataFrame(rows)


def _scoped_query(table, tournament_id: str, draw_id: str | None):
    query = table.eq("tournament_id", tournament_id)
    if draw_id:
        return query.eq("draw_id", draw_id)
    return query.is_("draw_id", "null")


def _teams_ready(teams_by_number: dict[int, dict], team_count: int, *, singles_division: bool) -> bool:
    if team_count <= 0 or len(teams_by_number) != team_count:
        return False
    for num in range(1, team_count + 1):
        team = teams_by_number.get(num)
        if not team:
            return False
        if not team.get("player1_id"):
            return False
        if not singles_division and not team.get("player2_id"):
            return False
    return True


def render(ctx):
    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell(
        "🏆 Tournaments",
        "Create tournament shells here, configure registration in Tournament Manager, then return here for division operations and scoring.",
        mode_label=mode_label,
    )

    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    supabase = getattr(ctx, "supabase", None)
    club_id = getattr(ctx, "club_id", None)
    df_players_all = getattr(ctx, "df_players_all", None)
    name_to_id = getattr(ctx, "name_to_id", {})
    id_to_name = getattr(ctx, "id_to_name", {})

    if supabase is None or club_id is None:
        st.error("Missing database context.")
        st.stop()

    if df_players_all is None or df_players_all.empty:
        st.warning("No players loaded.")
        st.stop()

    player_names = sorted(df_players_all["name"].dropna().astype(str).tolist())

    st.subheader("Create Tournament Shell")
    st.caption(
        "Stage 1: create the tournament shell and date window here. Stage 2: configure days, events, and divisions in Tournament Manager. Stage 3: return here to build division draws, import teams, and run scores."
    )
    c1, c2 = st.columns(2)
    with c1:
        tournament_name = st.text_input("Tournament name *", key="tourney_create_name")
        start_date = st.date_input("Start date", value=date.today(), key="tourney_create_start")
        end_date = st.date_input("End date", value=date.today(), min_value=start_date, key="tourney_create_end")
        registration_enabled = st.checkbox("Registration enabled", value=True, key="tourney_create_reg_enabled")
    with c2:
        status = st.selectbox("Status", TOURNAMENT_STATUS_OPTIONS, index=0, key="tourney_create_status")
        public_slug = st.text_input("Public slug (optional)", key="tourney_create_slug")
        locale = st.selectbox("Locale", TOURNAMENT_LOCALE_OPTIONS, index=0, key="tourney_create_locale")
        reg_open_at = st.text_input("Registration opens (optional ISO/local)", key="tourney_create_reg_open")
        reg_close_at = st.text_input("Registration closes (optional ISO/local)", key="tourney_create_reg_close")

    if st.button("Create Tournament", type="primary"):
        if not tournament_name.strip():
            st.error("Tournament name is required.")
        else:
            payload = {
                "club_id": str(club_id),
                "name": tournament_name.strip(),
                "status": status,
                "team_count": None,
                "start_date": _parse_optional_date(start_date),
                "end_date": _parse_optional_date(end_date),
                "registration_enabled": bool(registration_enabled),
                "public_slug": _safe_text(public_slug) or None,
                "locale": locale,
                "registration_open_at": _parse_optional_date(reg_open_at),
                "registration_close_at": _parse_optional_date(reg_close_at),
                "event_tags": normalize_event_tags({
                    "skill_levels": [],
                    "date_tags": derive_default_date_tags(start_date=start_date, end_date=end_date),
                }),
            }
            _insert_tournament_shell(supabase, payload)
            created = (
                supabase.table("tournaments")
                .select("id,name")
                .eq("club_id", str(club_id))
                .eq("name", tournament_name.strip())
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            created_row = (created.data or [None])[0]
            if created_row:
                reg_available, _ = registration_feature_available(supabase)
                if reg_available:
                    try:
                        upsert_registration_settings(
                            supabase,
                            {
                                "tournament_id": created_row["id"],
                                "registration_slug": _safe_text(public_slug) or None,
                                "locale": locale,
                                "registration_status": "open" if registration_enabled else "draft",
                                "registration_open_at": _parse_optional_date(reg_open_at),
                                "registration_close_at": _parse_optional_date(reg_close_at),
                            },
                        )
                    except Exception:
                        pass
            st.success("Tournament shell created.")
            st.info("Next step: open Tournament Manager and build the days → event families → divisions schedule.")
            if created_row:
                st.link_button("Configure Registration", f"?page=tournament_manager&tournament_id={created_row.get('id')}")
            st.rerun()

    st.divider()

    tournaments_resp = (
        supabase.table("tournaments")
        .select("*")
        .eq("club_id", str(club_id))
        .order("created_at", desc=True)
        .execute()
    )
    tournaments = tournaments_resp.data or []

    if not tournaments:
        st.info("No tournaments created yet.")
        st.stop()

    preselected = _safe_text(st.query_params.get("tournament_id"))
    tournament_labels = [f"{row['name']} ({row['status']})" for row in tournaments]
    default_index = 0
    if preselected:
        for idx, row in enumerate(tournaments):
            if str(row.get("id")) == preselected:
                default_index = idx
                break
    selected_label = st.selectbox("Select tournament", tournament_labels, index=default_index)
    tournament = tournaments[tournament_labels.index(selected_label)]
    tournament_id = tournament["id"]
    st.query_params["tournament_id"] = tournament_id

    available, _ = registration_feature_available(supabase)
    registration_bridge = None
    if available:
        try:
            reg_settings = get_registration_settings(supabase, tournament_id, tournament_name=_safe_text(tournament.get("name")))
            reg_days = list_registration_days(supabase, tournament_id)
            reg_events = list_registration_event_options(supabase, tournament_id)
            reg_state = build_registration_state(supabase, tournament, reg_settings, reg_days, reg_events)
            registration_bridge = {"settings": reg_settings, "days": reg_days, "events": reg_events, "state": reg_state}
        except Exception:
            registration_bridge = None

    st.subheader("Tournament Overview")
    summary_cols = st.columns(4)
    summary_cols[0].metric("Status", _safe_text(tournament.get("status") or "DRAFT"))
    summary_cols[1].metric("Start", _safe_text(tournament.get("start_date") or "—"))
    summary_cols[2].metric("End", _safe_text(tournament.get("end_date") or "—"))
    summary_cols[3].metric(
        "Configured divisions",
        len((registration_bridge or {}).get("events") or []),
    )
    date_window = _tournament_date_window_text(tournament)
    if date_window:
        st.caption(f"Tournament date window: {date_window}. Tournament Manager can auto-generate day rows from this range.")
    else:
        st.info("This tournament has no saved start/end dates yet. Add dates in Tournament Manager to auto-generate the default day schedule.")
    st.link_button("Open Tournament Manager", f"?page=tournament_manager&tournament_id={tournament_id}")

    draws = _list_draws(supabase, tournament_id)
    modern_mode = bool((registration_bridge or {}).get("events"))
    selected_draw: dict[str, Any] | None = None
    selected_draw_id: str | None = None
    selected_event_option_id: str | None = None
    selected_day_id: str | None = None
    selected_event_type = ""
    selected_event: dict[str, Any] | None = None

    st.subheader("Division Builder & Operations Bridge")
    if modern_mode:
        event_options = (registration_bridge or {}).get("events") or []
        day_map = {str(d.get("id")): d for d in ((registration_bridge or {}).get("days") or [])}
        event_labels = []
        for row in event_options:
            day = day_map.get(str(row.get("registration_day_id")), {})
            day_label = _safe_text(day.get("label") or "Day")
            family = _safe_text(row.get("event_family_label") or row.get("label") or "Event")
            division = _safe_text(row.get("division_name") or row.get("label") or "Division")
            event_labels.append(f"{day_label} • {family} • {division}")
        selected_event_label = st.selectbox("Select a scheduled division", event_labels, key=f"ops_event_{tournament_id}")
        selected_event = event_options[event_labels.index(selected_event_label)]
        selected_event_option_id = str(selected_event.get("id"))
        selected_day_id = str(selected_event.get("registration_day_id"))
        selected_event_type = _safe_text(selected_event.get("event_type"))

        info_cols = st.columns(3)
        info_cols[0].metric("Format", _safe_text(selected_event.get("event_format_override") or selected_event.get("event_format_default") or "—").replace("_", " "))
        info_cols[1].metric("Scoring", _safe_text(selected_event.get("scoring_override") or selected_event.get("scoring_default") or "—").replace("_", " "))
        info_cols[2].metric("Capacity", _safe_text(selected_event.get("capacity_teams") or "Open"))

        scoped_draws = [
            row
            for row in draws
            if str(row.get("event_option_id")) == selected_event_option_id and str(row.get("registration_day_id")) == selected_day_id
        ]
        if scoped_draws:
            draw_labels = [f"{_safe_text(row.get('name') or 'Draw')} ({_safe_text(row.get('status') or 'draft')})" for row in scoped_draws]
            chosen_draw = st.selectbox("Select division draw", draw_labels, key=f"ops_draw_{tournament_id}")
            selected_draw = scoped_draws[draw_labels.index(chosen_draw)]
            selected_draw_id = str(selected_draw.get("id"))

        action_cols = st.columns(3)
        if action_cols[0].button("Create draw for this division", key=f"create_draw_{tournament_id}"):
            _create_draw(
                supabase,
                {
                    "tournament_id": tournament_id,
                    "registration_day_id": selected_day_id,
                    "event_option_id": selected_event_option_id,
                    "name": f"{_safe_text(selected_event.get('division_name') or selected_event.get('label') or 'Division')} Ops Draw",
                    "status": "draft",
                },
            )
            st.success("Division draw created.")
            st.rerun()
        if action_cols[1].button("Configure registration", key=f"goto_reg_{tournament_id}"):
            st.query_params["page"] = "tournament_manager"
            st.query_params["tournament_id"] = tournament_id
            st.rerun()
        if action_cols[2].button("Refresh draw list", key=f"refresh_draws_{tournament_id}"):
            st.rerun()
    else:
        st.info("This tournament has not been configured in Tournament Manager yet. Create days, event families, and divisions there first.")

    _render_registration_bridge(tournament, registration_bridge)

    if modern_mode and not selected_draw_id:
        st.warning("Create or select a division draw before editing teams or generating brackets for this scheduled division.")

    team_query = _scoped_query(supabase.table("tournament_teams").select("*"), tournament_id, selected_draw_id)
    game_query = _scoped_query(supabase.table("tournament_games").select("*"), tournament_id, selected_draw_id)
    teams_resp = team_query.order("team_number").execute()
    teams = teams_resp.data or []
    teams_by_number = {int(row["team_number"]): row for row in teams if row.get("team_number") is not None}
    teams_by_id = {row["id"]: row for row in teams}

    games_resp = game_query.order("rr_round_number", desc=False).order("rr_slot_number", desc=False).execute()
    games = games_resp.data or []
    rr_games = [row for row in games if row.get("stage") == "ROUND_ROBIN"]
    playoff_games = [row for row in games if row.get("stage") == "PLAYOFF"]

    podium_resp = (
        supabase.table("tournament_podium")
        .select("*")
        .eq("tournament_id", tournament_id)
        .order("placement", desc=False)
        .execute()
    )
    podium_rows = podium_resp.data or []
    is_complete = tournament.get("status") == "COMPLETE"

    if modern_mode:
        st.caption("Operations below are scoped to the selected division draw whenever a draw is selected. Legacy tournaments without scheduled divisions still work tournament-wide.")

    ops_tabs = st.tabs(["Teams", "Round Robin", "Standings", "Playoffs", "Completion"])
    draw_team_count = _effective_team_count(selected_draw, tournament, teams_by_number)
    singles_division = _safe_text(selected_event_type).upper() == "SINGLES"

    with ops_tabs[0]:
        st.subheader("Teams")
        if modern_mode and not selected_draw_id:
            st.info("Create or select a division draw above to manage teams for this division.")
        else:
            st.caption(
                "Manual team editing and bulk import happen here. This is also where you move confirmed registrations into an operations-ready draw. Current scheduling engine supports these draw sizes: "
                + ", ".join(str(x) for x in SUPPORTED_TEAM_COUNTS)
                + "."
            )
            ops_cols = st.columns([2, 1, 1])
            with ops_cols[0]:
                roster = _find_event_roster(
                    registration_bridge,
                    event_option_id=_safe_text(selected_event_option_id),
                    registration_day_id=_safe_text(selected_day_id),
                ) if modern_mode else None
                confirmed_count = len([row for row in (roster or {}).get("entries", []) if _safe_text(row.get("status")) == "CONFIRMED"])
                st.metric("Confirmed registration entries", confirmed_count)
            with ops_cols[1]:
                draw_size = st.selectbox(
                    "Draw size / team slots",
                    SUPPORTED_TEAM_COUNTS,
                    index=SUPPORTED_TEAM_COUNTS.index(draw_team_count) if draw_team_count in SUPPORTED_TEAM_COUNTS else 0,
                    disabled=bool(games) or is_complete,
                    key=f"draw_size_{tournament_id}_{selected_draw_id or 'legacy'}",
                )
            with ops_cols[2]:
                if st.button("Save draw size", disabled=bool(games) or is_complete):
                    _update_ops_field(supabase, tournament_id, selected_draw_id, "team_count", int(draw_size))
                    st.success("Draw size updated.")
                    st.rerun()

            build_cols = st.columns(2)
            with build_cols[0]:
                import_mode = st.radio(
                    "Registration import mode",
                    ["Append", "Replace"],
                    horizontal=True,
                    key=f"reg_import_mode_{tournament_id}_{selected_draw_id or 'legacy'}",
                )
            with build_cols[1]:
                if modern_mode and selected_draw_id and st.button("Build draw from confirmed registrations"):
                    roster = _find_event_roster(
                        registration_bridge,
                        event_option_id=_safe_text(selected_event_option_id),
                        registration_day_id=_safe_text(selected_day_id),
                    ) or {}
                    roster_entries = roster.get("entries") or []
                    entries = [row for row in roster_entries if _safe_text(row.get("status")) == "CONFIRMED"]
                    if not entries:
                        unresolved_partner_count = sum(
                            1
                            for row in roster_entries
                            if _safe_text(row.get("status")).upper() in {"REVIEW", "NEEDS_PARTNER", "PARTNER_MISSING"}
                        )
                        if not roster_entries:
                            st.info("No registration entries exist for this division yet.")
                        elif unresolved_partner_count:
                            st.warning(
                                "No confirmed roster entries are ready yet because partner/roster issues still need admin review."
                            )
                            st.caption("Hint: Open Tournament Manager → Publish & QA → Registration admin review to resolve statuses.")
                        else:
                            st.info("No confirmed roster entries are ready for this division.")
                    else:
                        if import_mode == "Replace":
                            _scoped_query(supabase.table("tournament_teams").delete(), tournament_id, selected_draw_id).execute()
                        start_slot = max(teams_by_number.keys(), default=0) + 1 if import_mode == "Append" else 1
                        payload = []
                        unresolved_names = []
                        for offset, entry in enumerate(entries, start=0):
                            members = entry.get("members") or []
                            p1_name = _safe_text((members[0] if len(members) > 0 else {}).get("display_name"))
                            p2_name = _safe_text((members[1] if len(members) > 1 else {}).get("display_name"))
                            p1_id = _resolve_player_id(p1_name, name_to_id, id_to_name)
                            p2_id = _resolve_player_id(p2_name, name_to_id, id_to_name) if p2_name else None
                            if p1_name and not p1_id:
                                unresolved_names.append(p1_name)
                            if p2_name and not p2_id:
                                unresolved_names.append(p2_name)
                            payload.append(
                                {
                                    "tournament_id": tournament_id,
                                    "draw_id": selected_draw_id,
                                    "event_option_id": selected_event_option_id,
                                    "registration_day_id": selected_day_id,
                                    "team_number": start_slot + offset,
                                    "player1_id": p1_id,
                                    "player2_id": p2_id,
                                    "source": "REGISTRATION",
                                    "notes": None,
                                }
                            )
                        if unresolved_names:
                            st.error("Some confirmed registration names could not be matched to JUPR players: " + ", ".join(sorted(set(unresolved_names))))
                            st.caption("Hint: fix display names or player mappings first, then rerun this import.")
                        else:
                            supabase.table("tournament_teams").upsert(payload, on_conflict="tournament_id,draw_id,team_number").execute()
                            total_slots = max(draw_size, start_slot + len(payload) - 1)
                            if total_slots in SUPPORTED_TEAM_COUNTS:
                                _update_ops_field(supabase, tournament_id, selected_draw_id, "team_count", total_slots)
                            st.success(f"Imported {len(payload)} confirmed entries into the division draw.")
                            st.rerun()

            editor_df = st.data_editor(
                _team_rows_for_editor(teams_by_number, id_to_name, int(draw_size)),
                use_container_width=True,
                hide_index=True,
                key=f"tourney_team_editor_{tournament_id}_{selected_draw_id or 'legacy'}",
                disabled=is_complete,
                column_config={
                    "Team / Slot": st.column_config.NumberColumn("Team / Slot", step=1, min_value=1),
                    "Player 1": st.column_config.SelectboxColumn("Player 1", options=player_names, required=True),
                    "Player 2": st.column_config.SelectboxColumn("Player 2", options=player_names, required=not singles_division),
                    "Source": st.column_config.SelectboxColumn("Source", options=IMPORT_SOURCE_OPTIONS),
                    "Notes": st.column_config.TextColumn("Notes"),
                },
            )

            save_col, import_col = st.columns(2)
            with save_col:
                if st.button("Save teams", disabled=is_complete):
                    seen_slots: set[int] = set()
                    selected_ids: list[Any] = []
                    payload = []
                    for _, row in editor_df.iterrows():
                        slot = _safe_text(row.get("Team / Slot"))
                        if not slot:
                            continue
                        try:
                            slot_num = int(float(slot))
                        except Exception:
                            st.error("Each row needs a numeric Team / Slot value.")
                            st.stop()
                        if slot_num in seen_slots:
                            st.error(f"Duplicate slot detected: {slot_num}.")
                            st.stop()
                        seen_slots.add(slot_num)
                        p1 = _safe_text(row.get("Player 1"))
                        p2 = _safe_text(row.get("Player 2"))
                        if not p1 and not p2:
                            continue
                        if singles_division and not p1:
                            st.error("Singles divisions require Player 1.")
                            st.stop()
                        if not singles_division and (not p1 or not p2):
                            st.error("Doubles divisions require both Player 1 and Player 2.")
                            st.stop()
                        if p1 and p2 and p1 == p2:
                            st.error("A team cannot use the same player twice.")
                            st.stop()
                        p1_id = _resolve_player_id(p1, name_to_id, id_to_name)
                        p2_id = _resolve_player_id(p2, name_to_id, id_to_name) if p2 else None
                        if p1_id:
                            selected_ids.append(p1_id)
                        if p2_id:
                            selected_ids.append(p2_id)
                        payload.append(
                            {
                                "tournament_id": tournament_id,
                                "draw_id": selected_draw_id,
                                "event_option_id": selected_event_option_id,
                                "registration_day_id": selected_day_id,
                                "team_number": slot_num,
                                "player1_id": p1_id,
                                "player2_id": p2_id,
                                "source": _safe_text(row.get("Source") or "MANUAL"),
                                "notes": _safe_text(row.get("Notes")) or None,
                            }
                        )

                    duplicates = {pid for pid in selected_ids if pid is not None and selected_ids.count(pid) > 1}
                    if duplicates:
                        names = ", ".join(id_to_name.get(pid, str(pid)) for pid in duplicates)
                        st.error(f"Duplicate players detected in this draw: {names}.")
                        st.stop()
                    _scoped_query(supabase.table("tournament_teams").delete(), tournament_id, selected_draw_id).execute()
                    if payload:
                        supabase.table("tournament_teams").upsert(payload, on_conflict="tournament_id,draw_id,team_number").execute()
                    st.success("Teams saved.")
                    st.rerun()

            with import_col:
                with st.expander("Bulk upload teams"):
                    upload_mode = st.radio(
                        "Bulk upload save mode",
                        ["Append", "Replace"],
                        horizontal=True,
                        key=f"bulk_mode_{tournament_id}_{selected_draw_id or 'legacy'}",
                    )
                    uploaded = st.file_uploader("CSV or XLSX", type=["csv", "xlsx"], key=f"bulk_file_{tournament_id}_{selected_draw_id or 'legacy'}")
                    pasted = st.text_area("Or paste CSV/TSV", key=f"bulk_text_{tournament_id}_{selected_draw_id or 'legacy'}")
                    if st.button("Preview and save bulk upload", key=f"bulk_preview_{tournament_id}_{selected_draw_id or 'legacy'}"):
                        try:
                            parsed = _canonical_import_df(_parse_bulk_upload(uploaded, pasted))
                        except Exception as exc:
                            st.error(f"Could not parse upload: {exc}")
                            st.stop()
                        if parsed.empty:
                            st.warning("No rows found.")
                            st.stop()
                        parsed = parsed.fillna("")
                        unresolved = []
                        duplicate_names = []
                        id_counts: dict[Any, int] = {}
                        preview_rows = []
                        for _, row in parsed.iterrows():
                            p1_name = _safe_text(row.get("Player 1"))
                            p2_name = _safe_text(row.get("Player 2"))
                            p1_id = _resolve_player_id(p1_name, name_to_id, id_to_name)
                            p2_id = _resolve_player_id(p2_name, name_to_id, id_to_name) if p2_name else None
                            if not p1_id:
                                unresolved.append(p1_name)
                            if p2_name and not p2_id:
                                unresolved.append(p2_name)
                            for pid in [p1_id, p2_id]:
                                if pid:
                                    id_counts[pid] = id_counts.get(pid, 0) + 1
                            preview_rows.append(
                                {
                                    "Player 1": p1_name,
                                    "Player 2": p2_name,
                                    "Player 1 Resolved": bool(p1_id),
                                    "Player 2 Resolved": True if not p2_name else bool(p2_id),
                                    "Seed": row.get("Seed"),
                                    "Notes": _safe_text(row.get("Notes")),
                                }
                            )
                        duplicate_names = [id_to_name.get(pid, str(pid)) for pid, count in id_counts.items() if count > 1]
                        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)
                        if unresolved:
                            st.error("Unresolved player names: " + ", ".join(sorted({name for name in unresolved if name})))
                        if duplicate_names:
                            st.error("Duplicate players in upload: " + ", ".join(sorted(set(duplicate_names))))
                        if unresolved or duplicate_names:
                            st.stop()

                        if upload_mode == "Replace":
                            _scoped_query(supabase.table("tournament_teams").delete(), tournament_id, selected_draw_id).execute()
                            start_slot = 1
                        else:
                            start_slot = max(teams_by_number.keys(), default=0) + 1

                        payload = []
                        for offset, row in enumerate(preview_rows, start=0):
                            payload.append(
                                {
                                    "tournament_id": tournament_id,
                                    "draw_id": selected_draw_id,
                                    "event_option_id": selected_event_option_id,
                                    "registration_day_id": selected_day_id,
                                    "team_number": start_slot + offset,
                                    "player1_id": _resolve_player_id(_safe_text(row.get("Player 1")), name_to_id, id_to_name),
                                    "player2_id": _resolve_player_id(_safe_text(row.get("Player 2")), name_to_id, id_to_name) if _safe_text(row.get("Player 2")) else None,
                                    "seed": int(float(row.get("Seed"))) if _safe_text(row.get("Seed")).replace(".", "", 1).isdigit() else None,
                                    "source": "BULK_UPLOAD",
                                    "notes": _safe_text(row.get("Notes")) or None,
                                }
                            )
                        supabase.table("tournament_teams").upsert(payload, on_conflict="tournament_id,draw_id,team_number").execute()
                        final_slot = start_slot + len(payload) - 1
                        if final_slot in SUPPORTED_TEAM_COUNTS:
                            _update_ops_field(supabase, tournament_id, selected_draw_id, "team_count", final_slot)
                        st.success("Bulk upload saved.")
                        st.rerun()

    with ops_tabs[1]:
        st.subheader("Round Robin")
        if modern_mode and not selected_draw_id:
            st.info("Select a division draw first.")
        else:
            ready_teams = _teams_ready(teams_by_number, int(draw_size), singles_division=singles_division)
            if not rr_games:
                st.info("No round robin games created yet.")
            if not ready_teams:
                need_text = "one player" if singles_division else "two players"
                st.warning(f"Fill exactly {draw_size} slots with {need_text} per slot to generate the round robin schedule.")
            if st.button("Generate round robin schedule", disabled=bool(rr_games) or not ready_teams or is_complete):
                team_ids = {int(num): row["id"] for num, row in teams_by_number.items()}
                games_payload = build_round_robin_games(tournament_id=tournament_id, team_ids_by_number=team_ids)
                for row in games_payload:
                    row["draw_id"] = selected_draw_id
                    row["event_option_id"] = selected_event_option_id
                    row["registration_day_id"] = selected_day_id
                supabase.table("tournament_games").insert(games_payload).execute()
                supabase.table("tournaments").update({"status": "ROUND_ROBIN"}).eq("id", tournament_id).execute()
                st.success("Round robin schedule generated.")
                st.rerun()

            if rr_games:
                with st.expander("Regenerate schedule"):
                    st.warning("This deletes the existing round robin and playoff games for the active scope.")
                    confirm = st.text_input("Type RESET to confirm", key=f"rr_reset_confirm_{tournament_id}_{selected_draw_id or 'legacy'}")
                    if st.button("Regenerate round robin schedule", disabled=confirm.strip().upper() != "RESET" or is_complete):
                        _scoped_query(supabase.table("tournament_games").delete(), tournament_id, selected_draw_id).execute()
                        _scoped_query(supabase.table("tournament_teams").update({"seed": None}), tournament_id, selected_draw_id).execute()
                        supabase.table("tournaments").update({"status": "DRAFT", "playoff_advance_count": None}).eq("id", tournament_id).execute()
                        st.success("Schedule cleared.")
                        st.rerun()

            if rr_games:
                _render_games_table(
                    games=rr_games,
                    teams_by_id=teams_by_id,
                    id_to_name=id_to_name,
                    on_save=lambda updates: _save_games(
                        ctx,
                        {**tournament, "active_draw_id": selected_draw_id},
                        teams_by_id,
                        updates,
                        stage="ROUND_ROBIN",
                    ),
                    key_prefix="rr",
                    disabled=is_complete,
                )

    with ops_tabs[2]:
        st.subheader("Standings")
        if modern_mode and not selected_draw_id:
            st.info("Select a division draw first.")
        elif not rr_games:
            st.info("Generate the round robin schedule and enter scores to see standings.")
        else:
            standings = compute_round_robin_standings(list(teams_by_id.values()), rr_games)
            standings_df = pd.DataFrame(standings)
            if not standings_df.empty:
                display_df = standings_df.copy()
                display_df["Team"] = display_df["team_number"]
                display_df["Wins"] = display_df["wins"]
                display_df["Losses"] = display_df["losses"]
                display_df["PF"] = display_df["points_for"]
                display_df["PA"] = display_df["points_against"]
                display_df["Diff"] = display_df["differential"]
                display_df["Seed"] = display_df["seed"]
                st.dataframe(
                    display_df[["Team", "Wins", "Losses", "PF", "PA", "Diff", "Seed"]],
                    use_container_width=True,
                    hide_index=True,
                )
                if st.button("Update seeds", disabled=is_complete):
                    for row in standings:
                        supabase.table("tournament_teams").update({"seed": int(row["seed"])}).eq("id", row["team_id"]).execute()
                    st.success("Seeds updated.")
                    st.rerun()

    with ops_tabs[3]:
        st.subheader("Playoffs")
        if modern_mode and not selected_draw_id:
            st.info("Select a division draw first.")
        elif not rr_games:
            st.info("Generate round robin games first.")
        else:
            advance_options = [opt for opt in [4, 5, 6] if opt <= int(draw_size)]
            current_advance = _get_draw_setting(selected_draw, tournament, "playoff_advance_count", advance_options[0])
            selected_advance = st.selectbox(
                "Teams advancing",
                advance_options,
                index=advance_options.index(int(current_advance)) if int(current_advance) in advance_options else 0,
                key=f"playoff_advance_select_{tournament_id}_{selected_draw_id or 'legacy'}",
                disabled=is_complete,
            )
            if st.button("Save advance count", disabled=is_complete):
                _update_ops_field(supabase, tournament_id, selected_draw_id, "playoff_advance_count", int(selected_advance))
                st.success("Advance count saved.")
                st.rerun()

            if not playoff_games:
                st.info("No playoff bracket generated yet.")
            if st.button("Generate playoff bracket", disabled=bool(playoff_games) or is_complete):
                standings = compute_round_robin_standings(list(teams_by_id.values()), rr_games)
                if len(standings) < int(selected_advance):
                    st.error("Not enough seeded teams to generate the bracket.")
                    st.stop()
                for row in standings:
                    supabase.table("tournament_teams").update({"seed": int(row["seed"])}).eq("id", row["team_id"]).execute()
                games_payload = build_playoff_games(tournament_id=tournament_id, advance_count=int(selected_advance), standings=standings)
                for row in games_payload:
                    row["draw_id"] = selected_draw_id
                    row["event_option_id"] = selected_event_option_id
                    row["registration_day_id"] = selected_day_id
                supabase.table("tournament_games").insert(games_payload).execute()
                supabase.table("tournaments").update({"status": "PLAYOFFS", "playoff_advance_count": int(selected_advance)}).eq("id", tournament_id).execute()
                st.success("Playoff bracket generated.")
                st.rerun()

            if playoff_games:
                _render_playoff_bracket(
                    games=playoff_games,
                    teams_by_id=teams_by_id,
                    id_to_name=id_to_name,
                    on_save=lambda updates: _save_games(
                        ctx,
                        {**tournament, "active_draw_id": selected_draw_id},
                        teams_by_id,
                        updates,
                        stage="PLAYOFF",
                    ),
                    disabled=is_complete,
                )

    with ops_tabs[4]:
        st.subheader("Completion")
        if modern_mode:
            st.caption("Completion still uses the legacy tournament-wide podium tables unless draw-scoped podium storage is added later.")
        if is_complete:
            st.success("Tournament is complete. Editing is locked.")
            _render_podium_read_only(podium_rows, teams_by_id, id_to_name)
        else:
            if st.button("🏁 Complete tournament"):
                st.session_state[f"podium_review_open_{tournament_id}"] = True
            if st.session_state.get(f"podium_review_open_{tournament_id}"):
                _render_podium_review(
                    ctx,
                    tournament,
                    teams_by_id=teams_by_id,
                    id_to_name=id_to_name,
                    rr_games=rr_games,
                    playoff_games=playoff_games,
                )


def _render_registration_bridge(tournament: dict[str, Any], registration_bridge: dict[str, Any] | None) -> None:
    st.subheader("Registration Bridge")
    if not registration_bridge:
        st.warning("Registration has not been configured for this tournament yet.")
        st.info("Use Tournament Manager to define days, event families, divisions, and public registration links.")
        st.link_button("Configure Registration", f"?page=tournament_manager&tournament_id={tournament.get('id')}")
        return

    settings = registration_bridge.get("settings") or {}
    state = registration_bridge.get("state") or {}
    summary = state.get("summary") or {}
    links = build_public_urls(
        base_url=_safe_text(st.session_state.get("base_url")),
        tournament_id=str(tournament.get("id")),
        registration_slug=settings.get("registration_slug"),
    )

    cols = st.columns(4)
    cols[0].metric("Registrations", summary.get("total_registrations", 0))
    cols[1].metric("Confirmed", summary.get("confirmed_entries", 0))
    cols[2].metric("Needs partner", summary.get("needs_partner_entries", 0))
    cols[3].metric("Issues", summary.get("issue_count", 0))

    st.caption("Use Tournament Manager to set up registration. Use this page to build operations-ready division draws from those confirmed registrations.")
    with st.expander("Registration links"):
        st.text_input("Tournament Manager", value=links["admin_manager"], key=f"ops_reg_admin_{tournament.get('id')}")
        st.text_input("Public registration", value=links["registration"], key=f"ops_reg_public_{tournament.get('id')}")
        st.text_input("Public partner board", value=links["partner_board"], key=f"ops_reg_partner_{tournament.get('id')}")


def _render_games_table(*, games, teams_by_id, id_to_name, on_save, key_prefix: str, disabled: bool = False):
    rounds = sorted({int(game.get("rr_round_number", 0)) for game in games})
    for round_num in rounds:
        st.markdown(f"#### Round {round_num}")
        round_games = [game for game in games if int(game.get("rr_round_number", 0)) == round_num]
        scores = {}
        with st.form(key=f"{key_prefix}_round_{round_num}"):
            for game in round_games:
                team_a = teams_by_id.get(game.get("team_a_id"), {})
                team_b = teams_by_id.get(game.get("team_b_id"), {})
                label_a = _team_label(team_a, id_to_name)
                label_b = _team_label(team_b, id_to_name)
                slot = game.get("rr_slot_number")
                c1, c2, c3, c4 = st.columns([4, 1, 1, 2])
                c1.write(f"Game {slot}: {label_a} vs {label_b}")
                c2.number_input("Score A", min_value=0, value=int(game.get("score_a") or 0), key=f"{key_prefix}_a_{game['id']}", disabled=disabled)
                c3.number_input("Score B", min_value=0, value=int(game.get("score_b") or 0), key=f"{key_prefix}_b_{game['id']}", disabled=disabled)
                c4.caption("Final" if game.get("finalized_at") else "Open")
                scores[game["id"]] = game
            if st.form_submit_button("Save scores", disabled=disabled):
                on_save(scores)


def _render_playoff_bracket(*, games, teams_by_id, id_to_name, on_save, disabled: bool = False):
    round_order = ["QF", "SF", "Final", "Bronze"]
    for round_name in round_order:
        round_games = [game for game in games if game.get("playoff_round") == round_name]
        if not round_games:
            continue
        st.markdown(f"#### {round_name}")
        with st.form(key=f"playoff_{round_name}"):
            for game in round_games:
                team_a = teams_by_id.get(game.get("team_a_id"), {})
                team_b = teams_by_id.get(game.get("team_b_id"), {})
                label_a = _team_label(team_a, id_to_name)
                label_b = _team_label(team_b, id_to_name)
                c1, c2, c3, c4 = st.columns([4, 1, 1, 2])
                c1.write(f"{game.get('playoff_game_code')}: {label_a} vs {label_b}")
                c2.number_input("Score A", min_value=0, value=int(game.get("score_a") or 0), key=f"playoff_a_{game['id']}", disabled=disabled or not game.get("team_a_id") or not game.get("team_b_id"))
                c3.number_input("Score B", min_value=0, value=int(game.get("score_b") or 0), key=f"playoff_b_{game['id']}", disabled=disabled or not game.get("team_a_id") or not game.get("team_b_id"))
                c4.caption("Final" if game.get("finalized_at") else "Open")
            if st.form_submit_button("Save scores", disabled=disabled):
                on_save({game["id"]: game for game in round_games})


def _render_podium_review(
    ctx,
    tournament: dict[str, Any],
    *,
    teams_by_id: dict[str, dict],
    id_to_name: dict[Any, str],
    rr_games: list[dict],
    playoff_games: list[dict],
) -> None:
    tournament_id = tournament["id"]
    tournament_name = tournament.get("name", "Tournament")
    team_count = len(teams_by_id)
    max_places = min(3, team_count)

    rr_podium = compute_podium_from_rr(list(teams_by_id.values()), rr_games, max_placements=max_places) if rr_games else []
    playoff_podium = compute_podium_from_playoffs(playoff_games)
    default_mode = "PLAYOFF_RESULTS" if playoff_podium else "ROUND_ROBIN_STANDINGS"
    mode_labels = {"PLAYOFF_RESULTS": "Use Playoff Results", "ROUND_ROBIN_STANDINGS": "Use RR Standings", "MANUAL": "Manual"}
    mode_options = ["PLAYOFF_RESULTS", "ROUND_ROBIN_STANDINGS", "MANUAL"]
    mode = st.radio(
        "Podium review",
        mode_options,
        index=mode_options.index(default_mode),
        format_func=lambda value: mode_labels[value],
        key=f"podium_review_mode_{tournament_id}",
    )

    placements: list[dict[str, str | int]] = []
    source = "MANUAL"
    if mode == "PLAYOFF_RESULTS":
        source = "PLAYOFF"
        if not playoff_podium:
            st.error("Playoff results are not finalized yet.")
        else:
            placements = playoff_podium
            _render_podium_preview(placements, teams_by_id, id_to_name)
    elif mode == "ROUND_ROBIN_STANDINGS":
        source = "ROUND_ROBIN"
        if not rr_games:
            st.error("Round robin games are required to use standings.")
        else:
            if _has_unscored_rr_games(rr_games):
                st.warning("Some round robin games are unscored; podium reflects currently entered results.")
            placements = rr_podium
            _render_podium_preview(placements, teams_by_id, id_to_name)
    else:
        options = list(teams_by_id.keys())
        for placement in range(1, max_places + 1):
            selection = st.selectbox(
                f"{_placement_label(placement)}",
                options=[None, *options],
                format_func=lambda value: _team_label(teams_by_id.get(value, {}), id_to_name) if value else "Select team",
                key=f"podium_manual_{tournament_id}_{placement}",
            )
            if selection:
                placements.append({"placement": placement, "team_id": selection})

    if st.button("Finalize tournament", type="primary", disabled=max_places == 0):
        if mode == "MANUAL" and len(placements) < max_places:
            st.error("Select a team for each podium placement.")
            return
        try:
            payload = build_podium_payload(tournament_id, placements, source)
        except ValueError as exc:
            st.error(str(exc))
            return
        if not payload:
            st.error("Podium placements are required to complete the tournament.")
            return
        upsert_tournament_podium(ctx.supabase, tournament_id, payload)
        award_tournament_trophies_from_podium(ctx, tournament_id, tournament_name)
        ctx.supabase.table("tournaments").update({"status": "COMPLETE"}).eq("id", tournament_id).execute()
        st.success("Tournament completed and podium locked.")
        st.session_state.pop(f"podium_review_open_{tournament_id}", None)
        st.rerun()


def _render_podium_read_only(podium_rows: list[dict], teams_by_id: dict[str, dict], id_to_name: dict[Any, str]) -> None:
    if not podium_rows:
        st.info("No podium recorded yet.")
        return
    st.markdown("#### 🏆 Podium")
    _render_podium_preview(podium_rows, teams_by_id, id_to_name)


def _render_podium_preview(placements: list[dict], teams_by_id: dict[str, dict], id_to_name: dict[Any, str]) -> None:
    rows = []
    for row in sorted(placements, key=lambda item: int(item.get("placement", 0) or 0)):
        team = teams_by_id.get(row.get("team_id"), {})
        rows.append({"Placement": _placement_label(int(row.get("placement", 0) or 0)), "Team": _team_label(team, id_to_name)})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _placement_label(placement: int) -> str:
    medals = {1: "🥇 Champion", 2: "🥈 Runner-up", 3: "🥉 Bronze"}
    return medals.get(int(placement), f"Placement {placement}")


def _has_unscored_rr_games(games: list[dict]) -> bool:
    for game in games:
        score_a = game.get("score_a")
        score_b = game.get("score_b")
        if score_a is None or score_b is None:
            return True
        try:
            if int(score_a) == 0 and int(score_b) == 0:
                return True
        except Exception:
            return True
    return False


def _team_label(team: dict, id_to_name: dict[Any, str]) -> str:
    if not team:
        return "TBD"
    p1 = id_to_name.get(team.get("player1_id"), "?")
    p2 = id_to_name.get(team.get("player2_id"))
    if p2:
        return f"Team {team.get('team_number')}: {p1} / {p2}"
    return f"Team {team.get('team_number')}: {p1}"


def _save_games(ctx, tournament: dict[str, Any], teams_by_id: dict[str, dict], game_map: dict[str, dict], stage: str):
    supabase = ctx.supabase
    if tournament.get("status") == "COMPLETE":
        st.error("Tournament is complete. Scores are locked.")
        return
    updated_any = False
    for game_id, game in game_map.items():
        if game.get("finalized_at"):
            continue
        score_a = int(st.session_state.get(f"{_score_key(stage, 'a', game_id)}", 0))
        score_b = int(st.session_state.get(f"{_score_key(stage, 'b', game_id)}", 0))
        if score_a == 0 and score_b == 0:
            if game.get("score_a") or game.get("score_b"):
                supabase.table("tournament_games").update({"score_a": None, "score_b": None}).eq("id", game_id).execute()
                updated_any = True
            continue
        supabase.table("tournament_games").update({"score_a": score_a, "score_b": score_b}).eq("id", game_id).execute()
        updated_any = True
        try:
            finalize_payload = finalize_game({**game, "score_a": score_a, "score_b": score_b})
        except ValueError:
            continue
        match_payload = build_tournament_match_payload(tournament, game, teams_by_id, score_a=score_a, score_b=score_b)
        process_matches(
            [match_payload],
            supabase=supabase,
            club_id=str(ctx.club_id),
            name_to_id=ctx.name_to_id,
            df_players_all=ctx.df_players_all,
            df_leagues=ctx.df_leagues,
            df_meta=ctx.df_meta,
        )
        supabase.table("tournament_games").update(finalize_payload).eq("id", game_id).execute()

        if stage == "PLAYOFF":
            playoff_query = supabase.table("tournament_games").select("*").eq("tournament_id", tournament["id"]).eq("stage", "PLAYOFF")
            if tournament.get("active_draw_id"):
                playoff_query = playoff_query.eq("draw_id", tournament.get("active_draw_id"))
            else:
                playoff_query = playoff_query.is_("draw_id", "null")
            playoff_games = (playoff_query.execute().data or [])
            for update in resolve_playoff_dependencies(playoff_games):
                supabase.table("tournament_games").update(update).eq("id", update["id"]).execute()

    if updated_any:
        st.success("Scores saved.")
        st.session_state["force_data_refresh"] = True
        st.rerun()


def _score_key(stage: str, side: str, game_id: str) -> str:
    prefix = "playoff" if stage == "PLAYOFF" else "rr"
    return f"{prefix}_{side}_{game_id}"
