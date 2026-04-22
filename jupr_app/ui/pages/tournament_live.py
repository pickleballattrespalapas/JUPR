from __future__ import annotations

from datetime import date, datetime
from io import StringIO
from typing import Any

import pandas as pd
import streamlit as st

from jupr_app.domain.event_tags import derive_default_date_tags, normalize_event_tags
from jupr_app.domain.match_processing import process_matches
from jupr_app.domain.player_ops import safe_add_player
from jupr_app.domain.tournament_match_payload import build_tournament_match_payload
from jupr_app.domain.tournament_results_import import (
    STAGE_OPTIONS,
    build_draw_import_payload,
    parse_dupr_results_csv,
    suggest_player_matches,
)
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
    archive_tournament,
    build_public_urls,
    build_registration_state,
    delete_unused_draft_tournament,
    get_registration_settings,
    list_event_options as list_registration_event_options,
    list_existing_tournaments,
    list_registration_days,
    registration_feature_available,
    tournament_can_be_deleted,
    unarchive_tournament,
    upsert_registration_settings,
)
from jupr_app.ui.layout import page_shell

LEGACY_DEFAULT_TEAM_COUNT = 4
TOURNAMENT_STATUS_OPTIONS = ["DRAFT", "REGISTRATION", "REGISTRATION_OPEN", "REGISTRATION_CLOSED", "ARCHIVED"]
TOURNAMENT_LOCALE_OPTIONS = ["en", "es", "bilingual"]
IMPORT_SOURCE_OPTIONS = ["REGISTRATION", "MANUAL", "BULK_UPLOAD"]


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_name(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _coerce_date(value: Any, fallback: date) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return fallback


def _parse_optional_date(value: Any) -> str | None:
    if isinstance(value, date):
        return value.isoformat()
    text = _safe_text(value)
    return text or None


def _go_to_tournament_manager(tournament_id: Any) -> None:
    st.query_params["page"] = "tournament_manager"
    current_tournament_id = _safe_text(st.query_params.get("tournament_id"))
    if current_tournament_id != str(tournament_id):
        st.query_params["tournament_id"] = tournament_id
    st.rerun()




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
        "🔴 Tournament Live",
        "Run live scoring, standings, playoffs, and completion for tournament draws.",
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

    st.subheader("Tournament List")
    show_archived = st.checkbox("Show archived", value=False, key="tournaments_show_archived")
    st.caption("Archived tournaments are hidden from default selectors and public registration.")

    tournaments = list_existing_tournaments(supabase, str(club_id), include_archived=show_archived)

    if not tournaments:
        st.info("No tournaments available for this filter.")
        st.stop()

    preselected = _safe_text(st.query_params.get("tournament_id"))
    tournament_labels = []
    for row in tournaments:
        status = _safe_text(row.get("status") or "DRAFT")
        status_label = "ARCHIVED" if status.upper() == "ARCHIVED" else status
        tournament_labels.append(f"{row['name']} ({status_label})")
    default_index = 0
    if preselected:
        for idx, row in enumerate(tournaments):
            if str(row.get("id")) == preselected:
                default_index = idx
                break
    selected_label = st.selectbox("Select tournament", tournament_labels, index=default_index)
    tournament = tournaments[tournament_labels.index(selected_label)]
    tournament_id = tournament["id"]
    current_tournament_id = _safe_text(st.query_params.get("tournament_id"))
    if current_tournament_id != str(tournament_id):
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
    if st.button("Open Tournament Setup", key=f"open_tournament_manager_{tournament_id}"):
        _go_to_tournament_manager(tournament_id)

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
            current_tournament_id = _safe_text(st.query_params.get("tournament_id"))
            if current_tournament_id != str(tournament_id):
                st.query_params["tournament_id"] = tournament_id
            st.rerun()
        if action_cols[2].button("Refresh draw list", key=f"refresh_draws_{tournament_id}"):
            st.rerun()
    else:
        st.info("This tournament has not been configured in Tournament Manager yet. Create days, events, and divisions there first.")

    if modern_mode and not selected_draw_id:
        st.warning("Create or select a division draw before live scoring. Open Tournament Operations to create the draw and teams.")

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

    ops_tabs = st.tabs(["Round Robin", "Standings", "Playoffs", "Completion"])
    draw_team_count = _effective_team_count(selected_draw, tournament, teams_by_number)
    draw_size = draw_team_count
    singles_division = _safe_text(selected_event_type).upper() == "SINGLES"

    with ops_tabs[0]:
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

    with ops_tabs[1]:
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

    with ops_tabs[2]:
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

    with ops_tabs[3]:
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
