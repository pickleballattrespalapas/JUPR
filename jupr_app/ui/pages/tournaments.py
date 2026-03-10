from __future__ import annotations

from io import StringIO
import pandas as pd
import streamlit as st

from jupr_app.domain.match_processing import process_matches
from jupr_app.domain.tournament_match_payload import build_tournament_match_payload
from jupr_app.domain.tournaments import (
    build_playoff_games,
    SUPPORTED_TEAM_COUNTS,
    build_round_robin_games,
    build_podium_payload,
    compute_podium_from_playoffs,
    compute_podium_from_rr,
    compute_round_robin_standings,
    finalize_game,
    resolve_playoff_dependencies,
)
from jupr_app.domain.tournament_podium import award_tournament_trophies_from_podium, upsert_tournament_podium
from jupr_app.ui.layout import page_shell
from jupr_app.domain.tournament_registration_repo import (
    build_public_urls,
    build_registration_state,
    get_registration_settings,
    list_event_options as list_registration_event_options,
    list_registration_days,
    registration_feature_available,
    upsert_registration_settings,
)


LEGACY_DEFAULT_TEAM_COUNT = 4
TOURNAMENT_STATUS_OPTIONS = ["DRAFT", "REGISTRATION", "REGISTRATION_OPEN", "REGISTRATION_CLOSED"]
TOURNAMENT_LOCALE_OPTIONS = ["en", "es", "bilingual"]


def _normalize_name(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


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
    text = str(pasted_text or "").strip()
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


def _parse_optional_date(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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
        # team_count is required by the legacy tournament bracket schema.
        "team_count": int(payload.get("team_count") or LEGACY_DEFAULT_TEAM_COUNT),
    }
    supabase.table("tournaments").insert(fallback_payload).execute()


def render(ctx):
    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell("🏆 Tournaments", "Admin-only tournament manager.", mode_label=mode_label)

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
    st.caption("Stage 1 of 3: Create a tournament shell here. Configure registration details in Tournament Manager, then run brackets and scoring in Tournament Operations.")
    c1, c2 = st.columns(2)
    with c1:
        tournament_name = st.text_input("Tournament name *", key="tourney_create_name")
        start_date = st.text_input("Start date (recommended, YYYY-MM-DD)", key="tourney_create_start")
        end_date = st.text_input("End date (recommended, YYYY-MM-DD)", key="tourney_create_end")
        registration_enabled = st.checkbox("Registration enabled", value=True, key="tourney_create_reg_enabled")
    with c2:
        status = st.selectbox("Status", TOURNAMENT_STATUS_OPTIONS, index=0, key="tourney_create_status")
        public_slug = st.text_input("Public slug (optional)", key="tourney_create_slug")
        locale = st.selectbox("Locale", TOURNAMENT_LOCALE_OPTIONS, index=0, key="tourney_create_locale")
        reg_open_at = st.text_input("Registration open at (optional ISO)", key="tourney_create_reg_open")
        reg_close_at = st.text_input("Registration close at (optional ISO)", key="tourney_create_reg_close")

    if st.button("Create Tournament", type="primary"):
        if not tournament_name.strip():
            st.error("Tournament name is required.")
        else:
            payload = {
                "club_id": str(club_id),
                "name": tournament_name.strip(),
                "status": status,
                # Legacy tournament ops compatibility: bracket engine still expects this.
                "team_count": LEGACY_DEFAULT_TEAM_COUNT,
                "start_date": _parse_optional_date(start_date),
                "end_date": _parse_optional_date(end_date),
                "registration_enabled": bool(registration_enabled),
                "public_slug": str(public_slug or "").strip() or None,
                "locale": locale,
                "registration_open_at": _parse_optional_date(reg_open_at),
                "registration_close_at": _parse_optional_date(reg_close_at),
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
                                "registration_slug": str(public_slug or "").strip() or None,
                                "locale": locale,
                                "registration_status": "open" if registration_enabled else "draft",
                                "registration_open_at": _parse_optional_date(reg_open_at),
                                "registration_close_at": _parse_optional_date(reg_close_at),
                            },
                        )
                    except Exception:
                        pass
            st.success("Tournament shell created.")
            st.info("Next step: open Tournament Manager and configure registration days/events.")
            st.link_button("Configure Registration", f"?page=tournament_manager&tournament_id={(created_row or {}).get('id','')}")
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

    tournament_labels = [f"{t['name']} ({t['status']})" for t in tournaments]
    selected_label = st.selectbox("Select tournament", tournament_labels)
    selected_idx = tournament_labels.index(selected_label)
    tournament = tournaments[selected_idx]
    tournament_id = tournament["id"]

    st.subheader("Tournament Shell / Overview")
    st.caption("Stage 1: shell metadata • Stage 2: registration setup in Tournament Manager • Stage 3: operations below.")
    meta_cols = st.columns(4)
    meta_cols[0].metric("Status", str(tournament.get("status") or "DRAFT"))
    meta_cols[1].metric("Legacy team count", int(tournament.get("team_count") or LEGACY_DEFAULT_TEAM_COUNT))
    meta_cols[2].metric("Start", str(tournament.get("start_date") or "—"))
    meta_cols[3].metric("End", str(tournament.get("end_date") or "—"))

    draws_resp = (
        supabase.table("tournament_event_draws")
        .select("*")
        .eq("tournament_id", tournament_id)
        .order("created_at", desc=False)
        .execute()
    )
    draws = draws_resp.data or []

    available, _ = registration_feature_available(supabase)
    registration_bridge = None
    if available:
        try:
            reg_settings = get_registration_settings(supabase, tournament_id, tournament_name=str(tournament.get("name") or ""))
            reg_days = list_registration_days(supabase, tournament_id)
            reg_events = list_registration_event_options(supabase, tournament_id)
            reg_state = build_registration_state(supabase, tournament, reg_settings, reg_days, reg_events)
            registration_bridge = {
                "settings": reg_settings,
                "days": reg_days,
                "events": reg_events,
                "state": reg_state,
            }
        except Exception:
            registration_bridge = None

    selected_draw_id = None
    selected_event_option_id = None
    selected_day_id = None
    selected_event_type = ""

    st.subheader("Event Operations / Division Builder")
    day_options = (registration_bridge or {}).get("days") or []
    event_options = (registration_bridge or {}).get("events") or []
    day_map = {str(d.get("id")): d for d in day_options}

    if event_options:
        event_labels = []
        for row in event_options:
            day = day_map.get(str(row.get("registration_day_id")), {})
            event_labels.append(f"{day.get('label') or 'Day ?'} • {row.get('label') or 'Event'}")
        selected_event_label = st.selectbox("Select registration event/division", event_labels, key=f"ops_event_{tournament_id}")
        event_idx = event_labels.index(selected_event_label)
        selected_event = event_options[event_idx]
        selected_event_option_id = str(selected_event.get("id"))
        selected_day_id = str(selected_event.get("registration_day_id"))
        selected_event_type = str(selected_event.get("event_type") or "")

        scoped_draws = [
            d for d in draws
            if str(d.get("event_option_id")) == selected_event_option_id and str(d.get("registration_day_id")) == selected_day_id
        ]
        if scoped_draws:
            draw_labels = [f"{d.get('name') or 'Draw'} ({d.get('status') or 'draft'})" for d in scoped_draws]
            draw_pick = st.selectbox("Select ops draw", draw_labels, key=f"ops_draw_{tournament_id}")
            draw = scoped_draws[draw_labels.index(draw_pick)]
            selected_draw_id = str(draw.get("id"))
        if st.button("Create operations draw for selected division", key=f"create_draw_{tournament_id}"):
            draw_payload = {
                "tournament_id": tournament_id,
                "registration_day_id": selected_day_id,
                "event_option_id": selected_event_option_id,
                "name": f"{selected_event.get('label') or 'Division'} Ops Draw",
                "status": "draft",
            }
            supabase.table("tournament_event_draws").insert(draw_payload).execute()
            st.success("Division ops draw created.")
            st.rerun()

        if selected_draw_id and st.button("Build Division from Registrations", key=f"build_from_reg_{tournament_id}"):
            roster = _find_event_roster(registration_bridge, event_option_id=selected_event_option_id, registration_day_id=selected_day_id) or {}
            entries = [e for e in (roster.get("entries") or []) if str(e.get("status")) == "CONFIRMED"]
            payload = []
            for idx, entry in enumerate(entries, start=1):
                members = entry.get("members") or []
                p1_name = str((members[0] if len(members) > 0 else {}).get("display_name") or "").strip()
                p2_name = str((members[1] if len(members) > 1 else {}).get("display_name") or "").strip()
                payload.append(
                    {
                        "tournament_id": tournament_id,
                        "draw_id": selected_draw_id,
                        "event_option_id": selected_event_option_id,
                        "registration_day_id": selected_day_id,
                        "team_number": idx,
                        "player1_id": name_to_id.get(p1_name),
                        "player2_id": name_to_id.get(p2_name) if p2_name else None,
                        "source": "REGISTRATION",
                        "notes": None,
                    }
                )
            if payload:
                supabase.table("tournament_teams").upsert(payload, on_conflict="tournament_id,draw_id,team_number").execute()
                st.success(f"Imported {len(payload)} confirmed entries into this division draw.")
                st.rerun()
            st.info("No confirmed registration entries found for this event/division.")
    else:
        st.info("Configure registration days/events first in Tournament Manager to build division-scoped operations.")

    _render_registration_bridge(tournament, registration_bridge)

    team_query = supabase.table("tournament_teams").select("*").eq("tournament_id", tournament_id)
    game_query = supabase.table("tournament_games").select("*").eq("tournament_id", tournament_id)
    if selected_draw_id:
        team_query = team_query.eq("draw_id", selected_draw_id)
        game_query = game_query.eq("draw_id", selected_draw_id)
    else:
        team_query = team_query.is_("draw_id", "null")
        game_query = game_query.is_("draw_id", "null")

    teams_resp = team_query.order("team_number").execute()
    teams = teams_resp.data or []
    teams_by_number = {int(t["team_number"]): t for t in teams}
    teams_by_id = {t["id"]: t for t in teams}

    games_resp = game_query.order("rr_round_number", desc=False).order("rr_slot_number", desc=False).execute()
    games = games_resp.data or []

    rr_games = [g for g in games if g.get("stage") == "ROUND_ROBIN"]
    playoff_games = [g for g in games if g.get("stage") == "PLAYOFF"]

    podium_resp = (
        supabase.table("tournament_podium")
        .select("*")
        .eq("tournament_id", tournament_id)
        .order("placement", desc=False)
        .execute()
    )
    podium_rows = podium_resp.data or []
    is_complete = tournament.get("status") == "COMPLETE"


    st.subheader("Tournament Completion")
    if is_complete:
        st.success("Tournament is complete. Editing is locked.")
        _render_podium_read_only(podium_rows, teams_by_id, id_to_name)
    else:
        if st.button("🏁 Complete Tournament"):
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

    st.divider()

    tabs = st.tabs(["Tournament Operations (Teams)", "Round Robin", "Standings", "Playoffs"])

    with tabs[0]:
        st.subheader("Teams")
        singles_division = str(selected_event_type or "").upper() == "SINGLES"
        if selected_draw_id:
            st.caption("Editing teams for selected event/division draw.")
        else:
            st.caption("No division draw selected. Editing legacy tournament-wide team list.")

        team_count_locked = bool(games) or is_complete
        team_count_value = int(tournament.get("team_count", 4))

        c1, c2 = st.columns([2, 1])
        with c1:
            st.caption("Team count is an operations-draw setting used for bracket generation and locks once games are generated.")
        with c2:
            new_team_count = st.selectbox(
                "Team count",
                SUPPORTED_TEAM_COUNTS,
                index=SUPPORTED_TEAM_COUNTS.index(team_count_value) if team_count_value in SUPPORTED_TEAM_COUNTS else 0,
                disabled=team_count_locked,
                key="tourney_team_count_select",
            )
        if not team_count_locked and st.button("Update team count"):
            supabase.table("tournaments").update({"team_count": int(new_team_count)}).eq("id", tournament_id).execute()
            st.success("Team count updated.")
            st.rerun()

        rows = []
        for num in range(1, int(tournament.get("team_count", 4)) + 1):
            team = teams_by_number.get(num)
            p1_name = id_to_name.get(team.get("player1_id")) if team else None
            p2_name = id_to_name.get(team.get("player2_id")) if team else None
            rows.append({
                "Team / Slot": num,
                "Player 1": p1_name,
                "Player 2": p2_name,
                "Source": str(team.get("source") or "MANUAL") if team else "MANUAL",
                "Notes": str(team.get("notes") or "") if team else "",
            })

        editor_df = st.data_editor(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Player 1": st.column_config.SelectboxColumn("Player 1", options=player_names, required=True),
                "Player 2": st.column_config.SelectboxColumn("Player 2", options=player_names, required=not singles_division),
                "Source": st.column_config.SelectboxColumn("Source", options=["REGISTRATION", "MANUAL", "BULK_UPLOAD"]),
                "Notes": st.column_config.TextColumn("Notes"),
            },
            key="tourney_team_editor",
            disabled=is_complete,
        )

        save_col, import_col = st.columns([1, 1])
        with save_col:
            if st.button("Save Teams", disabled=is_complete):
                selected_ids = []
                for _, row in editor_df.iterrows():
                    p1 = row.get("Player 1")
                    p2 = row.get("Player 2")
                    if p1 and p2 and p1 == p2:
                        st.error("A team cannot use the same player twice.")
                        st.stop()
                    if singles_division and not p1:
                        st.error("Singles divisions require Player 1.")
                        st.stop()
                    if (not singles_division) and (not p1 or not p2):
                        st.error("Doubles divisions require both Player 1 and Player 2.")
                        st.stop()
                    if p1:
                        selected_ids.append(name_to_id.get(p1))
                    if p2:
                        selected_ids.append(name_to_id.get(p2))

                duplicates = {pid for pid in selected_ids if pid is not None and selected_ids.count(pid) > 1}
                if duplicates:
                    names = ", ".join(id_to_name.get(pid, str(pid)) for pid in duplicates)
                    st.error(f"Duplicate players detected: {names}.")
                    st.stop()

                payload = []
                for _, row in editor_df.iterrows():
                    payload.append(
                        {
                            "tournament_id": tournament_id,
                            "draw_id": selected_draw_id,
                            "event_option_id": selected_event_option_id,
                            "registration_day_id": selected_day_id,
                            "team_number": int(row.get("Team / Slot")),
                            "player1_id": name_to_id.get(row.get("Player 1")) if row.get("Player 1") else None,
                            "player2_id": name_to_id.get(row.get("Player 2")) if row.get("Player 2") else None,
                            "source": str(row.get("Source") or "MANUAL"),
                            "notes": str(row.get("Notes") or "").strip() or None,
                        }
                    )

                supabase.table("tournament_teams").upsert(payload, on_conflict="tournament_id,draw_id,team_number").execute()
                st.success("Teams saved.")
                st.rerun()

        with import_col:
            with st.expander("Bulk Upload Teams"):
                upload_mode = st.radio("Save mode", ["Append", "Replace"], horizontal=True, key=f"bulk_mode_{tournament_id}")
                uploaded = st.file_uploader("CSV/XLSX", type=["csv", "xlsx"], key=f"bulk_file_{tournament_id}")
                pasted = st.text_area("Or paste CSV/TSV", key=f"bulk_text_{tournament_id}")
                if st.button("Preview Import", key=f"bulk_preview_{tournament_id}"):
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
                    dupes = []
                    id_counts = {}
                    resolved_rows = []
                    for _, row in parsed.iterrows():
                        p1_name = str(row.get("Player 1") or "").strip()
                        p2_name = str(row.get("Player 2") or "").strip()
                        p1_id = name_to_id.get(p1_name)
                        p2_id = name_to_id.get(p2_name) if p2_name else None
                        if not p1_id:
                            unresolved.append(p1_name)
                        if p2_name and not p2_id:
                            unresolved.append(p2_name)
                        for pid in [p1_id, p2_id]:
                            if pid:
                                id_counts[pid] = id_counts.get(pid, 0) + 1
                        resolved_rows.append({
                            "Player 1": p1_name,
                            "Player 2": p2_name,
                            "Player 1 Resolved": bool(p1_id),
                            "Player 2 Resolved": True if not p2_name else bool(p2_id),
                            "Team Name": str(row.get("Team Name") or "").strip(),
                            "Seed": row.get("Seed"),
                            "Notes": str(row.get("Notes") or "").strip(),
                        })
                    dupes = [id_to_name.get(pid, str(pid)) for pid, count in id_counts.items() if count > 1]
                    st.dataframe(pd.DataFrame(resolved_rows), use_container_width=True, hide_index=True)
                    if unresolved:
                        st.error("Unresolved names: " + ", ".join(sorted({u for u in unresolved if u})))
                    if dupes:
                        st.error("Duplicate players in import: " + ", ".join(sorted(set(dupes))))
                    if unresolved or dupes:
                        st.stop()

                    if upload_mode == "Replace":
                        q = supabase.table("tournament_teams").delete().eq("tournament_id", tournament_id)
                        q = q.eq("draw_id", selected_draw_id) if selected_draw_id else q.is_("draw_id", "null")
                        q.execute()

                    payload = []
                    for i, row in enumerate(resolved_rows, start=1):
                        payload.append(
                            {
                                "tournament_id": tournament_id,
                                "draw_id": selected_draw_id,
                                "event_option_id": selected_event_option_id,
                                "registration_day_id": selected_day_id,
                                "team_number": i,
                                "player1_id": name_to_id.get(row.get("Player 1")),
                                "player2_id": name_to_id.get(row.get("Player 2")) if row.get("Player 2") else None,
                                "seed": int(row.get("Seed")) if str(row.get("Seed") or "").strip().isdigit() else None,
                                "source": "BULK_UPLOAD",
                                "notes": row.get("Notes") or None,
                            }
                        )
                    supabase.table("tournament_teams").upsert(payload, on_conflict="tournament_id,draw_id,team_number").execute()
                    st.success("Bulk upload saved.")
                    st.rerun()

    with tabs[1]:
        st.subheader("Round Robin")
        ready_teams = _teams_ready(teams_by_number, int(tournament.get("team_count", 4)))
        if not rr_games:
            st.info("No round robin games created yet.")
        if not ready_teams:
            st.warning("Assign exactly two players to every team to enable schedule generation.")

        if st.button("Generate RR Schedule", disabled=bool(rr_games) or not ready_teams or is_complete):
            team_ids = {int(num): t["id"] for num, t in teams_by_number.items()}
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
                st.warning("This will delete all existing round robin and playoff games.")
                confirm = st.text_input("Type RESET to confirm", key="rr_reset_confirm")
                if st.button(
                    "Regenerate RR Schedule",
                    disabled=confirm.strip().upper() != "RESET" or is_complete,
                ):
                    clear_games = supabase.table("tournament_games").delete().eq("tournament_id", tournament_id)
                    clear_teams = supabase.table("tournament_teams").update({"seed": None}).eq("tournament_id", tournament_id)
                    if selected_draw_id:
                        clear_games = clear_games.eq("draw_id", selected_draw_id)
                        clear_teams = clear_teams.eq("draw_id", selected_draw_id)
                    else:
                        clear_games = clear_games.is_("draw_id", "null")
                        clear_teams = clear_teams.is_("draw_id", "null")
                    clear_games.execute()
                    clear_teams.execute()
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

    with tabs[2]:
        st.subheader("Standings")
        if not rr_games:
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

                if st.button("Update Seeds", disabled=is_complete):
                    for row in standings:
                        supabase.table("tournament_teams").update({"seed": int(row["seed"])}).eq("id", row["team_id"]).execute()
                    st.success("Seeds updated.")
                    st.rerun()

    with tabs[3]:
        st.subheader("Playoffs")
        if not rr_games:
            st.info("Generate round robin games first.")
            st.stop()

        advance_options = [opt for opt in [4, 5, 6] if opt <= int(tournament.get("team_count", 4))]
        current_advance = tournament.get("playoff_advance_count")
        selected_advance = st.selectbox(
            "Teams advancing",
            advance_options,
            index=advance_options.index(int(current_advance)) if current_advance in advance_options else 0,
            key="playoff_advance_select",
            disabled=is_complete,
        )
        if st.button("Save advance count", disabled=is_complete):
            supabase.table("tournaments").update({"playoff_advance_count": int(selected_advance)}).eq("id", tournament_id).execute()
            st.success("Advance count saved.")
            st.rerun()

        if not playoff_games:
            st.info("No playoff bracket generated yet.")
        if st.button("Generate Playoff Bracket", disabled=bool(playoff_games) or is_complete):
            standings = compute_round_robin_standings(list(teams_by_id.values()), rr_games)
            if len(standings) < int(selected_advance):
                st.error("Not enough seeded teams to generate the bracket.")
                st.stop()
            for row in standings:
                supabase.table("tournament_teams").update({"seed": int(row["seed"])}).eq("id", row["team_id"]).execute()
            games_payload = build_playoff_games(
                tournament_id=tournament_id,
                advance_count=int(selected_advance),
                standings=standings,
            )
            for row in games_payload:
                row["draw_id"] = selected_draw_id
                row["event_option_id"] = selected_event_option_id
                row["registration_day_id"] = selected_day_id
            supabase.table("tournament_games").insert(games_payload).execute()
            supabase.table("tournaments").update(
                {"status": "PLAYOFFS", "playoff_advance_count": int(selected_advance)}
            ).eq("id", tournament_id).execute()
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



def _render_registration_bridge(tournament: dict, registration_bridge: dict | None) -> None:
    st.subheader("Registration Bridge (Stage 2)")
    if not registration_bridge:
        st.warning("Registration is not configured for this tournament yet.")
        st.info("Use Tournament Manager to define days, events, skill/age labels, and partner requirements.")
        st.link_button("Configure Registration", f"?page=tournament_manager&tournament_id={tournament.get('id')}")
        return

    settings = registration_bridge.get("settings") or {}
    state = registration_bridge.get("state") or {}
    summary = state.get("summary") or {}
    links = build_public_urls(
        base_url=str(st.session_state.get("base_url") or ""),
        tournament_id=str(tournament.get("id")),
        registration_slug=settings.get("registration_slug"),
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registrations", summary.get("total_registrations", 0))
    c2.metric("Confirmed", summary.get("confirmed_entries", 0))
    c3.metric("Needs partner", summary.get("needs_partner_entries", 0))
    c4.metric("Issues", summary.get("issue_count", 0))

    st.caption("Use Tournament Manager for registration setup and partner-board publishing. Return here once registrations are finalized to run teams, rounds, brackets, standings, and scoring.")
    st.link_button("Configure Registration", f"?page=tournament_manager&tournament_id={tournament.get('id')}")
    with st.expander("Registration links"):
        st.text_input("Registration setup", value=links["admin_manager"], key=f"ops_reg_admin_{tournament.get('id')}")
        st.text_input("Public registration", value=links["registration"], key=f"ops_reg_public_{tournament.get('id')}")
        st.text_input("Public partner board", value=links["partner_board"], key=f"ops_reg_partner_{tournament.get('id')}")

def _teams_ready(teams_by_number: dict[int, dict], team_count: int) -> bool:
    if len(teams_by_number) != team_count:
        return False
    for num in range(1, team_count + 1):
        team = teams_by_number.get(num)
        if not team:
            return False
        if not team.get("player1_id") or not team.get("player2_id"):
            return False
    return True


def _render_games_table(*, games, teams_by_id, id_to_name, on_save, key_prefix: str, disabled: bool = False):
    rounds = sorted({int(g.get("rr_round_number", 0)) for g in games})
    for round_num in rounds:
        st.markdown(f"#### Round {round_num}")
        round_games = [g for g in games if int(g.get("rr_round_number", 0)) == round_num]
        scores = {}
        with st.form(key=f"{key_prefix}_round_{round_num}"):
            for game in round_games:
                team_a = teams_by_id.get(game.get("team_a_id"), {})
                team_b = teams_by_id.get(game.get("team_b_id"), {})
                label_a = _team_label(team_a, id_to_name)
                label_b = _team_label(team_b, id_to_name)
                slot = game.get("rr_slot_number")

                col1, col2, col3, col4 = st.columns([4, 1, 1, 2])
                col1.write(f"Game {slot}: {label_a} vs {label_b}")
                col2.number_input(
                    "Score A",
                    min_value=0,
                    value=int(game.get("score_a") or 0),
                    key=f"{key_prefix}_a_{game['id']}",
                    disabled=disabled,
                )
                col3.number_input(
                    "Score B",
                    min_value=0,
                    value=int(game.get("score_b") or 0),
                    key=f"{key_prefix}_b_{game['id']}",
                    disabled=disabled,
                )
                status = "Final" if game.get("finalized_at") else "Open"
                col4.caption(status)
                scores[game["id"]] = game

            if st.form_submit_button("Save scores", disabled=disabled):
                on_save(scores)


def _render_playoff_bracket(*, games, teams_by_id, id_to_name, on_save, disabled: bool = False):
    round_order = ["QF", "SF", "Final", "Bronze"]
    for round_name in round_order:
        round_games = [g for g in games if g.get("playoff_round") == round_name]
        if not round_games:
            continue
        st.markdown(f"#### {round_name}")
        with st.form(key=f"playoff_{round_name}"):
            for game in round_games:
                team_a = teams_by_id.get(game.get("team_a_id"), {})
                team_b = teams_by_id.get(game.get("team_b_id"), {})
                label_a = _team_label(team_a, id_to_name)
                label_b = _team_label(team_b, id_to_name)
                col1, col2, col3, col4 = st.columns([4, 1, 1, 2])
                col1.write(f"{game.get('playoff_game_code')}: {label_a} vs {label_b}")
                col2.number_input(
                    "Score A",
                    min_value=0,
                    value=int(game.get("score_a") or 0),
                    key=f"playoff_a_{game['id']}",
                    disabled=disabled or not game.get("team_a_id") or not game.get("team_b_id"),
                )
                col3.number_input(
                    "Score B",
                    min_value=0,
                    value=int(game.get("score_b") or 0),
                    key=f"playoff_b_{game['id']}",
                    disabled=disabled or not game.get("team_a_id") or not game.get("team_b_id"),
                )
                status = "Final" if game.get("finalized_at") else "Open"
                col4.caption(status)

            if st.form_submit_button("Save scores", disabled=disabled):
                on_save({g["id"]: g for g in round_games})


def _render_podium_review(
    ctx,
    tournament: dict,
    *,
    teams_by_id: dict[str, dict],
    id_to_name: dict,
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
    mode_labels = {
        "PLAYOFF_RESULTS": "Use Playoff Results",
        "ROUND_ROBIN_STANDINGS": "Use RR Standings",
        "MANUAL": "Manual",
    }
    mode_options = ["PLAYOFF_RESULTS", "ROUND_ROBIN_STANDINGS", "MANUAL"]
    mode_index = mode_options.index(default_mode)
    mode = st.radio(
        "Podium Review",
        mode_options,
        index=mode_index,
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
            unscored = _has_unscored_rr_games(rr_games)
            if unscored:
                st.warning("Some RR games are unscored; podium reflects current entered results.")
            placements = rr_podium
            _render_podium_preview(placements, teams_by_id, id_to_name)
    else:
        source = "MANUAL"
        options = list(teams_by_id.keys())
        for placement in range(1, max_places + 1):
            key = f"podium_manual_{tournament_id}_{placement}"
            selection = st.selectbox(
                f"{_placement_label(placement)}",
                options=[None, *options],
                format_func=lambda value: _team_label(teams_by_id.get(value, {}), id_to_name)
                if value
                else "Select team",
                key=key,
            )
            if selection:
                placements.append({"placement": placement, "team_id": selection})

    if st.button("Finalize Tournament", type="primary", disabled=max_places == 0):
        if max_places == 0:
            st.error("No teams available for podium placement.")
            return
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


def _render_podium_read_only(podium_rows: list[dict], teams_by_id: dict[str, dict], id_to_name: dict) -> None:
    if not podium_rows:
        st.info("No podium recorded yet.")
        return
    st.markdown("#### 🏆 Podium")
    _render_podium_preview(podium_rows, teams_by_id, id_to_name)


def _render_podium_preview(placements: list[dict], teams_by_id: dict[str, dict], id_to_name: dict) -> None:
    rows = []
    for row in sorted(placements, key=lambda item: int(item.get("placement", 0) or 0)):
        placement = int(row.get("placement", 0) or 0)
        team_id = row.get("team_id")
        team = teams_by_id.get(team_id, {})
        rows.append(
            {
                "Placement": _placement_label(placement),
                "Team": _team_label(team, id_to_name),
            }
        )
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
            score_a = int(score_a)
            score_b = int(score_b)
        except Exception:
            return True
        if score_a == 0 and score_b == 0:
            return True
    return False


def _team_label(team: dict, id_to_name: dict) -> str:
    if not team:
        return "TBD"
    p1 = id_to_name.get(team.get("player1_id"), "?")
    p2 = id_to_name.get(team.get("player2_id"), "?")
    return f"Team {team.get('team_number')}: {p1} / {p2}"


def _save_games(ctx, tournament, teams_by_id, game_map, stage: str):
    supabase = ctx.supabase
    if tournament.get("status") == "COMPLETE":
        st.error("Tournament is complete. Scores are locked.")
        return
    df_players_all = ctx.df_players_all
    df_leagues = ctx.df_leagues
    df_meta = ctx.df_meta
    name_to_id = ctx.name_to_id

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

        match_payload = _build_match_payload(
            tournament,
            game,
            teams_by_id,
            score_a=score_a,
            score_b=score_b,
        )
        try:
            process_matches(
                [match_payload],
                supabase=supabase,
                club_id=str(ctx.club_id),
                name_to_id=name_to_id,
                df_players_all=df_players_all,
                df_leagues=df_leagues,
                df_meta=df_meta,
            )
        except Exception as exc:
            st.error(f"Could not create public match row for tournament game {game_id}: {exc}")
            raise

        supabase.table("tournament_games").update(finalize_payload).eq("id", game_id).execute()

        if stage == "PLAYOFF":
            playoff_query = (
                supabase.table("tournament_games")
                .select("*")
                .eq("tournament_id", tournament["id"])
                .eq("stage", "PLAYOFF")
            )
            if tournament.get("active_draw_id"):
                playoff_query = playoff_query.eq("draw_id", tournament.get("active_draw_id"))
            else:
                playoff_query = playoff_query.is_("draw_id", "null")
            playoff_games_resp = playoff_query.execute()
            playoff_games = playoff_games_resp.data or []
            updates = resolve_playoff_dependencies(playoff_games)
            for upd in updates:
                supabase.table("tournament_games").update(upd).eq("id", upd["id"]).execute()

    if updated_any:
        st.success("Scores saved.")
        st.session_state["force_data_refresh"] = True
        st.rerun()


def _build_match_payload(tournament, game, teams_by_id, *, score_a: int, score_b: int) -> dict:
    return build_tournament_match_payload(
        tournament,
        game,
        teams_by_id,
        score_a=score_a,
        score_b=score_b,
    )


def _score_key(stage: str, side: str, game_id: str) -> str:
    prefix = "playoff" if stage == "PLAYOFF" else "rr"
    return f"{prefix}_{side}_{game_id}"
