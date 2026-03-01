from __future__ import annotations

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

    st.subheader("Create Tournament")
    c1, c2, c3 = st.columns([3, 2, 1])
    with c1:
        tournament_name = st.text_input("Tournament name", key="tourney_create_name")
    with c2:
        team_count = st.selectbox("Team count", SUPPORTED_TEAM_COUNTS, key="tourney_create_team_count")
    with c3:
        if st.button("Create", type="primary"):
            if not tournament_name.strip():
                st.error("Tournament name is required.")
            else:
                supabase.table("tournaments").insert(
                    {
                        "club_id": str(club_id),
                        "name": tournament_name.strip(),
                        "status": "DRAFT",
                        "team_count": int(team_count),
                    }
                ).execute()
                st.success("Tournament created.")
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

    teams_resp = (
        supabase.table("tournament_teams")
        .select("*")
        .eq("tournament_id", tournament_id)
        .order("team_number")
        .execute()
    )
    teams = teams_resp.data or []
    teams_by_number = {int(t["team_number"]): t for t in teams}
    teams_by_id = {t["id"]: t for t in teams}

    games_resp = (
        supabase.table("tournament_games")
        .select("*")
        .eq("tournament_id", tournament_id)
        .order("rr_round_number", desc=False)
        .order("rr_slot_number", desc=False)
        .execute()
    )
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

    tabs = st.tabs(["Teams", "Round Robin", "Standings", "Playoffs"])

    with tabs[0]:
        st.subheader("Teams")
        team_count_locked = bool(games) or is_complete
        team_count_value = int(tournament.get("team_count", 4))

        c1, c2 = st.columns([2, 1])
        with c1:
            st.caption("Team count is locked once games are generated.")
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
            rows.append({"Team": num, "Player 1": p1_name, "Player 2": p2_name})

        editor_df = st.data_editor(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Player 1": st.column_config.SelectboxColumn("Player 1", options=player_names),
                "Player 2": st.column_config.SelectboxColumn("Player 2", options=player_names),
            },
            key="tourney_team_editor",
            disabled=is_complete,
        )

        if st.button("Save Teams", disabled=is_complete):
            selected_ids = []
            for _, row in editor_df.iterrows():
                p1 = row.get("Player 1")
                p2 = row.get("Player 2")
                if p1 and p2 and p1 == p2:
                    st.error("A team cannot use the same player twice.")
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
                        "team_number": int(row.get("Team")),
                        "player1_id": name_to_id.get(row.get("Player 1")) if row.get("Player 1") else None,
                        "player2_id": name_to_id.get(row.get("Player 2")) if row.get("Player 2") else None,
                    }
                )

            supabase.table("tournament_teams").upsert(payload, on_conflict="tournament_id,team_number").execute()
            st.success("Teams saved.")
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
                    supabase.table("tournament_games").delete().eq("tournament_id", tournament_id).execute()
                    supabase.table("tournament_teams").update({"seed": None}).eq("tournament_id", tournament_id).execute()
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
                    tournament,
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
                    tournament,
                    teams_by_id,
                    updates,
                    stage="PLAYOFF",
                ),
                disabled=is_complete,
            )


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
            playoff_games_resp = (
                supabase.table("tournament_games")
                .select("*")
                .eq("tournament_id", tournament["id"])
                .eq("stage", "PLAYOFF")
                .execute()
            )
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
