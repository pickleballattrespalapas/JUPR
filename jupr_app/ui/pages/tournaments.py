from __future__ import annotations

from jupr_app.data.sb_write import sb_insert, sb_update, sb_upsert

from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import pandas as pd
import streamlit as st

from jupr_app.domain.tournaments.sync import sync_tournament_game_to_match
from jupr_app.domain.tournaments import (
    build_playoff_games,
    build_round_robin_games,
    build_podium_payload,
    compute_podium_from_playoffs,
    compute_podium_from_rr,
    compute_round_robin_standings,
    finalize_game,
    resolve_series_results,
    resolve_playoff_dependencies,
)
from jupr_app.domain.tournament_podium import award_tournament_trophies_from_podium, upsert_tournament_podium
from jupr_app.ui.layout import page_shell


TOURNAMENT_CHARTS = [
    {
        "id": "rr-6",
        "name": "6-team RR",
        "file": Path(__file__).resolve().parents[1] / "assets" / "tournaments" / "rr-6.csv",
    }
]


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def _require_club_id_payload(payload):
    # NOTE: non-functional touch to retrigger CI checks.
    rows = payload if isinstance(payload, list) else [payload]
    for row in rows:
        if "club_id" not in row:
            raise RuntimeError("Missing club_id in tournament write payload.")


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
    st.caption("Tournament charts")
    chart_options = {f"{chart['name']} ({chart['id']})": chart for chart in TOURNAMENT_CHARTS}
    selected_chart_label = st.selectbox("Downloadable chart template", list(chart_options.keys()))
    selected_chart = chart_options[selected_chart_label]
    with selected_chart["file"].open("rb") as chart_file:
        st.download_button(
            "Download chart CSV",
            data=chart_file.read(),
            file_name=selected_chart["file"].name,
            mime="text/csv",
            key=f"download_chart_{selected_chart['id']}",
        )

    c1, c2, c3 = st.columns([3, 2, 1])
    with c1:
        tournament_name = st.text_input("Tournament name", key="tourney_create_name")
    with c2:
        team_count = st.selectbox("Team count", [4, 5, 6, 7, 8], key="tourney_create_team_count")
    with c3:
        if st.button("Create", type="primary"):
            if not tournament_name.strip():
                st.error("Tournament name is required.")
            else:
                sb_insert(
                    supabase,
                    "tournaments",
                    {
                        "club_id": str(club_id),
                        "name": tournament_name.strip(),
                        "status": "DRAFT",
                        "team_count": int(team_count),
                    },
                )
                st.success("Tournament created.")
                st.session_state["force_data_refresh"] = True

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
        .eq("club_id", str(club_id))
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
        .eq("club_id", str(club_id))
        .eq("tournament_id", tournament_id)
        .order("rr_round_number", desc=False)
        .order("rr_slot_number", desc=False)
        .order("id", desc=False)
        .execute()
    )
    games = games_resp.data or []
    ids = [g["id"] for g in games]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate tournament_game rows detected.")

    seen: set[tuple[object, object, object, object, object]] = set()
    for g in games:
        key = (
            g.get("stage"),
            g.get("rr_round_number"),
            g.get("rr_slot_number"),
            g.get("playoff_game_code"),
            g.get("series_game_number"),
        )
        if key in seen:
            raise RuntimeError("Duplicate tournament_game invariant violated.")
        seen.add(key)

    rr_games = [g for g in games if g.get("stage") == "ROUND_ROBIN"]
    playoff_games = [g for g in games if g.get("stage") == "PLAYOFF"]

    podium_resp = (
        supabase.table("tournament_podium")
        .select("*")
        .eq("club_id", str(club_id))
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
            podium_key = f"podium_review_open_{tournament_id}"
            if not st.session_state.get(podium_key):
                st.session_state[podium_key] = True
                _rerun()
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
                [4, 5, 6, 7, 8],
                index=[4, 5, 6, 7, 8].index(team_count_value),
                disabled=team_count_locked,
                key="tourney_team_count_select",
            )
        if not team_count_locked and st.button("Update team count"):
            sb_update(supabase, "tournaments", {"team_count": int(new_team_count)}, filters={"club_id": str(club_id), "id": tournament_id})
            st.success("Team count updated.")
            st.session_state["force_data_refresh"] = True

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
            assert club_id, "club_id must be present for tournament writes"
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
                        # Explicit club_id for tenant isolation (RLS + multi-club safety)
                        "club_id": str(club_id),
                        "tournament_id": tournament_id,
                        "team_number": int(row.get("Team")),
                        "player1_id": name_to_id.get(row.get("Player 1")) if row.get("Player 1") else None,
                        "player2_id": name_to_id.get(row.get("Player 2")) if row.get("Player 2") else None,
                    }
                )

            _require_club_id_payload(payload)

            sb_upsert(supabase, "tournament_teams", payload, conflict="club_id,tournament_id,team_number")
            st.success("Teams saved.")
            st.session_state["force_data_refresh"] = True

    with tabs[1]:
        st.subheader("Round Robin")
        ready_teams = _teams_ready(teams_by_number, int(tournament.get("team_count", 4)))
        if not rr_games:
            st.info("No round robin games created yet.")
        if not ready_teams:
            st.warning("Assign exactly two players to every team to enable schedule generation.")

        if st.button("Generate RR Schedule", disabled=bool(rr_games) or not ready_teams or is_complete):
            assert club_id, "club_id must be present for tournament writes"
            team_ids = {int(num): t["id"] for num, t in teams_by_number.items()}
            games_payload = build_round_robin_games(tournament_id=tournament_id, team_ids_by_number=team_ids)
            for game in games_payload:
                # Explicit club_id for tenant isolation (RLS + multi-club safety)
                game["club_id"] = str(club_id)
            _require_club_id_payload(games_payload)
            if getattr(ctx, "DEBUG_MODE", False):
                print(
                    "tournament_games upsert conflict:",
                    "club_id,tournament_id,stage,game_conflict_key",
                )
            supabase.table("tournament_games").upsert(
                games_payload,
                on_conflict="club_id,tournament_id,stage,game_conflict_key"
            ).execute()
            sb_update(supabase, "tournaments", {"status": "ROUND_ROBIN"}, filters={"club_id": str(club_id), "id": tournament_id})
            st.success("Round robin schedule generated.")
            st.session_state["force_data_refresh"] = True

        if rr_games:
            with st.expander("Regenerate schedule"):
                st.warning("This will delete all existing round robin and playoff games.")
                confirm = st.text_input("Type RESET to confirm", key="rr_reset_confirm")
                if st.button(
                    "Regenerate RR Schedule",
                    disabled=confirm.strip().upper() != "RESET" or is_complete,
                ):
                    supabase.table("tournament_games").delete().eq("club_id", str(club_id)).eq("tournament_id", tournament_id).execute()
                    sb_update(supabase, "tournament_teams", {"seed": None}, filters={"club_id": str(club_id), "tournament_id": tournament_id})
                    sb_update(supabase, "tournaments", {"status": "DRAFT", "playoff_advance_count": None}, filters={"club_id": str(club_id), "id": tournament_id})
                    st.success("Schedule cleared.")
                    st.session_state["force_data_refresh"] = True

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
                stage="ROUND_ROBIN",
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

                st.markdown("#### Manual Seeding")
                for row in standings:
                    team_id = row["team_id"]
                    st.number_input(
                        f"Team {row['team_number']} seed",
                        min_value=0,
                        max_value=32,
                        value=int(row.get("seed") or 0),
                        key=f"seed_input_{team_id}",
                        disabled=is_complete,
                    )

                if st.button("Save Manual Seeds", disabled=is_complete):
                    for row in standings:
                        team_id = row["team_id"]
                        new_seed = int(st.session_state.get(f"seed_input_{team_id}") or 0)
                        sb_update(
                            supabase,
                            "tournament_teams",
                            {"seed": new_seed if new_seed > 0 else None},
                            filters={"club_id": str(club_id), "tournament_id": tournament_id, "id": team_id},
                        )
                    st.success("Manual seeds updated.")
                    st.session_state["force_data_refresh"] = True

                if st.button("Update Seeds", disabled=is_complete):
                    for row in standings:
                        sb_update(
                            supabase,
                            "tournament_teams",
                            {"seed": int(row["seed"])},
                            filters={"club_id": str(club_id), "tournament_id": tournament_id, "id": row["team_id"]},
                        )
                    st.success("Recommended seeds updated.")
                    st.session_state["force_data_refresh"] = True

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
            sb_update(supabase, "tournaments", {"playoff_advance_count": int(selected_advance)}, filters={"club_id": str(club_id), "id": tournament_id})
            st.success("Advance count saved.")
            st.session_state["force_data_refresh"] = True

        best_of = st.selectbox(
            "Playoff Format",
            options=[1, 3],
            format_func=lambda x: "1 Game" if x == 1 else "Best 2 of 3",
            index=0 if tournament.get("playoff_best_of", 1) == 1 else 1,
            key="playoff_best_of_select",
            disabled=is_complete,
        )

        if st.button("Save format", disabled=is_complete):
            supabase.table("tournaments").update({"playoff_best_of": int(best_of)}).eq("club_id", str(club_id)).eq("id", tournament_id).execute()
            st.success("Playoff format saved.")
            st.session_state["force_data_refresh"] = True

        # --- Regenerate Playoff Bracket ---
        if playoff_games:
            if tournament.get("status") == "COMPLETE":
                st.info("Tournament is complete. Bracket cannot be regenerated.")
            else:
                existing_scored = any(
                    g.get("score_a") is not None or g.get("score_b") is not None
                    for g in playoff_games
                )

                if existing_scored:
                    st.warning("Playoff scores have already been entered. Bracket cannot be regenerated.")
                else:
                    existing_playoffs = (
                        supabase.table("tournament_games")
                        .select("id")
                        .eq("club_id", str(club_id))
                        .eq("tournament_id", tournament_id)
                        .eq("stage", "PLAYOFF")
                        .limit(1)
                        .execute()
                    )
                    requires_confirm = bool(existing_playoffs.data)
                    if requires_confirm:
                        st.warning("Playoffs already generated. Regenerate will overwrite.")
                    confirm_regen = st.checkbox(
                        "I understand this will overwrite the existing playoff bracket",
                        key=f"confirm_playoff_regen_{tournament_id}",
                    )
                    if st.button("Regenerate Playoff Bracket", type="secondary"):
                        if requires_confirm and not confirm_regen:
                            st.error("Confirm bracket overwrite before regenerating.")
                            st.stop()
                        assert club_id, "club_id must be present for tournament writes"
                        supabase.table("tournament_games").delete().eq("club_id", str(club_id)).eq("tournament_id", tournament_id).eq("stage", "PLAYOFF").execute()

                        standings = _load_seeded_standings(supabase, str(club_id), tournament_id)
                        if len(standings) < int(tournament.get("playoff_advance_count") or selected_advance):
                            st.error("Not enough stored seeds to regenerate the bracket.")
                            st.stop()

                        games_payload = build_playoff_games(
                            tournament_id=tournament_id,
                            advance_count=int(tournament.get("playoff_advance_count") or selected_advance),
                            standings=standings,
                            best_of=int(tournament.get("playoff_best_of", 1)),
                        )

                        for game in games_payload:
                            # Explicit club_id for tenant isolation (RLS + multi-club safety)
                            game["club_id"] = str(club_id)
                        _require_club_id_payload(games_payload)

                        if getattr(ctx, "DEBUG_MODE", False):
                            print("tournament_games upsert conflict:", "club_id,tournament_id,stage,game_conflict_key")
                        supabase.table("tournament_games").upsert(
                            games_payload,
                            on_conflict="club_id,tournament_id,stage,game_conflict_key"
                        ).execute()

                        st.success("Playoff bracket regenerated.")
                        st.session_state["force_data_refresh"] = True

        if not playoff_games:
            st.info("No playoff bracket generated yet.")
        if st.button("Generate Playoff Bracket", disabled=bool(playoff_games) or is_complete):
            assert club_id, "club_id must be present for tournament writes"
            standings = _load_seeded_standings(supabase, str(club_id), tournament_id)
            if len(standings) < int(selected_advance):
                st.error("Not enough seeded teams to generate the bracket.")
                st.stop()
            games_payload = build_playoff_games(
                tournament_id=tournament_id,
                advance_count=int(selected_advance),
                standings=standings,
                best_of=int(tournament.get("playoff_best_of", 1)),
            )
            for game in games_payload:
                # Explicit club_id for tenant isolation (RLS + multi-club safety)
                game["club_id"] = str(club_id)
            _require_club_id_payload(games_payload)
            if getattr(ctx, "DEBUG_MODE", False):
                print("tournament_games upsert conflict:", "club_id,tournament_id,stage,game_conflict_key")
            supabase.table("tournament_games").upsert(
                games_payload,
                on_conflict="club_id,tournament_id,stage,game_conflict_key"
            ).execute()
            sb_update(
                supabase,
                "tournaments",
                {"status": "PLAYOFFS", "playoff_advance_count": int(selected_advance)},
                filters={"club_id": str(club_id), "id": tournament_id},
            )
            st.success("Playoff bracket generated.")
            st.session_state["force_data_refresh"] = True

        if playoff_games:
            if st.button("🔄 Recompute Bracket Dependencies", disabled=is_complete):
                playoff_games_resp = (
                    supabase.table("tournament_games")
                    .select("*")
                    .eq("club_id", str(club_id))
                    .eq("tournament_id", tournament_id)
                    .eq("stage", "PLAYOFF")
                    .execute()
                )
                playoff_games = playoff_games_resp.data or []

                series_updates = resolve_series_results(playoff_games)
                for upd in series_updates:
                    supabase.table("tournament_games").update(upd).eq("club_id", str(club_id)).eq("id", upd["id"]).execute()

                playoff_games_resp = (
                    supabase.table("tournament_games")
                    .select("*")
                    .eq("club_id", str(club_id))
                    .eq("tournament_id", tournament_id)
                    .eq("stage", "PLAYOFF")
                    .execute()
                )
                playoff_games = playoff_games_resp.data or []

                dependency_updates = resolve_playoff_dependencies(playoff_games)
                for upd in dependency_updates:
                    supabase.table("tournament_games").update(upd).eq("club_id", str(club_id)).eq("id", upd["id"]).execute()

                st.success("Bracket dependencies recomputed.")
                st.session_state["force_data_refresh"] = True

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


def _load_seeded_standings(supabase, club_id: str, tournament_id: str) -> list[dict]:
    seeded_resp = (
        supabase.table("tournament_teams")
        .select("id, seed")
        .eq("club_id", club_id)
        .eq("tournament_id", tournament_id)
        .not_.is_("seed", "null")
        .order("seed", desc=False)
        .execute()
    )

    standings = []
    for row in (seeded_resp.data or []):
        standings.append({"team_id": row["id"], "seed": int(row["seed"])})
    return standings


def _render_games_table(*, games, teams_by_id, id_to_name, on_save, stage: str, disabled: bool = False):
    rounds = sorted({int(g.get("rr_round_number", 0)) for g in games})
    for round_num in rounds:
        st.markdown(f"#### Round {round_num}")
        round_games = [g for g in games if int(g.get("rr_round_number", 0)) == round_num]
        scores = {}
        with st.form(key=f"{_score_key(stage, 'round', str(round_num))}"):
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
                    key=_score_key(stage, "a", game["id"]),
                    disabled=disabled,
                )
                col3.number_input(
                    "Score B",
                    min_value=0,
                    value=int(game.get("score_b") or 0),
                    key=_score_key(stage, "b", game["id"]),
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

        series_groups = defaultdict(list)
        for game in round_games:
            series_groups[game.get("playoff_game_code")].append(game)

        with st.form(key=f"playoff_{round_name}"):
            for series_code, series_games in series_groups.items():
                series_len = max((g.get("series_game_number") or 1) for g in series_games)
                st.markdown(f"### {series_code}")
                for game in sorted(series_games, key=lambda x: x.get("series_game_number") or 1):
                    team_a = teams_by_id.get(game.get("team_a_id"), {})
                    team_b = teams_by_id.get(game.get("team_b_id"), {})
                    label_a = _team_label(team_a, id_to_name)
                    label_b = _team_label(team_b, id_to_name)
                    game_number = game.get("series_game_number") or 1
                    col1, col2, col3, col4 = st.columns([4, 1, 1, 2])
                    col1.write(f"{game.get('playoff_game_code')} (Game {game_number}/{series_len}): {label_a} vs {label_b}")
                    col2.number_input(
                        "Score A",
                        min_value=0,
                        value=int(game.get("score_a") or 0),
                        key=_score_key("PLAYOFF", "a", game["id"]),
                        disabled=disabled or not game.get("team_a_id") or not game.get("team_b_id"),
                    )
                    col3.number_input(
                        "Score B",
                        min_value=0,
                        value=int(game.get("score_b") or 0),
                        key=_score_key("PLAYOFF", "b", game["id"]),
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
        assert ctx.club_id, "club_id must be present for tournament writes"
        if max_places == 0:
            st.error("No teams available for podium placement.")
            return
        if mode == "MANUAL" and len(placements) < max_places:
            st.error("Select a team for each podium placement.")
            return
        try:
            payload = build_podium_payload(tournament_id, placements, source)
            for row in payload:
                # Explicit club_id for tenant isolation (RLS + multi-club safety)
                row["club_id"] = str(ctx.club_id)
            _require_club_id_payload(payload)
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
            return
        if not payload:
            st.error("Podium placements are required to complete the tournament.")
            return

        upsert_tournament_podium(ctx.supabase, str(ctx.club_id), tournament_id, payload)
        award_tournament_trophies_from_podium(ctx, tournament_id, tournament_name)
        ctx.supabase.table("tournaments") \
            .update({"status": "COMPLETE"}) \
            .eq("club_id", str(ctx.club_id)) \
            .eq("id", tournament_id) \
            .execute()
        st.success("Tournament completed and podium locked.")
        st.session_state.pop(f"podium_review_open_{tournament_id}", None)
        st.session_state["force_data_refresh"] = True
        _rerun()


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

    updated_any = False

    for game_id in game_map.keys():
        score_a = int(st.session_state.get(_score_key(stage, "a", game_id), 0))
        score_b = int(st.session_state.get(_score_key(stage, "b", game_id), 0))
        if getattr(ctx, "DEBUG_MODE", False):
            print("Saving game:", game_id, "Scores:", score_a, score_b)

        fresh_game = (
            supabase.table("tournament_games")
            .select("*")
            .eq("club_id", str(ctx.club_id))
            .eq("id", game_id)
            .single()
            .execute()
            .data
        )

        if score_a == 0 and score_b == 0:
            sb_update(
                supabase,
                "tournament_games",
                {
                    "score_a": None,
                    "score_b": None,
                    "winner_team_id": None,
                    "loser_team_id": None,
                    "finalized_at": None,
                },
                filters={"club_id": str(ctx.club_id), "id": game_id},
            )

            updated_any = True
            continue

        finalize_payload = finalize_game(
            {
                "team_a_id": fresh_game.get("team_a_id"),
                "team_b_id": fresh_game.get("team_b_id"),
                "score_a": score_a,
                "score_b": score_b,
            }
        )

        sb_update(
            supabase,
            "tournament_games",
            finalize_payload,
            filters={"club_id": str(ctx.club_id), "id": game_id},
        )

        match_payload = _build_match_payload(
            tournament,
            fresh_game,
            teams_by_id,
            score_a=score_a,
            score_b=score_b,
        )

        sync_tournament_game_to_match(
            supabase=supabase,
            club_id=str(ctx.club_id),
            game=fresh_game,
            match_payload=match_payload,
            name_to_id=ctx.name_to_id,
            df_players_all=ctx.df_players_all,
            df_leagues=ctx.df_leagues,
            df_meta=ctx.df_meta,
        )

        updated_any = True

    if updated_any:
        if stage == "PLAYOFF":
            # --- Fetch latest playoff games ---
            playoff_games_resp = (
                supabase.table("tournament_games")
                .select("*")
                .eq("club_id", str(ctx.club_id))
                .eq("tournament_id", tournament["id"])
                .eq("stage", "PLAYOFF")
                .execute()
            )
            playoff_games = playoff_games_resp.data or []

            # --- First resolve series winners (best-of logic) ---
            series_updates = resolve_series_results(playoff_games)
            for upd in series_updates:
                supabase.table("tournament_games").update(upd).eq("club_id", str(ctx.club_id)).eq("id", upd["id"]).execute()

            # --- Re-fetch after series resolution ---
            playoff_games_resp = (
                supabase.table("tournament_games")
                .select("*")
                .eq("club_id", str(ctx.club_id))
                .eq("tournament_id", tournament["id"])
                .eq("stage", "PLAYOFF")
                .execute()
            )
            playoff_games = playoff_games_resp.data or []

            # --- Then resolve bracket dependencies ---
            dependency_updates = resolve_playoff_dependencies(playoff_games)
            for upd in dependency_updates:
                supabase.table("tournament_games").update(upd).eq("club_id", str(ctx.club_id)).eq("id", upd["id"]).execute()

        st.success("Scores saved.")
        st.session_state["force_data_refresh"] = True



def _build_match_payload(tournament, game, teams_by_id, *, score_a: int, score_b: int) -> dict:
    team_a = teams_by_id.get(game.get("team_a_id"))
    team_b = teams_by_id.get(game.get("team_b_id"))

    return {
        "t1_p1": team_a.get("player1_id") if team_a else None,
        "t1_p2": team_a.get("player2_id") if team_a else None,
        "t2_p1": team_b.get("player1_id") if team_b else None,
        "t2_p2": team_b.get("player2_id") if team_b else None,
        "s1": int(score_a),
        "s2": int(score_b),
        "date": datetime.now(timezone.utc).isoformat(),
        "league": tournament.get("name", "Tournament"),
        "match_type": "Tournament",
        "week_tag": "Tournament",
        "is_popup": True,
        "context_type": "TOURNAMENT",
        "context_id": tournament["id"],
        "tournament_id": tournament["id"],
        "tournament_game_id": game["id"],
    }


def _score_key(stage: str, side: str, game_id: str) -> str:
    prefix = "playoff" if stage == "PLAYOFF" else "rr"
    return f"{prefix}_{side}_{game_id}"
