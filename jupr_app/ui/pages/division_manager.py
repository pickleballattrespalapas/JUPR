from __future__ import annotations

from uuid import uuid4

import pandas as pd
import streamlit as st

from jupr_app.data.retry import sb_retry
from jupr_app.ui.layout import page_shell


def _load_division(supabase, club_id: str, tournament_id: str, division_id: str) -> dict | None:
    resp = sb_retry(
        lambda: (
            supabase.table("tournament_divisions")
            .select("id,title,max_teams,tournament_id")
            .eq("club_id", club_id)
            .eq("tournament_id", tournament_id)
            .eq("id", division_id)
            .limit(1)
            .execute()
        )
    )
    rows = resp.data or []
    return rows[0] if rows else None


def _load_teams(supabase, club_id: str) -> list[dict]:
    resp = sb_retry(lambda: (supabase.table("teams").select("*").eq("club_id", club_id).execute()))
    return resp.data or []


def _load_entries(supabase, club_id: str, division_id: str) -> list[dict]:
    resp = sb_retry(
        lambda: (
            supabase.table("division_entries")
            .select("id,team_id,seed,created_at")
            .eq("club_id", club_id)
            .eq("division_id", division_id)
            .order("seed", desc=False, nullsfirst=False)
            .order("created_at", desc=False)
            .execute()
        )
    )
    return resp.data or []


def _team_label(team: dict) -> str:
    for key in ("name", "team_name", "title", "display_name"):
        value = str(team.get(key) or "").strip()
        if value:
            return value
    return str(team.get("id") or "Unnamed team")


def _insert_entry(supabase, club_id: str, division_id: str, team_id: str, seed: int | None) -> None:
    payload = {
        "id": str(uuid4()),
        "club_id": club_id,
        "division_id": division_id,
        "team_id": team_id,
        "seed": seed,
    }
    sb_retry(lambda: supabase.table("division_entries").insert(payload).execute())


def _update_entry_seed(supabase, club_id: str, entry_id: str, seed: int | None) -> None:
    sb_retry(
        lambda: (
            supabase.table("division_entries")
            .update({"seed": seed})
            .eq("club_id", club_id)
            .eq("id", entry_id)
            .execute()
        )
    )


def _remove_entry(supabase, club_id: str, entry_id: str) -> None:
    sb_retry(
        lambda: (
            supabase.table("division_entries")
            .delete()
            .eq("club_id", club_id)
            .eq("id", entry_id)
            .execute()
        )
    )


def render(ctx):
    mode_label = "Public" if bool(getattr(ctx, "public_mode", False)) else "Admin"
    page_shell("🏆 Division Manager", "Manage teams and seeds for a division.", mode_label=mode_label)

    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    supabase = getattr(ctx, "supabase", None)
    club_id = str(getattr(ctx, "club_id", "") or "")
    if supabase is None or not club_id:
        st.error("Missing required application context.")
        st.stop()

    route = str(st.query_params.get("route", "") or "").strip("/")
    if route:
        parts = route.split("/")
        if len(parts) == 4 and parts[0] == "tournament" and parts[2] == "division":
            st.query_params["tournament_id"] = parts[1]
            st.query_params["division_id"] = parts[3]

    tournament_id = str(st.query_params.get("tournament_id", "") or "").strip()
    division_id = str(st.query_params.get("division_id", "") or "").strip()

    if not tournament_id or not division_id:
        st.info("Choose a division from Tournament Manager to manage entries.")
        return

    try:
        division = _load_division(supabase, club_id, tournament_id, division_id)
    except Exception as exc:
        st.error(f"Could not load division: {exc}")
        return

    if not division:
        st.error("Division not found for this tournament/club.")
        return

    st.subheader(str(division.get("title") or "Division"))
    max_teams = division.get("max_teams")
    st.caption(f"Max teams: {max_teams if max_teams is not None else 'No limit'}")

    try:
        teams = _load_teams(supabase, club_id)
        entries = _load_entries(supabase, club_id, division_id)
    except Exception as exc:
        st.error(f"Could not load division entry data: {exc}")
        return

    teams_by_id = {str(team.get("id")): team for team in teams if team.get("id")}
    existing_team_ids = {str(entry.get("team_id")) for entry in entries if entry.get("team_id")}

    st.markdown("### Add Team")
    remaining_slots = None
    if max_teams is not None:
        remaining_slots = max(0, int(max_teams) - len(entries))
        st.caption(f"Remaining slots: {remaining_slots}")

    available_teams = [team for team in teams if str(team.get("id")) not in existing_team_ids]
    team_option_ids = [str(team.get("id")) for team in available_teams if team.get("id")]
    option_labels = {team_id: _team_label(teams_by_id[team_id]) for team_id in team_option_ids}

    add_col1, add_col2, add_col3 = st.columns([3, 1, 1])
    with add_col1:
        selected_team_id = st.selectbox(
            "Team",
            options=team_option_ids,
            format_func=lambda tid: option_labels.get(tid, tid),
            index=None,
            placeholder="Select a team",
            key="division_manager_add_team",
        )
    with add_col2:
        seed_raw = st.text_input("Seed", value="", key="division_manager_add_seed")
    with add_col3:
        add_clicked = st.button("Add", use_container_width=True, type="primary")

    if add_clicked:
        if max_teams is not None and len(entries) >= int(max_teams):
            st.error("This division already reached its max teams limit.")
        elif not selected_team_id:
            st.error("Select a team first.")
        elif selected_team_id in existing_team_ids:
            st.error("That team is already in this division.")
        else:
            clean_seed: int | None = None
            raw = seed_raw.strip()
            if raw:
                if not raw.isdigit() or int(raw) <= 0:
                    st.error("Seed must be a positive whole number.")
                    st.stop()
                clean_seed = int(raw)

            try:
                _insert_entry(supabase, club_id, division_id, selected_team_id, clean_seed)
                st.success("Team added.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not add team: {exc}")

    st.markdown("### Existing Entries")
    if not entries:
        st.info("No entries in this division yet.")
        return

    rows = []
    for entry in entries:
        team_id = str(entry.get("team_id") or "")
        rows.append(
            {
                "Entry ID": str(entry.get("id") or ""),
                "Team": _team_label(teams_by_id.get(team_id, {"id": team_id})),
                "Seed": int(entry["seed"]) if entry.get("seed") is not None else None,
                "Remove": False,
            }
        )

    edited_df = st.data_editor(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Entry ID": st.column_config.TextColumn("Entry ID", disabled=True),
            "Team": st.column_config.TextColumn("Team", disabled=True),
            "Seed": st.column_config.NumberColumn("Seed", min_value=1, step=1),
            "Remove": st.column_config.CheckboxColumn("Remove"),
        },
        disabled=["Entry ID", "Team"],
        key="division_manager_entries_editor",
    )

    if st.button("Save Entry Changes", type="secondary"):
        try:
            for _, row in edited_df.iterrows():
                entry_id = str(row.get("Entry ID") or "")
                if not entry_id:
                    continue

                if bool(row.get("Remove")):
                    _remove_entry(supabase, club_id, entry_id)
                    continue

                seed_value = row.get("Seed")
                clean_seed = int(seed_value) if pd.notna(seed_value) else None
                _update_entry_seed(supabase, club_id, entry_id, clean_seed)

            st.success("Division entries updated.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not save entry changes: {exc}")
