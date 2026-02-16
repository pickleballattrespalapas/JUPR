from __future__ import annotations

from jupr_app.data.sb_write import sb_insert, sb_update, sb_upsert

import hashlib
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
import streamlit as st

from jupr_app.data.retry import sb_retry
from jupr_app.ui.layout import page_shell
from services.match_pipeline import submit_match


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
    sb_retry(lambda: sb_insert(supabase, "division_entries", payload))


def _update_entry_seed(supabase, club_id: str, entry_id: str, seed: int | None) -> None:
    sb_retry(
        lambda: sb_update(
            supabase,
            "division_entries",
            {"seed": seed},
            filters={"club_id": club_id, "id": entry_id},
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


def _load_division_matches(supabase, club_id: str, division_id: str) -> list[dict]:
    resp = sb_retry(
        lambda: (
            supabase.table("division_matches")
            .select(
                "id,club_id,division_id,round_number,bracket_position,team_a_id,team_b_id,winner_team_id,score_json,status,created_at"
            )
            .eq("club_id", club_id)
            .eq("division_id", division_id)
            .order("round_number", desc=False)
            .order("bracket_position", desc=False)
            .order("created_at", desc=False)
            .execute()
        )
    )
    return resp.data or []


def _load_division_match(supabase, club_id: str, division_match_id: str) -> dict | None:
    resp = sb_retry(
        lambda: (
            supabase.table("division_matches")
            .select(
                "id,club_id,division_id,round_number,bracket_position,team_a_id,team_b_id,winner_team_id,score_json,status"
            )
            .eq("club_id", club_id)
            .eq("id", division_match_id)
            .limit(1)
            .execute()
        )
    )
    rows = resp.data or []
    return rows[0] if rows else None


def _build_division_match_idempotency_key(club_id: str, division_id: str, division_match_id: str) -> str:
    raw = f"division_match_submission:{club_id}:{division_id}:{division_match_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _build_division_match_payload(
    division_title: str,
    division_id: str,
    division_match: dict,
    team_a: dict,
    team_b: dict,
    score_t1: int,
    score_t2: int,
) -> dict:
    return {
        "date": datetime.now(timezone.utc).isoformat(),
        "league": division_title,
        "match_type": "Tournament",
        "week_tag": "Tournament",
        "is_popup": True,
        "division_id": division_id,
        "division_match_id": division_match.get("id"),
        "round_number": division_match.get("round_number"),
        "bracket_position": division_match.get("bracket_position"),
        "match_context": "division_match",
        "competition_type": "tournament",
        "competition_id": division_id,
        "t1_p1": int(team_a.get("player1_id")),
        "t1_p2": int(team_a.get("player2_id")),
        "t2_p1": int(team_b.get("player1_id")),
        "t2_p2": int(team_b.get("player2_id")),
        "score_t1": int(score_t1),
        "score_t2": int(score_t2),
        "s1": int(score_t1),
        "s2": int(score_t2),
    }


def _complete_division_match(
    *,
    supabase,
    club_id: str,
    division_id: str,
    division_title: str,
    division_match: dict,
    team_a: dict,
    team_b: dict,
    score_t1: int,
    score_t2: int,
) -> None:
    match_id = str(division_match.get("id") or "")
    if not match_id:
        raise ValueError("Division match id is missing.")

    latest_row = _load_division_match(supabase, club_id, match_id)
    if not latest_row:
        raise ValueError("Division match no longer exists.")

    latest_status = str(latest_row.get("status") or "").lower()
    if latest_status == "completed":
        raise ValueError("This division match has already been submitted.")

    winner_team_id = str(latest_row.get("team_a_id") if score_t1 > score_t2 else latest_row.get("team_b_id"))
    score_json = {
        "team_a": int(score_t1),
        "team_b": int(score_t2),
        "winner_team_id": winner_team_id,
    }

    update_resp = sb_retry(
        lambda: sb_update(
            supabase,
            "division_matches",
            {
                "winner_team_id": winner_team_id,
                "score_json": score_json,
                "status": "completed",
            },
            filters={"club_id": club_id, "id": match_id, "status": latest_row.get("status")},
        )
    )

    if not (getattr(update_resp, "data", None) or []):
        raise ValueError("This division match was already updated by another submission.")

    payload = _build_division_match_payload(
        division_title=division_title,
        division_id=division_id,
        division_match=latest_row,
        team_a=team_a,
        team_b=team_b,
        score_t1=int(score_t1),
        score_t2=int(score_t2),
    )

    submit_match(
        club_id=club_id,
        context_type="tournament",
        context_id=division_id,
        match_payload=payload,
        idempotency_key=_build_division_match_idempotency_key(club_id, division_id, match_id),
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

    st.markdown("### Record Result")

    try:
        division_matches = _load_division_matches(supabase, club_id, division_id)
    except Exception as exc:
        st.error(f"Could not load division matches: {exc}")
        return

    if not division_matches:
        st.info("No bracket matches have been generated for this division yet.")
        return

    completed_count = sum(1 for m in division_matches if str(m.get("status") or "").lower() == "completed")
    st.caption(f"Completed matches: {completed_count}/{len(division_matches)}")

    for match in division_matches:
        match_id = str(match.get("id") or "")
        if not match_id:
            continue

        team_a_id = str(match.get("team_a_id") or "")
        team_b_id = str(match.get("team_b_id") or "")
        team_a = teams_by_id.get(team_a_id, {"id": team_a_id})
        team_b = teams_by_id.get(team_b_id, {"id": team_b_id})
        team_a_label = _team_label(team_a)
        team_b_label = _team_label(team_b)
        match_status = str(match.get("status") or "scheduled").lower()

        with st.container(border=True):
            st.markdown(
                f"**Round {int(match.get('round_number') or 0)} · Match {int(match.get('bracket_position') or 0)}**"
            )
            st.write(f"{team_a_label} vs {team_b_label}")

            if match_status == "completed":
                score_json = match.get("score_json") if isinstance(match.get("score_json"), dict) else {}
                st.success(
                    f"Completed · {int(score_json.get('team_a', 0) or 0)} - {int(score_json.get('team_b', 0) or 0)}"
                )
                continue

            players_ready = all(
                team.get("player1_id") is not None and team.get("player2_id") is not None
                for team in (team_a, team_b)
            )
            if not players_ready:
                st.warning("Both teams must have player1_id and player2_id assigned before recording this result.")
                continue

            score_cols = st.columns(2)
            score_t1 = score_cols[0].number_input(
                f"{team_a_label} score",
                min_value=0,
                max_value=99,
                value=0,
                step=1,
                key=f"division_match_score_a_{match_id}",
            )
            score_t2 = score_cols[1].number_input(
                f"{team_b_label} score",
                min_value=0,
                max_value=99,
                value=0,
                step=1,
                key=f"division_match_score_b_{match_id}",
            )

            if score_t1 == score_t2:
                st.caption("Division matches require a winner (no ties).")

            if st.button(
                "Record Result",
                key=f"division_match_submit_{match_id}",
                type="primary",
                disabled=score_t1 == score_t2,
            ):
                try:
                    _complete_division_match(
                        supabase=supabase,
                        club_id=club_id,
                        division_id=division_id,
                        division_title=str(division.get("title") or "Division"),
                        division_match=match,
                        team_a=team_a,
                        team_b=team_b,
                        score_t1=int(score_t1),
                        score_t2=int(score_t2),
                    )
                    st.success("Division match recorded and submitted through canonical match pipeline.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not record division result: {exc}")
