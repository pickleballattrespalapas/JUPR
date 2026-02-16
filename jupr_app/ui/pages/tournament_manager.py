from __future__ import annotations

from jupr_app.data.sb_write import sb_insert, sb_update, sb_upsert

from collections import Counter

import streamlit as st

from jupr_app.domain.tournaments.bracket_builder import generate_single_elim
from jupr_app.ui.layout import page_shell

DIVISION_FORMATS = ["single_elim", "double_elim", "rr", "pool_to_bracket"]
TAB_OPTIONS = {"overview": "Overview", "divisions": "Divisions"}


def _get_tournaments(supabase, club_id: str) -> list[dict]:
    resp = (
        supabase.table("tournaments")
        .select("id,name,status,created_at")
        .eq("club_id", club_id)
        .order("created_at", desc=True)
        .execute()
    )
    return resp.data or []


def _get_divisions(supabase, club_id: str, tournament_id: str) -> list[dict]:
    resp = (
        supabase.table("tournament_divisions")
        .select("id,title,format,max_teams,status,created_at")
        .eq("club_id", club_id)
        .eq("tournament_id", tournament_id)
        .order("created_at", desc=False)
        .execute()
    )
    return resp.data or []


def _get_entry_counts(supabase, club_id: str, division_ids: list[str]) -> dict[str, int]:
    if not division_ids:
        return {}
    resp = (
        supabase.table("division_entries")
        .select("division_id")
        .eq("club_id", club_id)
        .in_("division_id", division_ids)
        .execute()
    )
    rows = resp.data or []
    return dict(Counter(str(row.get("division_id")) for row in rows if row.get("division_id")))


def _insert_division(supabase, club_id: str, tournament_id: str, title: str, fmt: str, max_teams: int | None) -> None:
    payload = {
        "club_id": club_id,
        "tournament_id": tournament_id,
        "title": title,
        "format": fmt,
        "status": "draft",
    }
    if max_teams is not None:
        payload["max_teams"] = int(max_teams)
    sb_insert(supabase, "tournament_divisions", payload)


def _render_division_modal(supabase, club_id: str, tournament_id: str) -> None:
    @st.dialog("Add Division")
    def add_division_dialog() -> None:
        with st.form("add_division_form"):
            title = st.text_input("Division Title", max_chars=120)
            fmt = st.selectbox("Format", options=DIVISION_FORMATS, index=0)
            max_teams_raw = st.text_input("Max Teams (optional)", value="")
            submitted = st.form_submit_button("Create Division", use_container_width=True)

        if not submitted:
            return

        clean_title = title.strip()
        if not clean_title:
            st.error("Division Title is required.")
            return

        clean_max: int | None = None
        raw = max_teams_raw.strip()
        if raw:
            if not raw.isdigit() or int(raw) <= 0:
                st.error("Max Teams must be a positive whole number.")
                return
            clean_max = int(raw)

        try:
            _insert_division(supabase, club_id, tournament_id, clean_title, fmt, clean_max)
            st.success("Division created.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not create division: {exc}")

    add_division_dialog()


def _render_divisions_tab(supabase, club_id: str, tournament: dict) -> None:
    tournament_id = str(tournament["id"])

    cta_col, _ = st.columns([1, 3])
    with cta_col:
        if st.button("+ Add Division", use_container_width=True, key="tm_add_division_btn"):
            st.session_state["tm_show_add_division_dialog"] = True

    if st.session_state.get("tm_show_add_division_dialog", False):
        st.session_state["tm_show_add_division_dialog"] = False
        _render_division_modal(supabase, club_id, tournament_id)

    try:
        divisions = _get_divisions(supabase, club_id, tournament_id)
    except Exception as exc:
        st.error(f"Could not load divisions: {exc}")
        return

    if not divisions:
        st.info("No divisions yet. Create one to get started.")
        return

    division_ids = [str(d["id"]) for d in divisions if d.get("id")]
    try:
        counts = _get_entry_counts(supabase, club_id, division_ids)
    except Exception as exc:
        st.error(f"Could not load division entry counts: {exc}")
        counts = {}

    for row in divisions:
        division_id = str(row.get("id", ""))
        with st.container(border=True):
            left, right = st.columns([3, 2])
            with left:
                st.subheader(str(row.get("title") or "Untitled Division"))
                st.caption(f"Format: {row.get('format') or '—'}")
                st.caption(f"Team count: {counts.get(division_id, 0)}")
                st.caption(f"Status: {row.get('status') or 'draft'}")
            with right:
                if st.button(
                    "Manage Entries",
                    key=f"manage_entries_{division_id}",
                    use_container_width=True,
                    type="secondary",
                ):
                    st.query_params["page"] = "division_manager"
                    st.query_params["route"] = f"tournament/{tournament_id}/division/{division_id}"
                    st.query_params["tournament_id"] = tournament_id
                    st.query_params["division_id"] = division_id
                    st.rerun()
                generate_clicked = st.button(
                    "Generate Bracket",
                    key=f"generate_bracket_{division_id}",
                    use_container_width=True,
                    type="primary",
                    disabled=str(row.get("format") or "") != "single_elim",
                    help="Only available for single elimination divisions.",
                )
                if generate_clicked:
                    try:
                        result = generate_single_elim(supabase, division_id=division_id, club_id=club_id)
                        st.success(
                            "Bracket generated "
                            f"({result['match_count']} matches, {result['entry_count']} teams, size {result['bracket_size']})."
                        )
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"Could not generate bracket: {exc}")


def render(ctx):
    mode_label = "Public" if bool(getattr(ctx, "public_mode", False)) else "Admin"
    page_shell("🏆 Tournament Manager", "Admin-only tournament operations.", mode_label=mode_label)

    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    supabase = getattr(ctx, "supabase", None)
    club_id = getattr(ctx, "club_id", None)
    if supabase is None or club_id is None:
        st.error("Missing required application context.")
        st.stop()

    requested_tab = str(st.query_params.get("tm_tab", "divisions"))
    if requested_tab not in TAB_OPTIONS:
        requested_tab = "divisions"

    selected_tab = st.radio(
        "Tournament Manager Sections",
        options=list(TAB_OPTIONS.keys()),
        index=list(TAB_OPTIONS.keys()).index(requested_tab),
        format_func=lambda key: TAB_OPTIONS[key],
        horizontal=True,
        label_visibility="collapsed",
        key="tm_tab_selector",
    )
    st.query_params["tm_tab"] = selected_tab

    tournaments = _get_tournaments(supabase, str(club_id))
    if not tournaments:
        st.info("No tournaments found for this club yet.")
        return

    selected_tournament_id = str(st.query_params.get("tournament_id", str(tournaments[0]["id"])))
    tournament_ids = [str(t["id"]) for t in tournaments]
    if selected_tournament_id not in tournament_ids:
        selected_tournament_id = tournament_ids[0]

    current_idx = tournament_ids.index(selected_tournament_id)
    tournament_label_options = [f"{t['name']} ({t.get('status', 'draft')})" for t in tournaments]
    selected_idx = st.selectbox(
        "Tournament",
        options=list(range(len(tournaments))),
        index=current_idx,
        format_func=lambda i: tournament_label_options[i],
        key="tm_tournament_selector",
    )

    tournament = tournaments[int(selected_idx)]
    st.query_params["tournament_id"] = str(tournament["id"])

    if selected_tab == "divisions":
        _render_divisions_tab(supabase, str(club_id), tournament)
        return

    st.info("Overview is coming soon.")
