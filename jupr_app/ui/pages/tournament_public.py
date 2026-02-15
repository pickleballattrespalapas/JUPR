from __future__ import annotations

from collections import defaultdict

import streamlit as st

from jupr_app.data.retry import sb_retry
from jupr_app.ui.layout import page_shell


def _load_tournament(supabase, club_id: str, tournament_id: str) -> dict | None:
    resp = sb_retry(
        lambda: (
            supabase.table("tournaments")
            .select("id,name,status")
            .eq("club_id", club_id)
            .eq("id", tournament_id)
            .limit(1)
            .execute()
        )
    )
    rows = resp.data or []
    return rows[0] if rows else None


def _load_divisions(supabase, club_id: str, tournament_id: str) -> list[dict]:
    resp = sb_retry(
        lambda: (
            supabase.table("tournament_divisions")
            .select("id,title,format,status")
            .eq("club_id", club_id)
            .eq("tournament_id", tournament_id)
            .order("created_at", desc=False)
            .execute()
        )
    )
    return resp.data or []


def _load_division_matches(supabase, club_id: str, division_id: str) -> list[dict]:
    resp = sb_retry(
        lambda: (
            supabase.table("division_matches")
            .select(
                "id,round_number,bracket_position,team_a_id,team_b_id,winner_team_id,score_json,status,created_at"
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


def _load_teams_by_id(supabase, club_id: str, team_ids: list[str]) -> dict[str, dict]:
    if not team_ids:
        return {}
    resp = sb_retry(
        lambda: (
            supabase.table("teams")
            .select("id,name,team_name,title,display_name")
            .eq("club_id", club_id)
            .in_("id", team_ids)
            .execute()
        )
    )
    rows = resp.data or []
    return {str(row.get("id")): row for row in rows if row.get("id")}


def _team_label(team: dict | None, fallback: str) -> str:
    if not team:
        return fallback
    for key in ("name", "team_name", "title", "display_name"):
        value = str(team.get(key) or "").strip()
        if value:
            return value
    return fallback


def _inject_public_bracket_css() -> None:
    st.markdown(
        """
        <style>
          .tpub-grid { display: grid; grid-auto-flow: column; grid-auto-columns: minmax(220px, 1fr); gap: 0.75rem; overflow-x: auto; padding-bottom: 0.4rem; }
          .tpub-round { display: flex; flex-direction: column; gap: 0.65rem; }
          .tpub-round-title { font-weight: 700; font-size: 0.95rem; opacity: 0.86; }
          .tpub-match {
            border: 1px solid rgba(127, 127, 127, 0.28);
            border-radius: 10px;
            padding: 0.55rem 0.65rem;
            background: rgba(127, 127, 127, 0.06);
          }
          .tpub-team { display: flex; justify-content: space-between; gap: 0.5rem; font-size: 0.93rem; }
          .tpub-team + .tpub-team { margin-top: 0.25rem; }
          .tpub-win { font-weight: 700; }
          .tpub-status { margin-top: 0.35rem; font-size: 0.78rem; opacity: 0.7; }

          html[data-theme="dark"] .tpub-match {
            border-color: rgba(255, 255, 255, 0.20);
            background: rgba(255, 255, 255, 0.04);
          }

          @media (prefers-color-scheme: dark) {
            .tpub-match {
              border-color: rgba(255, 255, 255, 0.20);
              background: rgba(255, 255, 255, 0.04);
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render(ctx):
    page_shell("🏆 Tournament Bracket", "Public read-only tournament view.", mode_label="Public")

    supabase = getattr(ctx, "supabase", None)
    club_id = str(getattr(ctx, "club_id", "") or "")
    if supabase is None or not club_id:
        st.error("Missing required application context.")
        st.stop()

    route = str(st.query_params.get("route", "") or "").strip("/")
    if route:
        parts = route.split("/")
        if len(parts) == 2 and parts[0] == "tournament":
            st.query_params["tournament_id"] = parts[1]
        if len(parts) == 4 and parts[0] == "tournament" and parts[2] == "division":
            st.query_params["tournament_id"] = parts[1]
            st.query_params["division_id"] = parts[3]

    tournament_id = str(st.query_params.get("tournament_id", "") or "").strip()
    division_id = str(st.query_params.get("division_id", "") or "").strip()

    if not tournament_id:
        st.info("Tournament link is missing. Open a valid public tournament URL.")
        return

    try:
        tournament = _load_tournament(supabase, club_id, tournament_id)
    except Exception as exc:
        st.error(f"Could not load tournament: {exc}")
        return

    if not tournament:
        st.error("Tournament not found.")
        return

    st.subheader(str(tournament.get("name") or "Tournament"))
    st.caption(f"Status: {str(tournament.get('status') or 'draft').upper()}")

    try:
        divisions = _load_divisions(supabase, club_id, tournament_id)
    except Exception as exc:
        st.error(f"Could not load divisions: {exc}")
        return

    if not divisions:
        st.info("No divisions found for this tournament.")
        return

    divisions_by_id = {str(d.get("id")): d for d in divisions if d.get("id")}
    if division_id not in divisions_by_id:
        division_id = next(iter(divisions_by_id.keys()))
        st.query_params["division_id"] = division_id

    st.markdown("### Divisions")
    div_cols = st.columns(max(1, min(len(divisions), 4)))
    for idx, division in enumerate(divisions):
        with div_cols[idx % len(div_cols)]:
            row_division_id = str(division.get("id") or "")
            label = str(division.get("title") or "Untitled Division")
            selected = row_division_id == division_id
            if st.button(
                f"{'✅ ' if selected else ''}{label}",
                key=f"tournament_public_division_{row_division_id}",
                use_container_width=True,
                type="primary" if selected else "secondary",
            ):
                st.query_params["division_id"] = row_division_id
                st.query_params["route"] = f"tournament/{tournament_id}/division/{row_division_id}"
                st.rerun()

    selected_division = divisions_by_id.get(division_id)
    if not selected_division:
        st.warning("Choose a division to view bracket details.")
        return

    st.markdown("### Bracket")
    st.caption(
        f"{selected_division.get('title') or 'Division'} · "
        f"{str(selected_division.get('format') or 'single_elim').replace('_', ' ').title()} · read-only"
    )

    try:
        matches = _load_division_matches(supabase, club_id, division_id)
    except Exception as exc:
        st.error(f"Could not load bracket matches: {exc}")
        return

    if not matches:
        st.info("Bracket has not been generated for this division yet.")
        return

    team_ids = {
        str(match.get("team_a_id") or "")
        for match in matches
        if match.get("team_a_id")
    } | {
        str(match.get("team_b_id") or "")
        for match in matches
        if match.get("team_b_id")
    }

    try:
        teams_by_id = _load_teams_by_id(supabase, club_id, sorted(team_ids))
    except Exception:
        teams_by_id = {}

    rounds: dict[int, list[dict]] = defaultdict(list)
    for row in matches:
        rounds[int(row.get("round_number") or 0)].append(row)

    _inject_public_bracket_css()
    round_html_blocks: list[str] = []
    for round_num in sorted(rounds.keys()):
        round_matches = sorted(rounds[round_num], key=lambda m: int(m.get("bracket_position") or 0))
        match_html: list[str] = []
        for match in round_matches:
            team_a_id = str(match.get("team_a_id") or "")
            team_b_id = str(match.get("team_b_id") or "")
            winner_id = str(match.get("winner_team_id") or "")
            score_json = match.get("score_json") if isinstance(match.get("score_json"), dict) else {}
            score_a = score_json.get("team_a")
            score_b = score_json.get("team_b")

            team_a_label = _team_label(teams_by_id.get(team_a_id), "TBD")
            team_b_label = _team_label(teams_by_id.get(team_b_id), "TBD")

            team_a_class = "tpub-team tpub-win" if winner_id and winner_id == team_a_id else "tpub-team"
            team_b_class = "tpub-team tpub-win" if winner_id and winner_id == team_b_id else "tpub-team"
            status = str(match.get("status") or "scheduled").lower()

            match_html.append(
                "<div class='tpub-match'>"
                f"<div class='{team_a_class}'><span>{team_a_label}</span><span>{'' if score_a is None else int(score_a)}</span></div>"
                f"<div class='{team_b_class}'><span>{team_b_label}</span><span>{'' if score_b is None else int(score_b)}</span></div>"
                f"<div class='tpub-status'>Match {int(match.get('bracket_position') or 0)} · {status}</div>"
                "</div>"
            )

        round_html_blocks.append(
            "<section class='tpub-round'>"
            f"<div class='tpub-round-title'>Round {round_num}</div>"
            + "".join(match_html)
            + "</section>"
        )

    st.markdown(f"<div class='tpub-grid'>{''.join(round_html_blocks)}</div>", unsafe_allow_html=True)

    st.markdown("### Results")
    completed = [m for m in matches if str(m.get("status") or "").lower() == "completed"]
    if not completed:
        st.caption("No completed results yet.")
        return

    completed = sorted(
        completed,
        key=lambda m: (
            int(m.get("round_number") or 0),
            int(m.get("bracket_position") or 0),
        ),
    )
    for match in completed:
        team_a_id = str(match.get("team_a_id") or "")
        team_b_id = str(match.get("team_b_id") or "")
        winner_id = str(match.get("winner_team_id") or "")
        score_json = match.get("score_json") if isinstance(match.get("score_json"), dict) else {}

        team_a_label = _team_label(teams_by_id.get(team_a_id), "TBD")
        team_b_label = _team_label(teams_by_id.get(team_b_id), "TBD")
        winner_label = _team_label(teams_by_id.get(winner_id), "Winner TBD")

        st.write(
            f"Round {int(match.get('round_number') or 0)} · Match {int(match.get('bracket_position') or 0)} — "
            f"{team_a_label} {int(score_json.get('team_a', 0) or 0)} : "
            f"{int(score_json.get('team_b', 0) or 0)} {team_b_label} · Winner: {winner_label}"
        )
