from __future__ import annotations

from datetime import datetime, timezone
import re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from jupr_app.domain.live_beta_engine import (
    SUPPORTED_RR_FORMATS,
    SUPPORTED_TOURNAMENT_TEAM_COUNTS,
    build_league_movement,
    create_league_event,
    create_round_robin_event,
    create_tournament_event,
    current_league_round,
    export_event_json,
    is_league_round_complete,
    league_aggregate_standings,
    league_round_summary,
    mark_tournament_matches_saved,
    match_payloads_from_current_league_round,
    match_payloads_from_rr,
    normalize_name,
    resolve_payload_player_ids,
    round_robin_standings,
    set_pending_assignment,
    start_next_league_round,
    standings_csv_rows,
    suggest_exact_league_court_sizes,
    tournament_bracket_rows,
    tournament_champion,
    tournament_completed_match_payloads,
    update_league_score,
    update_round_robin_score,
    update_tournament_score,
    validate_assignments,
)
from jupr_app.domain.match_processing import process_matches
from jupr_app.ui.layout import page_shell


LIVE_STATE_KEY = "jupr_live_beta_state"
SUPPORTED_TYPES = {
    "Round Robin": "round_robin",
    "League / Ladder": "league",
    "Tournament": "tournament",
}
RUN_MODES = ["Quick", "Official"]


def _default_state() -> dict:
    return {
        "event": None,
        "type_label": "Round Robin",
        "run_mode": "Quick",
        "event_name": "Saturday Event",
        "participant_count": 8,
        "participant_text": "",
        "league_rounds": 3,
        "official_league": "",
        "official_week_tag": "Week 1",
        "last_saved_rounds": [],
    }


def _state() -> dict:
    state = st.session_state.setdefault(LIVE_STATE_KEY, _default_state())
    for key, value in _default_state().items():
        state.setdefault(key, value)
    return state



def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .jupr-live-card {
            border: 1px solid rgba(47,111,237,0.12);
            border-radius: 18px;
            padding: 1rem 1rem 0.75rem;
            background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(245,249,255,0.98));
            box-shadow: 0 8px 28px rgba(17, 24, 39, 0.06);
            margin-bottom: 1rem;
        }
        .jupr-live-kicker { font-size: 0.8rem; font-weight: 700; color: #2F6FED; text-transform: uppercase; letter-spacing: 0.08em; }
        .jupr-live-score-shell {
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            background: #fff;
            margin-bottom: 0.9rem;
        }
        .jupr-live-team { font-weight: 700; font-size: 1rem; }
        .jupr-live-vs { text-align: center; font-size: 0.9rem; font-weight: 700; color: #64748b; margin-top: 1.9rem; }
        .jupr-live-actions button[kind="primary"] {
            min-height: 3rem;
            font-weight: 700;
        }
        .jupr-live-pill {
            display:inline-block; padding:0.35rem 0.7rem; border-radius:999px; background:#eaf2ff; color:#2F6FED; font-weight:600; margin-right:0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _inject_score_keyboard_nav() -> None:
    components.html(
        """
        <script>
        const doc = window.parent.document;
        const inputs = Array.from(doc.querySelectorAll('input[aria-label^="JUPR Live Score"]'));
        inputs.forEach((input, idx) => {
          if (input.dataset.juprLiveNavBound === '1') return;
          input.dataset.juprLiveNavBound = '1';
          input.addEventListener('keydown', (event) => {
            if (event.key !== 'Enter') return;
            event.preventDefault();
            const next = inputs[idx + 1];
            if (next) {
              next.focus();
              if (typeof next.select === 'function') next.select();
            }
          });
        });
        </script>
        """,
        height=0,
    )


def _participant_lines(value: str) -> list[str]:
    return [normalize_name(x) for x in str(value or "").replace(",", "\n").splitlines() if normalize_name(x)]


def _team_entry_lines(value: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for idx, raw_line in enumerate(str(value or "").splitlines(), start=1):
        line = normalize_name(raw_line)
        if not line:
            continue
        players = [normalize_name(part) for part in re.split(r"\s*(?:/|&|\+)\s*", line) if normalize_name(part)]
        if len(players) != 2:
            raise ValueError(
                f"Team line {idx} must contain exactly two player names separated by '/' (example: Amy / Brooke)."
            )
        entries.append(
            {
                "name": " / ".join(players),
                "player1_name": players[0],
                "player2_name": players[1],
            }
        )
    return entries


def _active_league_options(df_meta: pd.DataFrame | None) -> list[str]:
    if df_meta is None or df_meta.empty or "league_name" not in df_meta.columns:
        return []
    meta = df_meta.copy()
    meta["league_name"] = meta["league_name"].fillna("").astype(str).str.strip()
    if "is_active" in meta.columns:
        meta = meta[meta["is_active"] == True]
    return sorted([x for x in meta["league_name"].tolist() if x])


def _week_tag_options() -> list[str]:
    return [f"Week {i}" for i in range(1, 13)] + ["Playoffs", "Finals", "Event"]


def _resolved_ids_for_official(names: list[str], name_to_id: dict[str, int]) -> tuple[dict[str, int], list[str]]:
    normalized_map = {normalize_name(k): int(v) for k, v in (name_to_id or {}).items() if normalize_name(k)}
    resolved: dict[str, int] = {}
    missing: list[str] = []
    for name in names:
        pid = normalized_map.get(normalize_name(name))
        if pid is None:
            missing.append(name)
        else:
            resolved[name] = int(pid)
    return resolved, missing


def _official_context_ui(ctx, state: dict) -> tuple[dict, bool]:
    disabled = not bool(getattr(ctx, "admin_logged_in", False))
    opts = _active_league_options(getattr(ctx, "df_meta", None))
    if not opts:
        opts = ["Default"]
    if state["official_league"] not in opts:
        state["official_league"] = opts[0]
    st.markdown("#### Official save context")
    c1, c2 = st.columns(2)
    state["official_league"] = c1.selectbox(
        "League",
        opts,
        index=opts.index(state["official_league"]),
        disabled=disabled,
        key="jupr_live_official_league",
    )
    week_options = _week_tag_options()
    if state["official_week_tag"] not in week_options:
        state["official_week_tag"] = week_options[0]
    state["official_week_tag"] = c2.selectbox(
        "Week / Session",
        week_options,
        index=week_options.index(state["official_week_tag"]),
        disabled=disabled,
        key="jupr_live_official_week",
    )
    if disabled:
        st.warning("Official mode requires admin login before event creation or save.")
    return {
        "league": state["official_league"],
        "week_tag": state["official_week_tag"],
        "match_type": "Live Match",
        "is_popup": False,
    }, not disabled


def _render_setup(ctx, state: dict) -> None:
    st.markdown('<div class="jupr-live-card">', unsafe_allow_html=True)
    st.markdown('<div class="jupr-live-kicker">Setup</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.1, 1, 1])
    type_label = c1.radio(
        "Event type",
        list(SUPPORTED_TYPES.keys()),
        index=list(SUPPORTED_TYPES.keys()).index(state["type_label"]),
        horizontal=True,
        key="jupr_live_type",
    )
    state["type_label"] = type_label
    run_mode = c2.radio(
        "Run mode",
        RUN_MODES,
        index=RUN_MODES.index(state["run_mode"]),
        horizontal=True,
        key="jupr_live_run_mode",
    )
    state["run_mode"] = run_mode
    default_count = int(state["participant_count"])
    if state["type_label"] == "Round Robin":
        participant_count = c3.selectbox(
            "Count",
            SUPPORTED_RR_FORMATS,
            index=SUPPORTED_RR_FORMATS.index(default_count) if default_count in SUPPORTED_RR_FORMATS else 2,
            key="jupr_live_count_rr",
        )
    elif state["type_label"] == "Tournament":
        participant_count = c3.selectbox(
            "Teams",
            SUPPORTED_TOURNAMENT_TEAM_COUNTS,
            index=SUPPORTED_TOURNAMENT_TEAM_COUNTS.index(default_count)
            if default_count in SUPPORTED_TOURNAMENT_TEAM_COUNTS
            else 0,
            key="jupr_live_count_tn",
        )
    else:
        participant_count = c3.number_input(
            "Count",
            min_value=4,
            max_value=40,
            value=max(4, default_count),
            step=1,
            key="jupr_live_count_lg",
        )
    state["participant_count"] = int(participant_count)
    state["event_name"] = st.text_input("Event name", value=state["event_name"], key="jupr_live_event_name")
    if state["type_label"] == "Round Robin":
        help_text = "Enter one participant per line. JUPR Live Beta uses current JUPR doubles schedules for 4, 5, 6, 8, 9, 12, and 14 participants."
        placeholder = "Amy\nBrooke\nChris\nDana"
    elif state["type_label"] == "Tournament":
        help_text = "Enter one doubles team per line using 'Player 1 / Player 2'. Tournament brackets support 4 to 8 fixed teams."
        placeholder = "Amy / Brooke\nChris / Dana\nEli / Finn\nGia / Hugo"
    else:
        exact_sizes = suggest_exact_league_court_sizes(int(state["participant_count"]))
        if exact_sizes:
            help_text = f"League / Ladder will start with courts sized {', '.join(map(str, exact_sizes))}."
        else:
            help_text = "League / Ladder currently requires an exact 4-player / 5-player court fit."
        placeholder = "Amy\nBrooke\nChris\nDana"
    st.caption(help_text)
    state["participant_text"] = st.text_area(
        "Names or roster entry",
        value=state["participant_text"],
        height=180,
        placeholder=placeholder,
        key="jupr_live_participants",
    )
    if state["type_label"] == "League / Ladder":
        state["league_rounds"] = int(
            st.number_input(
                "Total rounds",
                min_value=1,
                max_value=12,
                value=int(state["league_rounds"]),
                step=1,
                key="jupr_live_total_rounds",
            )
        )
    can_create = True
    official_context = {}
    if state["run_mode"] == "Official":
        official_context, can_create = _official_context_ui(ctx, state)
    participant_names = _participant_lines(state["participant_text"])
    team_entries: list[dict[str, str]] = []
    team_parse_error: str | None = None
    if state["type_label"] == "Tournament":
        try:
            team_entries = _team_entry_lines(state["participant_text"])
        except ValueError as exc:
            team_parse_error = str(exc)
        if team_parse_error:
            st.info(team_parse_error)
            can_create = False
        elif team_entries and len(team_entries) != int(state["participant_count"]):
            st.info(f"You entered {len(team_entries)} team(s); count is set to {int(state['participant_count'])}.")
            can_create = False
    elif participant_names and len(participant_names) != int(state["participant_count"]):
        st.info(f"You entered {len(participant_names)} name(s); count is set to {int(state['participant_count'])}.")
        can_create = False
    action_cols = st.columns([1, 1, 3])
    if action_cols[0].button("Create event", type="primary", disabled=not can_create, key="jupr_live_create_btn"):
        try:
            if state["type_label"] == "Round Robin":
                resolved_ids = None
                if state["run_mode"] == "Official":
                    resolved_ids, missing = _resolved_ids_for_official(participant_names, getattr(ctx, "name_to_id", {}))
                    if missing:
                        raise ValueError("Official mode could not resolve: " + ", ".join(missing))
                state["event"] = create_round_robin_event(
                    name=state["event_name"],
                    participant_names=participant_names,
                    resolved_ids=resolved_ids,
                    official_context=official_context,
                )
            elif state["type_label"] == "Tournament":
                if state["run_mode"] == "Official":
                    resolved_team_entries: list[dict[str, str | int]] = []
                    missing_players: list[str] = []
                    name_to_id = getattr(ctx, "name_to_id", {})
                    for team in team_entries:
                        resolved_ids, missing = _resolved_ids_for_official(
                            [str(team["player1_name"]), str(team["player2_name"])],
                            name_to_id,
                        )
                        if missing:
                            missing_players.extend(missing)
                            continue
                        resolved_team_entries.append(
                            {
                                **team,
                                "player1_id": int(resolved_ids[str(team["player1_name"])]),
                                "player2_id": int(resolved_ids[str(team["player2_name"])]),
                            }
                        )
                    if missing_players:
                        raise ValueError("Official mode could not resolve: " + ", ".join(sorted(dict.fromkeys(missing_players))))
                    state["event"] = create_tournament_event(
                        name=state["event_name"],
                        team_entries=resolved_team_entries,
                        official_context=official_context,
                    )
                else:
                    state["event"] = create_tournament_event(
                        name=state["event_name"],
                        team_entries=team_entries,
                        official_context=official_context,
                    )
            else:
                exact_sizes = suggest_exact_league_court_sizes(len(participant_names))
                if not exact_sizes:
                    raise ValueError("League / Ladder requires an exact 4-player / 5-player court fit.")
                resolved_ids = None
                if state["run_mode"] == "Official":
                    resolved_ids, missing = _resolved_ids_for_official(participant_names, getattr(ctx, "name_to_id", {}))
                    if missing:
                        raise ValueError("Official mode could not resolve: " + ", ".join(missing))
                state["event"] = create_league_event(
                    name=state["event_name"],
                    participant_names=participant_names,
                    total_rounds=int(state["league_rounds"]),
                    resolved_ids=resolved_ids,
                    court_sizes=exact_sizes,
                    official_context=official_context,
                )
            state["last_saved_rounds"] = []
            st.success("Event created.")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if action_cols[1].button("Reset", key="jupr_live_reset_btn"):
        st.session_state[LIVE_STATE_KEY] = _default_state()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def _participant_name_map(event: dict) -> dict[str, str]:
    return {str(p["id"]): str(p.get("name", p["id"])) for p in event.get("participants") or []}


def _team_label(event: dict, ids: list[str]) -> str:
    name_map = _participant_name_map(event)
    return " / ".join(name_map.get(str(pid), str(pid)) for pid in (ids or []))


def _render_event_exports(event: dict, standings: list[dict]) -> None:
    c1, c2 = st.columns([1, 1])
    c1.download_button(
        "Download event JSON",
        data=export_event_json(event).encode("utf-8"),
        file_name=f"{normalize_name(event.get('name', 'jupr-live')).lower().replace(' ', '-')}.json",
        mime="application/json",
        key=f"jupr_live_export_{event.get('type')}",
    )
    csv_df = pd.DataFrame(standings_csv_rows(standings))
    c2.download_button(
        "Download standings CSV",
        data=csv_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{normalize_name(event.get('name', 'jupr-live')).lower().replace(' ', '-')}-standings.csv",
        mime="text/csv",
        key=f"jupr_live_csv_{event.get('type')}",
    )
    st.caption("Printing is best handled with your browser’s print dialog after expanding the sections you want to keep.")


def _render_event_csv_export(event: dict, rows: list[dict], *, label: str, suffix: str) -> None:
    if not rows:
        return
    st.download_button(
        label,
        data=pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
        file_name=f"{normalize_name(event.get('name', 'jupr-live')).lower().replace(' ', '-')}-{suffix}.csv",
        mime="text/csv",
        key=f"jupr_live_csv_{event.get('type')}_{suffix}",
    )


def _render_standings_table(standings: list[dict], title: str) -> None:
    st.markdown(f"#### {title}")
    df = pd.DataFrame(standings_csv_rows(standings))
    if df.empty:
        st.info("No standings yet.")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _official_base_payload(state: dict) -> dict:
    return {
        "date": _utc_now_iso(),
        "league": state.get("official_league") or "",
        "match_type": "Live Match",
        "week_tag": state.get("official_week_tag") or "",
        "is_popup": False,
    }


def _save_rr_official(ctx, state: dict, event: dict) -> None:
    if "rr" in set(state.get("last_saved_rounds") or []):
        st.info("Official round robin results were already saved in this session.")
        return
    payloads = resolve_payload_player_ids(event, match_payloads_from_rr(event))
    payloads = [{**payload, **_official_base_payload(state)} for payload in payloads]
    if not payloads:
        st.warning("Enter at least one scored match before saving officially.")
        return
    res = process_matches(
        payloads,
        supabase=ctx.supabase,
        club_id=str(ctx.club_id),
        name_to_id=ctx.name_to_id,
        df_players_all=ctx.df_players_all,
        df_leagues=ctx.df_leagues,
        df_meta=ctx.df_meta,
    )
    state["last_saved_rounds"] = ["rr"]
    st.session_state["force_data_refresh"] = True
    st.success(f"Official results saved ({res['inserted']} matches).")


def _save_league_round_official(ctx, state: dict, event: dict) -> bool:
    current_round_number = int(event.get("currentRoundNumber") or 1)
    saved_rounds = set(state.get("last_saved_rounds") or [])
    if current_round_number in saved_rounds:
        return True
    payloads = resolve_payload_player_ids(event, match_payloads_from_current_league_round(event))
    payloads = [{**payload, **_official_base_payload(state)} for payload in payloads]
    if not payloads:
        st.warning("Enter at least one scored match before finalizing this round.")
        return False
    res = process_matches(
        payloads,
        supabase=ctx.supabase,
        club_id=str(ctx.club_id),
        name_to_id=ctx.name_to_id,
        df_players_all=ctx.df_players_all,
        df_leagues=ctx.df_leagues,
        df_meta=ctx.df_meta,
    )
    saved_rounds.add(current_round_number)
    state["last_saved_rounds"] = sorted(saved_rounds)
    st.session_state["force_data_refresh"] = True
    st.success(f"Official round {current_round_number} saved ({res['inserted']} matches).")
    return True


def _save_tournament_official(ctx, state: dict, event: dict) -> None:
    payloads = tournament_completed_match_payloads(event, unsaved_only=True)
    if not payloads:
        st.info("No newly completed tournament matches to save yet.")
        return
    res = process_matches(
        payloads,
        supabase=ctx.supabase,
        club_id=str(ctx.club_id),
        name_to_id=ctx.name_to_id,
        df_players_all=ctx.df_players_all,
        df_leagues=ctx.df_leagues,
        df_meta=ctx.df_meta,
    )
    mark_tournament_matches_saved(event, payloads)
    st.session_state["force_data_refresh"] = True
    st.success(f"Official tournament results saved ({res['inserted']} matches).")


def _render_rr_scoring(ctx, state: dict, event: dict) -> None:
    standings = round_robin_standings(event)
    leader = standings[0]["name"] if standings else "—"
    c_left, c_right = st.columns([1.5, 1])
    with c_left:
        st.markdown('<div class="jupr-live-card">', unsafe_allow_html=True)
        st.markdown(f"<div class='jupr-live-kicker'>Live scoring</div><h3 style='margin-top:0.25rem'>{event['name']}</h3>", unsafe_allow_html=True)
        st.markdown(f"<span class='jupr-live-pill'>{state['run_mode']}</span><span class='jupr-live-pill'>Leader: {leader}</span>", unsafe_allow_html=True)
        with st.form("jupr_live_rr_form"):
            for round_data in event.get("rounds") or []:
                st.markdown(f"#### Round {int(round_data.get('number') or 0)}")
                for match in round_data.get("matches") or []:
                    st.markdown('<div class="jupr-live-score-shell">', unsafe_allow_html=True)
                    cols = st.columns([3.6, 1.1, 0.6, 1.1, 3.6])
                    cols[0].markdown(f"<div class='jupr-live-team'>{_team_label(event, match.get('teamA') or [])}</div>", unsafe_allow_html=True)
                    score_a = cols[1].number_input(
                        f"JUPR Live Score {match['id']} A",
                        min_value=0,
                        max_value=99,
                        value=int(match.get("scoreA") or 0),
                        step=1,
                        key=f"rr_{match['id']}_a",
                    )
                    cols[2].markdown("<div class='jupr-live-vs'>vs</div>", unsafe_allow_html=True)
                    score_b = cols[3].number_input(
                        f"JUPR Live Score {match['id']} B",
                        min_value=0,
                        max_value=99,
                        value=int(match.get("scoreB") or 0),
                        step=1,
                        key=f"rr_{match['id']}_b",
                    )
                    cols[4].markdown(f"<div class='jupr-live-team'>{_team_label(event, match.get('teamB') or [])}</div>", unsafe_allow_html=True)
                    st.caption(str(match.get("desc") or ""))
                    st.markdown("</div>", unsafe_allow_html=True)
                st.divider()
            submit_label = "Save official results" if state["run_mode"] == "Official" else "Update live standings"
            submitted = st.form_submit_button(submit_label, type="primary")
        if submitted:
            for round_data in event.get("rounds") or []:
                for match in round_data.get("matches") or []:
                    a_val = int(st.session_state.get(f"rr_{match['id']}_a", 0) or 0)
                    b_val = int(st.session_state.get(f"rr_{match['id']}_b", 0) or 0)
                    if a_val == 0 and b_val == 0:
                        update_round_robin_score(event, match["id"], None, None)
                    else:
                        update_round_robin_score(event, match["id"], a_val, b_val)
            if state["run_mode"] == "Official":
                if not bool(getattr(ctx, "admin_logged_in", False)):
                    st.error("Admin login required to save official results.")
                else:
                    _save_rr_official(ctx, state, event)
            st.rerun()
        _render_event_exports(event, standings)
        st.markdown("</div>", unsafe_allow_html=True)
    with c_right:
        _render_standings_table(standings, "Live standings")


def _movement_rows_to_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Player": row["name"],
                "Court": int(row["currentCourt"]),
                "Rank": int(row["currentRank"]),
                "W": int(row["wins"]),
                "L": int(row["losses"]),
                "T": int(row["ties"]),
                "PF": int(row["pointsFor"]),
                "PA": int(row["pointsAgainst"]),
                "Diff": int(row["differential"]),
                "Next Court": int(row["proposedCourt"]),
            }
            for row in rows
        ]
    )


def _render_league_scoring(ctx, state: dict, event: dict) -> None:
    round_data = current_league_round(event)
    if round_data is None:
        st.error("Current round not found.")
        return
    summary = league_round_summary(event)
    aggregate = league_aggregate_standings(event)
    movement = build_league_movement(event)
    if not event.get("pendingAssignments"):
        event["pendingAssignments"] = dict(movement["assignments"])
    for row in movement["rows"]:
        row["proposedCourt"] = int(event["pendingAssignments"].get(str(row["participantId"]), row["proposedCourt"]))
    current_round_number = int(event.get("currentRoundNumber") or 1)
    total_rounds = int(event.get("totalRounds") or 1)
    c_left, c_right = st.columns([1.5, 1])
    with c_left:
        st.markdown('<div class="jupr-live-card">', unsafe_allow_html=True)
        st.markdown(
            f"<div class='jupr-live-kicker'>Round {current_round_number} of {total_rounds}</div><h3 style='margin-top:0.25rem'>{event['name']}</h3>",
            unsafe_allow_html=True,
        )
        with st.form(f"jupr_live_league_form_r{current_round_number}"):
            for court in round_data.get("courts") or []:
                st.markdown(f"#### Court {int(court.get('courtNumber') or 0)}")
                for mini_round in court.get("miniRounds") or []:
                    bye_pid = mini_round.get("byeParticipantId")
                    label = f"Mini-round {int(mini_round.get('number') or 0)}"
                    if bye_pid:
                        label += f" • Bye: {_participant_name_map(event).get(str(bye_pid), str(bye_pid))}"
                    st.caption(label)
                    for match in mini_round.get("matches") or []:
                        st.markdown('<div class="jupr-live-score-shell">', unsafe_allow_html=True)
                        cols = st.columns([3.6, 1.1, 0.6, 1.1, 3.6])
                        cols[0].markdown(f"<div class='jupr-live-team'>{_team_label(event, match.get('teamA') or [])}</div>", unsafe_allow_html=True)
                        score_a = cols[1].number_input(
                            f"JUPR Live Score {match['id']} A",
                            min_value=0,
                            max_value=99,
                            value=int(match.get("scoreA") or 0),
                            step=1,
                            key=f"lg_{match['id']}_a",
                        )
                        cols[2].markdown("<div class='jupr-live-vs'>vs</div>", unsafe_allow_html=True)
                        score_b = cols[3].number_input(
                            f"JUPR Live Score {match['id']} B",
                            min_value=0,
                            max_value=99,
                            value=int(match.get("scoreB") or 0),
                            step=1,
                            key=f"lg_{match['id']}_b",
                        )
                        cols[4].markdown(f"<div class='jupr-live-team'>{_team_label(event, match.get('teamB') or [])}</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Update round & movement", type="primary")
        if submitted:
            for court in round_data.get("courts") or []:
                for mini_round in court.get("miniRounds") or []:
                    for match in mini_round.get("matches") or []:
                        a_val = int(st.session_state.get(f"lg_{match['id']}_a", 0) or 0)
                        b_val = int(st.session_state.get(f"lg_{match['id']}_b", 0) or 0)
                        if a_val == 0 and b_val == 0:
                            update_league_score(event, match["id"], None, None)
                        else:
                            update_league_score(event, match["id"], a_val, b_val)
            event["pendingAssignments"] = None
            st.rerun()
        st.divider()
        st.markdown("#### Ladder movement preview")
        if is_league_round_complete(event):
            movement = build_league_movement(event)
            for row in movement["rows"]:
                current_assignment = int(event["pendingAssignments"].get(str(row["participantId"]), row["proposedCourt"]))
                c1, c2 = st.columns([4, 1.2])
                c1.write(
                    f"**{row['name']}** — Court {int(row['currentCourt'])} Rank {int(row['currentRank'])} | W {int(row['wins'])} / L {int(row['losses'])} / T {int(row['ties'])} | Diff {int(row['differential'])}"
                )
                selected = c2.selectbox(
                    f"Next court {row['participantId']}",
                    list(range(1, len(event.get('courtSizes') or []) + 1)),
                    index=list(range(1, len(event.get('courtSizes') or []) + 1)).index(current_assignment),
                    key=f"move_{row['participantId']}",
                    label_visibility="collapsed",
                )
                set_pending_assignment(event, str(row["participantId"]), int(selected))
            validation = validate_assignments(event, dict(event.get("pendingAssignments") or {}))
            if validation["ok"]:
                st.success("Court counts are valid. You can finalize this round.")
            else:
                st.error(" ".join(validation["errors"]))
            action_cols = st.columns([1.2, 1.2, 3])
            next_label = "Finish league night" if current_round_number >= total_rounds else "Finalize round & start next"
            if action_cols[0].button(next_label, type="primary", key=f"jupr_live_next_round_{current_round_number}"):
                try:
                    if state["run_mode"] == "Official":
                        if not bool(getattr(ctx, "admin_logged_in", False)):
                            raise ValueError("Admin login required to save official round results.")
                        if not _save_league_round_official(ctx, state, event):
                            st.stop()
                    if current_round_number < total_rounds:
                        start_next_league_round(event)
                        st.success(f"Round {current_round_number} finalized.")
                    else:
                        st.success("League night complete.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
            if action_cols[1].button("Reset to auto", key=f"jupr_live_reset_move_{current_round_number}"):
                event["pendingAssignments"] = dict(build_league_movement(event)["assignments"])
                st.rerun()
            st.dataframe(_movement_rows_to_df(build_league_movement(event)["rows"]), use_container_width=True, hide_index=True)
        else:
            st.info("Complete every score in the current round to unlock movement preview.")
        _render_event_exports(event, aggregate)
        st.markdown("</div>", unsafe_allow_html=True)
    with c_right:
        for court_info in summary:
            _render_standings_table(court_info.get("standings") or [], f"Court {int(court_info['courtNumber'])} standings")
        _render_standings_table(aggregate, "Cumulative ladder standings")


def _render_tournament_scoring(ctx, state: dict, event: dict) -> None:
    bracket_rows = tournament_bracket_rows(event)
    champion_id = tournament_champion(event)
    team_map = {str(team["id"]): team for team in (event.get("teams") or [])}
    champion_name = team_map.get(str(champion_id), {}).get("name") if champion_id else None
    c_left, c_right = st.columns([1.5, 1])
    with c_left:
        st.markdown('<div class="jupr-live-card">', unsafe_allow_html=True)
        pills = [f"<span class='jupr-live-pill'>{state['run_mode']}</span>"]
        pills.append(f"<span class='jupr-live-pill'>Champion: {champion_name or 'Pending'}</span>")
        st.markdown(
            f"<div class='jupr-live-kicker'>Tournament bracket</div><h3 style='margin-top:0.25rem'>{event['name']}</h3>{''.join(pills)}",
            unsafe_allow_html=True,
        )
        with st.form("jupr_live_tournament_form"):
            for round_data in event.get("rounds") or []:
                st.markdown(f"#### Round {int(round_data.get('number') or 0)}")
                for match in round_data.get("matches") or []:
                    match_id = f"r{int(round_data.get('number') or 0)}-s{int(match.get('slot') or 0)}"
                    st.markdown('<div class="jupr-live-score-shell">', unsafe_allow_html=True)
                    cols = st.columns([3.6, 1.1, 0.6, 1.1, 3.6])
                    cols[0].markdown(
                        f"<div class='jupr-live-team'>{team_map.get(str(match.get('participantAId')), {}).get('name', 'TBD')}</div>",
                        unsafe_allow_html=True,
                    )
                    score_a = cols[1].number_input(
                        f"JUPR Live Score {match_id} A",
                        min_value=0,
                        max_value=99,
                        value=int(match.get("scoreA") or 0),
                        step=1,
                        key=f"tn_{match_id}_a",
                        disabled=not (match.get("participantAId") and match.get("participantBId")),
                    )
                    cols[2].markdown("<div class='jupr-live-vs'>vs</div>", unsafe_allow_html=True)
                    score_b = cols[3].number_input(
                        f"JUPR Live Score {match_id} B",
                        min_value=0,
                        max_value=99,
                        value=int(match.get("scoreB") or 0),
                        step=1,
                        key=f"tn_{match_id}_b",
                        disabled=not (match.get("participantAId") and match.get("participantBId")),
                    )
                    cols[4].markdown(
                        f"<div class='jupr-live-team'>{team_map.get(str(match.get('participantBId')), {}).get('name', 'TBD')}</div>",
                        unsafe_allow_html=True,
                    )
                    winner = team_map.get(str(match.get("winnerId")), {}).get("name")
                    st.caption(f"{match.get('name')} • Winner: {winner or 'Pending'}")
                    st.markdown("</div>", unsafe_allow_html=True)
                st.divider()
            submitted = st.form_submit_button(
                "Save official results" if state["run_mode"] == "Official" else "Update bracket",
                type="primary",
            )
        if submitted:
            for round_data in event.get("rounds") or []:
                for match in round_data.get("matches") or []:
                    round_number = int(round_data.get("number") or 0)
                    slot = int(match.get("slot") or 0)
                    match_id = f"r{round_number}-s{slot}"
                    if not (match.get("participantAId") and match.get("participantBId")):
                        update_tournament_score(event, round_number, slot, None, None)
                        continue
                    a_val = int(st.session_state.get(f"tn_{match_id}_a", 0) or 0)
                    b_val = int(st.session_state.get(f"tn_{match_id}_b", 0) or 0)
                    if a_val == 0 and b_val == 0:
                        update_tournament_score(event, round_number, slot, None, None)
                    else:
                        update_tournament_score(event, round_number, slot, a_val, b_val)
            if state["run_mode"] == "Official":
                if not bool(getattr(ctx, "admin_logged_in", False)):
                    st.error("Admin login required to save official results.")
                else:
                    _save_tournament_official(ctx, state, event)
            st.rerun()
        export_cols = st.columns([1, 1])
        export_cols[0].download_button(
            "Download event JSON",
            data=export_event_json(event).encode("utf-8"),
            file_name=f"{normalize_name(event.get('name', 'jupr-live')).lower().replace(' ', '-')}.json",
            mime="application/json",
            key="jupr_live_export_tournament",
        )
        with export_cols[1]:
            _render_event_csv_export(event, bracket_rows, label="Download bracket CSV", suffix="bracket")
        st.markdown("</div>", unsafe_allow_html=True)
    with c_right:
        st.markdown("#### Bracket status")
        if bracket_rows:
            st.dataframe(pd.DataFrame(bracket_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No bracket rows yet.")


def render(ctx):
    mode_label = "Public" if bool(ctx.public_mode) else ("Admin" if bool(ctx.admin_logged_in) else "Guest")
    page_shell(
        "🔴 JUPR Live Beta",
        "Run Round Robin or League / Ladder events with Quick or Official workflows.",
        mode_label=mode_label,
    )
    _inject_styles()
    _inject_score_keyboard_nav()
    state = _state()

    st.markdown(
        "- **Quick** keeps everything in session only: no DB writes, no ratings, no player creation.\n"
        "- **Official** resolves players to JUPR entities and uses the current official processing path.")

    _render_setup(ctx, state)

    event = state.get("event")
    if not event:
        return

    st.divider()
    if str(event.get("type")) == "round_robin":
        _render_rr_scoring(ctx, state, event)
    elif str(event.get("type")) == "tournament":
        _render_tournament_scoring(ctx, state, event)
    else:
        _render_league_scoring(ctx, state, event)
