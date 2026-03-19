from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Callable

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from jupr_app.domain.live_beta_engine import (
    SUPPORTED_RR_FORMATS,
    SUPPORTED_TOURNAMENT_TEAM_COUNTS,
    apply_round_substitution,
    apply_single_game_substitution,
    build_league_movement,
    clear_expired_substitutions,
    create_league_event,
    create_round_robin_event,
    create_tournament_event,
    current_league_round,
    export_event_json,
    find_match_by_id,
    get_active_sub_for_match,
    is_league_round_complete,
    league_aggregate_standings,
    league_round_summary,
    mark_tournament_matches_saved,
    match_is_scored,
    match_payloads_from_current_league_round,
    match_payloads_from_rr,
    matches_for_round,
    normalize_name,
    resolve_display_name,
    resolve_payload_player_ids,
    round_robin_current_round_number,
    round_robin_standings,
    set_pending_assignment,
    start_next_league_round,
    standings_csv_rows,
    substitution_is_active,
    substitution_is_locked,
    suggest_exact_league_court_sizes,
    tournament_bracket_rows,
    tournament_champion,
    tournament_completed_match_payloads,
    update_league_score,
    update_round_robin_score,
    update_tournament_score,
    validate_assignments,
)


SaveCallback = Callable[[object, dict, dict], bool | None]


@dataclass(frozen=True)
class LivePageConfig:
    state_key: str
    intro_markdown: str
    event_types: tuple[str, ...] = ("Round Robin", "League / Ladder")
    mode_pill_label: str = "Live"
    allow_official: bool = False
    allow_tournament: bool = False
    show_official_context: bool = False


def _default_state(config: LivePageConfig) -> dict:
    default_type = config.event_types[0] if config.event_types else "Round Robin"
    return {
        "event": None,
        "type_label": default_type,
        "event_name": "Saturday Event",
        "participant_count": 8,
        "participant_text": "",
        "league_rounds": 3,
        "official_league": "",
        "official_week_tag": "Week 1",
        "last_saved_rounds": [],
        "editing_substitution_id": None,
    }


def _state(config: LivePageConfig) -> dict:
    state = st.session_state.setdefault(config.state_key, _default_state(config))
    for key, value in _default_state(config).items():
        state.setdefault(key, value)
    if state.get("type_label") not in config.event_types:
        state["type_label"] = config.event_types[0]
        state["event"] = None
    return state


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_official(config: LivePageConfig) -> bool:
    return bool(config.allow_official)


def inject_styles() -> None:
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


def inject_score_keyboard_nav() -> None:
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
    return [
        normalize_name(x)
        for x in str(value or "").replace(",", "\n").splitlines()
        if normalize_name(x)
    ]


def _team_entry_lines(value: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for idx, raw_line in enumerate(str(value or "").splitlines(), start=1):
        line = normalize_name(raw_line)
        if not line:
            continue
        players = [
            normalize_name(part)
            for part in re.split(r"\s*(?:/|&|\+)\s*", line)
            if normalize_name(part)
        ]
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


def _resolved_ids_for_official(
    names: list[str], name_to_id: dict[str, int]
) -> tuple[dict[str, int], list[str]]:
    normalized_map = {
        normalize_name(k): int(v)
        for k, v in (name_to_id or {}).items()
        if normalize_name(k)
    }
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


def render_setup(ctx, state: dict, config: LivePageConfig) -> None:
    st.markdown('<div class="jupr-live-card">', unsafe_allow_html=True)
    st.markdown('<div class="jupr-live-kicker">Setup</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.3, 1])
    type_label = c1.radio(
        "Event type",
        list(config.event_types),
        index=list(config.event_types).index(state["type_label"]),
        horizontal=True,
        key=f"{config.state_key}_type",
    )
    state["type_label"] = type_label
    default_count = int(state["participant_count"])
    if state["type_label"] == "Round Robin":
        participant_count = c2.selectbox(
            "Count",
            SUPPORTED_RR_FORMATS,
            index=(
                SUPPORTED_RR_FORMATS.index(default_count)
                if default_count in SUPPORTED_RR_FORMATS
                else 2
            ),
            key=f"{config.state_key}_count_rr",
        )
    elif state["type_label"] == "Tournament":
        participant_count = c2.selectbox(
            "Teams",
            SUPPORTED_TOURNAMENT_TEAM_COUNTS,
            index=(
                SUPPORTED_TOURNAMENT_TEAM_COUNTS.index(default_count)
                if default_count in SUPPORTED_TOURNAMENT_TEAM_COUNTS
                else 0
            ),
            key=f"{config.state_key}_count_tn",
        )
    else:
        participant_count = c2.number_input(
            "Count",
            min_value=4,
            max_value=40,
            value=max(4, default_count),
            step=1,
            key=f"{config.state_key}_count_lg",
        )
    state["participant_count"] = int(participant_count)
    state["event_name"] = st.text_input(
        "Event name",
        value=state["event_name"],
        key=f"{config.state_key}_event_name",
    )
    if state["type_label"] == "Round Robin":
        help_text = "Enter one participant per line. JUPR Live Beta now supports every current JUPR doubles schedule from 4 through 20 participants."
        help_text += " Organized RR prioritizes maximum player exposure in ~8 rounds."
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
        key=f"{config.state_key}_participants",
    )
    if state["type_label"] == "League / Ladder":
        state["league_rounds"] = int(
            st.number_input(
                "Total rounds",
                min_value=1,
                max_value=12,
                value=int(state["league_rounds"]),
                step=1,
                key=f"{config.state_key}_total_rounds",
            )
        )
    can_create = True
    official_context: dict = {}
    if config.show_official_context:
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
            st.info(
                f"You entered {len(team_entries)} team(s); count is set to {int(state['participant_count'])}."
            )
            can_create = False
    elif participant_names and len(participant_names) != int(
        state["participant_count"]
    ):
        st.info(
            f"You entered {len(participant_names)} name(s); count is set to {int(state['participant_count'])}."
        )
        can_create = False
    action_cols = st.columns([1, 1, 3])
    if action_cols[0].button(
        "Create event",
        type="primary",
        disabled=not can_create,
        key=f"{config.state_key}_create_btn",
    ):
        try:
            if state["type_label"] == "Round Robin":
                resolved_ids = None
                if _is_official(config):
                    resolved_ids, missing = _resolved_ids_for_official(
                        participant_names, getattr(ctx, "name_to_id", {})
                    )
                    if missing:
                        raise ValueError(
                            "Official mode could not resolve: " + ", ".join(missing)
                        )
                state["event"] = create_round_robin_event(
                    name=state["event_name"],
                    participant_names=participant_names,
                    resolved_ids=resolved_ids,
                    official_context=official_context,
                )
            elif state["type_label"] == "Tournament":
                if _is_official(config):
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
                                "player1_id": int(
                                    resolved_ids[str(team["player1_name"])]
                                ),
                                "player2_id": int(
                                    resolved_ids[str(team["player2_name"])]
                                ),
                            }
                        )
                    if missing_players:
                        raise ValueError(
                            "Official mode could not resolve: "
                            + ", ".join(sorted(dict.fromkeys(missing_players)))
                        )
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
                    raise ValueError(
                        "League / Ladder requires an exact 4-player / 5-player court fit."
                    )
                resolved_ids = None
                if _is_official(config):
                    resolved_ids, missing = _resolved_ids_for_official(
                        participant_names, getattr(ctx, "name_to_id", {})
                    )
                    if missing:
                        raise ValueError(
                            "Official mode could not resolve: " + ", ".join(missing)
                        )
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
    if action_cols[1].button("Reset", key=f"{config.state_key}_reset_btn"):
        st.session_state[config.state_key] = _default_state(config)
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def _participant_name_map(event: dict) -> dict[str, str]:
    return {
        str(p["id"]): str(p.get("name", p["id"]))
        for p in event.get("participants") or []
    }


def _team_label(event: dict, ids: list[str]) -> str:
    name_map = _participant_name_map(event)
    return " / ".join(name_map.get(str(pid), str(pid)) for pid in (ids or []))


def _current_round_number(event: dict) -> int:
    if str(event.get("type")) == "league":
        return int(event.get("currentRoundNumber") or 1)
    return round_robin_current_round_number(event)


def _match_round_number(event: dict, match_id: str) -> int:
    for round_data in event.get("rounds") or []:
        matches = round_data.get("matches") or []
        if str(event.get("type")) == "league":
            matches = matches_for_round(event, int(round_data.get("number") or 0))
        for match in matches:
            if str(match.get("id")) == str(match_id):
                return int(round_data.get("number") or 0)
    return _current_round_number(event)


def _display_team_label(event: dict, match: dict, ids: list[str]) -> str:
    return " / ".join(
        resolve_display_name(event, str(match.get("id")), str(pid))
        for pid in (ids or [])
    )


def _player_directory(ctx) -> tuple[list[str], dict[str, int]]:
    df_players_all = getattr(ctx, "df_players_all", pd.DataFrame())
    if df_players_all is None or df_players_all.empty:
        return [], {}
    if "name" not in df_players_all.columns or "id" not in df_players_all.columns:
        return [], {}
    frame = df_players_all[["name", "id"]].copy()
    frame["name"] = frame["name"].fillna("").astype(str).map(normalize_name)
    frame = frame[frame["name"] != ""]
    frame = frame.drop_duplicates(subset=["name"], keep="first")
    options = sorted(frame["name"].tolist())
    return options, {str(row["name"]): int(row["id"]) for _, row in frame.iterrows()}


def _upsert_substitution(event: dict, substitution: dict) -> None:
    substitutions = list(event.get("substitutions") or [])
    replacements: list[dict] = []
    for existing in substitutions:
        same_target = (
            str(existing.get("scope")) == str(substitution.get("scope"))
            and int(existing.get("round_number") or 0) == int(substitution.get("round_number") or 0)
            and str(existing.get("original_participant_id")) == str(substitution.get("original_participant_id"))
            and str(existing.get("match_id") or "") == str(substitution.get("match_id") or "")
        )
        if str(existing.get("id")) == str(substitution.get("id")) or same_target:
            continue
        replacements.append(existing)
    replacements.append(substitution)
    replacements.sort(key=lambda item: str(item.get("created_at") or ""))
    event["substitutions"] = replacements


def _remove_substitution(event: dict, substitution_id: str) -> None:
    event["substitutions"] = [
        item
        for item in (event.get("substitutions") or [])
        if str(item.get("id")) != str(substitution_id)
    ]


def _substitution_summary(substitution: dict, event: dict) -> str:
    participant_map = _participant_name_map(event)
    replaced = participant_map.get(
        str(substitution.get("original_participant_id")),
        str(substitution.get("original_participant_id") or "Unknown"),
    )
    scope = "Round" if substitution.get("scope") == "round" else "Game"
    match_part = f" • {substitution.get('match_id')}" if substitution.get("match_id") else ""
    note = f" • Note: {substitution.get('note')}" if substitution.get("note") else ""
    return (
        f"{scope} sub — {replaced} → {substitution.get('substitute_name')} "
        f"(Round {int(substitution.get('round_number') or 0)}, {len(substitution.get('affected_match_ids') or [])} match(es){match_part}){note}"
    )


def _render_substitution_badges(event: dict, match: dict) -> None:
    substitutions = [
        sub
        for sub in (event.get("substitutions") or [])
        if str(match.get("id")) in {str(x) for x in (sub.get("affected_match_ids") or [])}
    ]
    if not substitutions:
        return
    for substitution in substitutions:
        scope = "Round sub" if substitution.get("scope") == "round" else "Game sub"
        st.caption(
            f"{scope}: {resolve_display_name(event, str(match.get('id')), str(substitution.get('original_participant_id')))}"
        )


def _render_substitutions_area(ctx, state: dict, event: dict, config: LivePageConfig) -> None:
    if not _is_official(config):
        return
    clear_expired_substitutions(event)
    round_number = _current_round_number(event)
    substitutions = list(event.get("substitutions") or [])
    st.markdown("#### Substitutions")
    player_options, player_name_to_id = _player_directory(ctx)
    if not player_options:
        st.warning("No substitute player directory is available.")
        return
    participant_map = _participant_name_map(event)
    round_matches = matches_for_round(event, round_number)
    available_round_participants: list[str] = []
    for match in round_matches:
        if match_is_scored(match):
            continue
        for pid in (match.get("teamA") or []) + (match.get("teamB") or []):
            pid = str(pid)
            if pid not in available_round_participants:
                available_round_participants.append(pid)
    if not available_round_participants:
        st.info("No unscored matches remain in the current round for substitutions.")
    else:
        cols = st.columns([1.2, 1.2, 1.2, 1.4, 0.9])
        round_labels = [participant_map.get(pid, pid) for pid in available_round_participants]
        selected_label = cols[0].selectbox(
            "Replace for round",
            round_labels,
            key=f"{config.state_key}_round_sub_out_r{round_number}",
        )
        selected_pid = available_round_participants[round_labels.index(selected_label)]
        substitute_name = cols[1].selectbox(
            "Substitute",
            player_options,
            key=f"{config.state_key}_round_sub_in_r{round_number}",
        )
        note = cols[2].text_input(
            "Note",
            key=f"{config.state_key}_round_sub_note_r{round_number}",
        )
        affected_matches = [
            str(match.get("id"))
            for match in round_matches
            if not match_is_scored(match)
            and selected_pid in [str(x) for x in (match.get("teamA") or []) + (match.get("teamB") or [])]
        ]
        cols[3].metric("Affected games", len(affected_matches))
        if cols[4].button("Sub for round", key=f"{config.state_key}_round_sub_apply_r{round_number}", type="primary"):
            try:
                if not substitute_name:
                    raise ValueError("Choose a substitute player first.")
                substitution = apply_round_substitution(
                    event,
                    round_number=round_number,
                    original_participant_id=selected_pid,
                    substitute_player_id=int(player_name_to_id[substitute_name]),
                    substitute_name=substitute_name,
                    created_by="admin",
                    created_at=_utc_now_iso(),
                    note=note,
                    substitution_id=state.get("editing_substitution_id"),
                )
                _upsert_substitution(event, substitution)
                state["editing_substitution_id"] = None
                st.success("Round substitution applied.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    if substitutions:
        st.caption("Active / pending substitution state")
        for substitution in substitutions:
            c1, c2, c3 = st.columns([6, 1.2, 1.2])
            c1.write(_substitution_summary(substitution, event))
            can_edit = substitution_is_active(event, substitution) and not substitution_is_locked(event, substitution)
            if c2.button(
                "Edit",
                key=f"{config.state_key}_edit_sub_{substitution['id']}",
                disabled=not can_edit,
            ):
                state["editing_substitution_id"] = str(substitution["id"])
                if substitution.get("scope") == "round":
                    participant_map_reverse = {
                        v: k for k, v in _participant_name_map(event).items()
                    }
                    player_name = participant_map.get(
                        str(substitution.get("original_participant_id")),
                        participant_map_reverse.get(
                            str(substitution.get("original_participant_id")),
                            str(substitution.get("original_participant_id")),
                        ),
                    )
                    st.session_state[
                        f"{config.state_key}_round_sub_out_r{int(substitution.get('round_number') or 0)}"
                    ] = player_name
                    st.session_state[
                        f"{config.state_key}_round_sub_in_r{int(substitution.get('round_number') or 0)}"
                    ] = str(substitution.get("substitute_name") or "")
                    st.session_state[
                        f"{config.state_key}_round_sub_note_r{int(substitution.get('round_number') or 0)}"
                    ] = str(substitution.get("note") or "")
                elif substitution.get("match_id"):
                    match_id = str(substitution.get("match_id"))
                    st.session_state[
                        f"{config.state_key}_game_sub_note_{match_id}"
                    ] = str(substitution.get("note") or "")
                    st.session_state[
                        f"{config.state_key}_game_sub_in_{match_id}"
                    ] = str(substitution.get("substitute_name") or "")
                st.rerun()
            if c3.button(
                "Remove",
                key=f"{config.state_key}_remove_sub_{substitution['id']}",
                disabled=not can_edit,
            ):
                _remove_substitution(event, str(substitution["id"]))
                if state.get("editing_substitution_id") == str(substitution["id"]):
                    state["editing_substitution_id"] = None
                st.rerun()


def _render_game_substitution_controls(ctx, event: dict, config: LivePageConfig, match: dict) -> None:
    player_options, player_name_to_id = _player_directory(ctx)
    if not player_options:
        return
    match_id = str(match.get("id"))
    round_number = _match_round_number(event, match_id)
    participant_map = _participant_name_map(event)
    existing_subs = [
        sub
        for sub in (event.get("substitutions") or [])
        if str(sub.get("scope")) == "game" and str(sub.get("match_id")) == match_id
    ]
    with st.expander("Sub for game", expanded=False):
        match_participants = [
            str(pid) for pid in (match.get("teamA") or []) + (match.get("teamB") or [])
        ]
        player_labels = [participant_map.get(pid, pid) for pid in match_participants]
        replace_label = st.radio(
            "Replace player",
            player_labels,
            horizontal=True,
            key=f"{config.state_key}_game_sub_out_{match_id}",
        )
        replace_pid = match_participants[player_labels.index(replace_label)]
        substitute_name = st.selectbox(
            "Substitute player",
            player_options,
            key=f"{config.state_key}_game_sub_in_{match_id}",
        )
        note = st.text_input("Note", key=f"{config.state_key}_game_sub_note_{match_id}")
        if existing_subs:
            for sub in existing_subs:
                st.caption(_substitution_summary(sub, event))
        apply_col, remove_col = st.columns([1, 1])
        if apply_col.button("Apply game sub", key=f"{config.state_key}_game_sub_apply_{match_id}"):
            try:
                if not substitute_name:
                    raise ValueError("Choose a substitute player first.")
                current_sub = get_active_sub_for_match(
                    event,
                    match_id,
                    replace_pid,
                    include_inactive=True,
                )
                substitution = apply_single_game_substitution(
                    event,
                    round_number=round_number,
                    match_id=match_id,
                    original_participant_id=replace_pid,
                    substitute_player_id=int(player_name_to_id[substitute_name]),
                    substitute_name=substitute_name,
                    created_by="admin",
                    created_at=_utc_now_iso(),
                    note=note,
                    substitution_id=(str(current_sub["id"]) if current_sub else None),
                )
                _upsert_substitution(event, substitution)
                st.success("Game substitution applied.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        removable = next(
            (
                sub for sub in existing_subs
                if str(sub.get("original_participant_id")) == replace_pid
                and not substitution_is_locked(event, sub)
            ),
            None,
        )
        if remove_col.button(
            "Remove game sub",
            key=f"{config.state_key}_game_sub_remove_{match_id}",
            disabled=removable is None,
        ):
            _remove_substitution(event, str(removable["id"]))
            st.rerun()


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
    st.caption(
        "Printing is best handled with your browser’s print dialog after expanding the sections you want to keep."
    )


def _render_event_csv_export(
    event: dict, rows: list[dict], *, label: str, suffix: str
) -> None:
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


def official_base_payload(state: dict) -> dict:
    return {
        "date": _utc_now_iso(),
        "league": state.get("official_league") or "",
        "match_type": "Live Match",
        "week_tag": state.get("official_week_tag") or "",
        "is_popup": False,
    }


def build_rr_official_payloads(state: dict, event: dict) -> list[dict]:
    payloads = resolve_payload_player_ids(
        event,
        match_payloads_from_rr(event),
        materialize_substitutions=True,
    )
    return [{**payload, **official_base_payload(state)} for payload in payloads]


def build_league_round_official_payloads(state: dict, event: dict) -> list[dict]:
    payloads = resolve_payload_player_ids(
        event,
        match_payloads_from_current_league_round(event),
        materialize_substitutions=True,
    )
    return [{**payload, **official_base_payload(state)} for payload in payloads]


def build_tournament_official_payloads(event: dict) -> list[dict]:
    return tournament_completed_match_payloads(event, unsaved_only=True)


def mark_tournament_payloads_saved(event: dict, payloads: list[dict]) -> None:
    mark_tournament_matches_saved(event, payloads)


def _render_rr_scoring(
    ctx,
    state: dict,
    event: dict,
    config: LivePageConfig,
    on_save_rr: SaveCallback | None,
) -> None:
    standings = round_robin_standings(event)
    leader = standings[0]["name"] if standings else "—"
    c_left, c_right = st.columns([1.5, 1])
    with c_left:
        st.markdown('<div class="jupr-live-card">', unsafe_allow_html=True)
        st.markdown(
            f"<div class='jupr-live-kicker'>Live scoring</div><h3 style='margin-top:0.25rem'>{event['name']}</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<span class='jupr-live-pill'>{config.mode_pill_label}</span><span class='jupr-live-pill'>Leader: {leader}</span>",
            unsafe_allow_html=True,
        )
        _render_substitutions_area(ctx, state, event, config)
        for round_data in event.get("rounds") or []:
            st.markdown(f"#### Round {int(round_data.get('number') or 0)}")
            for match in round_data.get("matches") or []:
                st.markdown(
                    '<div class="jupr-live-score-shell">', unsafe_allow_html=True
                )
                cols = st.columns([3.6, 1.1, 0.6, 1.1, 3.6])
                cols[0].markdown(
                    f"<div class='jupr-live-team'>{_display_team_label(event, match, match.get('teamA') or [])}</div>",
                    unsafe_allow_html=True,
                )
                cols[1].number_input(
                    f"JUPR Live Score {match['id']} A",
                    min_value=0,
                    max_value=99,
                    value=int(match.get("scoreA") or 0),
                    step=1,
                    key=f"{config.state_key}_rr_{match['id']}_a",
                )
                cols[2].markdown(
                    "<div class='jupr-live-vs'>vs</div>", unsafe_allow_html=True
                )
                cols[3].number_input(
                    f"JUPR Live Score {match['id']} B",
                    min_value=0,
                    max_value=99,
                    value=int(match.get("scoreB") or 0),
                    step=1,
                    key=f"{config.state_key}_rr_{match['id']}_b",
                )
                cols[4].markdown(
                    f"<div class='jupr-live-team'>{_display_team_label(event, match, match.get('teamB') or [])}</div>",
                    unsafe_allow_html=True,
                )
                _render_substitution_badges(event, match)
                st.caption(str(match.get("desc") or ""))
                if _is_official(config) and not match_is_scored(match):
                    _render_game_substitution_controls(ctx, event, config, match)
                st.markdown("</div>", unsafe_allow_html=True)
            st.divider()
        submit_label = (
            "Save official results"
            if _is_official(config)
            else "Update live standings"
        )
        submitted = st.button(
            submit_label,
            type="primary",
            key=f"{config.state_key}_rr_submit",
        )
        if submitted:
            for round_data in event.get("rounds") or []:
                for match in round_data.get("matches") or []:
                    a_val = int(
                        st.session_state.get(
                            f"{config.state_key}_rr_{match['id']}_a", 0
                        )
                        or 0
                    )
                    b_val = int(
                        st.session_state.get(
                            f"{config.state_key}_rr_{match['id']}_b", 0
                        )
                        or 0
                    )
                    if a_val == 0 and b_val == 0:
                        update_round_robin_score(event, match["id"], None, None)
                    else:
                        update_round_robin_score(event, match["id"], a_val, b_val)
            if _is_official(config) and on_save_rr is not None:
                if not bool(getattr(ctx, "admin_logged_in", False)):
                    st.error("Admin login required to save official results.")
                else:
                    on_save_rr(ctx, state, event)
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


def _render_league_scoring(
    ctx,
    state: dict,
    event: dict,
    config: LivePageConfig,
    on_save_league: SaveCallback | None,
) -> None:
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
        row["proposedCourt"] = int(
            event["pendingAssignments"].get(
                str(row["participantId"]), row["proposedCourt"]
            )
        )
    current_round_number = int(event.get("currentRoundNumber") or 1)
    total_rounds = int(event.get("totalRounds") or 1)
    c_left, c_right = st.columns([1.5, 1])
    with c_left:
        st.markdown('<div class="jupr-live-card">', unsafe_allow_html=True)
        st.markdown(
            f"<div class='jupr-live-kicker'>Round {current_round_number} of {total_rounds}</div><h3 style='margin-top:0.25rem'>{event['name']}</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<span class='jupr-live-pill'>{config.mode_pill_label}</span>",
            unsafe_allow_html=True,
        )
        _render_substitutions_area(ctx, state, event, config)
        for court in round_data.get("courts") or []:
            st.markdown(f"#### Court {int(court.get('courtNumber') or 0)}")
            for mini_round in court.get("miniRounds") or []:
                bye_pid = mini_round.get("byeParticipantId")
                label = f"Mini-round {int(mini_round.get('number') or 0)}"
                if bye_pid:
                    label += f" • Bye: {_participant_name_map(event).get(str(bye_pid), str(bye_pid))}"
                st.caption(label)
                for match in mini_round.get("matches") or []:
                    st.markdown(
                        '<div class="jupr-live-score-shell">',
                        unsafe_allow_html=True,
                    )
                    cols = st.columns([3.6, 1.1, 0.6, 1.1, 3.6])
                    cols[0].markdown(
                        f"<div class='jupr-live-team'>{_display_team_label(event, match, match.get('teamA') or [])}</div>",
                        unsafe_allow_html=True,
                    )
                    cols[1].number_input(
                        f"JUPR Live Score {match['id']} A",
                        min_value=0,
                        max_value=99,
                        value=int(match.get("scoreA") or 0),
                        step=1,
                        key=f"{config.state_key}_lg_{match['id']}_a",
                    )
                    cols[2].markdown(
                        "<div class='jupr-live-vs'>vs</div>", unsafe_allow_html=True
                    )
                    cols[3].number_input(
                        f"JUPR Live Score {match['id']} B",
                        min_value=0,
                        max_value=99,
                        value=int(match.get("scoreB") or 0),
                        step=1,
                        key=f"{config.state_key}_lg_{match['id']}_b",
                    )
                    cols[4].markdown(
                        f"<div class='jupr-live-team'>{_display_team_label(event, match, match.get('teamB') or [])}</div>",
                        unsafe_allow_html=True,
                    )
                    _render_substitution_badges(event, match)
                    if _is_official(config) and not match_is_scored(match):
                        _render_game_substitution_controls(ctx, event, config, match)
                    st.markdown("</div>", unsafe_allow_html=True)
        submitted = st.button(
            "Update round & movement",
            type="primary",
            key=f"{config.state_key}_league_submit_r{current_round_number}",
        )
        if submitted:
            for court in round_data.get("courts") or []:
                for mini_round in court.get("miniRounds") or []:
                    for match in mini_round.get("matches") or []:
                        a_val = int(
                            st.session_state.get(
                                f"{config.state_key}_lg_{match['id']}_a", 0
                            )
                            or 0
                        )
                        b_val = int(
                            st.session_state.get(
                                f"{config.state_key}_lg_{match['id']}_b", 0
                            )
                            or 0
                        )
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
                current_assignment = int(
                    event["pendingAssignments"].get(
                        str(row["participantId"]), row["proposedCourt"]
                    )
                )
                c1, c2 = st.columns([4, 1.2])
                c1.write(
                    f"**{row['name']}** — Court {int(row['currentCourt'])} Rank {int(row['currentRank'])} | W {int(row['wins'])} / L {int(row['losses'])} / T {int(row['ties'])} | Diff {int(row['differential'])}"
                )
                selected = c2.selectbox(
                    f"Next court {row['participantId']}",
                    list(range(1, len(event.get("courtSizes") or []) + 1)),
                    index=list(range(1, len(event.get("courtSizes") or []) + 1)).index(
                        current_assignment
                    ),
                    key=f"{config.state_key}_move_{row['participantId']}",
                    label_visibility="collapsed",
                )
                set_pending_assignment(event, str(row["participantId"]), int(selected))
            validation = validate_assignments(
                event, dict(event.get("pendingAssignments") or {})
            )
            if validation["ok"]:
                st.success("Court counts are valid. You can finalize this round.")
            else:
                st.error(" ".join(validation["errors"]))
            action_cols = st.columns([1.3, 1.2, 3])
            next_label = (
                "Finish league night"
                if current_round_number >= total_rounds
                else "Finalize round & start next"
            )
            if action_cols[0].button(
                next_label,
                type="primary",
                key=f"{config.state_key}_next_round_{current_round_number}",
            ):
                try:
                    if _is_official(config) and on_save_league is not None:
                        if not bool(getattr(ctx, "admin_logged_in", False)):
                            raise ValueError(
                                "Admin login required to save official round results."
                            )
                        if on_save_league(ctx, state, event) is False:
                            st.stop()
                    if current_round_number < total_rounds:
                        start_next_league_round(event)
                        st.success(f"Round {current_round_number} finalized.")
                    else:
                        st.success("League night complete.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
            if action_cols[1].button(
                "Reset to auto",
                key=f"{config.state_key}_reset_move_{current_round_number}",
            ):
                event["pendingAssignments"] = dict(
                    build_league_movement(event)["assignments"]
                )
                st.rerun()
            st.dataframe(
                _movement_rows_to_df(build_league_movement(event)["rows"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "Complete every score in the current round to unlock movement preview."
            )
        _render_event_exports(event, aggregate)
        st.markdown("</div>", unsafe_allow_html=True)
    with c_right:
        for court_info in summary:
            _render_standings_table(
                court_info.get("standings") or [],
                f"Court {int(court_info['courtNumber'])} standings",
            )
        _render_standings_table(aggregate, "Cumulative ladder standings")


def _render_tournament_scoring(
    ctx,
    state: dict,
    event: dict,
    config: LivePageConfig,
    on_save_tournament: SaveCallback | None,
) -> None:
    bracket_rows = tournament_bracket_rows(event)
    champion_id = tournament_champion(event)
    team_map = {str(team["id"]): team for team in (event.get("teams") or [])}
    champion_name = (
        team_map.get(str(champion_id), {}).get("name") if champion_id else None
    )
    c_left, c_right = st.columns([1.5, 1])
    with c_left:
        st.markdown('<div class="jupr-live-card">', unsafe_allow_html=True)
        pills = [f"<span class='jupr-live-pill'>{config.mode_pill_label}</span>"]
        pills.append(
            f"<span class='jupr-live-pill'>Champion: {champion_name or 'Pending'}</span>"
        )
        st.markdown(
            f"<div class='jupr-live-kicker'>Tournament bracket</div><h3 style='margin-top:0.25rem'>{event['name']}</h3>{''.join(pills)}",
            unsafe_allow_html=True,
        )
        with st.form(f"{config.state_key}_tournament_form"):
            for round_data in event.get("rounds") or []:
                st.markdown(f"#### Round {int(round_data.get('number') or 0)}")
                for match in round_data.get("matches") or []:
                    match_id = f"r{int(round_data.get('number') or 0)}-s{int(match.get('slot') or 0)}"
                    st.markdown(
                        '<div class="jupr-live-score-shell">', unsafe_allow_html=True
                    )
                    cols = st.columns([3.6, 1.1, 0.6, 1.1, 3.6])
                    cols[0].markdown(
                        f"<div class='jupr-live-team'>{team_map.get(str(match.get('participantAId')), {}).get('name', 'TBD')}</div>",
                        unsafe_allow_html=True,
                    )
                    cols[1].number_input(
                        f"JUPR Live Score {match_id} A",
                        min_value=0,
                        max_value=99,
                        value=int(match.get("scoreA") or 0),
                        step=1,
                        key=f"{config.state_key}_tn_{match_id}_a",
                        disabled=not (
                            match.get("participantAId") and match.get("participantBId")
                        ),
                    )
                    cols[2].markdown(
                        "<div class='jupr-live-vs'>vs</div>", unsafe_allow_html=True
                    )
                    cols[3].number_input(
                        f"JUPR Live Score {match_id} B",
                        min_value=0,
                        max_value=99,
                        value=int(match.get("scoreB") or 0),
                        step=1,
                        key=f"{config.state_key}_tn_{match_id}_b",
                        disabled=not (
                            match.get("participantAId") and match.get("participantBId")
                        ),
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
                "Save official results" if _is_official(config) else "Update bracket",
                type="primary",
            )
        if submitted:
            for round_data in event.get("rounds") or []:
                for match in round_data.get("matches") or []:
                    round_number = int(round_data.get("number") or 0)
                    slot = int(match.get("slot") or 0)
                    match_id = f"r{round_number}-s{slot}"
                    if not (
                        match.get("participantAId") and match.get("participantBId")
                    ):
                        update_tournament_score(event, round_number, slot, None, None)
                        continue
                    a_val = int(
                        st.session_state.get(f"{config.state_key}_tn_{match_id}_a", 0)
                        or 0
                    )
                    b_val = int(
                        st.session_state.get(f"{config.state_key}_tn_{match_id}_b", 0)
                        or 0
                    )
                    if a_val == 0 and b_val == 0:
                        update_tournament_score(event, round_number, slot, None, None)
                    else:
                        update_tournament_score(event, round_number, slot, a_val, b_val)
            if _is_official(config) and on_save_tournament is not None:
                if not bool(getattr(ctx, "admin_logged_in", False)):
                    st.error("Admin login required to save official results.")
                else:
                    on_save_tournament(ctx, state, event)
            st.rerun()
        export_cols = st.columns([1, 1])
        export_cols[0].download_button(
            "Download event JSON",
            data=export_event_json(event).encode("utf-8"),
            file_name=f"{normalize_name(event.get('name', 'jupr-live')).lower().replace(' ', '-')}.json",
            mime="application/json",
            key=f"{config.state_key}_export_tournament",
        )
        with export_cols[1]:
            _render_event_csv_export(
                event, bracket_rows, label="Download bracket CSV", suffix="bracket"
            )
        st.markdown("</div>", unsafe_allow_html=True)
    with c_right:
        st.markdown("#### Bracket status")
        if bracket_rows:
            st.dataframe(
                pd.DataFrame(bracket_rows), use_container_width=True, hide_index=True
            )
        else:
            st.info("No bracket rows yet.")


def render_live_page(
    ctx,
    config: LivePageConfig,
    *,
    on_save_rr: SaveCallback | None = None,
    on_save_league: SaveCallback | None = None,
    on_save_tournament: SaveCallback | None = None,
) -> None:
    inject_styles()
    inject_score_keyboard_nav()
    state = _state(config)

    if config.intro_markdown:
        st.markdown(config.intro_markdown)

    render_setup(ctx, state, config)

    event = state.get("event")
    if not event:
        return

    st.divider()
    event_type = str(event.get("type"))
    if event_type == "round_robin":
        _render_rr_scoring(ctx, state, event, config, on_save_rr)
    elif event_type == "tournament":
        _render_tournament_scoring(ctx, state, event, config, on_save_tournament)
    else:
        _render_league_scoring(ctx, state, event, config, on_save_league)
