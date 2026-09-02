from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import difflib
import math
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
    resolve_active_player_name,
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
from jupr_app.domain.player_ops import safe_add_player


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
    show_rating_mode: bool = False
    persistent_save_label: str | None = None
    requires_roster_resolution: bool = False
    use_admin_roster_builder: bool = False


def _default_state(config: LivePageConfig) -> dict:
    default_type = config.event_types[0] if config.event_types else "Round Robin"
    return {
        "event": None,
        "type_label": default_type,
        "event_name": "Saturday Event",
        "participant_count": 8,
        "participant_text": "",
        "selected_existing_players": [],
        "league_rounds": 3,
        "official_league": "",
        "official_week_tag": "Week 1",
        "rating_mode": "Rated",
        "last_saved_rounds": [],
        "editing_substitution_id": None,
        "parsed_roster_lines": [],
        "roster_candidates": [],
        "confirmed_roster_rows": [],
        "roster_confirmed": False,
        "resolved_roster_ids": {},
        "admin_roster_rows": [],
        "default_new_player_rating": None,
        "quick_paste_nonce": 0,
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


def _save_button_label(
    config: LivePageConfig, callback_present: bool, *, default_non_official: str
) -> str:
    if _is_official(config):
        return "Save official results"
    if callback_present and config.persistent_save_label:
        return str(config.persistent_save_label)
    return default_non_official


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .jupr-live-card {
            border: 1px solid var(--accent-border);
            border-radius: 18px;
            padding: 1rem 1rem 0.75rem;
            background: linear-gradient(180deg, var(--panel), var(--accent-soft));
            box-shadow: var(--shadow-lg);
            margin-bottom: 1rem;
        }
        .jupr-live-kicker { font-size: 0.8rem; font-weight: 700; color: var(--accent); text-transform: uppercase; letter-spacing: 0.08em; }
        .jupr-live-score-shell {
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            background: var(--panel);
            color: var(--text-primary);
            margin-bottom: 0.9rem;
        }
        .jupr-live-team { color: var(--text-primary); font-weight: 700; font-size: 1rem; }
        .jupr-live-vs { text-align: center; font-size: 0.9rem; font-weight: 700; color: var(--text-secondary); margin-top: 1.9rem; }
        .jupr-live-actions button[kind="primary"] {
            min-height: 3rem;
            font-weight: 700;
        }
        .jupr-live-pill {
            display:inline-block; padding:0.35rem 0.7rem; border-radius:999px; background:var(--accent-soft); color:var(--accent); font-weight:600; margin-right:0.4rem;
        }
        .jupr-live-slot {
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.6rem 0.7rem;
            background: var(--pill-bg);
            margin-bottom: 0.45rem;
        }
        .jupr-live-slot-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-secondary);
            margin-bottom: 0.15rem;
        }
        .jupr-live-slot-name {
            color: var(--text-primary);
            font-weight: 700;
            font-size: 0.98rem;
            line-height: 1.3;
        }
        .jupr-live-sub-summary {
            border: 1px solid var(--accent-border);
            border-radius: 14px;
            background: var(--accent-soft);
            padding: 0.65rem 0.8rem;
            margin: 0.4rem 0 0.8rem;
            font-size: 0.92rem;
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


def _dedupe_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for raw_name in names:
        clean = normalize_name(raw_name)
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(clean)
    return deduped


def _merge_participant_text(participant_text: str, selected_names: list[str]) -> str:
    existing_lines = _participant_lines(participant_text)
    merged_lines = _dedupe_names(existing_lines + list(selected_names or []))
    return "\n".join(merged_lines)


def _normalized_person_key(value: object) -> str:
    return normalize_name(value).casefold()


def _best_fuzzy_score(target: str, candidate: str) -> float:
    if not target or not candidate:
        return 0.0
    return float(difflib.SequenceMatcher(None, target, candidate).ratio())


def _existing_player_rating_jupr(ctx, player_id: int | None) -> float | None:
    if player_id is None:
        return None
    df_players_all = getattr(ctx, "df_players_all", pd.DataFrame())
    if df_players_all is None or df_players_all.empty:
        return None
    if "id" not in df_players_all.columns or "rating" not in df_players_all.columns:
        return None
    matches = df_players_all[df_players_all["id"] == int(player_id)]
    if matches.empty:
        return None
    rating_elo = pd.to_numeric(matches["rating"], errors="coerce").dropna()
    if rating_elo.empty:
        return None
    return round(float(rating_elo.iloc[0]) / 400.0, 2)


def _new_player_rating(value: object, *, fallback: object = None) -> float | None:
    for candidate in (value, fallback):
        if candidate is None:
            continue
        try:
            rating = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(rating) and 1.0 <= rating <= 7.0:
            return rating
    return None


def _build_roster_candidate_rows(
    participant_names: list[str],
    player_name_to_id: dict[str, int],
    *,
    suggestion_cap: int = 5,
) -> list[dict]:
    normalized_to_names: dict[str, list[str]] = {}
    for player_name in player_name_to_id.keys():
        key = _normalized_person_key(player_name)
        if not key:
            continue
        normalized_to_names.setdefault(key, []).append(str(player_name))
    normalized_keys = list(normalized_to_names.keys())

    rows: list[dict] = []
    for pasted_name in participant_names:
        normalized_pasted = _normalized_person_key(pasted_name)
        exact_names = normalized_to_names.get(normalized_pasted, [])
        candidate_names: list[str] = []
        if exact_names:
            candidate_names.extend(exact_names)
        else:
            close_keys = difflib.get_close_matches(
                normalized_pasted,
                normalized_keys,
                n=max(1, int(suggestion_cap)),
                cutoff=0.65,
            )
            for key in close_keys:
                for match_name in normalized_to_names.get(key, []):
                    if match_name not in candidate_names:
                        candidate_names.append(match_name)
        candidate_names = candidate_names[: max(1, int(suggestion_cap))]
        top_score = 0.0
        if candidate_names:
            top_score = max(
                _best_fuzzy_score(normalized_pasted, _normalized_person_key(name))
                for name in candidate_names
            )
        if exact_names:
            status = "exact match"
        elif candidate_names:
            status = "suggested match"
        else:
            status = "new social-only"

        create_social_token = f"new::{normalized_pasted or pasted_name}"
        options: list[dict[str, str | int | None]] = [
            {"token": f"player::{int(player_name_to_id[name])}", "label": str(name)}
            for name in candidate_names
        ]
        options.append(
            {
                "token": create_social_token,
                "label": f"Create new social-only player: {normalize_name(pasted_name)}",
            }
        )
        default_token = create_social_token
        if exact_names:
            default_token = f"player::{int(player_name_to_id[exact_names[0]])}"
        duplicate_warning = ""
        if (not exact_names) and candidate_names and top_score >= 0.9:
            duplicate_warning = (
                "Possible duplicate: this name is very close to an existing rated player. "
                "If you still need a separate social-only profile, provide a confirmation note."
            )
        rows.append(
            {
                "original": str(pasted_name),
                "normalized": normalize_name(pasted_name),
                "status": status,
                "default_token": default_token,
                "options": options,
                "duplicate_warning": duplicate_warning,
            }
        )
    return rows


def _resolved_participants_from_confirmation(
    confirmed_rows: list[dict],
    player_name_to_id: dict[str, int],
) -> tuple[list[str], dict[str, int], list[dict]]:
    id_to_name = {int(pid): str(name) for name, pid in player_name_to_id.items()}
    names: list[str] = []
    resolved_ids: dict[str, int] = {}
    resolved_rows: list[dict] = []
    for row in confirmed_rows:
        selected_token = str(row.get("selection") or "")
        original = normalize_name(row.get("original"))
        if selected_token.startswith("player::"):
            player_id = int(selected_token.split("::", 1)[1])
            canonical = normalize_name(id_to_name.get(player_id, original))
            if canonical:
                names.append(canonical)
                resolved_ids[canonical] = int(player_id)
                resolved_rows.append(
                    {
                        "name": canonical,
                        "player_id": int(player_id),
                        "source_name": original,
                        "match_status": "matched_existing",
                    }
                )
            continue
        social_name = original
        if social_name:
            names.append(social_name)
            resolved_rows.append(
                {
                    "name": social_name,
                    "player_id": None,
                    "source_name": original,
                    "match_status": "new_social",
                    "duplicate_confirmed": bool(row.get("duplicate_confirmed", False)),
                    "duplicate_note": str(row.get("duplicate_note") or "").strip(),
                }
            )
    return names, resolved_ids, resolved_rows


def _fetch_player_by_exact_name(supabase, *, club_id: str, display_name: str) -> dict | None:
    rows = (
        supabase.table("players")
        .select("id,name")
        .eq("club_id", str(club_id))
        .eq("name", normalize_name(display_name))
        .limit(2)
        .execute()
        .data
        or []
    )
    if not rows:
        return None
    return dict(rows[0])


def _default_admin_roster_row(
    ctx,
    display_name: str,
    *,
    order: int,
    player_name_to_id: dict[str, int],
    default_new_player_rating: float | None,
) -> dict:
    normalized_display = normalize_name(display_name)
    exact_pid = player_name_to_id.get(normalized_display)
    suggestion = ""
    if exact_pid is None and normalized_display:
        suggestion = next(
            iter(
                difflib.get_close_matches(
                    normalized_display,
                    list(player_name_to_id.keys()),
                    n=1,
                    cutoff=0.8,
                )
            ),
            "",
        )
    if exact_pid is not None:
        status = "existing_player"
        selected_name = normalized_display
        starting_rating = _existing_player_rating_jupr(ctx, exact_pid)
    elif suggestion:
        status = "needs_review"
        selected_name = suggestion
        suggested_pid = player_name_to_id.get(suggestion)
        starting_rating = (
            _existing_player_rating_jupr(ctx, suggested_pid)
            if suggested_pid is not None
            else None
        )
    else:
        status = "create_new_player"
        selected_name = ""
        starting_rating = _new_player_rating(default_new_player_rating)
    return {
        "order": int(order),
        "display_name": normalized_display,
        "resolution_status": status,
        "player_id": int(exact_pid) if exact_pid is not None else None,
        "selected_existing_name": selected_name,
        "suggested_existing_name": suggestion,
        "starting_jupr_rating": starting_rating,
    }


def _append_roster_names(
    ctx,
    roster_rows: list[dict],
    incoming_names: list[str],
    *,
    player_name_to_id: dict[str, int],
    default_new_player_rating: float | None,
) -> list[dict]:
    existing_keys = {
        normalize_name(row.get("display_name")).casefold()
        for row in roster_rows
        if normalize_name(row.get("display_name"))
    }
    next_order = len(roster_rows) + 1
    updated = list(roster_rows)
    for raw_name in incoming_names:
        name = normalize_name(raw_name)
        if not name:
            continue
        key = name.casefold()
        if key in existing_keys:
            continue
        updated.append(
            _default_admin_roster_row(
                ctx,
                name,
                order=next_order,
                player_name_to_id=player_name_to_id,
                default_new_player_rating=default_new_player_rating,
            )
        )
        existing_keys.add(key)
        next_order += 1
    return updated


def _rows_from_admin_editor_df(
    editor_df: pd.DataFrame,
    *,
    player_name_to_id: dict[str, int],
    ctx,
    default_new_player_rating: float | None,
) -> list[dict]:
    rows: list[dict] = []
    for _, row in editor_df.iterrows():
        display_name = normalize_name(row.get("Name"))
        if not display_name:
            continue
        try:
            order = int(row.get("Order"))
        except Exception:
            order = len(rows) + 1
        resolution_status = str(row.get("Resolution") or "create_new_player")
        if resolution_status not in {"existing_player", "create_new_player", "needs_review"}:
            resolution_status = "create_new_player"
        selected_existing_name = normalize_name(row.get("Matched Player"))
        selected_player_id = (
            int(player_name_to_id[selected_existing_name])
            if selected_existing_name and selected_existing_name in player_name_to_id
            else None
        )
        if resolution_status == "existing_player" and selected_player_id is not None:
            starting_rating = _existing_player_rating_jupr(ctx, selected_player_id)
        elif resolution_status == "create_new_player":
            starting_rating = _new_player_rating(
                row.get("Current / Starting JUPR"),
                fallback=default_new_player_rating,
            )
        else:
            starting_rating = _new_player_rating(row.get("Current / Starting JUPR"))
        rows.append(
            {
                "order": int(order),
                "display_name": display_name,
                "resolution_status": resolution_status,
                "player_id": selected_player_id,
                "selected_existing_name": selected_existing_name,
                "starting_jupr_rating": starting_rating,
            }
        )
    rows.sort(key=lambda item: (int(item.get("order") or 0), str(item.get("display_name") or "")))
    for idx, row in enumerate(rows, start=1):
        row["order"] = idx
    return rows


def _create_and_resolve_admin_players(
    ctx,
    *,
    roster_rows: list[dict],
    default_new_player_rating: float | None,
    player_name_to_id: dict[str, int],
) -> tuple[list[str], dict[str, int], list[str], list[str]]:
    sorted_rows = sorted(
        list(roster_rows or []),
        key=lambda item: (int(item.get("order") or 0), str(item.get("display_name") or "")),
    )
    participant_names: list[str] = []
    resolved_ids: dict[str, int] = {}
    review_messages: list[str] = []
    created_names: list[str] = []
    for row in sorted_rows:
        display_name = normalize_name(row.get("display_name"))
        if not display_name:
            continue
        participant_names.append(display_name)
        status = str(row.get("resolution_status") or "create_new_player")
        selected_existing_name = normalize_name(row.get("selected_existing_name"))
        selected_pid = row.get("player_id")
        if selected_pid is None and selected_existing_name in player_name_to_id:
            selected_pid = int(player_name_to_id[selected_existing_name])

        if status == "needs_review":
            suggestion = selected_existing_name or "an existing player"
            review_messages.append(
                f"Review roster: {display_name} is close to {suggestion}. Choose existing player or create a new player."
            )
            continue

        if status == "existing_player":
            if selected_pid is None:
                review_messages.append(
                    f"Review roster: {display_name} must select a matched existing player."
                )
                continue
            resolved_ids[display_name] = int(selected_pid)
            continue

        rating_jupr = _new_player_rating(
            row.get("starting_jupr_rating"),
            fallback=default_new_player_rating,
        )
        if rating_jupr is None:
            review_messages.append(
                f"Review roster: {display_name} needs an explicit Starting JUPR before a new player can be created."
            )
            continue
        ok, err = safe_add_player(
            supabase=ctx.supabase,
            club_id=str(ctx.club_id),
            name=display_name,
            rating_jupr=rating_jupr,
        )
        if not ok:
            raise RuntimeError(err or f"Unable to create rated player for {display_name}.")
        created = _fetch_player_by_exact_name(
            ctx.supabase,
            club_id=str(ctx.club_id),
            display_name=display_name,
        )
        if created is None or created.get("id") is None:
            raise RuntimeError(f"Unable to resolve created rated player for {display_name}.")
        pid = int(created["id"])
        resolved_ids[display_name] = pid
        player_name_to_id[display_name] = pid
        created_names.append(display_name)
        if isinstance(getattr(ctx, "name_to_id", None), dict):
            ctx.name_to_id[display_name] = pid
    return participant_names, resolved_ids, review_messages, created_names

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


def _rating_mode_context_ui(ctx, state: dict) -> tuple[dict, bool]:
    disabled = not bool(getattr(ctx, "admin_logged_in", False))
    options = ["Rated", "Unrated"]
    current_mode = str(state.get("rating_mode") or "Rated")
    if current_mode not in options:
        current_mode = "Rated"
    state["rating_mode"] = st.radio(
        "Rating mode",
        options,
        horizontal=True,
        index=options.index(current_mode),
        key="jupr_live_rating_mode",
        disabled=disabled,
    )
    if disabled:
        st.warning("Official mode requires admin login before event creation or save.")
    if state["rating_mode"] == "Unrated":
        return {
            "league": "JUPR Live",
            "week_tag": "",
            "match_type": "JUPR Live Unrated",
            "is_popup": False,
            "rating_scope": "unrated",
        }, not disabled
    return {
        "league": "JUPR Live",
        "week_tag": "",
        "match_type": "JUPR Live Rated",
        "is_popup": False,
        "rating_scope": "overall_only",
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
    player_options, player_name_to_id = _player_directory(ctx)
    player_name_to_id = player_name_to_id or {}
    selected_players_key = f"{config.state_key}_selected_existing_players"
    participants_key = f"{config.state_key}_participants"
    if selected_players_key not in st.session_state:
        st.session_state[selected_players_key] = state.get("selected_existing_players", [])
    default_rating_key = f"{config.state_key}_default_new_player_rating"
    previous_selected_existing_players = list(state.get("selected_existing_players") or [])
    selected_existing_players = st.multiselect(
        "Add from current players",
        options=player_options,
        key=selected_players_key,
        help="Search and select existing player names to quickly append them to the roster.",
    )
    state["selected_existing_players"] = list(selected_existing_players)
    use_admin_roster_builder = bool(config.use_admin_roster_builder) and state["type_label"] in {
        "Round Robin",
        "League / Ladder",
    }
    if use_admin_roster_builder:
        quick_paste_nonce = int(state.get("quick_paste_nonce") or 0)
        quick_paste_key = f"{config.state_key}_quick_paste_{quick_paste_nonce}"
        state["default_new_player_rating"] = _new_player_rating(
            st.number_input(
                "New-player starting JUPR (optional batch value)",
                min_value=1.0,
                max_value=7.0,
                value=_new_player_rating(state.get("default_new_player_rating")),
                step=0.05,
                placeholder="Enter a batch value",
                key=default_rating_key,
                help="If entered, this pre-fills new-player rows. Every new player must have an explicit starting JUPR before creation.",
            )
        )
        st.caption(
            "Existing players keep their current overall JUPR. Enter a starting JUPR for every new-player row."
        )
        quick_paste = st.text_area(
            "Quick paste names (one per line)",
            value="",
            height=120,
            placeholder=placeholder,
            key=quick_paste_key,
        )
        if st.button("Append pasted names", key=f"{config.state_key}_append_paste"):
            incoming = _participant_lines(quick_paste)
            state["admin_roster_rows"] = _append_roster_names(
                ctx,
                list(state.get("admin_roster_rows") or []),
                incoming,
                player_name_to_id=player_name_to_id,
                default_new_player_rating=state.get("default_new_player_rating"),
            )
            state["quick_paste_nonce"] = quick_paste_nonce + 1
            st.rerun()
        prev_selected = set(previous_selected_existing_players)
        newly_added = [
            name
            for name in state["selected_existing_players"]
            if name not in prev_selected
        ]
        if newly_added:
            state["admin_roster_rows"] = _append_roster_names(
                ctx,
                list(state.get("admin_roster_rows") or []),
                newly_added,
                player_name_to_id=player_name_to_id,
                default_new_player_rating=state.get("default_new_player_rating"),
            )
        roster_rows = list(state.get("admin_roster_rows") or [])
        if not roster_rows and state.get("participant_text"):
            roster_rows = _append_roster_names(
                ctx,
                [],
                _participant_lines(state["participant_text"]),
                player_name_to_id=player_name_to_id,
                default_new_player_rating=state.get("default_new_player_rating"),
            )
        if not roster_rows:
            st.caption("Add current players or paste names to build the roster.")
        editor_source = pd.DataFrame(
            [
                {
                    "Order": int(row.get("order") or (idx + 1)),
                    "Name": str(row.get("display_name") or ""),
                    "Resolution": str(row.get("resolution_status") or "create_new_player"),
                    "Matched Player": str(row.get("selected_existing_name") or ""),
                    "Current / Starting JUPR": _new_player_rating(
                        row.get("starting_jupr_rating"),
                        fallback=(
                            state.get("default_new_player_rating")
                            if row.get("resolution_status") == "create_new_player"
                            else None
                        ),
                    ),
                }
                for idx, row in enumerate(roster_rows)
            ],
            columns=["Order", "Name", "Resolution", "Matched Player", "Current / Starting JUPR"],
        )
        edited_df = st.data_editor(
            editor_source,
            num_rows="dynamic",
            hide_index=True,
            key=f"{config.state_key}_admin_roster_editor",
            column_config={
                "Order": st.column_config.NumberColumn("Order", min_value=1, step=1),
                "Name": st.column_config.TextColumn("Name"),
                "Resolution": st.column_config.SelectboxColumn(
                    "Resolution",
                    options=["existing_player", "create_new_player", "needs_review"],
                ),
                "Matched Player": st.column_config.SelectboxColumn(
                    "Matched Player",
                    options=[""] + player_options,
                ),
                "Current / Starting JUPR": st.column_config.NumberColumn(
                    "Current / Starting JUPR",
                    min_value=1.0,
                    max_value=7.0,
                    step=0.05,
                    required=True,
                ),
            },
        )
        state["admin_roster_rows"] = _rows_from_admin_editor_df(
            edited_df,
            player_name_to_id=player_name_to_id,
            ctx=ctx,
            default_new_player_rating=state.get("default_new_player_rating"),
        )
        state["participant_text"] = "\n".join(
            row["display_name"] for row in (state.get("admin_roster_rows") or [])
        )
        st.session_state[participants_key] = state["participant_text"]
    else:
        st.caption(
            "Search current players to add them quickly. You can still type guest names below."
        )
        merged_participant_text = _merge_participant_text(
            state["participant_text"], state["selected_existing_players"]
        )
        if merged_participant_text != state["participant_text"]:
            state["participant_text"] = merged_participant_text
            st.session_state[participants_key] = merged_participant_text
        state["participant_text"] = st.text_area(
            "Names or roster entry",
            value=state["participant_text"],
            height=180,
            placeholder=placeholder,
            key=participants_key,
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
    elif config.show_rating_mode:
        official_context, can_create = _rating_mode_context_ui(ctx, state)
    participant_names = _participant_lines(state["participant_text"])
    if use_admin_roster_builder:
        participant_names = [
            str(row.get("display_name") or "")
            for row in (state.get("admin_roster_rows") or [])
            if normalize_name(row.get("display_name"))
        ]
    roster_requires_resolution = bool(config.requires_roster_resolution) and state["type_label"] in {
        "Round Robin",
        "League / Ladder",
    }
    if roster_requires_resolution and not use_admin_roster_builder:
        if state.get("parsed_roster_lines") != participant_names:
            state["parsed_roster_lines"] = list(participant_names)
            state["roster_candidates"] = _build_roster_candidate_rows(
                participant_names,
                player_name_to_id,
            )
            state["confirmed_roster_rows"] = []
            state["roster_confirmed"] = False
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
    elif use_admin_roster_builder:
        required_count = int(state["participant_count"])
        if not participant_names:
            st.info("Add current players or paste names to build the roster.")
            can_create = False
        elif len(participant_names) != required_count:
            st.info(
                f"You entered {len(participant_names)} name(s); count is set to {required_count}."
            )
            can_create = False
    elif participant_names and len(participant_names) != int(state["participant_count"]):
        st.info(
            f"You entered {len(participant_names)} name(s); count is set to {int(state['participant_count'])}."
        )
        can_create = False
    if roster_requires_resolution and participant_names and not use_admin_roster_builder:
        roster_candidates = list(state.get("roster_candidates") or [])
        st.markdown("#### Step 1: Review roster matches")
        st.caption(
            "Confirm each pasted name once. Choose an existing rated profile or create a social-only profile."
        )
        with st.form(f"{config.state_key}_roster_resolution_form"):
            confirmed_rows: list[dict] = []
            for idx, row in enumerate(roster_candidates):
                status_col, selector_col = st.columns([1.3, 3.7])
                status_col.caption(f"**{row.get('original', '')}**")
                status_col.caption(f"Status: {row.get('status', 'create rated')}")
                duplicate_warning = str(row.get("duplicate_warning") or "").strip()
                if duplicate_warning:
                    status_col.warning(duplicate_warning)
                options = list(row.get("options") or [])
                labels = [str(opt.get("label") or "") for opt in options]
                tokens = [str(opt.get("token") or "") for opt in options]
                default_token = str(row.get("default_token") or "")
                default_index = tokens.index(default_token) if default_token in tokens else max(len(tokens) - 1, 0)
                selected_label = selector_col.selectbox(
                    f"Match choice {idx + 1}",
                    options=labels,
                    index=default_index,
                    key=f"{config.state_key}_roster_choice_{idx}",
                    label_visibility="collapsed",
                )
                selected_token = tokens[labels.index(selected_label)] if selected_label in labels else default_token
                duplicate_confirmed = False
                duplicate_note = ""
                if duplicate_warning and selected_token.startswith("new::"):
                    duplicate_confirmed = selector_col.checkbox(
                        "I intentionally want a separate social-only profile for this similar name.",
                        value=False,
                        key=f"{config.state_key}_dup_confirm_{idx}",
                    )
                    duplicate_note = selector_col.text_input(
                        "Duplicate confirmation note (required)",
                        value="",
                        key=f"{config.state_key}_dup_note_{idx}",
                    ).strip()
                confirmed_rows.append(
                    {
                        "original": str(row.get("original") or ""),
                        "selection": selected_token,
                        "status": str(row.get("status") or ""),
                        "duplicate_confirmed": duplicate_confirmed,
                        "duplicate_note": duplicate_note,
                    }
                )
            confirmed = st.form_submit_button("Confirm roster", type="primary")
        if confirmed:
            duplicate_errors: list[str] = []
            for row in confirmed_rows:
                if not str(row.get("selection") or "").startswith("new::"):
                    continue
                if row.get("duplicate_confirmed") and str(row.get("duplicate_note") or "").strip():
                    continue
                for source_row in roster_candidates:
                    if str(source_row.get("original") or "") == str(row.get("original") or "") and str(
                        source_row.get("duplicate_warning") or ""
                    ).strip():
                        duplicate_errors.append(
                            f"{row.get('original')}: check duplicate confirmation and add a note to create a separate social-only profile."
                        )
                        break
            if duplicate_errors:
                st.error("\n".join(duplicate_errors))
                st.stop()
            canonical_names, resolved_ids, resolved_rows = _resolved_participants_from_confirmation(
                confirmed_rows,
                player_name_to_id,
            )
            if len(canonical_names) != int(state["participant_count"]):
                st.error(
                    f"Confirmed roster has {len(canonical_names)} entries; expected {int(state['participant_count'])}."
                )
            else:
                state["confirmed_roster_rows"] = resolved_rows
                state["roster_confirmed"] = True
                state["participant_text"] = "\n".join(canonical_names)
                state["resolved_roster_ids"] = resolved_ids
                st.success("Step 2 complete: roster confirmed. You can now create the event.")
                st.rerun()
        if state.get("roster_confirmed"):
            confirmed_rows = list(state.get("confirmed_roster_rows") or [])
            matched_rows = [row for row in confirmed_rows if row.get("player_id") is not None]
            social_only_rows = [row for row in confirmed_rows if row.get("player_id") is None]
            st.caption(
                f"Confirmed roster: {len(matched_rows)} matched existing rated players, "
                f"{len(social_only_rows)} social-only players for admin review."
            )
            if matched_rows:
                st.caption("Matched: " + ", ".join(str(row.get("name") or "") for row in matched_rows))
            if social_only_rows:
                st.caption("Social-only fallback: " + ", ".join(str(row.get("name") or "") for row in social_only_rows))

    action_cols = st.columns([1, 1, 3])
    if roster_requires_resolution and not use_admin_roster_builder:
        st.caption("Step 3: Create event → enter scores → submit Club Social results.")
    create_disabled = not can_create or (
        (roster_requires_resolution and not use_admin_roster_builder and not bool(state.get("roster_confirmed")))
    )
    if action_cols[0].button(
        "Create event",
        type="primary",
        disabled=create_disabled,
        key=f"{config.state_key}_create_btn",
    ):
        try:
            create_participant_names = list(participant_names)
            create_resolved_ids: dict[str, int] | None = None
            created_player_names: list[str] = []
            confirmed_roster_rows = list(state.get("confirmed_roster_rows") or [])
            if use_admin_roster_builder:
                (
                    create_participant_names,
                    admin_resolved_ids,
                    review_messages,
                    created_player_names,
                ) = _create_and_resolve_admin_players(
                    ctx,
                    roster_rows=list(state.get("admin_roster_rows") or []),
                    default_new_player_rating=state.get("default_new_player_rating"),
                    player_name_to_id=player_name_to_id,
                )
                if review_messages:
                    raise ValueError("\n".join(review_messages))
                create_resolved_ids = dict(admin_resolved_ids)
            elif roster_requires_resolution and confirmed_roster_rows:
                create_participant_names, create_resolved_ids, resolved_rows = _resolved_participants_from_confirmation(
                    [
                        {
                            "original": row.get("source_name") or row.get("name"),
                            "selection": (
                                f"player::{int(row.get('player_id'))}"
                                if row.get("player_id") is not None
                                else f"new::{_normalized_person_key(row.get('name'))}"
                            ),
                            "duplicate_confirmed": bool(row.get("duplicate_confirmed", False)),
                            "duplicate_note": str(row.get("duplicate_note") or "").strip(),
                        }
                        for row in confirmed_roster_rows
                    ],
                    player_name_to_id,
                )
            if state["type_label"] == "Round Robin":
                resolved_ids = None
                if _is_official(config) and use_admin_roster_builder:
                    resolved_ids = dict(create_resolved_ids or {})
                elif _is_official(config):
                    resolved_ids, missing = _resolved_ids_for_official(
                        create_participant_names, getattr(ctx, "name_to_id", {})
                    )
                    if missing:
                        raise ValueError(
                            "Official mode could not resolve: " + ", ".join(missing)
                        )
                elif create_resolved_ids:
                    resolved_ids = dict(create_resolved_ids)
                state["event"] = create_round_robin_event(
                    name=state["event_name"],
                    participant_names=create_participant_names,
                    resolved_ids=resolved_ids,
                    official_context=official_context,
                )
                if roster_requires_resolution and confirmed_roster_rows:
                    participant_by_name = {
                        str(p.get("name") or ""): p for p in (state["event"].get("participants") or [])
                    }
                    for row in confirmed_roster_rows:
                        participant = participant_by_name.get(str(row.get("name") or ""))
                        if not participant:
                            continue
                        participant["social_resolution"] = dict(row)
                        participant["match_status"] = str(row.get("match_status") or "")
                        participant["source_name"] = str(row.get("source_name") or "")
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
                exact_sizes = suggest_exact_league_court_sizes(len(create_participant_names))
                if not exact_sizes:
                    raise ValueError(
                        "League / Ladder requires an exact 4-player / 5-player court fit."
                    )
                resolved_ids = None
                if _is_official(config) and use_admin_roster_builder:
                    resolved_ids = dict(create_resolved_ids or {})
                elif _is_official(config):
                    resolved_ids, missing = _resolved_ids_for_official(
                        create_participant_names, getattr(ctx, "name_to_id", {})
                    )
                    if missing:
                        raise ValueError(
                            "Official mode could not resolve: " + ", ".join(missing)
                        )
                elif create_resolved_ids:
                    resolved_ids = dict(create_resolved_ids)
                state["event"] = create_league_event(
                    name=state["event_name"],
                    participant_names=create_participant_names,
                    total_rounds=int(state["league_rounds"]),
                    resolved_ids=resolved_ids,
                    court_sizes=exact_sizes,
                    official_context=official_context,
                )
                if roster_requires_resolution and confirmed_roster_rows:
                    participant_by_name = {
                        str(p.get("name") or ""): p for p in (state["event"].get("participants") or [])
                    }
                    for row in confirmed_roster_rows:
                        participant = participant_by_name.get(str(row.get("name") or ""))
                        if not participant:
                            continue
                        participant["social_resolution"] = dict(row)
                        participant["match_status"] = str(row.get("match_status") or "")
                        participant["source_name"] = str(row.get("source_name") or "")
            state["last_saved_rounds"] = []
            if created_player_names:
                st.session_state["force_data_refresh"] = True
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


def _active_round_substitutions(event: dict, round_number: int) -> list[dict]:
    substitutions = [
        sub
        for sub in (event.get("substitutions") or [])
        if str(sub.get("scope")) == "round"
        and int(sub.get("round_number") or 0) == int(round_number)
        and substitution_is_active(event, sub)
    ]
    substitutions.sort(
        key=lambda item: (
            str(item.get("original_player_name") or ""),
            str(item.get("created_at") or ""),
        )
    )
    return substitutions


def _render_round_sub_summary(event: dict, round_number: int) -> None:
    substitutions = _active_round_substitutions(event, round_number)
    if not substitutions:
        return
    summary = " · ".join(
        f"{sub.get('substitute_name')} for {sub.get('original_player_name')}"
        for sub in substitutions
    )
    st.markdown(
        f"<div class='jupr-live-sub-summary'><strong>Active round subs:</strong> {summary}</div>",
        unsafe_allow_html=True,
    )


def _find_substitution(
    event: dict,
    *,
    scope: str,
    original_participant_id: str,
    match_id: str | None = None,
    round_number: int | None = None,
) -> dict | None:
    substitutions = list(event.get("substitutions") or [])
    substitutions.sort(key=lambda item: str(item.get("created_at") or ""))
    selected: dict | None = None
    for substitution in substitutions:
        if str(substitution.get("scope")) != str(scope):
            continue
        if str(substitution.get("original_participant_id")) != str(original_participant_id):
            continue
        if match_id is not None and str(substitution.get("match_id") or "") != str(match_id):
            continue
        if round_number is not None and int(substitution.get("round_number") or 0) != int(round_number):
            continue
        selected = substitution
    return selected


def _render_substitutions_area(ctx, state: dict, event: dict, config: LivePageConfig) -> None:
    if not _is_official(config):
        return
    clear_expired_substitutions(event)
    substitutions = list(event.get("substitutions") or [])
    st.markdown("#### Substitutions")
    if not substitutions:
        st.caption("Use the edit icon on a player slot to swap in a sub for this game or the rest of the round.")
        return
    with st.expander("View substitution audit", expanded=False):
        for substitution in substitutions:
            c1, c2 = st.columns([6, 1.2])
            c1.write(_substitution_summary(substitution, event))
            can_remove = substitution_is_active(event, substitution) and not substitution_is_locked(event, substitution)
            if c2.button(
                "Remove",
                key=f"{config.state_key}_remove_sub_{substitution['id']}",
                disabled=not can_remove,
            ):
                _remove_substitution(event, str(substitution["id"]))
                st.rerun()


def _render_player_slot_editor(
    ctx,
    event: dict,
    config: LivePageConfig,
    match: dict,
    *,
    participant_id: str,
    slot_label: str,
) -> None:
    match_id = str(match.get("id"))
    slot_markup = (
        f"<div class='jupr-live-slot'><div class='jupr-live-slot-label'>{slot_label}</div>"
        f"<div class='jupr-live-slot-name'>{resolve_display_name(event, match_id, participant_id)}</div></div>"
    )
    with st.container():
        if not _is_official(config) or match_is_scored(match):
            st.markdown(slot_markup, unsafe_allow_html=True)
            return
        player_options, player_name_to_id = _player_directory(ctx)
        if not player_options:
            st.markdown(slot_markup, unsafe_allow_html=True)
            return
        round_number = _match_round_number(event, match_id)
        original_name = _participant_name_map(event).get(participant_id, participant_id)
        round_sub = _find_substitution(
            event,
            scope="round",
            original_participant_id=participant_id,
            round_number=round_number,
        )
        game_sub = _find_substitution(
            event,
            scope="game",
            original_participant_id=participant_id,
            match_id=match_id,
        )
        active_sub = get_active_sub_for_match(event, match_id, participant_id, include_inactive=True)
        active_scope_label = "This game only" if game_sub else ("Rest of round" if round_sub else "Original player")
        status = original_name if active_sub is None else f"{resolve_active_player_name(event, match_id, participant_id)} • {active_scope_label}"
        editor_key = f"{config.state_key}_slot_editor_{match_id}_{participant_id}"
        c_name, c_action = st.columns([5, 1])
        c_name.markdown(slot_markup, unsafe_allow_html=True)
        if c_action.button("Edit", key=f"{editor_key}_toggle"):
            st.session_state[editor_key] = not bool(st.session_state.get(editor_key, False))
            st.rerun()
        if not bool(st.session_state.get(editor_key, False)):
            return
        st.caption(f"Slot owner: {original_name} • Active: {status}")
        scope_options = ["This game only", "Rest of round"]
        preferred_scope = 0 if game_sub or not round_sub else 1
        scope_key = f"{editor_key}_scope"
        if scope_key not in st.session_state:
            st.session_state[scope_key] = scope_options[preferred_scope]
        elif st.session_state.get(scope_key) not in scope_options:
            st.session_state[scope_key] = scope_options[preferred_scope]
        selected_scope = st.radio(
            "Apply change to",
            scope_options,
            horizontal=True,
            key=scope_key,
        )
        scoped_sub = game_sub if selected_scope == "This game only" else round_sub
        scoped_sub_data = scoped_sub or {}
        default_substitute = ""
        if scoped_sub_data:
            default_substitute = str(scoped_sub_data.get("substitute_name") or "")
        elif active_sub:
            default_substitute = str(active_sub.get("substitute_name") or "")
        scope_token = "game" if selected_scope == "This game only" else "round"
        scoped_sub_id = str(scoped_sub_data.get("id") or "none")
        substitute_key = f"{editor_key}_substitute_{scope_token}_{scoped_sub_id}"
        note_key = f"{editor_key}_note_{scope_token}_{scoped_sub_id}"
        if substitute_key not in st.session_state:
            st.session_state[substitute_key] = (
                default_substitute if default_substitute in player_options else player_options[0]
            )
        elif st.session_state.get(substitute_key) not in player_options:
            st.session_state[substitute_key] = player_options[0]
        if note_key not in st.session_state:
            st.session_state[note_key] = str(scoped_sub_data.get("note") or "")
        selected_name = st.selectbox(
            "Replacement player",
            player_options,
            key=substitute_key,
        )
        note_value = st.text_input("Note", key=note_key)
        apply_col, clear_col, cancel_col = st.columns([1, 1, 1])
        if apply_col.button("Apply", type="primary", key=f"{editor_key}_apply"):
            try:
                if selected_scope == "Rest of round":
                    substitution = apply_round_substitution(
                        event,
                        round_number=round_number,
                        original_participant_id=participant_id,
                        substitute_player_id=int(player_name_to_id[selected_name]),
                        substitute_name=selected_name,
                        created_by="admin",
                        created_at=_utc_now_iso(),
                        note=str(note_value or ""),
                        substitution_id=(str(round_sub["id"]) if round_sub else None),
                    )
                else:
                    substitution = apply_single_game_substitution(
                        event,
                        round_number=round_number,
                        match_id=match_id,
                        original_participant_id=participant_id,
                        substitute_player_id=int(player_name_to_id[selected_name]),
                        substitute_name=selected_name,
                        created_by="admin",
                        created_at=_utc_now_iso(),
                        note=str(note_value or ""),
                        substitution_id=(str(game_sub["id"]) if game_sub else None),
                    )
                _upsert_substitution(event, substitution)
                st.session_state[editor_key] = False
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        removable = game_sub if selected_scope == "This game only" else round_sub
        if clear_col.button(
            "Clear",
            key=f"{editor_key}_clear",
            disabled=removable is None or substitution_is_locked(event, removable),
        ):
            _remove_substitution(event, str(removable["id"]))
            st.session_state[editor_key] = False
            st.rerun()
        if cancel_col.button("Done", key=f"{editor_key}_done"):
            st.session_state[editor_key] = False
            st.rerun()


def _render_match_team(
    ctx,
    event: dict,
    config: LivePageConfig,
    match: dict,
    *,
    team_label: str,
    participant_ids: list[str],
) -> None:
    for index, pid in enumerate(participant_ids, start=1):
        _render_player_slot_editor(
            ctx,
            event,
            config,
            match,
            participant_id=str(pid),
            slot_label=f"{team_label} • Player {index}",
        )


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


def official_context_payload(event: dict, state: dict) -> dict:
    context = dict(event.get("official_context") or {})
    if context:
        return context
    return official_base_payload(state)


def build_rr_official_payloads(state: dict, event: dict) -> list[dict]:
    payloads = resolve_payload_player_ids(
        event,
        match_payloads_from_rr(event),
        materialize_substitutions=True,
    )
    context = official_context_payload(event, state)
    return [{**payload, **context} for payload in payloads]


def build_league_round_official_payloads(state: dict, event: dict) -> list[dict]:
    payloads = resolve_payload_player_ids(
        event,
        match_payloads_from_current_league_round(event),
        materialize_substitutions=True,
    )
    context = official_context_payload(event, state)
    return [{**payload, **context} for payload in payloads]


def build_tournament_official_payloads(state: dict, event: dict) -> list[dict]:
    payloads = tournament_completed_match_payloads(event, unsaved_only=True)
    context = official_context_payload(event, state)
    return [{**payload, **context} for payload in payloads]


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
            if int(round_data.get("number") or 0) == _current_round_number(event):
                _render_round_sub_summary(event, int(round_data.get("number") or 0))
            for match in round_data.get("matches") or []:
                st.markdown(
                    '<div class="jupr-live-score-shell">', unsafe_allow_html=True
                )
                cols = st.columns([3.6, 1.1, 0.6, 1.1, 3.6])
                with cols[0]:
                    _render_match_team(
                        ctx,
                        event,
                        config,
                        match,
                        team_label="Team A",
                        participant_ids=[str(pid) for pid in (match.get("teamA") or [])],
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
                with cols[4]:
                    _render_match_team(
                        ctx,
                        event,
                        config,
                        match,
                        team_label="Team B",
                        participant_ids=[str(pid) for pid in (match.get("teamB") or [])],
                    )
                st.caption(str(match.get("desc") or ""))
                st.markdown("</div>", unsafe_allow_html=True)
            st.divider()
        submit_label = _save_button_label(
            config,
            on_save_rr is not None,
            default_non_official="Update live standings",
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
            if on_save_rr is not None:
                if _is_official(config) and not bool(getattr(ctx, "admin_logged_in", False)):
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
        _render_round_sub_summary(event, current_round_number)
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
                    with cols[0]:
                        _render_match_team(
                            ctx,
                            event,
                            config,
                            match,
                            team_label="Team A",
                            participant_ids=[str(pid) for pid in (match.get("teamA") or [])],
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
                    with cols[4]:
                        _render_match_team(
                            ctx,
                            event,
                            config,
                            match,
                            team_label="Team B",
                            participant_ids=[str(pid) for pid in (match.get("teamB") or [])],
                        )
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
                    _maybe_save_league_before_advance(
                        ctx, state, event, config, on_save_league
                    )
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


def _maybe_save_league_before_advance(
    ctx,
    state: dict,
    event: dict,
    config: LivePageConfig,
    on_save_league: SaveCallback | None,
) -> None:
    if on_save_league is None:
        return
    if _is_official(config) and not bool(getattr(ctx, "admin_logged_in", False)):
        raise ValueError("Admin login required to save official round results.")
    if on_save_league(ctx, state, event) is False:
        st.stop()


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
