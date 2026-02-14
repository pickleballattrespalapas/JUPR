from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from jupr_app.domain.leagues import get_league_meta_row
from jupr_app.domain.ratings import calculate_hybrid_elo
from jupr_app.ui.layout import page_shell
from jupr_app.ui.theme_tokens import get_theme_tokens
from services.match_pipeline import submit_match


WIZARD_STEP_KEY = "record_match_wizard_step"
SELECTED_TYPE_KEY = "record_match_competition_type"

COMPETITION_TYPES: list[dict[str, str]] = [
    {
        "id": "ladder_league",
        "title": "Ladder League",
        "icon": "🪜",
        "description": "Structured weekly ladder play with tracked standings.",
    },
    {
        "id": "challenge_ladder",
        "title": "Challenge Ladder",
        "icon": "⚔️",
        "description": "Open challenges where players climb by results.",
    },
    {
        "id": "tournament",
        "title": "Tournament",
        "icon": "🏆",
        "description": "Bracketed or pool-based tournament results.",
    },
    {
        "id": "round_robin",
        "title": "Round Robin",
        "icon": "🔄",
        "description": "Everyone plays everyone within a group.",
    },
    {
        "id": "moneyball",
        "title": "Moneyball",
        "icon": "💰",
        "description": "Moneyball format sessions and side-game outcomes.",
    },
    {
        "id": "bulk_match_entry",
        "title": "Bulk Match Entry",
        "icon": "📥",
        "description": "Fast multi-match input for admins and captains.",
    },
]


def _ensure_state() -> None:
    st.session_state.setdefault(WIZARD_STEP_KEY, 1)
    st.session_state.setdefault(SELECTED_TYPE_KEY, None)


def _step_1_competition_type(tokens: dict[str, str]) -> None:
    st.markdown("### Step 1 · Choose competition type")
    st.caption("Select the format first. The wizard will tailor next steps to this selection.")

    st.markdown(
        f"""
        <style>
        .record-match-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
            margin: 0.5rem 0 1rem 0;
        }}
        .record-match-card {{
            border: 1px solid {tokens['border_subtle']};
            border-radius: 12px;
            background: {tokens['card_bg']};
            padding: 12px;
            color: {tokens['text_primary']};
            min-height: 112px;
        }}
        .record-match-card h4 {{
            margin: 0 0 6px 0;
            color: {tokens['text_primary']};
        }}
        .record-match-card p {{
            margin: 0;
            color: {tokens['text_secondary']};
            font-size: 0.9rem;
            line-height: 1.35;
        }}
        .record-match-success-card {{
            border: 1px solid {tokens['border_subtle']};
            border-radius: 12px;
            background: {tokens['card_bg']};
            padding: 14px;
            margin-top: 0.8rem;
            color: {tokens['text_primary']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    card_html = "".join(
        [
            (
                "<div class='record-match-card'>"
                f"<h4>{option['icon']} {option['title']}</h4>"
                f"<p>{option['description']}</p>"
                "</div>"
            )
            for option in COMPETITION_TYPES
        ]
    )
    st.markdown(f"<div class='record-match-grid'>{card_html}</div>", unsafe_allow_html=True)

    selector_options = ["Select one…"] + [option["title"] for option in COMPETITION_TYPES]
    selected_title = st.selectbox(
        "Competition type",
        options=selector_options,
        key="record_match_competition_selector",
    )

    if selected_title != "Select one…":
        selected_option = next(opt for opt in COMPETITION_TYPES if opt["title"] == selected_title)
        st.session_state[SELECTED_TYPE_KEY] = selected_option

    controls = st.columns([1, 1, 3])
    with controls[0]:
        next_disabled = st.session_state.get(SELECTED_TYPE_KEY) is None
        if st.button("Next →", type="primary", disabled=next_disabled):
            st.session_state[WIZARD_STEP_KEY] = 2
            st.rerun()


def _clean_divisions(meta_row: dict[str, Any] | None) -> list[str]:
    if not meta_row:
        return []
    rules_cfg = meta_row.get("rules_config")
    parsed: dict[str, Any] = {}
    if isinstance(rules_cfg, dict):
        parsed = rules_cfg
    elif isinstance(rules_cfg, str) and rules_cfg.strip():
        try:
            parsed = json.loads(rules_cfg)
        except Exception:
            parsed = {}

    overview = parsed.get("overview") if isinstance(parsed, dict) else {}
    divisions = overview.get("divisions") if isinstance(overview, dict) else []
    if not isinstance(divisions, list):
        return []
    cleaned = [str(d).strip() for d in divisions if str(d).strip()]
    return cleaned


def _league_options(ctx) -> list[str]:
    league_names: set[str] = set()
    df_meta = getattr(ctx, "df_meta", None)
    if isinstance(df_meta, pd.DataFrame) and not df_meta.empty and "league_name" in df_meta.columns:
        league_names |= {
            str(x).strip()
            for x in df_meta["league_name"].dropna().astype(str).tolist()
            if str(x).strip() and str(x).strip().upper() != "OVERALL"
        }

    df_leagues = getattr(ctx, "df_leagues", None)
    if isinstance(df_leagues, pd.DataFrame) and not df_leagues.empty and "league_name" in df_leagues.columns:
        league_names |= {
            str(x).strip()
            for x in df_leagues["league_name"].dropna().astype(str).tolist()
            if str(x).strip() and str(x).strip().upper() != "OVERALL"
        }

    return sorted(league_names)


def _player_options_for_league(ctx, league_name: str) -> list[tuple[str, int]]:
    df_players = getattr(ctx, "df_players_active", None)
    if not isinstance(df_players, pd.DataFrame) or df_players.empty:
        df_players = getattr(ctx, "df_players", None)

    if not isinstance(df_players, pd.DataFrame) or df_players.empty:
        return []

    player_ids: set[int] = set()
    df_leagues = getattr(ctx, "df_leagues", None)
    if isinstance(df_leagues, pd.DataFrame) and not df_leagues.empty and {"league_name", "player_id"}.issubset(df_leagues.columns):
        scoped = df_leagues[df_leagues["league_name"].astype(str).str.strip() == str(league_name).strip()].copy()
        for pid in scoped["player_id"].dropna().tolist():
            try:
                player_ids.add(int(pid))
            except Exception:
                pass

    if player_ids and "id" in df_players.columns:
        active = df_players[df_players["id"].apply(lambda x: str(x).isdigit() and int(x) in player_ids)].copy()
    else:
        active = df_players.copy()

    if active.empty or "name" not in active.columns or "id" not in active.columns:
        return []

    rows: list[tuple[str, int]] = []
    for _, row in active.iterrows():
        try:
            pid = int(row.get("id"))
        except Exception:
            continue
        name = str(row.get("name") or "").strip() or f"Player {pid}"
        rows.append((f"{name} (#{pid})", pid))
    rows.sort(key=lambda item: item[0].lower())
    return rows


def _league_rating_map(ctx, league_name: str) -> dict[int, float]:
    df_leagues = getattr(ctx, "df_leagues", None)
    if not isinstance(df_leagues, pd.DataFrame) or df_leagues.empty:
        return {}
    needed = {"league_name", "player_id", "rating"}
    if not needed.issubset(df_leagues.columns):
        return {}

    scoped = df_leagues[df_leagues["league_name"].astype(str).str.strip() == str(league_name).strip()].copy()
    if scoped.empty:
        return {}

    ratings: dict[int, float] = {}
    for _, row in scoped.iterrows():
        try:
            ratings[int(row.get("player_id"))] = float(row.get("rating") or 1200.0)
        except Exception:
            continue
    return ratings


def _fallback_overall_rating_map(ctx) -> dict[int, float]:
    df_players = getattr(ctx, "df_players_all", None)
    if not isinstance(df_players, pd.DataFrame) or df_players.empty or "id" not in df_players.columns:
        return {}
    rating_col = None
    for col in ("rating", "jupr", "overall_rating"):
        if col in df_players.columns:
            rating_col = col
            break

    ratings: dict[int, float] = {}
    for _, row in df_players.iterrows():
        try:
            pid = int(row.get("id"))
        except Exception:
            continue
        if rating_col is None:
            ratings[pid] = 1200.0
            continue
        try:
            ratings[pid] = float(row.get(rating_col) or 1200.0)
        except Exception:
            ratings[pid] = 1200.0
    return ratings


def _deterministic_idempotency_key(
    *,
    club_id: str,
    league_id: str,
    player_ids: list[int],
    score_t1: int,
    score_t2: int,
    match_date: str,
) -> str:
    seed = "|".join(
        [
            str(club_id).strip(),
            str(league_id).strip(),
            "-".join(str(int(pid)) for pid in player_ids),
            f"{int(score_t1)}-{int(score_t2)}",
            str(match_date).strip(),
        ]
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _step_2_ladder_league(ctx, tokens: dict[str, str]) -> None:
    st.markdown("### Step 2 · Ladder League details")
    st.caption("Select league/division, assign teams, enter score, then review rating projection before confirming.")

    leagues = _league_options(ctx)
    if not leagues:
        st.warning("No leagues found. Create a league first in League Manager.")
        controls = st.columns([1, 4])
        with controls[0]:
            if st.button("← Back", key="rm_step2_back_no_league"):
                st.session_state[WIZARD_STEP_KEY] = 1
                st.rerun()
        return

    selected_league = st.selectbox("Select League", leagues, key="record_match_ll_league")
    meta_row = get_league_meta_row(getattr(ctx, "df_meta", None), selected_league)
    divisions = _clean_divisions(meta_row)

    selected_division: str | None = None
    if divisions:
        division_options = ["All divisions"] + divisions
        selected_division = st.selectbox("Select Division", division_options, key="record_match_ll_division")

    player_options = _player_options_for_league(ctx, selected_league)
    if not player_options:
        st.warning("No eligible players found for this league.")
        return

    labels = [label for label, _ in player_options]
    label_to_pid = {label: pid for label, pid in player_options}

    st.markdown("#### Teams")
    c1, c2 = st.columns(2)
    with c1:
        t1_p1_label = st.selectbox("Team 1 · Player 1", labels, key="rm_ll_t1_p1")
        t1_p2_label = st.selectbox("Team 1 · Player 2", labels, key="rm_ll_t1_p2")
    with c2:
        t2_p1_label = st.selectbox("Team 2 · Player 1", labels, key="rm_ll_t2_p1")
        t2_p2_label = st.selectbox("Team 2 · Player 2", labels, key="rm_ll_t2_p2")

    score_cols = st.columns(2)
    score_t1 = score_cols[0].number_input("Team 1 Score", min_value=0, max_value=99, value=0, step=1, key="rm_ll_s1")
    score_t2 = score_cols[1].number_input("Team 2 Score", min_value=0, max_value=99, value=0, step=1, key="rm_ll_s2")

    t1_p1 = label_to_pid[t1_p1_label]
    t1_p2 = label_to_pid[t1_p2_label]
    t2_p1 = label_to_pid[t2_p1_label]
    t2_p2 = label_to_pid[t2_p2_label]

    selected_ids = [t1_p1, t1_p2, t2_p1, t2_p2]
    unique_ids = len(set(selected_ids)) == 4
    has_score = int(score_t1) + int(score_t2) > 0

    ratings_map = _league_rating_map(ctx, selected_league)
    if not ratings_map:
        ratings_map = _fallback_overall_rating_map(ctx)

    def r(pid: int) -> float:
        return float(ratings_map.get(int(pid), 1200.0))

    d_t1, d_t2 = calculate_hybrid_elo(
        (r(t1_p1) + r(t1_p2)) / 2.0,
        (r(t2_p1) + r(t2_p2)) / 2.0,
        int(score_t1),
        int(score_t2),
    )

    projections = [
        (t1_p1_label, t1_p1, r(t1_p1), r(t1_p1) + float(d_t1)),
        (t1_p2_label, t1_p2, r(t1_p2), r(t1_p2) + float(d_t1)),
        (t2_p1_label, t2_p1, r(t2_p1), r(t2_p1) + float(d_t2)),
        (t2_p2_label, t2_p2, r(t2_p2), r(t2_p2) + float(d_t2)),
    ]

    st.markdown("### Step 3 · Rating preview & confirmation")
    st.markdown(
        f"""
        <div style="
            border: 1px solid {tokens['border_subtle']};
            border-radius: 12px;
            background: {tokens['card_bg']};
            padding: 14px;
            color: {tokens['text_primary']};
            margin: 0.25rem 0 0.75rem 0;
        ">
            <strong>League:</strong> {selected_league}<br/>
            <strong>Division:</strong> {selected_division or 'N/A'}
        </div>
        """,
        unsafe_allow_html=True,
    )

    preview_df = pd.DataFrame(
        [
            {
                "Player": label,
                "Current": round(current, 2),
                "Projected": round(projected, 2),
                "Δ": round(projected - current, 2),
            }
            for label, _pid, current, projected in projections
        ]
    )
    st.dataframe(preview_df, use_container_width=True, hide_index=True)

    if not unique_ids:
        st.error("Select 4 distinct players (2 per team).")
    if not has_score:
        st.error("Enter a non-zero score before confirming.")

    can_submit = unique_ids and has_score and bool(getattr(ctx, "club_id", None))

    controls = st.columns([1, 1, 4])
    with controls[0]:
        if st.button("← Back", key="rm_step2_back"):
            st.session_state[WIZARD_STEP_KEY] = 1
            st.rerun()
    with controls[1]:
        if st.button("Confirm & Submit", type="primary", disabled=not can_submit, key="rm_ll_submit"):
            club_id = str(getattr(ctx, "club_id", "")).strip()
            match_date = datetime.utcnow().isoformat()
            idem_key = _deterministic_idempotency_key(
                club_id=club_id,
                league_id=selected_league,
                player_ids=[t1_p1, t1_p2, t2_p1, t2_p2],
                score_t1=int(score_t1),
                score_t2=int(score_t2),
                match_date=match_date,
            )

            payload = {
                "date": match_date,
                "league": selected_league,
                "division": selected_division if selected_division and selected_division != "All divisions" else None,
                "t1_p1": int(t1_p1),
                "t1_p2": int(t1_p2),
                "t2_p1": int(t2_p1),
                "t2_p2": int(t2_p2),
                "score_t1": int(score_t1),
                "score_t2": int(score_t2),
                "t1_p1_r": float(r(t1_p1)),
                "t1_p2_r": float(r(t1_p2)),
                "t2_p1_r": float(r(t2_p1)),
                "t2_p2_r": float(r(t2_p2)),
                "t1_p1_r_end": float(r(t1_p1) + float(d_t1)),
                "t1_p2_r_end": float(r(t1_p2) + float(d_t1)),
                "t2_p1_r_end": float(r(t2_p1) + float(d_t2)),
                "t2_p2_r_end": float(r(t2_p2) + float(d_t2)),
            }

            try:
                result = submit_match(
                    club_id=club_id,
                    context_type="league",
                    context_id=selected_league,
                    match_payload=payload,
                    idempotency_key=idem_key,
                )
                st.session_state["record_match_last_submit"] = {
                    "league": selected_league,
                    "division": selected_division,
                    "score": f"{int(score_t1)} - {int(score_t2)}",
                    "idempotency_key": idem_key,
                    "result": result,
                }
                st.rerun()
            except Exception as exc:
                st.error(f"Submit failed: {exc}")

    last_submit = st.session_state.get("record_match_last_submit")
    if isinstance(last_submit, dict):
        st.markdown(
            (
                "<div class='record-match-success-card'>"
                "<h4 style='margin:0 0 6px 0;'>✅ Match submitted</h4>"
                f"<div><strong>League:</strong> {last_submit.get('league')}</div>"
                f"<div><strong>Division:</strong> {last_submit.get('division') or 'N/A'}</div>"
                f"<div><strong>Score:</strong> {last_submit.get('score')}</div>"
                f"<div><strong>Idempotency Key:</strong> <code>{last_submit.get('idempotency_key')}</code></div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def _step_2_placeholder(tokens: dict[str, str]) -> None:
    selected = st.session_state.get(SELECTED_TYPE_KEY)
    selected_label = selected["title"] if isinstance(selected, dict) else "Not selected"

    st.markdown("### Step 2 · Match details")
    st.caption("Scaffold only: non-Ladder-League flows are intentionally pending.")

    st.markdown(
        f"""
        <div style="
            border: 1px dashed {tokens['border_subtle']};
            border-radius: 12px;
            background: {tokens['card_bg']};
            padding: 14px;
            color: {tokens['text_primary']};
            margin: 0.5rem 0 1rem 0;
        ">
            <strong>Selected competition type:</strong> {selected_label}<br/>
            <span style="color: {tokens['text_secondary']};">
                This flow is currently implemented only for Ladder League.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    controls = st.columns([1, 1, 3])
    with controls[0]:
        if st.button("← Back"):
            st.session_state[WIZARD_STEP_KEY] = 1
            st.rerun()
    with controls[1]:
        st.button("Submit (coming soon)", type="primary", disabled=True)


def _step_2_challenge_ladder(ctx, tokens: dict[str, str]) -> None:
    st.markdown("### Step 2 · Challenge Ladder result")
    st.caption("Select a pending challenge. Players are auto-filled from the challenge record.")

    supabase = getattr(ctx, "supabase", None)
    club_id = str(getattr(ctx, "club_id", "") or "").strip()
    if not supabase or not club_id:
        st.error("Challenge Ladder submission requires club and database context.")
        return

    try:
        response = (
            supabase.table("ladder_challenges")
            .select("id,challenger_id,defender_id,status,accepted_at,play_by,created_at,tier_id")
            .eq("club_id", club_id)
            .in_("status", ["ACCEPTED", "ACCEPTED_SCHEDULING"])
            .is_("winner_id", "null")
            .is_("completed_at", "null")
            .order("created_at", desc=False)
            .limit(500)
            .execute()
        )
        pending_rows = getattr(response, "data", None) or []
    except Exception as exc:
        st.error(f"Could not load pending challenges: {exc}")
        pending_rows = []

    if not pending_rows:
        st.info("No pending accepted challenges available to record.")
        controls = st.columns([1, 4])
        with controls[0]:
            if st.button("← Back", key="rm_step2_back_challenge_empty"):
                st.session_state[WIZARD_STEP_KEY] = 1
                st.rerun()
        return

    id_to_name = getattr(ctx, "id_to_name", None) or {}

    def player_name(pid: Any) -> str:
        try:
            parsed = int(pid)
        except Exception:
            return "Unknown"
        return str(id_to_name.get(parsed) or f"Player #{parsed}")

    challenge_rows: list[dict[str, Any]] = []
    for raw in pending_rows:
        try:
            ch_id = int(raw.get("id"))
            challenger_id = int(raw.get("challenger_id"))
            defender_id = int(raw.get("defender_id"))
        except Exception:
            continue
        challenge_rows.append(
            {
                "challenge_id": ch_id,
                "challenger_id": challenger_id,
                "defender_id": defender_id,
                "challenger": player_name(challenger_id),
                "defender": player_name(defender_id),
                "status": str(raw.get("status") or ""),
                "play_by": raw.get("play_by"),
                "tier": str(raw.get("tier_id") or ""),
            }
        )

    if not challenge_rows:
        st.warning("Pending challenge rows are missing required players.")
        return

    st.dataframe(
        pd.DataFrame(challenge_rows),
        use_container_width=True,
        hide_index=True,
    )

    challenge_options = {
        f"#{row['challenge_id']} · {row['challenger']} vs {row['defender']} ({row['status']})": row
        for row in challenge_rows
    }
    selected_label = st.selectbox(
        "Select pending challenge",
        options=list(challenge_options.keys()),
        key="record_match_challenge_selector",
    )
    selected_challenge = challenge_options[selected_label]

    st.markdown(
        f"""
        <div style="
            border: 1px solid {tokens['border_subtle']};
            border-radius: 12px;
            background: {tokens['card_bg']};
            padding: 14px;
            color: {tokens['text_primary']};
            margin: 0.25rem 0 0.75rem 0;
        ">
            <strong>Challenge:</strong> #{selected_challenge['challenge_id']}<br/>
            <strong>Challenger:</strong> {selected_challenge['challenger']} (#{selected_challenge['challenger_id']})<br/>
            <strong>Defender:</strong> {selected_challenge['defender']} (#{selected_challenge['defender_id']})<br/>
            <strong>Tier:</strong> {selected_challenge['tier'] or 'N/A'}
        </div>
        """,
        unsafe_allow_html=True,
    )

    score_cols = st.columns(2)
    score_t1 = score_cols[0].number_input(
        "Challenger score",
        min_value=0,
        max_value=99,
        value=0,
        step=1,
        key="rm_cl_s1",
    )
    score_t2 = score_cols[1].number_input(
        "Defender score",
        min_value=0,
        max_value=99,
        value=0,
        step=1,
        key="rm_cl_s2",
    )

    has_score = int(score_t1) + int(score_t2) > 0
    if not has_score:
        st.error("Enter a non-zero score before confirming.")

    controls = st.columns([1, 1, 4])
    with controls[0]:
        if st.button("← Back", key="rm_step2_back_challenge"):
            st.session_state[WIZARD_STEP_KEY] = 1
            st.rerun()
    with controls[1]:
        if st.button(
            "Confirm & Submit",
            type="primary",
            disabled=not has_score,
            key="rm_cl_submit",
        ):
            match_date = datetime.utcnow().isoformat()
            idem_key = _deterministic_idempotency_key(
                club_id=club_id,
                league_id=f"challenge:{selected_challenge['challenge_id']}",
                player_ids=[selected_challenge["challenger_id"], selected_challenge["defender_id"]],
                score_t1=int(score_t1),
                score_t2=int(score_t2),
                match_date=match_date,
            )

            payload = {
                "date": match_date,
                "league": "OVERALL",
                "match_type": "ChallengeLadder",
                "t1_p1": int(selected_challenge["challenger_id"]),
                "t1_p2": None,
                "t2_p1": int(selected_challenge["defender_id"]),
                "t2_p2": None,
                "score_t1": int(score_t1),
                "score_t2": int(score_t2),
                "s1": int(score_t1),
                "s2": int(score_t2),
            }

            try:
                result = submit_match(
                    club_id=club_id,
                    context_type="ladder",
                    context_id=str(selected_challenge["challenge_id"]),
                    match_payload=payload,
                    idempotency_key=idem_key,
                )
                st.session_state["record_match_last_submit"] = {
                    "challenge": selected_challenge,
                    "score": f"{int(score_t1)} - {int(score_t2)}",
                    "idempotency_key": idem_key,
                    "result": result,
                }
                st.rerun()
            except Exception as exc:
                st.error(f"Submit failed: {exc}")

    last_submit = st.session_state.get("record_match_last_submit")
    if isinstance(last_submit, dict) and isinstance(last_submit.get("challenge"), dict):
        challenge = last_submit["challenge"]
        st.markdown(
            (
                "<div class='record-match-success-card'>"
                "<h4 style='margin:0 0 6px 0;'>✅ Challenge result submitted</h4>"
                f"<div><strong>Challenge:</strong> #{challenge.get('challenge_id')}</div>"
                f"<div><strong>Players:</strong> {challenge.get('challenger')} vs {challenge.get('defender')}</div>"
                f"<div><strong>Score:</strong> {last_submit.get('score')}</div>"
                f"<div><strong>Idempotency Key:</strong> <code>{last_submit.get('idempotency_key')}</code></div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def render(ctx) -> None:
    mode_label = "Public" if bool(getattr(ctx, "public_mode", False)) else "Admin"
    page_shell("🧾 Record Match", "Unified wizard for recording results across competition types.", mode_label=mode_label)

    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin only.")
        return

    _ensure_state()
    tokens = get_theme_tokens()

    step = int(st.session_state.get(WIZARD_STEP_KEY, 1))
    if step <= 1:
        _step_1_competition_type(tokens)
        return

    selected = st.session_state.get(SELECTED_TYPE_KEY)
    selected_id = selected.get("id") if isinstance(selected, dict) else None
    if selected_id == "ladder_league":
        _step_2_ladder_league(ctx, tokens)
    elif selected_id == "challenge_ladder":
        _step_2_challenge_ladder(ctx, tokens)
    else:
        _step_2_placeholder(tokens)
