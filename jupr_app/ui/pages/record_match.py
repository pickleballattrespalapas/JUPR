from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from jupr_app.domain.leagues import get_league_meta_row
from jupr_app.domain.ratings import calculate_hybrid_elo
from jupr_app.domain.tournaments import finalize_game, resolve_playoff_dependencies
from jupr_app.ui.layout import page_shell
from jupr_app.ui.theme_tokens import get_theme_tokens
from services.match_pipeline import submit_match


WIZARD_STEP_KEY = "record_match_wizard_step"
WIZARD_VIEW_STEP_KEY = "record_match_view_step"
WIZARD_LAST_VIEW_STEP_KEY = "record_match_last_view_step"
SELECTED_TYPE_KEY = "record_match_competition_type"
BULK_UPLOAD_STATE_KEY = "record_match_bulk_upload"
BULK_CHUNK_SIZE = 200
UNDO_BANNER_SECONDS = 10

WIZARD_PROGRESS_STEPS: list[str] = ["Competition", "Participants", "Score", "Confirm"]

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
    st.session_state.setdefault(WIZARD_VIEW_STEP_KEY, 1)
    st.session_state.setdefault(WIZARD_LAST_VIEW_STEP_KEY, 1)
    st.session_state.setdefault(SELECTED_TYPE_KEY, None)


def _render_motion_css(tokens: dict[str, str]) -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --motion-duration-fast: 140ms;
            --motion-duration-medium: 240ms;
            --motion-duration-slow: 340ms;
            --motion-ease-standard: cubic-bezier(0.22, 1, 0.36, 1);
        }}
        .record-match-success-card {{
            border: 1px solid {tokens['border_subtle']};
            border-radius: 12px;
            background: color-mix(in srgb, {tokens['card_bg']} 80%, #1f9d55 20%);
            padding: 14px;
            margin-top: 0.8rem;
            color: {tokens['text_primary']};
            max-height: 0;
            opacity: 0;
            overflow: hidden;
            transform: translateY(4px);
            transition:
                max-height var(--motion-duration-slow) var(--motion-ease-standard),
                opacity var(--motion-duration-medium) var(--motion-ease-standard),
                transform var(--motion-duration-medium) var(--motion-ease-standard);
        }}
        .record-match-success-card.is-visible {{
            max-height: 240px;
            opacity: 1;
            transform: translateY(0);
        }}
        .record-match-check {{
            opacity: 0;
            margin-right: 0.3rem;
            transition: opacity var(--motion-duration-slow) var(--motion-ease-standard);
        }}
        .record-match-success-card.is-visible .record-match-check {{
            opacity: 1;
        }}
        .record-match-loading-btn {{
            border: 1px solid {tokens['border_subtle']};
            border-radius: 8px;
            padding: 0.42rem 0.85rem;
            color: {tokens['text_secondary']};
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.92rem;
            background: {tokens['card_bg']};
        }}
        .record-match-loading-dot {{
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: {tokens['text_secondary']};
            animation: record-match-pulse 1s ease-in-out infinite;
        }}
        @keyframes record-match-pulse {{
            0%, 100% {{ opacity: 0.35; transform: scale(0.8); }}
            50% {{ opacity: 1; transform: scale(1); }}
        }}
        .record-match-undo {{
            border: 1px solid {tokens['border_subtle']};
            border-radius: 10px;
            padding: 0.6rem 0.8rem;
            margin-top: 0.6rem;
            background: {tokens['card_bg']};
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: {tokens['text_secondary']};
            transition: opacity var(--motion-duration-medium) var(--motion-ease-standard);
        }}
        .record-match-progress {{
            margin: 0.25rem 0 1rem;
            padding: 0.75rem;
            border: 1px solid {tokens['border_subtle']};
            border-radius: 12px;
            background: color-mix(in srgb, {tokens['card_bg']} 92%, {tokens['bg']} 8%);
        }}
        .record-match-progress-track {{
            position: relative;
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.55rem;
        }}
        .record-match-progress-step {{
            min-width: 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            opacity: 0.7;
            transition: opacity var(--motion-duration-fast) var(--motion-ease-standard);
        }}
        .record-match-progress-step.is-active,
        .record-match-progress-step.is-complete {{
            opacity: 1;
        }}
        .record-match-progress-dot {{
            width: 1.5rem;
            height: 1.5rem;
            border-radius: 999px;
            border: 1px solid {tokens['border_subtle']};
            background: {tokens['card_bg']};
            color: {tokens['text_secondary']};
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.78rem;
            font-weight: 600;
            flex-shrink: 0;
            transition:
                color var(--motion-duration-medium) var(--motion-ease-standard),
                background-color var(--motion-duration-medium) var(--motion-ease-standard),
                border-color var(--motion-duration-medium) var(--motion-ease-standard);
        }}
        .record-match-progress-step.is-active .record-match-progress-dot {{
            border-color: color-mix(in srgb, {tokens['text_primary']} 28%, {tokens['border_subtle']} 72%);
            color: {tokens['text_primary']};
            background: color-mix(in srgb, {tokens['card_bg']} 86%, {tokens['text_primary']} 14%);
        }}
        .record-match-progress-step.is-complete .record-match-progress-dot {{
            border-color: color-mix(in srgb, #1f9d55 35%, {tokens['border_subtle']} 65%);
            background: color-mix(in srgb, #1f9d55 18%, {tokens['card_bg']} 82%);
            color: {tokens['text_primary']};
        }}
        .record-match-progress-label {{
            color: {tokens['text_secondary']};
            font-size: 0.82rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            transition: color var(--motion-duration-medium) var(--motion-ease-standard);
        }}
        .record-match-progress-step.is-active .record-match-progress-label,
        .record-match-progress-step.is-complete .record-match-progress-label {{
            color: {tokens['text_primary']};
        }}
        .record-match-step-shell {{
            animation: record-match-step-fade var(--motion-duration-medium) var(--motion-ease-standard);
            will-change: opacity;
        }}
        .record-match-step-shell-steady {{
            animation: none;
        }}
        @keyframes record-match-step-fade {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _resolve_progress_step() -> int:
    if isinstance(st.session_state.get("record_match_last_submit"), dict):
        return 4
    wizard_step = int(st.session_state.get(WIZARD_STEP_KEY, 1))
    if wizard_step <= 1:
        return 1
    return 2


def _render_progress_indicator(current_step: int) -> None:
    current_step = max(1, min(current_step, len(WIZARD_PROGRESS_STEPS)))
    steps_html: list[str] = []
    for idx, label in enumerate(WIZARD_PROGRESS_STEPS, start=1):
        state = "is-pending"
        if idx < current_step:
            state = "is-complete"
        elif idx == current_step:
            state = "is-active"
        steps_html.append(
            (
                f"<div class='record-match-progress-step {state}'>"
                f"<span class='record-match-progress-dot'>{idx}</span>"
                f"<span class='record-match-progress-label'>Step {idx}: {label}</span>"
                "</div>"
            )
        )

    st.markdown(
        "<div class='record-match-progress'><div class='record-match-progress-track'>"
        + "".join(steps_html)
        + "</div></div>",
        unsafe_allow_html=True,
    )


def _confirm_loading_key(button_key: str) -> str:
    return f"{button_key}__loading"


def _render_confirm_submit_button(button_key: str, disabled: bool) -> bool:
    loading_key = _confirm_loading_key(button_key)
    if st.session_state.get(loading_key, False):
        st.markdown(
            "<div class='record-match-loading-btn'><span class='record-match-loading-dot'></span>Submitting…</div>",
            unsafe_allow_html=True,
        )
        return True
    if st.button("Confirm & Submit", type="primary", disabled=disabled, key=button_key):
        st.session_state[loading_key] = True
        st.rerun()
    return False


def _clear_confirm_loading(button_key: str) -> None:
    st.session_state[_confirm_loading_key(button_key)] = False


def _set_submit_feedback(payload: dict[str, Any], *, undo_label: str = "submission") -> None:
    st.session_state["record_match_last_submit"] = payload
    st.session_state["record_match_undo_state"] = {
        "expires_at": time.time() + UNDO_BANNER_SECONDS,
        "label": undo_label,
    }


def _render_undo_banner() -> None:
    undo_state = st.session_state.get("record_match_undo_state")
    if not isinstance(undo_state, dict):
        return
    remaining = int((undo_state.get("expires_at") or 0) - time.time())
    if remaining <= 0:
        st.session_state.pop("record_match_undo_state", None)
        return
    cols = st.columns([4, 1])
    with cols[0]:
        st.markdown(
            (
                "<div class='record-match-undo'>"
                f"<span>Saved {undo_state.get('label')}. Undo available for {remaining}s.</span>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    with cols[1]:
        if st.button("Undo", key="record_match_undo_btn"):
            st.session_state.pop("record_match_last_submit", None)
            st.session_state.pop("record_match_undo_state", None)
            st.info("Submission feedback cleared.")
            st.rerun()


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


def _normalize_bulk_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        col: str(col).strip().lower().replace(" ", "_") for col in df.columns
    }
    out = df.rename(columns=renamed).copy()
    alias_map = {
        "s1": "score_t1",
        "s2": "score_t2",
        "date_utc": "date",
        "match_date": "date",
        "league_name": "league",
    }
    for src, dst in alias_map.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]
    return out


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(float(text))


def _coerce_required_int(value: Any, field: str, row_number: int) -> int:
    try:
        parsed = _coerce_optional_int(value)
    except Exception as exc:
        raise ValueError(f"row {row_number}: invalid {field} ({exc})") from exc
    if parsed is None:
        raise ValueError(f"row {row_number}: missing required field '{field}'")
    return int(parsed)


def _step_2_bulk_match_entry(ctx, tokens: dict[str, str]) -> None:
    st.markdown("### Step 2 · Bulk Match Entry")
    st.caption("Upload CSV, validate rows, then submit in chunks via submit_match().")

    club_id = str(getattr(ctx, "club_id", "") or "").strip()
    if not club_id:
        st.error("Bulk Match Entry requires a valid club context.")
        return

    st.markdown(
        (
            f"<div style='border:1px solid {tokens['border_subtle']};border-radius:12px;"
            f"background:{tokens['card_bg']};padding:12px;margin-bottom:0.75rem;'>"
            "<strong>Required CSV columns:</strong> "
            "<code>league,t1_p1,t1_p2,t2_p1,t2_p2,score_t1,score_t2</code><br/>"
            "<span style='font-size:0.9rem;'>Optional: <code>date,match_type,week_tag,division,notes,context_type,context_id</code>.</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload match CSV",
        type=["csv"],
        key="record_match_bulk_csv",
        help="One row per match. Player fields should be numeric player IDs.",
    )

    parsed_rows: list[dict[str, Any]] = []
    structure_errors: list[str] = []
    preview_df = pd.DataFrame()

    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            raw_df = pd.read_csv(uploaded_file)
            normalized_df = _normalize_bulk_columns(raw_df)
        except Exception as exc:
            st.error(f"Could not parse CSV: {exc}")
            normalized_df = pd.DataFrame()

        required_cols = ["league", "t1_p1", "t1_p2", "t2_p1", "t2_p2", "score_t1", "score_t2"]
        missing_cols = [c for c in required_cols if c not in normalized_df.columns]
        if missing_cols:
            structure_errors.append(f"Missing required columns: {', '.join(missing_cols)}")

        if not normalized_df.empty and not missing_cols:
            for idx, row in normalized_df.iterrows():
                row_number = int(idx) + 2
                try:
                    league = str(row.get("league") or "").strip()
                    if not league:
                        raise ValueError(f"row {row_number}: missing required field 'league'")

                    parsed = {
                        "row_number": row_number,
                        "league": league,
                        "t1_p1": _coerce_required_int(row.get("t1_p1"), "t1_p1", row_number),
                        "t1_p2": _coerce_required_int(row.get("t1_p2"), "t1_p2", row_number),
                        "t2_p1": _coerce_required_int(row.get("t2_p1"), "t2_p1", row_number),
                        "t2_p2": _coerce_required_int(row.get("t2_p2"), "t2_p2", row_number),
                        "score_t1": _coerce_required_int(row.get("score_t1"), "score_t1", row_number),
                        "score_t2": _coerce_required_int(row.get("score_t2"), "score_t2", row_number),
                        "date": str(row.get("date") or "").strip() or datetime.utcnow().isoformat(),
                        "match_type": str(row.get("match_type") or "").strip() or "BulkEntry",
                        "week_tag": str(row.get("week_tag") or "").strip() or None,
                        "division": str(row.get("division") or "").strip() or None,
                        "notes": str(row.get("notes") or "").strip() or None,
                        "context_type": str(row.get("context_type") or "").strip() or "league",
                        "context_id": str(row.get("context_id") or "").strip() or None,
                    }
                    if parsed["score_t1"] == 0 and parsed["score_t2"] == 0:
                        raise ValueError(f"row {row_number}: both scores cannot be zero")
                    if len({parsed["t1_p1"], parsed["t1_p2"], parsed["t2_p1"], parsed["t2_p2"]}) != 4:
                        raise ValueError(f"row {row_number}: players must be 4 distinct IDs")
                    parsed_rows.append(parsed)
                except Exception as exc:
                    structure_errors.append(str(exc))

        preview_cols = [
            "row_number",
            "league",
            "t1_p1",
            "t1_p2",
            "t2_p1",
            "t2_p2",
            "score_t1",
            "score_t2",
            "date",
            "context_type",
            "context_id",
        ]
        if parsed_rows:
            preview_df = pd.DataFrame(parsed_rows)[preview_cols]
            st.session_state[BULK_UPLOAD_STATE_KEY] = parsed_rows
        else:
            st.session_state[BULK_UPLOAD_STATE_KEY] = []

    if not preview_df.empty:
        st.markdown("#### Preview")
        st.dataframe(preview_df.head(200), hide_index=True, use_container_width=True)
        if len(preview_df) > 200:
            st.caption(f"Showing first 200 of {len(preview_df)} parsed rows.")

    if structure_errors:
        st.error("CSV validation failed for one or more rows.")
        for message in structure_errors[:50]:
            st.caption(f"• {message}")
        if len(structure_errors) > 50:
            st.caption(f"… plus {len(structure_errors) - 50} more validation errors.")

    valid_rows = st.session_state.get(BULK_UPLOAD_STATE_KEY) or []
    can_submit = bool(valid_rows) and not structure_errors

    controls = st.columns([1, 1, 4])
    with controls[0]:
        if st.button("← Back", key="rm_step2_back_bulk"):
            st.session_state[WIZARD_STEP_KEY] = 1
            st.rerun()
    with controls[1]:
        confirm = _render_confirm_submit_button("rm_bulk_submit", disabled=not can_submit)

    if confirm:
        success_count = 0
        error_rows: list[str] = []
        total_rows = len(valid_rows)
        progress = st.progress(0.0)

        for chunk_start in range(0, total_rows, BULK_CHUNK_SIZE):
            chunk = valid_rows[chunk_start : chunk_start + BULK_CHUNK_SIZE]
            for row in chunk:
                context_type = str(row.get("context_type") or "league").strip().lower() or "league"
                if context_type not in {"league", "ladder", "tournament", "round_robin", "moneyball", "admin"}:
                    context_type = "admin"
                context_id = row.get("context_id") or (row.get("league") if context_type == "league" else None)
                row_seed_date = str(row.get("date") or "").strip()
                idem_key = _deterministic_idempotency_key(
                    club_id=club_id,
                    league_id=f"{row.get('league')}|{row.get('row_number')}",
                    player_ids=[row["t1_p1"], row["t1_p2"], row["t2_p1"], row["t2_p2"]],
                    score_t1=int(row["score_t1"]),
                    score_t2=int(row["score_t2"]),
                    match_date=row_seed_date,
                )
                payload = {
                    "date": row_seed_date,
                    "league": row["league"],
                    "division": row.get("division"),
                    "match_type": row.get("match_type"),
                    "week_tag": row.get("week_tag"),
                    "notes": row.get("notes"),
                    "t1_p1": int(row["t1_p1"]),
                    "t1_p2": int(row["t1_p2"]),
                    "t2_p1": int(row["t2_p1"]),
                    "t2_p2": int(row["t2_p2"]),
                    "score_t1": int(row["score_t1"]),
                    "score_t2": int(row["score_t2"]),
                    "s1": int(row["score_t1"]),
                    "s2": int(row["score_t2"]),
                }
                try:
                    submit_match(
                        club_id=club_id,
                        context_type=context_type,
                        context_id=str(context_id) if context_id is not None else None,
                        match_payload=payload,
                        idempotency_key=idem_key,
                    )
                    success_count += 1
                except Exception as exc:
                    error_rows.append(f"row {row.get('row_number')}: {exc}")

                processed = success_count + len(error_rows)
                progress.progress(min(1.0, processed / max(total_rows, 1)))

        _set_submit_feedback(
            {
                "bulk_summary": {
                    "total": total_rows,
                    "success": success_count,
                    "errors": error_rows,
                }
            },
            undo_label="bulk submission",
        )
        _clear_confirm_loading("rm_bulk_submit")
        st.rerun()

    last_submit = st.session_state.get("record_match_last_submit")
    summary = (last_submit or {}).get("bulk_summary") if isinstance(last_submit, dict) else None
    if isinstance(summary, dict):
        total = int(summary.get("total") or 0)
        success = int(summary.get("success") or 0)
        errors = summary.get("errors") or []
        st.markdown("### Submission summary")
        st.success(f"Submitted successfully: {success} / {total}")
        if errors:
            st.error(f"Failed rows: {len(errors)}")
            for err in errors[:50]:
                st.caption(f"• {err}")
            if len(errors) > 50:
                st.caption(f"… plus {len(errors) - 50} more errors.")


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
        if _render_confirm_submit_button("rm_ll_submit", disabled=not can_submit):
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
                _set_submit_feedback(
                    {
                        "league": selected_league,
                        "division": selected_division,
                        "score": f"{int(score_t1)} - {int(score_t2)}",
                        "idempotency_key": idem_key,
                        "result": result,
                    },
                    undo_label="league match",
                )
                _clear_confirm_loading("rm_ll_submit")
                st.rerun()
            except Exception as exc:
                _clear_confirm_loading("rm_ll_submit")
                st.error(f"Submit failed: {exc}")

    last_submit = st.session_state.get("record_match_last_submit")
    if isinstance(last_submit, dict):
        st.markdown(
            (
                "<div class='record-match-success-card is-visible'>"
                "<h4 style='margin:0 0 6px 0;'><span class='record-match-check'>✓</span>Match submitted</h4>"
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
        if _render_confirm_submit_button("rm_cl_submit", disabled=not has_score):
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
                _set_submit_feedback(
                    {
                        "challenge": selected_challenge,
                        "score": f"{int(score_t1)} - {int(score_t2)}",
                        "idempotency_key": idem_key,
                        "result": result,
                    },
                    undo_label="challenge result",
                )
                _clear_confirm_loading("rm_cl_submit")
                st.rerun()
            except Exception as exc:
                _clear_confirm_loading("rm_cl_submit")
                st.error(f"Submit failed: {exc}")

    last_submit = st.session_state.get("record_match_last_submit")
    if isinstance(last_submit, dict) and isinstance(last_submit.get("challenge"), dict):
        challenge = last_submit["challenge"]
        st.markdown(
            (
                "<div class='record-match-success-card is-visible'>"
                "<h4 style='margin:0 0 6px 0;'><span class='record-match-check'>✓</span>Challenge result submitted</h4>"
                f"<div><strong>Challenge:</strong> #{challenge.get('challenge_id')}</div>"
                f"<div><strong>Players:</strong> {challenge.get('challenger')} vs {challenge.get('defender')}</div>"
                f"<div><strong>Score:</strong> {last_submit.get('score')}</div>"
                f"<div><strong>Idempotency Key:</strong> <code>{last_submit.get('idempotency_key')}</code></div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def _team_label(team: dict[str, Any] | None, id_to_name: dict[int, str]) -> str:
    if not isinstance(team, dict):
        return "TBD"
    p1_id = team.get("player1_id")
    p2_id = team.get("player2_id")
    p1 = id_to_name.get(int(p1_id), f"#{p1_id}") if p1_id is not None else "TBD"
    p2 = id_to_name.get(int(p2_id), f"#{p2_id}") if p2_id is not None else "TBD"
    return f"Team {team.get('team_number')}: {p1} / {p2}"


def _step_2_tournament(ctx, tokens: dict[str, str]) -> None:
    st.markdown("### Step 2 · Tournament match entry")
    st.caption("Pick an active tournament match node. Teams are auto-filled from the bracket.")

    supabase = getattr(ctx, "supabase", None)
    club_id = str(getattr(ctx, "club_id", "") or "").strip()
    if not supabase or not club_id:
        st.error("Tournament submission requires club and database context.")
        return

    try:
        tournaments_resp = (
            supabase.table("tournaments")
            .select("id,name,status,created_at")
            .eq("club_id", club_id)
            .neq("status", "COMPLETE")
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )
        active_tournaments = getattr(tournaments_resp, "data", None) or []
    except Exception as exc:
        st.error(f"Could not load active tournaments: {exc}")
        active_tournaments = []

    if not active_tournaments:
        st.info("No active tournament found. Create or reopen a tournament first.")
        controls = st.columns([1, 4])
        with controls[0]:
            if st.button("← Back", key="rm_step2_back_tourney_empty"):
                st.session_state[WIZARD_STEP_KEY] = 1
                st.rerun()
        return

    tournament_options = {
        f"{row.get('name', 'Tournament')} · {row.get('status', 'UNKNOWN')} (#{row.get('id')})": row
        for row in active_tournaments
    }
    selected_tournament_label = st.selectbox(
        "Active tournament",
        options=list(tournament_options.keys()),
        key="record_match_tournament_selector",
    )
    selected_tournament = tournament_options[selected_tournament_label]
    tournament_id = selected_tournament.get("id")

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
            <strong>Tournament:</strong> {selected_tournament.get('name', 'Tournament')}<br/>
            <strong>Status:</strong> {selected_tournament.get('status', 'UNKNOWN')}<br/>
            <strong>ID:</strong> {tournament_id}
        </div>
        """,
        unsafe_allow_html=True,
    )

    teams_resp = (
        supabase.table("tournament_teams")
        .select("id,team_number,player1_id,player2_id,seed")
        .eq("tournament_id", tournament_id)
        .order("team_number", desc=False)
        .execute()
    )
    teams = getattr(teams_resp, "data", None) or []
    teams_by_id = {row.get("id"): row for row in teams}

    games_resp = (
        supabase.table("tournament_games")
        .select("id,stage,rr_round_number,rr_slot_number,playoff_round_label,playoff_slot_number,playoff_game_code,team_a_id,team_b_id,score_a,score_b,winner_team_id,loser_team_id,finalized_at")
        .eq("tournament_id", tournament_id)
        .order("stage", desc=False)
        .order("rr_round_number", desc=False)
        .order("playoff_round_label", desc=False)
        .order("rr_slot_number", desc=False)
        .order("playoff_slot_number", desc=False)
        .execute()
    )
    games = getattr(games_resp, "data", None) or []

    if not games:
        st.warning("No tournament games generated yet.")
        return

    id_to_name = getattr(ctx, "id_to_name", None) or {}
    rows = []
    pending_games: list[dict[str, Any]] = []
    for game in games:
        team_a = teams_by_id.get(game.get("team_a_id"))
        team_b = teams_by_id.get(game.get("team_b_id"))
        teams_ready = bool(team_a) and bool(team_b) and all(team_a.get(k) is not None for k in ("player1_id", "player2_id")) and all(team_b.get(k) is not None for k in ("player1_id", "player2_id"))
        is_pending = not game.get("finalized_at") and teams_ready
        if is_pending:
            pending_games.append(game)
        rows.append(
            {
                "Game": game.get("playoff_game_code") or f"RR-{game.get('rr_round_number', '?')}-{game.get('rr_slot_number', '?')}",
                "Stage": game.get("stage"),
                "Round": game.get("playoff_round_label") or game.get("rr_round_number"),
                "Team A": _team_label(team_a, id_to_name),
                "Team B": _team_label(team_b, id_to_name),
                "Score": f"{int(game.get('score_a') or 0)} - {int(game.get('score_b') or 0)}" if game.get("finalized_at") else "Pending",
                "Status": "Pending" if is_pending else ("Final" if game.get("finalized_at") else "Waiting on bracket"),
            }
        )

    st.markdown("#### Bracket / pending matches")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if not pending_games:
        st.info("No scoreable tournament matches are currently pending. Finalize earlier bracket nodes first.")
        return

    selectable = {}
    for game in pending_games:
        team_a = teams_by_id.get(game.get("team_a_id"))
        team_b = teams_by_id.get(game.get("team_b_id"))
        game_code = game.get("playoff_game_code") or f"RR R{game.get('rr_round_number')} · S{game.get('rr_slot_number')}"
        label = f"{game_code} · {_team_label(team_a, id_to_name)} vs {_team_label(team_b, id_to_name)}"
        selectable[label] = game

    selected_match_label = st.selectbox(
        "Select match node",
        options=list(selectable.keys()),
        key="record_match_tournament_game_selector",
    )
    selected_game = selectable[selected_match_label]
    team_a = teams_by_id.get(selected_game.get("team_a_id"))
    team_b = teams_by_id.get(selected_game.get("team_b_id"))

    st.markdown("#### Teams (auto-filled from tournament node)")
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
            <strong>Team A:</strong> {_team_label(team_a, id_to_name)}<br/>
            <strong>Team B:</strong> {_team_label(team_b, id_to_name)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    score_cols = st.columns(2)
    score_t1 = score_cols[0].number_input(
        "Team A score",
        min_value=0,
        max_value=99,
        value=0,
        step=1,
        key=f"rm_tour_s1_{selected_game['id']}",
    )
    score_t2 = score_cols[1].number_input(
        "Team B score",
        min_value=0,
        max_value=99,
        value=0,
        step=1,
        key=f"rm_tour_s2_{selected_game['id']}",
    )

    has_score = int(score_t1) + int(score_t2) > 0
    has_winner = int(score_t1) != int(score_t2)
    if not has_score:
        st.error("Enter a non-zero score before confirming.")
    if has_score and not has_winner:
        st.error("Tournament matches cannot end in a tie.")

    controls = st.columns([1, 1, 4])
    with controls[0]:
        if st.button("← Back", key="rm_step2_back_tournament"):
            st.session_state[WIZARD_STEP_KEY] = 1
            st.rerun()
    with controls[1]:
        if _render_confirm_submit_button("rm_tournament_submit", disabled=not (has_score and has_winner)):
            match_date = datetime.utcnow().isoformat()
            idem_key = _deterministic_idempotency_key(
                club_id=club_id,
                league_id=f"tournament:{tournament_id}:game:{selected_game['id']}",
                player_ids=[
                    int(team_a.get("player1_id")),
                    int(team_a.get("player2_id")),
                    int(team_b.get("player1_id")),
                    int(team_b.get("player2_id")),
                ],
                score_t1=int(score_t1),
                score_t2=int(score_t2),
                match_date=match_date,
            )

            payload = {
                "date": match_date,
                "league": selected_tournament.get("name", "Tournament"),
                "match_type": "Tournament",
                "t1_p1": int(team_a.get("player1_id")),
                "t1_p2": int(team_a.get("player2_id")),
                "t2_p1": int(team_b.get("player1_id")),
                "t2_p2": int(team_b.get("player2_id")),
                "score_t1": int(score_t1),
                "score_t2": int(score_t2),
                "s1": int(score_t1),
                "s2": int(score_t2),
                "week_tag": "Tournament",
                "is_popup": True,
                "tournament_id": tournament_id,
                "tournament_game_id": selected_game["id"],
            }

            try:
                result = submit_match(
                    club_id=club_id,
                    context_type="tournament",
                    context_id=str(tournament_id),
                    match_payload=payload,
                    idempotency_key=idem_key,
                )

                finalize_payload = finalize_game({**selected_game, "score_a": int(score_t1), "score_b": int(score_t2)})
                supabase.table("tournament_games").update(finalize_payload).eq("id", selected_game["id"]).execute()

                if str(selected_game.get("stage") or "").upper() == "PLAYOFF":
                    playoff_games_resp = (
                        supabase.table("tournament_games")
                        .select("*")
                        .eq("tournament_id", tournament_id)
                        .eq("stage", "PLAYOFF")
                        .execute()
                    )
                    playoff_games = getattr(playoff_games_resp, "data", None) or []
                    updates = resolve_playoff_dependencies(playoff_games)
                    for upd in updates:
                        supabase.table("tournament_games").update(upd).eq("id", upd["id"]).execute()

                _set_submit_feedback(
                    {
                        "tournament": selected_tournament,
                        "game": selected_game,
                        "score": f"{int(score_t1)} - {int(score_t2)}",
                        "idempotency_key": idem_key,
                        "result": result,
                    },
                    undo_label="tournament result",
                )
                _clear_confirm_loading("rm_tournament_submit")
                st.rerun()
            except Exception as exc:
                _clear_confirm_loading("rm_tournament_submit")
                st.error(f"Submit failed: {exc}")

    last_submit = st.session_state.get("record_match_last_submit")
    if isinstance(last_submit, dict) and isinstance(last_submit.get("tournament"), dict):
        game = last_submit.get("game") or {}
        st.markdown(
            (
                "<div class='record-match-success-card is-visible'>"
                "<h4 style='margin:0 0 6px 0;'><span class='record-match-check'>✓</span>Tournament match submitted</h4>"
                f"<div><strong>Tournament:</strong> {last_submit['tournament'].get('name')}</div>"
                f"<div><strong>Game:</strong> {game.get('playoff_game_code') or game.get('id')}</div>"
                f"<div><strong>Score:</strong> {last_submit.get('score')}</div>"
                f"<div><strong>Idempotency Key:</strong> <code>{last_submit.get('idempotency_key')}</code></div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _first_int(row: dict[str, Any], keys: list[str]) -> int | None:
    for key in keys:
        if key in row:
            parsed = _safe_int(row.get(key))
            if parsed is not None:
                return parsed
    return None


def _round_robin_team_ids(match_row: dict[str, Any], prefix: str) -> tuple[int | None, int | None]:
    p1 = _first_int(match_row, [f"{prefix}_p1_id", f"{prefix}_player1_id", f"{prefix}1_id"])
    p2 = _first_int(match_row, [f"{prefix}_p2_id", f"{prefix}_player2_id", f"{prefix}2_id"])
    return p1, p2


def _round_robin_team_label(match_row: dict[str, Any], prefix: str, id_to_name: dict[int, str]) -> str:
    p1, p2 = _round_robin_team_ids(match_row, prefix)
    if p1 is None and p2 is None:
        return "TBD"
    left = id_to_name.get(int(p1), f"#{p1}") if p1 is not None else "TBD"
    right = id_to_name.get(int(p2), f"#{p2}") if p2 is not None else "TBD"
    return f"{left} / {right}"


def _compute_round_robin_pool_standings(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    standings: dict[int, dict[str, Any]] = {}

    def ensure(pid: int) -> dict[str, Any]:
        if pid not in standings:
            standings[pid] = {
                "player_id": int(pid),
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "matches_played": 0,
                "points_for": 0,
                "points_against": 0,
                "point_diff": 0,
            }
        return standings[pid]

    for match in matches:
        score_a = _safe_int(match.get("score_a"))
        score_b = _safe_int(match.get("score_b"))
        if score_a is None or score_b is None:
            continue

        a1, a2 = _round_robin_team_ids(match, "team_a")
        b1, b2 = _round_robin_team_ids(match, "team_b")
        team_a_ids = [pid for pid in (a1, a2) if pid is not None]
        team_b_ids = [pid for pid in (b1, b2) if pid is not None]
        if not team_a_ids or not team_b_ids:
            continue

        for pid in team_a_ids:
            row = ensure(pid)
            row["matches_played"] += 1
            row["points_for"] += int(score_a)
            row["points_against"] += int(score_b)
            if score_a > score_b:
                row["wins"] += 1
            elif score_a < score_b:
                row["losses"] += 1
            else:
                row["ties"] += 1

        for pid in team_b_ids:
            row = ensure(pid)
            row["matches_played"] += 1
            row["points_for"] += int(score_b)
            row["points_against"] += int(score_a)
            if score_b > score_a:
                row["wins"] += 1
            elif score_b < score_a:
                row["losses"] += 1
            else:
                row["ties"] += 1

    for row in standings.values():
        row["point_diff"] = int(row["points_for"]) - int(row["points_against"])

    return sorted(
        standings.values(),
        key=lambda row: (
            -int(row["wins"]),
            -int(row["point_diff"]),
            -int(row["points_for"]),
            int(row["player_id"]),
        ),
    )


def _step_2_round_robin(ctx, tokens: dict[str, str]) -> None:
    st.markdown("### Step 2 · Round Robin match entry")
    st.caption("Select an active session, choose a pool, pick a pairing, then submit score.")

    supabase = getattr(ctx, "supabase", None)
    club_id = str(getattr(ctx, "club_id", "") or "").strip()
    if not supabase or not club_id:
        st.error("Round Robin submission requires club and database context.")
        return

    try:
        sessions_resp = (
            supabase.table("round_robin_sessions")
            .select("id,name,status,created_at")
            .eq("club_id", club_id)
            .in_("status", ["ACTIVE", "IN_PROGRESS"])
            .order("created_at", desc=True)
            .limit(100)
            .execute()
        )
        sessions = getattr(sessions_resp, "data", None) or []
    except Exception as exc:
        st.error(f"Could not load active round robin sessions: {exc}")
        sessions = []

    if not sessions:
        st.info("No active Round Robin session found.")
        controls = st.columns([1, 4])
        with controls[0]:
            if st.button("← Back", key="rm_step2_back_rr_empty"):
                st.session_state[WIZARD_STEP_KEY] = 1
                st.rerun()
        return

    session_options = {
        f"{row.get('name') or 'Round Robin'} · {row.get('status')} (#{row.get('id')})": row
        for row in sessions
    }
    selected_session_label = st.selectbox(
        "Active Round Robin session",
        options=list(session_options.keys()),
        key="record_match_rr_session_selector",
    )
    selected_session = session_options[selected_session_label]
    session_id = str(selected_session.get("id"))

    pools_resp = (
        supabase.table("round_robin_pools")
        .select("id,name,pool_number")
        .eq("session_id", session_id)
        .order("pool_number", desc=False)
        .order("name", desc=False)
        .execute()
    )
    pools = getattr(pools_resp, "data", None) or []
    if not pools:
        st.warning("No pools found for this session.")
        return

    pool_options = {
        f"Pool {row.get('pool_number') or '?'} · {row.get('name') or 'Unnamed'} (#{row.get('id')})": row
        for row in pools
    }
    selected_pool_label = st.selectbox(
        "Pool",
        options=list(pool_options.keys()),
        key=f"record_match_rr_pool_selector_{session_id}",
    )
    selected_pool = pool_options[selected_pool_label]
    pool_id = str(selected_pool.get("id"))

    matches_resp = (
        supabase.table("round_robin_matches")
        .select("*")
        .eq("pool_id", pool_id)
        .order("match_number", desc=False)
        .order("created_at", desc=False)
        .execute()
    )
    pool_matches = getattr(matches_resp, "data", None) or []
    if not pool_matches:
        st.info("No matches generated yet for this pool.")
        return

    id_to_name = getattr(ctx, "id_to_name", None) or {}
    pending_matches: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    for match in pool_matches:
        score_a = _safe_int(match.get("score_a"))
        score_b = _safe_int(match.get("score_b"))
        done = bool(match.get("finalized_at")) or (score_a is not None and score_b is not None)
        if not done:
            pending_matches.append(match)
        table_rows.append(
            {
                "Match": match.get("match_number") or match.get("id"),
                "Team A": _round_robin_team_label(match, "team_a", id_to_name),
                "Team B": _round_robin_team_label(match, "team_b", id_to_name),
                "Score": f"{score_a} - {score_b}" if done and score_a is not None and score_b is not None else "Pending",
                "Status": "Final" if done else "Pending",
            }
        )

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    if not pending_matches:
        st.success("All pairings in this pool already have scores.")
        return

    pairing_options: dict[str, dict[str, Any]] = {}
    for match in pending_matches:
        match_id = match.get("id")
        pairing_options[
            f"#{match.get('match_number') or match_id} · {_round_robin_team_label(match, 'team_a', id_to_name)} vs {_round_robin_team_label(match, 'team_b', id_to_name)}"
        ] = match

    selected_pairing_label = st.selectbox(
        "Match pairing",
        options=list(pairing_options.keys()),
        key=f"record_match_rr_pairing_selector_{pool_id}",
    )
    selected_pairing = pairing_options[selected_pairing_label]

    score_cols = st.columns(2)
    score_t1 = score_cols[0].number_input(
        "Team A score",
        min_value=0,
        max_value=99,
        value=0,
        step=1,
        key=f"rm_rr_s1_{selected_pairing['id']}",
    )
    score_t2 = score_cols[1].number_input(
        "Team B score",
        min_value=0,
        max_value=99,
        value=0,
        step=1,
        key=f"rm_rr_s2_{selected_pairing['id']}",
    )

    has_score = int(score_t1) + int(score_t2) > 0
    if not has_score:
        st.error("Enter a non-zero score before confirming.")

    controls = st.columns([1, 1, 4])
    with controls[0]:
        if st.button("← Back", key="rm_step2_back_round_robin"):
            st.session_state[WIZARD_STEP_KEY] = 1
            st.rerun()
    with controls[1]:
        if _render_confirm_submit_button("rm_round_robin_submit", disabled=not has_score):
            team_a_p1, team_a_p2 = _round_robin_team_ids(selected_pairing, "team_a")
            team_b_p1, team_b_p2 = _round_robin_team_ids(selected_pairing, "team_b")
            if None in (team_a_p1, team_a_p2, team_b_p1, team_b_p2):
                st.error("Selected pairing is missing one or more player assignments.")
                return

            match_date = datetime.utcnow().isoformat()
            idem_key = _deterministic_idempotency_key(
                club_id=club_id,
                league_id=f"round_robin:{session_id}:pool:{pool_id}:match:{selected_pairing['id']}",
                player_ids=[int(team_a_p1), int(team_a_p2), int(team_b_p1), int(team_b_p2)],
                score_t1=int(score_t1),
                score_t2=int(score_t2),
                match_date=match_date,
            )

            payload = {
                "date": match_date,
                "league": selected_session.get("name") or "Round Robin",
                "match_type": "Round Robin",
                "week_tag": f"Round Robin Session {session_id}",
                "is_popup": True,
                "round_robin_session_id": session_id,
                "round_robin_pool_id": pool_id,
                "round_robin_match_id": selected_pairing["id"],
                "t1_p1": int(team_a_p1),
                "t1_p2": int(team_a_p2),
                "t2_p1": int(team_b_p1),
                "t2_p2": int(team_b_p2),
                "score_t1": int(score_t1),
                "score_t2": int(score_t2),
                "s1": int(score_t1),
                "s2": int(score_t2),
            }

            try:
                result = submit_match(
                    club_id=club_id,
                    context_type="round_robin",
                    context_id=session_id,
                    match_payload=payload,
                    idempotency_key=idem_key,
                )

                update_payload = {
                    "score_a": int(score_t1),
                    "score_b": int(score_t2),
                    "winner_side": "A" if int(score_t1) > int(score_t2) else ("B" if int(score_t2) > int(score_t1) else "TIE"),
                    "updated_at": datetime.utcnow().isoformat(),
                    "finalized_at": datetime.utcnow().isoformat(),
                }
                supabase.table("round_robin_matches").update(update_payload).eq("id", selected_pairing["id"]).execute()

                refreshed_matches_resp = supabase.table("round_robin_matches").select("*").eq("pool_id", pool_id).execute()
                refreshed_matches = getattr(refreshed_matches_resp, "data", None) or []
                standings_rows = _compute_round_robin_pool_standings(refreshed_matches)
                for rank, row in enumerate(standings_rows, start=1):
                    upsert_payload = {
                        "session_id": session_id,
                        "pool_id": pool_id,
                        "player_id": int(row["player_id"]),
                        "rank": int(rank),
                        "wins": int(row["wins"]),
                        "losses": int(row["losses"]),
                        "ties": int(row["ties"]),
                        "matches_played": int(row["matches_played"]),
                        "points_for": int(row["points_for"]),
                        "points_against": int(row["points_against"]),
                        "point_diff": int(row["point_diff"]),
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                    supabase.table("round_robin_standings").upsert(
                        upsert_payload,
                        on_conflict="session_id,pool_id,player_id",
                    ).execute()

                _set_submit_feedback(
                    {
                        "session": selected_session,
                        "pool": selected_pool,
                        "pairing": selected_pairing_label,
                        "score": f"{int(score_t1)} - {int(score_t2)}",
                        "idempotency_key": idem_key,
                        "result": result,
                    },
                    undo_label="round robin result",
                )
                _clear_confirm_loading("rm_round_robin_submit")
                st.rerun()
            except Exception as exc:
                _clear_confirm_loading("rm_round_robin_submit")
                st.error(f"Submit failed: {exc}")

    last_submit = st.session_state.get("record_match_last_submit")
    if isinstance(last_submit, dict) and isinstance(last_submit.get("session"), dict):
        session = last_submit.get("session") or {}
        pool = last_submit.get("pool") or {}
        st.markdown(
            (
                "<div class='record-match-success-card is-visible'>"
                "<h4 style='margin:0 0 6px 0;'><span class='record-match-check'>✓</span>Round Robin match submitted</h4>"
                f"<div><strong>Session:</strong> {session.get('name')}</div>"
                f"<div><strong>Pool:</strong> {pool.get('name') or pool.get('pool_number')}</div>"
                f"<div><strong>Pairing:</strong> {last_submit.get('pairing')}</div>"
                f"<div><strong>Score:</strong> {last_submit.get('score')}</div>"
                f"<div><strong>Idempotency Key:</strong> <code>{last_submit.get('idempotency_key')}</code></div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def _step_2_moneyball(ctx, tokens: dict[str, str]) -> None:
    st.markdown("### Step 2 · Moneyball result")
    st.caption("Select an active Moneyball event, assign players, set score + bonus, then confirm.")

    supabase = getattr(ctx, "supabase", None)
    club_id = str(getattr(ctx, "club_id", "") or "").strip()
    if not supabase or not club_id:
        st.error("Moneyball submission requires club and database context.")
        return

    try:
        events_resp = (
            supabase.table("events")
            .select("id,name,event_type,is_active,created_at")
            .eq("club_id", club_id)
            .eq("is_active", True)
            .in_("event_type", ["moneyball", "Moneyball", "popup_rr"])
            .order("created_at", desc=True)
            .limit(100)
            .execute()
        )
        active_events = getattr(events_resp, "data", None) or []
    except Exception as exc:
        st.error(f"Could not load active Moneyball events: {exc}")
        active_events = []

    if not active_events:
        st.info("No active Moneyball event found.")
        controls = st.columns([1, 4])
        with controls[0]:
            if st.button("← Back", key="rm_step2_back_moneyball_empty"):
                st.session_state[WIZARD_STEP_KEY] = 1
                st.rerun()
        return

    event_options = {
        f"{row.get('name') or 'Moneyball'} (#{row.get('id')})": row for row in active_events
    }
    selected_event_label = st.selectbox(
        "Active Moneyball event",
        options=list(event_options.keys()),
        key="record_match_moneyball_event_selector",
    )
    selected_event = event_options[selected_event_label]
    event_id = str(selected_event.get("id") or "").strip()

    player_options = _player_options_for_league(ctx, "OVERALL")
    if not player_options:
        st.warning("No active players available for Moneyball entry.")
        return

    labels = [label for label, _ in player_options]
    label_to_pid = {label: pid for label, pid in player_options}

    st.markdown("#### Players")
    c1, c2 = st.columns(2)
    with c1:
        t1_p1_label = st.selectbox("Team 1 · Player 1", labels, key="rm_mb_t1_p1")
        t1_p2_label = st.selectbox("Team 1 · Player 2", labels, key="rm_mb_t1_p2")
    with c2:
        t2_p1_label = st.selectbox("Team 2 · Player 1", labels, key="rm_mb_t2_p1")
        t2_p2_label = st.selectbox("Team 2 · Player 2", labels, key="rm_mb_t2_p2")

    st.markdown("#### Score + Moneyball bonus")
    score_cols = st.columns(2)
    score_t1 = score_cols[0].number_input("Team 1 Score", min_value=0, max_value=99, value=0, step=1, key="rm_mb_s1")
    score_t2 = score_cols[1].number_input("Team 2 Score", min_value=0, max_value=99, value=0, step=1, key="rm_mb_s2")

    bonus_cols = st.columns(2)
    bonus_t1 = bonus_cols[0].number_input("Team 1 Bonus", min_value=0, max_value=30, value=0, step=1, key="rm_mb_bonus_t1")
    bonus_t2 = bonus_cols[1].number_input("Team 2 Bonus", min_value=0, max_value=30, value=0, step=1, key="rm_mb_bonus_t2")

    total_t1 = int(score_t1) + int(bonus_t1)
    total_t2 = int(score_t2) + int(bonus_t2)
    total_delta = int(total_t1) - int(total_t2)

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
            <strong>Event:</strong> {selected_event.get('name') or 'Moneyball'}<br/>
            <strong>Raw score:</strong> {int(score_t1)} - {int(score_t2)}<br/>
            <strong>Bonus:</strong> +{int(bonus_t1)} / +{int(bonus_t2)}<br/>
            <strong>Total points impact:</strong> {int(total_t1)} - {int(total_t2)} (Δ {int(total_delta):+d})
        </div>
        """,
        unsafe_allow_html=True,
    )

    t1_p1 = label_to_pid[t1_p1_label]
    t1_p2 = label_to_pid[t1_p2_label]
    t2_p1 = label_to_pid[t2_p1_label]
    t2_p2 = label_to_pid[t2_p2_label]

    selected_ids = [t1_p1, t1_p2, t2_p1, t2_p2]
    unique_ids = len(set(selected_ids)) == 4
    has_score = int(total_t1) + int(total_t2) > 0

    if not unique_ids:
        st.error("Select 4 distinct players (2 per team).")
    if not has_score:
        st.error("Enter score/bonus points before confirming.")

    controls = st.columns([1, 1, 4])
    with controls[0]:
        if st.button("← Back", key="rm_step2_back_moneyball"):
            st.session_state[WIZARD_STEP_KEY] = 1
            st.rerun()
    with controls[1]:
        if _render_confirm_submit_button("rm_moneyball_submit", disabled=not (unique_ids and has_score and bool(event_id))):
            match_date = datetime.utcnow().isoformat()
            idem_key = _deterministic_idempotency_key(
                club_id=club_id,
                league_id=f"moneyball:{event_id}",
                player_ids=[t1_p1, t1_p2, t2_p1, t2_p2],
                score_t1=int(total_t1),
                score_t2=int(total_t2),
                match_date=match_date,
            )

            payload = {
                "date": match_date,
                "league": str(selected_event.get("name") or "Moneyball"),
                "week_tag": f"Moneyball {event_id}",
                "match_type": "Moneyball",
                "is_popup": True,
                "t1_p1": int(t1_p1),
                "t1_p2": int(t1_p2),
                "t2_p1": int(t2_p1),
                "t2_p2": int(t2_p2),
                "score_t1": int(total_t1),
                "score_t2": int(total_t2),
                "s1": int(total_t1),
                "s2": int(total_t2),
                "notes": (
                    f"moneyball_raw_score={int(score_t1)}-{int(score_t2)};"
                    f"moneyball_bonus={int(bonus_t1)}-{int(bonus_t2)}"
                ),
            }

            try:
                result = submit_match(
                    club_id=club_id,
                    context_type="moneyball",
                    context_id=event_id,
                    match_payload=payload,
                    idempotency_key=idem_key,
                )
                _set_submit_feedback(
                    {
                        "event": selected_event,
                        "score": f"{int(total_t1)} - {int(total_t2)}",
                        "raw_score": f"{int(score_t1)} - {int(score_t2)}",
                        "bonus": f"+{int(bonus_t1)} / +{int(bonus_t2)}",
                        "idempotency_key": idem_key,
                        "result": result,
                    },
                    undo_label="moneyball result",
                )
                _clear_confirm_loading("rm_moneyball_submit")
                st.rerun()
            except Exception as exc:
                _clear_confirm_loading("rm_moneyball_submit")
                st.error(f"Submit failed: {exc}")

    last_submit = st.session_state.get("record_match_last_submit")
    if isinstance(last_submit, dict) and isinstance(last_submit.get("event"), dict):
        st.markdown(
            (
                "<div class='record-match-success-card is-visible'>"
                "<h4 style='margin:0 0 6px 0;'><span class='record-match-check'>✓</span>Moneyball result submitted</h4>"
                f"<div><strong>Event:</strong> {last_submit['event'].get('name')}</div>"
                f"<div><strong>Raw score:</strong> {last_submit.get('raw_score')}</div>"
                f"<div><strong>Bonus:</strong> {last_submit.get('bonus')}</div>"
                f"<div><strong>Submitted total:</strong> {last_submit.get('score')}</div>"
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
    _render_motion_css(tokens)

    current_view_step = _resolve_progress_step()
    previous_view_step = int(st.session_state.get(WIZARD_LAST_VIEW_STEP_KEY, current_view_step))
    st.session_state[WIZARD_VIEW_STEP_KEY] = current_view_step
    _render_progress_indicator(current_view_step)

    shell_class = "record-match-step-shell"
    if current_view_step == previous_view_step:
        shell_class += " record-match-step-shell-steady"
    st.markdown(f"<div class='{shell_class}'>", unsafe_allow_html=True)

    step = int(st.session_state.get(WIZARD_STEP_KEY, 1))
    if step <= 1:
        _step_1_competition_type(tokens)
        st.markdown("</div>", unsafe_allow_html=True)
        st.session_state[WIZARD_LAST_VIEW_STEP_KEY] = current_view_step
        return

    selected = st.session_state.get(SELECTED_TYPE_KEY)
    selected_id = selected.get("id") if isinstance(selected, dict) else None
    if selected_id == "ladder_league":
        _step_2_ladder_league(ctx, tokens)
    elif selected_id == "challenge_ladder":
        _step_2_challenge_ladder(ctx, tokens)
    elif selected_id == "tournament":
        _step_2_tournament(ctx, tokens)
    elif selected_id == "round_robin":
        _step_2_round_robin(ctx, tokens)
    elif selected_id == "moneyball":
        _step_2_moneyball(ctx, tokens)
    elif selected_id == "bulk_match_entry":
        _step_2_bulk_match_entry(ctx, tokens)
    else:
        _step_2_placeholder(tokens)

    _render_undo_banner()
    st.markdown("</div>", unsafe_allow_html=True)
    st.session_state[WIZARD_LAST_VIEW_STEP_KEY] = current_view_step
