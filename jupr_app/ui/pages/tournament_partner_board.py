from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd
import streamlit as st

from jupr_app.domain.tournament_registration_repo import (
    build_registration_state,
    get_public_tournament_bundle,
    list_open_public_tournaments,
    registration_feature_available,
)
from jupr_app.ui.layout import page_shell


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _pick_tournament(ctx, supabase):
    club_id = _safe_text(getattr(ctx, "club_id", ""))
    choices = list_open_public_tournaments(supabase, club_id)
    if not choices:
        return None, None, [], []
    labels = [f"{row['tournament'].get('name')}" for row in choices]
    selected_label = st.selectbox("Choose a tournament", labels)
    idx = labels.index(selected_label)
    selected = choices[idx]
    tournament = selected["tournament"]
    settings = selected["settings"]
    return get_public_tournament_bundle(
        supabase,
        club_id=club_id,
        tournament_id=str(tournament.get("id")),
        registration_slug=settings.get("registration_slug"),
    )


def render(ctx):
    mode_label = "Public" if bool(getattr(ctx, "public_mode", False)) else "Admin"
    page_shell(
        "🤝 Partner Board",
        "See who is looking for a partner, grouped by day and division.",
        mode_label=mode_label,
    )

    supabase = getattr(ctx, "supabase", None)
    club_id = _safe_text(getattr(ctx, "club_id", ""))
    if supabase is None or not club_id:
        st.error("Missing database context.")
        st.stop()

    available, detail = registration_feature_available(supabase)
    if not available:
        st.error("Tournament registration is not enabled yet. Apply the registration SQL migration first.")
        if detail:
            st.caption(detail)
        st.stop()

    qp_tournament_id = _safe_text(st.query_params.get("tournament_id"))
    qp_slug = _safe_text(st.query_params.get("tournament"))
    tournament, settings, days, event_options = get_public_tournament_bundle(
        supabase,
        club_id=club_id,
        tournament_id=qp_tournament_id or None,
        registration_slug=qp_slug or None,
    )
    if not tournament:
        tournament, settings, days, event_options = _pick_tournament(ctx, supabase)
        if not tournament:
            st.info("No open tournaments are currently using the partner board.")
            st.stop()

    state = build_registration_state(supabase, tournament, settings, days, event_options)
    board = state.get("partner_board", [])

    st.subheader(_safe_text(tournament.get("name") or "Tournament"))
    if not board:
        st.info("Nobody is currently published on the partner board for this tournament.")
        st.stop()

    day_labels = ["All days"] + sorted({_safe_text(row.get("event_day_label")) for row in board if _safe_text(row.get("event_day_label"))})
    event_labels = ["All events"] + sorted({_safe_text(row.get("event_label")) for row in board if _safe_text(row.get("event_label"))})
    c1, c2 = st.columns(2)
    with c1:
        selected_day = st.selectbox("Filter by day", day_labels)
    with c2:
        selected_event = st.selectbox("Filter by event", event_labels)

    filtered = []
    for row in board:
        day_label = _safe_text(row.get("event_day_label"))
        event_label = _safe_text(row.get("event_label"))
        if selected_day != "All days" and day_label != selected_day:
            continue
        if selected_event != "All events" and event_label != selected_event:
            continue
        filtered.append(row)

    if not filtered:
        st.info("No public partner-board entries match the current filters.")
        st.stop()

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in filtered:
        grouped[_safe_text(row.get("event_day_label"))][_safe_text(row.get("event_label"))].append(row)

    for day_label, events in grouped.items():
        st.markdown(f"### {day_label}")
        for event_label, rows in events.items():
            st.markdown(f"**{event_label}**")
            table_rows = []
            for row in rows:
                player = row.get("player") or {}
                table_rows.append(
                    {
                        "Player": player.get("display_name"),
                        "Skill": player.get("skill"),
                        "Age": player.get("age"),
                        "Contact": player.get("email") if row.get("show_contact_email") else "Contact hidden",
                        "Note": row.get("note"),
                    }
                )
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
