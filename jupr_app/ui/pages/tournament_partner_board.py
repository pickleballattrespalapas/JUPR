from __future__ import annotations

import pandas as pd
import streamlit as st

from jupr_app.domain.tournament_registration_repo import (
    build_registration_state,
    get_public_tournament_bundle,
    list_open_public_tournaments,
    registration_feature_available,
)
from jupr_app.ui.layout import page_shell


def _pick_tournament(ctx, supabase):
    club_id = str(getattr(ctx, "club_id", ""))
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
        "Players who marked themselves as needing a partner, grouped by event.",
        mode_label=mode_label,
    )

    supabase = getattr(ctx, "supabase", None)
    club_id = str(getattr(ctx, "club_id", ""))
    if supabase is None or not club_id:
        st.error("Missing database context.")
        st.stop()

    available, detail = registration_feature_available(supabase)
    if not available:
        st.error("Tournament registration is not enabled yet. Apply the registration SQL migration first.")
        if detail:
            st.caption(detail)
        st.stop()

    qp_tournament_id = str(st.query_params.get("tournament_id", "")).strip()
    qp_slug = str(st.query_params.get("tournament", "")).strip()
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

    st.subheader(str(tournament.get("name") or "Tournament"))
    if not board:
        st.info("Nobody is currently published on the partner board for this tournament.")
        st.stop()

    day_labels = ["All days"] + sorted({str(row.get("event_day_label") or "") for row in board})
    event_labels = ["All events"] + sorted({str(row.get("event_label") or "") for row in board})
    c1, c2 = st.columns(2)
    with c1:
        selected_day = st.selectbox("Filter by day", day_labels)
    with c2:
        selected_event = st.selectbox("Filter by event", event_labels)

    filtered = []
    for row in board:
        if selected_day != "All days" and str(row.get("event_day_label") or "") != selected_day:
            continue
        if selected_event != "All events" and str(row.get("event_label") or "") != selected_event:
            continue
        filtered.append(row)

    if not filtered:
        st.info("No public partner-board entries match the current filters.")
        st.stop()

    rows = []
    for row in filtered:
        player = row.get("player") or {}
        rows.append(
            {
                "Day": row.get("event_day_label"),
                "Event": row.get("event_label"),
                "Player": player.get("display_name"),
                "Email": player.get("email") if row.get("show_contact_email") else None,
                "Skill": player.get("skill"),
                "Age": player.get("age"),
                "Note": row.get("note"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
