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


def _public_tournament_label(choice: dict[str, Any]) -> str:
    tournament = choice.get("tournament") or {}
    settings = choice.get("settings") or {}
    name = _safe_text(tournament.get("name") or f"Tournament #{tournament.get('id')}")
    start_date = _safe_text(tournament.get("start_date"))
    slug = _safe_text(settings.get("registration_slug"))
    details = " • ".join(part for part in [start_date, slug] if part)
    return f"{name} ({details})" if details else name


def _resolve_public_tournament_id(choices: list[dict[str, Any]], *, qp_tournament_id: str, qp_slug: str) -> str:
    by_id = {str((row.get("tournament") or {}).get("id")): row for row in choices}
    by_slug = {
        _safe_text((row.get("settings") or {}).get("registration_slug")): row
        for row in choices
        if _safe_text((row.get("settings") or {}).get("registration_slug"))
    }
    if qp_tournament_id and qp_tournament_id in by_id:
        return qp_tournament_id
    if qp_slug and qp_slug in by_slug:
        return str((by_slug[qp_slug].get("tournament") or {}).get("id"))
    first = choices[0] if choices else {}
    return str((first.get("tournament") or {}).get("id") or "")


def _set_public_tournament_query_params(*, page_key: str, registration_slug: str | None) -> None:
    st.query_params["page"] = page_key
    if registration_slug:
        st.query_params["tournament"] = registration_slug
    else:
        st.query_params.pop("tournament", None)
    st.query_params.pop("tournament_id", None)


def _select_public_tournament(ctx, supabase, *, page_key: str):
    club_id = _safe_text(getattr(ctx, "club_id", ""))
    choices = list_open_public_tournaments(supabase, club_id)
    if not choices:
        return None, None, [], []

    qp_tournament_id = _safe_text(st.query_params.get("tournament_id"))
    qp_slug = _safe_text(st.query_params.get("tournament"))
    selected_id = _resolve_public_tournament_id(choices, qp_tournament_id=qp_tournament_id, qp_slug=qp_slug)

    by_id = {str((row.get("tournament") or {}).get("id")): row for row in choices}
    selected_choice = by_id.get(selected_id) or choices[0]
    selected_id = str((selected_choice.get("tournament") or {}).get("id") or "")

    if len(choices) > 1:
        selected_id = st.selectbox(
            "Choose a tournament",
            options=[str((row.get("tournament") or {}).get("id")) for row in choices],
            index=max(0, [str((row.get("tournament") or {}).get("id")) for row in choices].index(selected_id)),
            format_func=lambda tid: _public_tournament_label(by_id[tid]),
        )
        selected_choice = by_id[selected_id]

    selected_settings = selected_choice.get("settings") or {}
    selected_slug = _safe_text(selected_settings.get("registration_slug"))

    should_update_qp = (
        _safe_text(st.query_params.get("page")) != page_key
        or _safe_text(st.query_params.get("tournament")) != selected_slug
        or bool(_safe_text(st.query_params.get("tournament_id")))
    )
    if should_update_qp:
        _set_public_tournament_query_params(page_key=page_key, registration_slug=selected_slug or None)
        st.rerun()

    return get_public_tournament_bundle(
        supabase,
        club_id=club_id,
        tournament_id=selected_id or None,
        registration_slug=selected_slug or None,
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
    tournament, settings, days, event_options = _select_public_tournament(
        ctx,
        supabase,
        page_key="tournament_partner_board",
    )
    if not tournament:
        st.info("No open tournaments are currently using the partner board.")
        st.stop()
    if qp_tournament_id and str(tournament.get("id")) != qp_tournament_id:
        st.warning("The requested tournament_id is unavailable. Showing the selected open tournament instead.")
    elif qp_slug and _safe_text(settings.get("registration_slug")) != qp_slug:
        st.warning("The requested tournament link is unavailable. Showing the selected open tournament instead.")

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
                        "Note": row.get("note"),
                    }
                )
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
