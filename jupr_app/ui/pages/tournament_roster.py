from __future__ import annotations

from collections import defaultdict
from typing import Any

import streamlit as st

from jupr_app.domain.tournament_registration_repo import (
    build_public_tournament_roster_state,
    build_public_urls,
    get_public_tournament_bundle,
    list_open_public_tournaments,
    registration_feature_available,
)
from jupr_app.ui.layout import page_shell
from jupr_app.ui.public_links import navigate_same_tab


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


def render(ctx, *, focus_partners: bool = False, legacy_partner_board: bool = False):
    mode_label = "Public" if bool(getattr(ctx, "public_mode", False)) else "Admin"
    page_shell(
        "📋 Tournament Roster",
        "Public roster and partner-seeking players for current open tournaments.",
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
    tournament, settings, days, event_options = _select_public_tournament(ctx, supabase, page_key="tournament_roster")
    if not tournament:
        st.info("No open tournament registrations are currently published.")
        st.stop()

    if legacy_partner_board:
        st.info("Partner listings now live inside the Tournament Roster.")
    if qp_tournament_id and str(tournament.get("id")) != qp_tournament_id:
        st.warning("The requested tournament_id is unavailable. Showing the selected open tournament instead.")
    elif qp_slug and _safe_text(settings.get("registration_slug")) != qp_slug:
        st.warning("The requested tournament link is unavailable. Showing the selected open tournament instead.")

    state = build_public_tournament_roster_state(supabase, tournament, settings, days, event_options)
    public_urls = build_public_urls(
        base_url=_safe_text(st.session_state.get("base_url")),
        tournament_id=str(tournament.get("id")),
        registration_slug=settings.get("registration_slug"),
    )

    top_cols = st.columns([3, 1])
    with top_cols[0]:
        st.subheader(_safe_text(tournament.get("name") or "Tournament"))
        if settings.get("registration_open_at") or settings.get("registration_close_at"):
            window_bits = []
            if settings.get("registration_open_at"):
                window_bits.append(f"Opens: {_safe_text(settings.get('registration_open_at'))}")
            if settings.get("registration_close_at"):
                window_bits.append(f"Closes: {_safe_text(settings.get('registration_close_at'))}")
            st.caption(" • ".join(window_bits))
    with top_cols[1]:
        if st.button("Register", key=f"register_from_roster_{tournament.get('id')}"):
            nav_params = {"tournament_id": str(tournament.get("id"))}
            slug = _safe_text(settings.get("registration_slug"))
            if slug:
                nav_params["tournament"] = slug
            navigate_same_tab(
                page="tournament_registration",
                params=nav_params,
                public_mode=True,
                source_label="tournament_roster:register",
            )

    summary = state.get("summary") or {}
    registrations = state.get("registrations_by_event") or []
    partner_rows = state.get("players_needing_partners") or []

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Registered players/teams", int(summary.get("total_registrations") or 0))
    m2.metric("Public players", int(summary.get("total_players") or 0))
    m3.metric("Events", len({(_safe_text(r.get('event_day_label')), _safe_text(r.get('event_label'))) for r in registrations}))
    m4.metric("Looking for partners", int(summary.get("players_needing_partners") or 0))

    st.markdown("### Players Looking for Partners")
    if not partner_rows:
        st.info("No players are currently listed as looking for a partner.")
    else:
        for row in partner_rows:
            extras = []
            if row.get("skill") not in (None, ""):
                extras.append(f"Skill: {row.get('skill')}")
            if row.get("age_bracket"):
                extras.append(f"Age bracket: {row.get('age_bracket')}")
            elif row.get("age") not in (None, ""):
                extras.append(f"Age: {row.get('age')}")
            if row.get("note"):
                extras.append(f"Note: {row.get('note')}")
            st.markdown(
                f"- **{_safe_text(row.get('player_name') or 'Player')}** — "
                f"{_safe_text(row.get('event_day_label'))} / {_safe_text(row.get('event_family'))} / {_safe_text(row.get('division'))}"
            )
            if extras:
                st.caption(" • ".join(extras))

    if focus_partners:
        st.caption("Showing the partner-seeking section first from a legacy Partner Board link.")

    if not registrations:
        st.info("No players are publicly listed for this tournament yet.")
        st.stop()

    st.markdown("### Tournament Roster")

    day_filters = ["All"] + sorted({_safe_text(row.get("event_day_label")) for row in registrations if _safe_text(row.get("event_day_label"))})
    family_filters = ["All"] + sorted({_safe_text(row.get("event_family")) for row in registrations if _safe_text(row.get("event_family"))})
    division_filters = ["All"] + sorted({_safe_text(row.get("division")) for row in registrations if _safe_text(row.get("division"))})

    c1, c2, c3 = st.columns(3)
    with c1:
        selected_day = st.selectbox("Day", day_filters)
    with c2:
        selected_family = st.selectbox("Event", family_filters)
    with c3:
        selected_division = st.selectbox("Division", division_filters)

    grouped: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in registrations:
        if selected_day != "All" and _safe_text(row.get("event_day_label")) != selected_day:
            continue
        if selected_family != "All" and _safe_text(row.get("event_family")) != selected_family:
            continue
        if selected_division != "All" and _safe_text(row.get("division")) != selected_division:
            continue
        grouped[_safe_text(row.get("event_day_label"))][_safe_text(row.get("event_family"))][_safe_text(row.get("division"))].append(row)

    if not grouped:
        st.info("No roster entries match the selected filters.")
        st.stop()

    for day_label, family_rows in grouped.items():
        with st.expander(day_label or "Day", expanded=True):
            for family, division_rows in family_rows.items():
                st.markdown(f"**{family or 'Event'}**")
                for division, rows in division_rows.items():
                    st.markdown(f"_{division or 'Division'}_")
                    for row in rows:
                        names = " / ".join(
                            _safe_text(member.get("display_name") or "Player") for member in (row.get("members") or [])
                        )
                        details = []
                        if row.get("status"):
                            details.append(str(row.get("status")))
                        skills = [member.get("skill") for member in (row.get("members") or []) if member.get("skill") not in (None, "")]
                        if skills:
                            details.append("Skill " + " / ".join(str(skill) for skill in skills))
                        st.markdown(f"- {names}" + (f" — {' • '.join(details)}" if details else ""))
