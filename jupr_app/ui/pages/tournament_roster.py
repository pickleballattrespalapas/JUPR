from __future__ import annotations

from collections import defaultdict
from typing import Any

import streamlit as st

from jupr_app.domain.tournament_registration_repo import (
    build_public_tournament_roster_state,
    get_public_tournament_bundle,
    list_open_public_tournaments,
    registration_feature_available,
)
from jupr_app.ui.layout import page_shell
from jupr_app.ui.public_links import build_public_url, navigate_same_tab


_STATUS_SORT_ORDER = {
    "Needs Partner": 0,
    "Pending Partner Request": 1,
    "Waitlist": 2,
    "Review": 3,
    "Registered": 4,
}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _md(value: Any) -> str:
    text = _safe_text(value)
    for char in ["\\", "`", "*", "_", "{", "}", "[", "]", "(", ")", "#", "+", "-", ".", "!", "|"]:
        text = text.replace(char, f"\\{char}")
    return text


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


def _ordered_unique(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _safe_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _display_status(row: dict[str, Any]) -> str:
    status = _safe_text(row.get("status"))
    if status:
        return status
    entry_type = _safe_text(row.get("entry_type")).lower()
    if entry_type == "needs_partner":
        return "Needs Partner"
    if entry_type == "pending_partner_request":
        return "Pending Partner Request"
    if entry_type in {"unresolved_partner", "partner_missing"}:
        return "Review"
    return "Registered"


def _format_skill(value: Any) -> str:
    text = _safe_text(value)
    if not text:
        return "—"
    try:
        number = float(text)
        return f"{number:.2f}".rstrip("0").rstrip(".")
    except Exception:
        return text


def _format_age(member: dict[str, Any]) -> str:
    age_bracket = _safe_text(member.get("age_bracket"))
    if age_bracket:
        return age_bracket
    age = _safe_text(member.get("age"))
    return age or "—"


def _team_names(members: list[dict[str, Any]]) -> str:
    names = [_safe_text(member.get("display_name") or "Player") for member in members]
    return " / ".join(name for name in names if name) or "Player"


def _team_skills(members: list[dict[str, Any]]) -> str:
    skills = [_format_skill(member.get("skill")) for member in members if member.get("skill") not in (None, "")]
    return " / ".join(skills) if skills else "—"


def _team_ages(members: list[dict[str, Any]]) -> str:
    ages = [_format_age(member) for member in members if _format_age(member) != "—"]
    return " / ".join(ages) if ages else "—"


def _target_selection_id_from_row(row: dict[str, Any]) -> str:
    direct = _safe_text(row.get("selection_id"))
    if direct:
        return direct
    public_board_entry_key = _safe_text(row.get("board_entry_key"))
    if public_board_entry_key:
        return public_board_entry_key
    source_selection_ids = row.get("source_selection_ids") or []
    if source_selection_ids:
        return _safe_text(source_selection_ids[0])
    members = row.get("members") or []
    if members:
        return _safe_text((members[0] or {}).get("selection_id"))
    return ""


def _request_player_name(row: dict[str, Any]) -> str:
    if _safe_text(row.get("player_name")):
        return _safe_text(row.get("player_name"))
    members = row.get("members") or []
    if members:
        return _safe_text((members[0] or {}).get("display_name") or "Player")
    return "Player"


def _partner_request_params(*, tournament_id: str, registration_slug: str, target_selection_id: str) -> dict[str, str]:
    params = {"tournament_id": tournament_id, "target_selection_id": target_selection_id}
    if registration_slug:
        params["tournament"] = registration_slug
    return params


def _partner_request_url(*, tournament_id: str, registration_slug: str, target_selection_id: str) -> str:
    return build_public_url(
        page="tournament_partner_board",
        params=_partner_request_params(
            tournament_id=tournament_id,
            registration_slug=registration_slug,
            target_selection_id=target_selection_id,
        ),
    )


def _status_with_request_link(row: dict[str, Any], *, tournament_id: str, registration_slug: str) -> str:
    status = _display_status(row)
    target_selection_id = _target_selection_id_from_row(row)
    if status != "Needs Partner" or not target_selection_id:
        return _md(status)
    url = _partner_request_url(
        tournament_id=tournament_id,
        registration_slug=registration_slug,
        target_selection_id=target_selection_id,
    )
    return f"**Needs Partner** · [request partner]({url})"


def _compact_roster_line(row: dict[str, Any], *, tournament_id: str, registration_slug: str) -> str:
    members = [member or {} for member in (row.get("members") or [])]
    details = []
    skills = _team_skills(members)
    ages = _team_ages(members)
    if skills != "—":
        details.append(f"Skill {skills}")
    if ages != "—":
        details.append(f"Age {ages}")
    suffix = f" · {_md(' · '.join(details))}" if details else ""
    return f"- **{_md(_team_names(members))}** — {_status_with_request_link(row, tournament_id=tournament_id, registration_slug=registration_slug)}{suffix}"


def _compact_partner_division_line(row: dict[str, Any], *, tournament_id: str, registration_slug: str) -> str:
    target_selection_id = _target_selection_id_from_row(row)
    division = _safe_text(row.get("division") or row.get("event_label") or "Division")
    context = [
        _safe_text(row.get("event_day_label")),
        _safe_text(row.get("event_family")),
    ]
    context_text = " · ".join(part for part in context if part)
    extras = []
    if row.get("skill") not in (None, ""):
        extras.append(f"Skill {_format_skill(row.get('skill'))}")
    age = _safe_text(row.get("age_bracket") or row.get("age"))
    if age:
        extras.append(f"Age {age}")
    note = _safe_text(row.get("note"))
    if note:
        extras.append(f"Note: {note}")
    request_link = ""
    if target_selection_id:
        request_link = f" · [request partner]({_partner_request_url(tournament_id=tournament_id, registration_slug=registration_slug, target_selection_id=target_selection_id)})"
    extras_text = f" · {_md(' · '.join(extras))}" if extras else ""
    context_suffix = f" — {_md(context_text)}" if context_text else ""
    return f"- **{_md(division)}**{context_suffix}{extras_text}{request_link}"


def _group_partner_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        player_entry_key = _safe_text(row.get("player_entry_key"))
        board_entry_key = _safe_text(row.get("board_entry_key"))
        group_key = player_entry_key or (f"entry:{board_entry_key}" if board_entry_key else f"row:{index}")
        if group_key not in grouped:
            grouped[group_key] = {
                "player_entry_key": player_entry_key or None,
                "player_name": _request_player_name(row),
                "entries": [],
            }
        grouped[group_key]["entries"].append(row)
    return sorted(grouped.values(), key=lambda group: _safe_text(group.get("player_name")).casefold())


def _filter_options(rows: list[dict[str, Any]], field: str) -> list[str]:
    return ["All"] + _ordered_unique([row.get(field) for row in rows])


def _status_options(rows: list[dict[str, Any]]) -> list[str]:
    statuses = _ordered_unique([_display_status(row) for row in rows])
    return ["All"] + sorted(statuses, key=lambda value: _STATUS_SORT_ORDER.get(value, 99))


def _apply_roster_filters(rows: list[dict[str, Any]], *, day: str, event: str, division: str, status: str) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if day != "All" and _safe_text(row.get("event_day_label")) != day:
            continue
        if event != "All" and _safe_text(row.get("event_family")) != event:
            continue
        if division != "All" and _safe_text(row.get("division")) != division:
            continue
        if status != "All" and _display_status(row) != status:
            continue
        filtered.append(row)
    return filtered


def _apply_partner_filters(rows: list[dict[str, Any]], *, day: str, event: str, division: str) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if day != "All" and _safe_text(row.get("event_day_label")) != day:
            continue
        if event != "All" and _safe_text(row.get("event_family")) != event:
            continue
        if division != "All" and _safe_text(row.get("division")) != division:
            continue
        filtered.append(row)
    return filtered


def _render_roster_tab(registrations: list[dict[str, Any]], *, tournament_id: str, registration_slug: str) -> None:
    if not registrations:
        st.info("No players are publicly listed for this tournament yet.")
        return

    st.markdown("### Tournament Roster")
    filter_cols = st.columns(4)
    with filter_cols[0]:
        selected_day = st.selectbox("Day", _filter_options(registrations, "event_day_label"), key="roster_day_filter")
    with filter_cols[1]:
        selected_event = st.selectbox("Event", _filter_options(registrations, "event_family"), key="roster_event_filter")
    with filter_cols[2]:
        selected_division = st.selectbox("Division", _filter_options(registrations, "division"), key="roster_division_filter")
    with filter_cols[3]:
        selected_status = st.selectbox("Status", _status_options(registrations), key="roster_status_filter")

    filtered = _apply_roster_filters(
        registrations,
        day=selected_day,
        event=selected_event,
        division=selected_division,
        status=selected_status,
    )
    if not filtered:
        st.info("No roster entries match the selected filters.")
        return

    grouped: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in filtered:
        grouped[_safe_text(row.get("event_day_label"))][_safe_text(row.get("event_family"))][_safe_text(row.get("division"))].append(row)

    for day_label, family_rows in grouped.items():
        with st.expander(day_label or "Day", expanded=True):
            for family, division_rows in family_rows.items():
                st.markdown(f"**{_md(family or 'Event')}**")
                for division, rows in division_rows.items():
                    st.caption(division or "Division")
                    sorted_rows = sorted(
                        rows,
                        key=lambda row: (_STATUS_SORT_ORDER.get(_display_status(row), 99), _team_names(row.get("members") or [])),
                    )
                    for row in sorted_rows:
                        st.markdown(_compact_roster_line(row, tournament_id=tournament_id, registration_slug=registration_slug))


def _render_partner_tab(partner_rows: list[dict[str, Any]], *, tournament_id: str, registration_slug: str) -> None:
    st.markdown("### Players Needing Partners")
    if not partner_rows:
        st.info("No players are currently looking for a partner.")
        return

    filter_cols = st.columns(3)
    with filter_cols[0]:
        selected_day = st.selectbox("Day", _filter_options(partner_rows, "event_day_label"), key="partner_day_filter")
    with filter_cols[1]:
        selected_event = st.selectbox("Event", _filter_options(partner_rows, "event_family"), key="partner_event_filter")
    with filter_cols[2]:
        selected_division = st.selectbox("Division", _filter_options(partner_rows, "division"), key="partner_division_filter")

    filtered = _apply_partner_filters(
        partner_rows,
        day=selected_day,
        event=selected_event,
        division=selected_division,
    )
    if not filtered:
        st.info("No partner-needed entries match the selected filters.")
        return

    for player in _group_partner_rows(filtered):
        with st.container(border=True):
            st.markdown(f"#### {_md(player.get('player_name') or 'Player')}")
            for row in player.get("entries") or []:
                st.markdown(
                    _compact_partner_division_line(
                        row,
                        tournament_id=tournament_id,
                        registration_slug=registration_slug,
                    )
                )


def render(ctx, *, focus_partners: bool = False, legacy_partner_board: bool = False):
    mode_label = "Public" if bool(getattr(ctx, "public_mode", False)) else "Admin"
    page_title = "🤝 Players Needing Partners" if focus_partners else "📋 Tournament Roster"
    page_description = (
        "Find players who have consented to receive partner requests."
        if focus_partners
        else "See registered players and teams."
    )
    page_shell(
        page_title,
        page_description,
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
    page_key = "tournament_partner_board" if focus_partners else "tournament_roster"
    tournament, settings, days, event_options = _select_public_tournament(ctx, supabase, page_key=page_key)
    if not tournament:
        st.info("No open tournament registrations are currently published.")
        st.stop()

    if qp_tournament_id and str(tournament.get("id")) != qp_tournament_id:
        st.warning("The requested tournament_id is unavailable. Showing the selected open tournament instead.")
    elif qp_slug and _safe_text(settings.get("registration_slug")) != qp_slug:
        st.warning("The requested tournament link is unavailable. Showing the selected open tournament instead.")

    state = build_public_tournament_roster_state(supabase, tournament, settings, days, event_options)
    if state.get("partner_link_schema_available") is False and not bool(getattr(ctx, "public_mode", False)):
        st.warning("Partner request features are unavailable until the partner-link migration is applied.")

    tournament_id = _safe_text(tournament.get("id"))
    registration_slug = _safe_text((settings or {}).get("registration_slug"))
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
            nav_params = {"tournament_id": tournament_id}
            if registration_slug:
                nav_params["tournament"] = registration_slug
            navigate_same_tab(page="tournament_registration", params=nav_params, public_mode=True)

    registrations = state.get("registrations_by_event") or []
    partner_rows = (state.get("partner_board_entries") or []) if focus_partners else []
    event_count = len({(_safe_text(r.get("event_day_label")), _safe_text(r.get("event_family")), _safe_text(r.get("division"))) for r in registrations})

    if focus_partners:
        player_count = len(_group_partner_rows(partner_rows))
        metric_cols = st.columns(2)
        metric_cols[0].metric("Players needing partners", player_count)
        metric_cols[1].metric("Division requests", len(partner_rows))
        _render_partner_tab(partner_rows, tournament_id=tournament_id, registration_slug=registration_slug)
    else:
        metric_cols = st.columns(2)
        metric_cols[0].metric("Registered entries", len(registrations))
        metric_cols[1].metric("Events", event_count or len(event_options or []))
        _render_roster_tab(registrations, tournament_id=tournament_id, registration_slug=registration_slug)
