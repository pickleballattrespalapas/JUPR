from __future__ import annotations

from typing import Any
import uuid

import streamlit as st

from jupr_app.domain.tournament_registration_repo import (
    build_public_urls,
    get_public_tournament_bundle,
    list_open_public_tournaments,
    registration_feature_available,
    registration_is_open,
    save_registration,
)
from jupr_app.ui.layout import page_shell


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _coerce_float(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _coerce_int(value: str) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def _show_tournament_picker(ctx, supabase) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    club_id = str(getattr(ctx, "club_id", ""))
    choices = list_open_public_tournaments(supabase, club_id)
    if not choices:
        st.info("No open tournament registrations are currently published.")
        return None, None, [], []

    labels = [f"{row['tournament'].get('name')}" for row in choices]
    selected_label = st.selectbox("Choose a tournament", labels)
    idx = labels.index(selected_label)
    selected = choices[idx]
    tournament = selected["tournament"]
    settings = selected["settings"]
    st.query_params["tournament"] = settings.get("registration_slug") or ""
    return get_public_tournament_bundle(
        supabase,
        club_id=club_id,
        tournament_id=str(tournament.get("id")),
        registration_slug=settings.get("registration_slug"),
    )


def render(ctx):
    mode_label = "Public" if bool(getattr(ctx, "public_mode", False)) else "Admin"
    page_shell(
        "📝 Tournament Registration",
        "Register players directly inside JUPR and collect partner-needed requests without spreadsheets.",
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
        tournament, settings, days, event_options = _show_tournament_picker(ctx, supabase)
        if not tournament:
            st.stop()

    public_urls = build_public_urls(
        base_url=str(st.session_state.get("base_url") or ""),
        tournament_id=str(tournament.get("id")),
        registration_slug=settings.get("registration_slug"),
    )
    st.subheader(str(tournament.get("name") or "Tournament"))
    c1, c2 = st.columns([2, 1])
    with c1:
        st.caption(f"Tournament ID: {tournament.get('id')}")
        if tournament.get("status"):
            st.caption(f"Operations status: {tournament.get('status')}")
    with c2:
        st.link_button("Partner board", public_urls["partner_board"])

    if settings.get("sponsor_markdown"):
        st.markdown(str(settings.get("sponsor_markdown")))
    if settings.get("rules_markdown"):
        with st.expander("Rules and registration notes", expanded=False):
            st.markdown(str(settings.get("rules_markdown")))
    if settings.get("refund_policy_markdown"):
        with st.expander("Refund policy", expanded=False):
            st.markdown(str(settings.get("refund_policy_markdown")))

    is_open, message = registration_is_open(settings)
    if not is_open:
        st.warning(message or "Registration is not open.")
        st.stop()

    if not days or not event_options:
        st.warning("This tournament does not have a registration form configured yet.")
        st.stop()

    day_events: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for event in event_options:
        day_id = str(event.get("registration_day_id"))
        family = str(event.get("event_family_label") or event.get("label") or "Event")
        day_events.setdefault(day_id, {}).setdefault(family, []).append(event)

    with st.form(f"registration_form_{tournament.get('id')}"):
        st.markdown("### Player information")
        a1, a2 = st.columns(2)
        with a1:
            first_name = st.text_input("First name")
            email = st.text_input("Email *")
            phone = st.text_input("Phone / WhatsApp")
            doubles_skill = st.text_input("Doubles skill")
            age = st.text_input("Age")
        with a2:
            last_name = st.text_input("Last name")
            display_name = st.text_input("Display name")
            dupr_id = st.text_input("DUPR ID")
            singles_skill = st.text_input("Singles skill")
            age_bracket = st.text_input("Age bracket / division")
        gender = st.selectbox("Gender", ["", "Female", "Male", "Other", "Prefer not to say"])
        notes = st.text_area("Notes", height=80)
        wants_contact = st.checkbox(
            "Allow my email to appear on the public partner board if I mark that I need a partner.",
            value=False,
        )

        st.markdown("### Event selections")
        selections: list[dict[str, Any]] = []
        for day in days:
            day_id = str(day.get("id"))
            family_map = day_events.get(day_id, {})
            st.markdown(f"#### {day.get('label')}")
            for family, options in family_map.items():
                st.markdown(f"**{family}**")
                division_lookup = {"— Not playing this event —": None}
                for event in options:
                    division_lookup[str(event.get("division_name") or event.get("label"))] = event
                selected_label = st.selectbox(
                    f"Division for {family}",
                    list(division_lookup.keys()),
                    key=f"event_pick_{tournament.get('id')}_{day_id}_{family}",
                )
                selected_event = division_lookup[selected_label]
                if not selected_event:
                    continue

                selection_row = {
                    "id": _uid("sel"),
                    "registration_day_id": day_id,
                    "event_option_id": str(selected_event.get("id")),
                    "partner_mode": "NONE",
                }

                if bool(selected_event.get("partner_required")):
                    partner_mode_label = st.radio(
                        f"Partner status for {selected_event.get('division_name') or selected_event.get('label')}",
                        ["I already have a partner", "I need a partner"],
                        horizontal=True,
                        key=f"partner_mode_{day_id}_{selected_event.get('id')}",
                    )
                    if partner_mode_label == "I already have a partner":
                        selection_row["partner_mode"] = "HAS_PARTNER"
                        p1, p2 = st.columns(2)
                        with p1:
                            selection_row["partner_name"] = st.text_input("Partner name", key=f"partner_name_{selected_event.get('id')}")
                            selection_row["partner_email"] = st.text_input("Partner email", key=f"partner_email_{selected_event.get('id')}")
                            selection_row["partner_phone"] = st.text_input("Partner phone", key=f"partner_phone_{selected_event.get('id')}")
                        with p2:
                            selection_row["partner_dupr_id"] = st.text_input("Partner DUPR ID", key=f"partner_dupr_{selected_event.get('id')}")
                            selection_row["partner_skill"] = _coerce_float(st.text_input("Partner skill", key=f"partner_skill_{selected_event.get('id')}"))
                            selection_row["partner_age"] = _coerce_int(st.text_input("Partner age", key=f"partner_age_{selected_event.get('id')}"))
                    else:
                        selection_row["partner_mode"] = "NEEDS_PARTNER"
                        selection_row["show_on_partner_board"] = bool(settings.get("partner_board_enabled", True)) and wants_contact
                        selection_row["partner_note"] = st.text_input("Note for partner board", key=f"partner_note_{selected_event.get('id')}")
                selections.append(selection_row)

        submitted = st.form_submit_button("Submit registration", type="primary")

    if submitted:
        if not email.strip():
            st.error("Email is required.")
            st.stop()
        if not display_name.strip() and not (first_name.strip() or last_name.strip()):
            st.error("Enter at least a display name or first/last name.")
            st.stop()

        for selection in selections:
            if selection.get("partner_mode") == "HAS_PARTNER":
                if not str(selection.get("partner_name") or "").strip() and not str(selection.get("partner_email") or "").strip():
                    st.error("For doubles events with a named partner, enter at least the partner name or email.")
                    st.stop()

        try:
            result = save_registration(
                supabase,
                tournament_id=str(tournament.get("id")),
                payload={
                    "first_name": first_name,
                    "last_name": last_name,
                    "display_name": display_name,
                    "email": email,
                    "phone": phone,
                    "dupr_id": dupr_id,
                    "doubles_skill": _coerce_float(doubles_skill),
                    "singles_skill": _coerce_float(singles_skill),
                    "age": _coerce_int(age),
                    "age_bracket": age_bracket,
                    "gender": gender,
                    "notes": notes,
                    "wants_partner_board_contact": wants_contact,
                    "selections": selections,
                },
            )
            st.success(
                f"Registration saved. Confirmation record: {result.get('registration_id')}. If you submit again with the same email, your registration will update."
            )
            st.link_button("Open partner board", public_urls["partner_board"])
        except Exception as exc:
            st.error(f"Could not save registration: {exc}")
