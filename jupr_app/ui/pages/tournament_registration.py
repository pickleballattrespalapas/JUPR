from __future__ import annotations

from typing import Any
import json
import uuid

import streamlit as st

from jupr_app.domain.tournament_registration_compiler import validate_selection_against_skill
from jupr_app.domain.tournament_registration_repo import (
    build_registration_state,
    build_public_urls,
    get_public_tournament_bundle,
    is_day_enabled,
    list_open_public_tournaments,
    public_event_option_visibility,
    registration_feature_available,
    registration_is_open,
    save_registration,
)
from jupr_app.ui.layout import page_shell


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _coerce_float(value: str) -> float | None:
    text = _safe_text(value)
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _coerce_int(value: str) -> int | None:
    text = _safe_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def _show_tournament_picker(ctx, supabase) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    club_id = _safe_text(getattr(ctx, "club_id", ""))
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


def _group_events(days: list[dict[str, Any]], event_options: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    day_ids = {str(day.get("id")) for day in days}
    for event in event_options:
        day_id = str(event.get("registration_day_id"))
        if day_id not in day_ids:
            continue
        family = _safe_text(event.get("event_family_label") or event.get("label") or "Event")
        grouped.setdefault(day_id, {}).setdefault(family, []).append(event)
    return grouped


def _division_choice_label(event: dict[str, Any], *, eligible: bool = True) -> str:
    name = _safe_text(event.get("division_name") or event.get("label") or "Division")
    parts: list[str] = []
    skill = _safe_text(event.get("skill_label"))
    age = _safe_text(event.get("age_label"))
    price = event.get("price_usd")
    if skill and skill.lower() not in {"open", name.lower()}:
        parts.append(skill)
    if age and age.lower() not in {"all ages", name.lower()}:
        parts.append(age)
    if price not in (None, "", "None"):
        parts.append(f"${price}")
    label = f"{name} — {' • '.join(parts)}" if parts else name
    return label if eligible else f"{label} ⛔ Not eligible based on current rating"




def _preview_division_eligibility(event: dict[str, Any], player: dict[str, Any]) -> tuple[bool, str | None]:
    preview_selection = {"partner_mode": "NEEDS_PARTNER" if bool(event.get("partner_required")) else "NONE"}
    return validate_selection_against_skill(
        event=event,
        selection=preview_selection,
        player=player,
        partner=None,
        allow_missing_partner_for_preview=True,
    )

def _division_help(event: dict[str, Any]) -> str:
    details: list[str] = []
    age_mode = _safe_text(event.get("age_mode"))
    age_rules = _safe_text(event.get("age_rules"))
    event_format = _safe_text(event.get("event_format_override") or event.get("event_format_default"))
    scoring = _safe_text(event.get("scoring_override") or event.get("scoring_default"))
    capacity = event.get("capacity_teams")
    if event_format:
        details.append(event_format.replace("_", " ").title())
    if scoring:
        details.append(scoring.replace("_", " ").title())
    if capacity not in (None, "", "None"):
        details.append(f"Cap: {capacity}")
    if age_mode == "AUTO_AGE_SPLIT":
        details.append("Auto age split if minimum teams are met")
    elif age_mode == "SPLIT_AGE":
        details.append("Split-age partner rule")
    elif age_mode == "FIXED_AGE_BRACKET":
        details.append(_safe_text(event.get("age_label")))
    if age_rules:
        try:
            parsed_age_rules = json.loads(age_rules)
        except Exception:
            parsed_age_rules = {}
        min_teams = parsed_age_rules.get("min_teams_per_age_group")
        split_threshold = parsed_age_rules.get("split_age_threshold")
        notes = _safe_text(parsed_age_rules.get("notes"))
        if age_mode == "AUTO_AGE_SPLIT" and min_teams not in (None, ""):
            details.append(f"Auto-split minimum: {min_teams} teams/group")
        if age_mode == "SPLIT_AGE" and split_threshold not in (None, ""):
            details.append(f"Split-age threshold: {split_threshold}+")
        if notes:
            details.append(notes)
        if not any([min_teams not in (None, ""), split_threshold not in (None, ""), notes]):
            details.append("Age rule details available")
    return " • ".join(part for part in details if part)


def render(ctx):
    mode_label = "Public" if bool(getattr(ctx, "public_mode", False)) else "Admin"
    page_shell(
        "📝 Tournament Registration",
        "Register inside JUPR without spreadsheets. Choose your day, event, and division, then tell the organizer whether you already have a partner or need one.",
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
        if qp_tournament_id or qp_slug:
            st.warning("This tournament is unavailable for public registration.")
        tournament, settings, days, event_options = _show_tournament_picker(ctx, supabase)
        if not tournament:
            st.stop()

    public_urls = build_public_urls(
        base_url=_safe_text(st.session_state.get("base_url")),
        tournament_id=str(tournament.get("id")),
        registration_slug=settings.get("registration_slug"),
    )
    st.subheader(_safe_text(tournament.get("name") or "Tournament"))
    top_cols = st.columns([2, 1])
    with top_cols[0]:
        if settings.get("registration_open_at") or settings.get("registration_close_at"):
            window_bits = []
            if settings.get("registration_open_at"):
                window_bits.append(f"Opens: {_safe_text(settings.get('registration_open_at'))}")
            if settings.get("registration_close_at"):
                window_bits.append(f"Closes: {_safe_text(settings.get('registration_close_at'))}")
            st.caption(" • ".join(window_bits))
    with top_cols[1]:
        st.link_button("Partner board", public_urls["partner_board"])

    if settings.get("sponsor_markdown"):
        st.markdown(_safe_text(settings.get("sponsor_markdown")))
    if settings.get("rules_markdown"):
        with st.expander("Rules and registration notes", expanded=False):
            st.markdown(_safe_text(settings.get("rules_markdown")))
    if settings.get("refund_policy_markdown"):
        with st.expander("Refund policy", expanded=False):
            st.markdown(_safe_text(settings.get("refund_policy_markdown")))

    is_open, message = registration_is_open(settings)
    if not is_open:
        st.warning(message or "Registration is not open.")
        st.stop()

    days = [row for row in days if is_day_enabled(row)]
    selectable_event_options = [row for row in event_options if public_event_option_visibility(row) == "selectable"]
    blocked_visible_options = [row for row in event_options if public_event_option_visibility(row) == "visible_blocked"]

    if not days or not selectable_event_options:
        st.warning("This tournament does not have a registration form configured yet.")
        st.stop()

    grouped_events = _group_events(days, selectable_event_options)
    blocked_by_family: dict[tuple[str, str], list[str]] = {}
    for event in blocked_visible_options:
        day_id = str(event.get("registration_day_id"))
        family = _safe_text(event.get("event_family_label") or event.get("label") or "Event")
        division_name = _safe_text(event.get("division_name") or event.get("label") or "Division")
        blocked_by_family.setdefault((day_id, family), []).append(division_name)

    with st.form(f"registration_form_{tournament.get('id')}"):
        st.markdown("### 1. Player information")
        c1, c2 = st.columns(2)
        with c1:
            first_name = st.text_input("First name")
            email = st.text_input("Email *")
            phone = st.text_input("Phone / WhatsApp")
            doubles_skill = st.text_input("Doubles skill")
            age = st.text_input("Age")
        with c2:
            last_name = st.text_input("Last name")
            display_name = st.text_input("Display name")
            dupr_id = st.text_input("DUPR ID")
            singles_skill = st.text_input("Singles skill")
            age_bracket = st.text_input("Age bracket / age note")
        gender = st.selectbox("Gender", ["", "Female", "Male", "Other", "Prefer not to say"])
        notes = st.text_area("Notes for tournament staff", height=90)

        player_profile = {
            "doubles_skill": _coerce_float(doubles_skill),
            "singles_skill": _coerce_float(singles_skill),
        }

        st.markdown("### 2. Choose your divisions")
        st.caption("Work day by day. You can skip any event family you are not playing.")
        selections: list[dict[str, Any]] = []
        for day in days:
            day_id = str(day.get("id"))
            family_map = grouped_events.get(day_id, {})
            if not family_map:
                continue
            st.markdown(f"#### {day.get('label')}")
            for family, options in family_map.items():
                st.markdown(f"**{family}**")
                option_lookup = {"— Not playing this event —": None}
                ordered_labels = ["— Not playing this event —"]
                eligibility_lookup: dict[str, tuple[bool, str | None]] = {}
                for event in options:
                    eligible, reason = _preview_division_eligibility(event, player_profile)
                    label = _division_choice_label(event, eligible=eligible)
                    option_lookup[label] = event
                    ordered_labels.append(label)
                    eligibility_lookup[str(event.get("id"))] = (eligible, reason)
                selected_label = st.selectbox(
                    f"Choose your division for {family}",
                    ordered_labels,
                    key=f"event_pick_{tournament.get('id')}_{day_id}_{family}",
                )
                selected_event = option_lookup[selected_label]
                if not selected_event:
                    blocked_names = blocked_by_family.get((day_id, family), [])
                    if blocked_names:
                        st.caption("Closed divisions: " + ", ".join(sorted(blocked_names)))
                    continue

                help_text = _division_help(selected_event)
                if help_text:
                    st.caption(help_text)

                current_eligible, current_reason = eligibility_lookup.get(str(selected_event.get("id")), (True, None))
                if not current_eligible:
                    st.warning(current_reason or "Not eligible based on current rating.")
                elif bool(selected_event.get("partner_required")):
                    st.caption("For doubles, final eligibility is validated at submit time using both players' ratings when a partner is named.")

                selection_row: dict[str, Any] = {
                    "id": _uid("sel"),
                    "registration_day_id": day_id,
                    "event_option_id": str(selected_event.get("id")),
                    "partner_mode": "NONE",
                }

                if bool(selected_event.get("partner_required")):
                    partner_mode_label = st.radio(
                        f"Partner status for {_safe_text(selected_event.get('division_name') or selected_event.get('label'))}",
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
                        if bool(settings.get("partner_board_enabled", True)):
                            selection_row["show_on_partner_board"] = st.checkbox(
                                "Show me on the public partner board for this division",
                                value=False,
                                key=f"partner_board_optin_{selected_event.get('id')}",
                            )
                        else:
                            selection_row["show_on_partner_board"] = False
                        selection_row["partner_note"] = st.text_input(
                            "Short note for partner board (optional)",
                            key=f"partner_note_{selected_event.get('id')}",
                        )
                selections.append(selection_row)

        submitted = st.form_submit_button("Submit registration", type="primary")

    if submitted:
        final_display_name = _safe_text(display_name) or " ".join(part for part in [_safe_text(first_name), _safe_text(last_name)] if part)
        if not _safe_text(email):
            st.error("Email is required.")
            st.stop()
        if not final_display_name:
            st.error("Enter at least a display name or first/last name.")
            st.stop()
        if not selections:
            st.error("Choose at least one division before submitting.")
            st.stop()

        for selection in selections:
            if selection.get("partner_mode") == "HAS_PARTNER":
                if not _safe_text(selection.get("partner_name")) and not _safe_text(selection.get("partner_email")):
                    st.error("For doubles events with a named partner, enter at least the partner name or partner email.")
                    st.stop()

        event_lookup = {str(row.get("id")): row for row in event_options}
        submit_player = {
            "doubles_skill": _coerce_float(doubles_skill),
            "singles_skill": _coerce_float(singles_skill),
        }
        for selection in selections:
            event = event_lookup.get(str(selection.get("event_option_id") or ""))
            if not event:
                continue
            partner = None
            if _safe_text(selection.get("partner_mode")).upper() == "HAS_PARTNER":
                partner = {
                    "doubles_skill": selection.get("partner_skill"),
                    "singles_skill": selection.get("partner_skill"),
                }
            eligible, reason = validate_selection_against_skill(
                event=event,
                selection=selection,
                player=submit_player,
                partner=partner,
                allow_missing_partner_for_preview=False,
            )
            if not eligible:
                division_label = _safe_text(event.get("division_name") or event.get("label") or event.get("id"))
                st.error(f"{division_label}: {reason or 'Skill eligibility requirements were not met.'}")
                st.stop()

        try:
            state_before_submit = build_registration_state(supabase, tournament, settings, days, event_options)
        except Exception:
            state_before_submit = {}
        event_lookup = {str(row.get("id")): row for row in event_options}
        roster_lookup = {str(row.get("event_option_id")): row for row in (state_before_submit.get("event_rosters") or [])}
        at_capacity_warnings: list[str] = []
        for selection in selections:
            event_option_id = str(selection.get("event_option_id") or "")
            event = event_lookup.get(event_option_id) or {}
            capacity = event.get("capacity_teams")
            if not capacity:
                continue
            try:
                cap_value = int(capacity)
            except Exception:
                continue
            entries = (roster_lookup.get(event_option_id) or {}).get("entries") or []
            occupied_slots = sum(
                1
                for row in entries
                if _safe_text(row.get("status")).upper() not in {"NEEDS_PARTNER", "PARTNER_MISSING"}
            )
            if occupied_slots >= cap_value:
                at_capacity_warnings.append(_safe_text(event.get("division_name") or event.get("label") or event_option_id))
        if at_capacity_warnings:
            st.warning(
                "Heads up: these divisions appear full and this registration will likely be waitlisted: "
                + ", ".join(sorted(set(at_capacity_warnings)))
            )

        try:
            result = save_registration(
                supabase,
                tournament_id=str(tournament.get("id")),
                payload={
                    "first_name": first_name,
                    "last_name": last_name,
                    "display_name": final_display_name,
                    "email": email,
                    "phone": phone,
                    "dupr_id": dupr_id,
                    "doubles_skill": _coerce_float(doubles_skill),
                    "singles_skill": _coerce_float(singles_skill),
                    "age": _coerce_int(age),
                    "age_bracket": age_bracket,
                    "gender": gender,
                    "notes": notes,
                    "wants_partner_board_contact": any(bool(row.get("show_on_partner_board")) for row in selections),
                    "selections": selections,
                },
            )
            st.success(
                f"Registration saved. Confirmation record: {result.get('registration_id')}. Submitting again with the same email updates your registration. Final placement may still change after partner matching and waitlist review."
            )
            st.link_button("Open partner board", public_urls["partner_board"])
        except Exception as exc:
            st.error(f"Could not save registration: {exc}")
