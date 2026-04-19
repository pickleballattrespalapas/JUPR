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


def _active_players_from_ctx(ctx) -> list[dict[str, Any]]:
    df_players_active = getattr(ctx, "df_players_active", None)
    if df_players_active is None:
        return []
    try:
        if bool(getattr(df_players_active, "empty", True)):
            return []
        return [dict(row) for row in df_players_active.to_dict(orient="records")]
    except Exception:
        return []


def _load_active_players(supabase, *, club_id: str, ctx) -> list[dict[str, Any]]:
    rows = _active_players_from_ctx(ctx)
    if rows:
        return rows
    try:
        base_query = (
            supabase.table("players")
            .select("id,name,display_name,email,phone,whatsapp,dupr_id,doubles_skill,singles_skill,gender,age,inactive_at,active")
            .eq("club_id", str(club_id))
            .order("name")
            .limit(2000)
        )
        try:
            resp = base_query.is_("inactive_at", None).execute()
        except Exception:
            resp = base_query.eq("active", True).execute()
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _player_label(player: dict[str, Any]) -> str:
    display_name = _safe_text(player.get("display_name") or player.get("name") or f"Player #{player.get('id')}")
    rating = player.get("doubles_skill") or player.get("singles_skill")
    if rating in (None, ""):
        return display_name
    return f"{display_name} · Rating {rating}"


def _family_key(day_id: str, family: str) -> str:
    return f"{day_id}::{family}"


def _normalize_name_for_match(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _player_full_name(player: dict[str, Any]) -> str:
    first = _safe_text(player.get("first_name"))
    last = _safe_text(player.get("last_name"))
    if first or last:
        return " ".join(part for part in [first, last] if part)
    return _safe_text(player.get("display_name") or player.get("name"))


def _likely_active_player_matches(players: list[dict[str, Any]], *, first_name: str, last_name: str) -> list[dict[str, Any]]:
    first = _normalize_name_for_match(first_name)
    last = _normalize_name_for_match(last_name)
    if not first or not last:
        return []

    target_full = _normalize_name_for_match(f"{first} {last}")
    exact: list[dict[str, Any]] = []
    contains: list[dict[str, Any]] = []
    for row in players:
        full_name = _normalize_name_for_match(_player_full_name(row))
        if not full_name:
            continue
        if full_name == target_full:
            exact.append(row)
            continue
        tokens = full_name.split()
        if first in tokens and last in tokens:
            contains.append(row)

    matches = exact or contains
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in matches:
        pid = str(row.get("id") or "")
        key = pid or _normalize_name_for_match(_player_full_name(row))
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped[:8]


def _gender_filter_allows_event(event: dict[str, Any], gender: str) -> bool:
    g = _normalize_name_for_match(gender)
    if g not in {"male", "female"}:
        return True
    restriction = _normalize_name_for_match(event.get("gender_restriction"))
    if not restriction:
        return True
    female_only = {"female", "women", "womens", "woman", "girls", "f"}
    male_only = {"male", "men", "mens", "man", "boys", "m"}
    if g == "male" and restriction in female_only:
        return False
    if g == "female" and restriction in male_only:
        return False
    return True


def _rating_filter_allows_event(event: dict[str, Any], player: dict[str, Any]) -> bool:
    eligible, reason = _preview_division_eligibility(event, player)
    if eligible:
        return True
    reason_text = _normalize_name_for_match(reason)
    if "rated above" in reason_text or "one player is rated above" in reason_text or "please register for" in reason_text:
        return False
    return True


def _visible_division_options(options: list[dict[str, Any]], *, gender: str, player: dict[str, Any]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for event in options:
        if not _gender_filter_allows_event(event, gender):
            continue
        if not _rating_filter_allows_event(event, player):
            continue
        filtered.append(event)
    return filtered


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
        active_players = _load_active_players(supabase, club_id=club_id, ctx=ctx)

        c1, c2 = st.columns(2)
        with c1:
            first_name = st.text_input("First name")
        with c2:
            last_name = st.text_input("Last name")

        likely_matches = _likely_active_player_matches(active_players, first_name=first_name, last_name=last_name)
        selected_existing_player: dict[str, Any] | None = None

        if likely_matches:
            if len(likely_matches) == 1:
                selected_existing_player = likely_matches[0]
            else:
                match_choices: dict[str, dict[str, Any] | None] = {"Continue as new player": None}
                for row in likely_matches:
                    label = _player_label(row)
                    if label in match_choices:
                        label = f"{label} (#{row.get('id')})"
                    match_choices[label] = row
                selected_label = st.selectbox(
                    "We found multiple active JUPR players with this name",
                    list(match_choices.keys()),
                    key=f"match_existing_player_{tournament.get('id')}",
                )
                selected_existing_player = match_choices.get(selected_label)

        using_existing_player = selected_existing_player is not None
        existing_first = _safe_text((selected_existing_player or {}).get("first_name")) or first_name
        existing_last = _safe_text((selected_existing_player or {}).get("last_name")) or last_name
        existing_display = _safe_text((selected_existing_player or {}).get("display_name") or (selected_existing_player or {}).get("name"))
        existing_email = _safe_text((selected_existing_player or {}).get("email"))
        existing_phone = _safe_text((selected_existing_player or {}).get("phone") or (selected_existing_player or {}).get("whatsapp"))

        if using_existing_player:
            matched_rating = _coerce_float(selected_existing_player.get("doubles_skill")) or _coerce_float(selected_existing_player.get("singles_skill"))
            rating_text = f" ({matched_rating:.2f})" if matched_rating is not None else ""
            st.success(f"Matched to active JUPR player: {_safe_text(existing_display) or _safe_text(existing_first + ' ' + existing_last)}{rating_text}")

        c1, c2 = st.columns(2)
        with c1:
            email = st.text_input("Email *", value=existing_email if using_existing_player else "")
            phone = st.text_input("Phone / WhatsApp", value=existing_phone if using_existing_player else "")
            age_default = _safe_text((selected_existing_player or {}).get("age")) if using_existing_player else ""
            age = st.text_input("Age", value=age_default)
            doubles_skill = ""
            if not using_existing_player:
                doubles_skill = st.text_input("Doubles skill")
        with c2:
            display_name = st.text_input("Display name", value=existing_display)
            dupr_id = ""
            singles_skill = ""
            if not using_existing_player:
                dupr_id = st.text_input("DUPR ID")
                singles_skill = st.text_input("Singles skill")

        gender_options = ["", "Female", "Male", "Other", "Prefer not to say"]
        existing_gender = _safe_text((selected_existing_player or {}).get("gender"))
        gender_index = gender_options.index(existing_gender) if existing_gender in gender_options else 0
        gender = st.selectbox("Gender", gender_options, index=gender_index)
        notes = st.text_area("Notes for tournament staff", height=90)

        if using_existing_player and selected_existing_player:
            profile_doubles = _coerce_float(selected_existing_player.get("doubles_skill"))
            profile_singles = _coerce_float(selected_existing_player.get("singles_skill"))
            first_name = existing_first
            last_name = existing_last
            dupr_id = _safe_text(selected_existing_player.get("dupr_id"))
        else:
            profile_doubles = _coerce_float(doubles_skill)
            profile_singles = _coerce_float(singles_skill)
        player_profile = {
            "doubles_skill": profile_doubles,
            "singles_skill": profile_singles,
        }

        visible_event_options = _visible_division_options(
            selectable_event_options,
            gender=gender,
            player=player_profile,
        )
        visible_grouped_events = _group_events(days, visible_event_options)

        st.markdown("### 2. Choose your divisions")
        st.caption("Work day by day. Check divisions you want to play.")
        selections: list[dict[str, Any]] = []
        family_selection_counts: dict[str, int] = {}
        for day in days:
            day_id = str(day.get("id"))
            family_map = visible_grouped_events.get(day_id, {})
            if not family_map:
                continue
            st.markdown(f"#### {day.get('label')}")
            for family, options in family_map.items():
                st.markdown(f"**{family}**")
                eligibility_lookup: dict[str, tuple[bool, str | None]] = {}
                chosen_in_family = 0
                for event in options:
                    eligible, reason = _preview_division_eligibility(event, player_profile)
                    eligibility_lookup[str(event.get("id"))] = (eligible, reason)
                    checked = st.checkbox(
                        _division_choice_label(event, eligible=eligible),
                        value=False,
                        key=f"event_pick_{tournament.get('id')}_{day_id}_{family}_{event.get('id')}",
                    )
                    help_text = _division_help(event)
                    if help_text:
                        st.caption(help_text)
                    if not checked:
                        continue
                    chosen_in_family += 1
                    selection_row: dict[str, Any] = {
                        "id": _uid("sel"),
                        "registration_day_id": day_id,
                        "event_option_id": str(event.get("id")),
                        "partner_mode": "NONE",
                    }
                    current_eligible, current_reason = eligibility_lookup.get(str(event.get("id")), (True, None))
                    if not current_eligible:
                        st.warning(current_reason or "Not eligible based on current rating.")
                    elif bool(event.get("partner_required")):
                        st.caption(
                            "For doubles, final eligibility is validated at submit time using both players' ratings when a partner is named."
                        )
                    if bool(event.get("partner_required")):
                        partner_mode_label = st.radio(
                            f"Partner status for {_safe_text(event.get('division_name') or event.get('label'))}",
                            ["I already have a partner", "I need a partner"],
                            horizontal=True,
                            key=f"partner_mode_{day_id}_{event.get('id')}",
                        )
                        if partner_mode_label == "I already have a partner":
                            selection_row["partner_mode"] = "HAS_PARTNER"
                            p1, p2 = st.columns(2)
                            with p1:
                                selection_row["partner_name"] = st.text_input("Partner name", key=f"partner_name_{event.get('id')}")
                                selection_row["partner_email"] = st.text_input(
                                    "Partner email", key=f"partner_email_{event.get('id')}"
                                )
                                selection_row["partner_phone"] = st.text_input(
                                    "Partner phone", key=f"partner_phone_{event.get('id')}"
                                )
                            with p2:
                                selection_row["partner_dupr_id"] = st.text_input(
                                    "Partner DUPR ID", key=f"partner_dupr_{event.get('id')}"
                                )
                                selection_row["partner_skill"] = _coerce_float(
                                    st.text_input("Partner skill", key=f"partner_skill_{event.get('id')}")
                                )
                                selection_row["partner_age"] = _coerce_int(
                                    st.text_input("Partner age", key=f"partner_age_{event.get('id')}")
                                )
                        else:
                            selection_row["partner_mode"] = "NEEDS_PARTNER"
                            if bool(settings.get("partner_board_enabled", True)):
                                selection_row["show_on_partner_board"] = st.checkbox(
                                    "Show me on the public partner board for this division",
                                    value=False,
                                    key=f"partner_board_optin_{event.get('id')}",
                                )
                            else:
                                selection_row["show_on_partner_board"] = False
                            selection_row["partner_note"] = st.text_input(
                                "Short note for partner board (optional)",
                                key=f"partner_note_{event.get('id')}",
                            )
                    selections.append(selection_row)
                family_selection_counts[_family_key(day_id, family)] = chosen_in_family
                blocked_names = blocked_by_family.get((day_id, family), [])
                if blocked_names:
                    st.caption("Closed divisions: " + ", ".join(sorted(blocked_names)))

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
        over_selected_groups: list[str] = []
        for day in days:
            day_id = str(day.get("id"))
            family_map = visible_grouped_events.get(day_id, {})
            for family in family_map.keys():
                selected_count = family_selection_counts.get(_family_key(day_id, family), 0)
                if selected_count > 1:
                    over_selected_groups.append(f"{_safe_text(day.get('label') or day_id)} / {family}")
        if over_selected_groups:
            st.error(
                "Choose only one division per day/event family group. Please fix: " + ", ".join(sorted(set(over_selected_groups)))
            )
            st.stop()

        for selection in selections:
            if selection.get("partner_mode") == "HAS_PARTNER":
                if not _safe_text(selection.get("partner_name")) and not _safe_text(selection.get("partner_email")):
                    st.error("For doubles events with a named partner, enter at least the partner name or partner email.")
                    st.stop()

        event_lookup = {str(row.get("id")): row for row in event_options}
        submit_player = dict(player_profile)
        for selection in selections:
            event = event_lookup.get(str(selection.get("event_option_id") or ""))
            if not event:
                st.error(
                    f"Selected division {selection.get('event_option_id')} is no longer available. Please refresh and try again."
                )
                st.stop()
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
                    "doubles_skill": submit_player.get("doubles_skill"),
                    "singles_skill": submit_player.get("singles_skill"),
                    "age": _coerce_int(age),
                    "age_bracket": None,
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
