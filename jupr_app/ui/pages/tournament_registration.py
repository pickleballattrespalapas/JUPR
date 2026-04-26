from __future__ import annotations

from typing import Any
import json
import uuid

import streamlit as st

from jupr_app.domain.tournament_registration_compiler import validate_selection_against_skill
from jupr_app.domain.tournament_registration_repo import (
    ADMIN_PAYMENT_STATUS_OPTIONS,
    ADMIN_REGISTRATION_STATUS_OPTIONS,
    PARTNER_MODE_OPTIONS,
    build_registration_state,
    build_public_urls,
    cancel_registration,
    create_admin_registration,
    delete_registration,
    get_public_tournament_bundle,
    get_registration_settings,
    is_day_enabled,
    list_event_options as list_registration_event_options,
    list_existing_tournaments,
    list_open_public_tournaments,
    list_registration_admin_rows,
    list_registration_days,
    public_event_option_visibility,
    registration_feature_available,
    registration_is_imported_to_draw,
    registration_is_open,
    save_registration,
    update_admin_registration,
    update_admin_registration_selection,
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


def _select_public_tournament(ctx, supabase, *, page_key: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    club_id = _safe_text(getattr(ctx, "club_id", ""))
    choices = list_open_public_tournaments(supabase, club_id)
    if not choices:
        st.info("No open tournament registrations are currently published.")
        return None, None, [], []

    qp_tournament_id = _safe_text(st.query_params.get("tournament_id"))
    qp_slug = _safe_text(st.query_params.get("tournament"))
    selected_id = _resolve_public_tournament_id(choices, qp_tournament_id=qp_tournament_id, qp_slug=qp_slug)

    by_id = {str((row.get("tournament") or {}).get("id")): row for row in choices}
    selected_choice = by_id.get(selected_id) or choices[0]
    selected_id = str((selected_choice.get("tournament") or {}).get("id") or "")

    show_selector = len(choices) > 1
    if show_selector:
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
            .select("id,name,display_name,email,phone,whatsapp,dupr_id,rating,doubles_skill,singles_skill,gender,age,inactive_at,active")
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


def _player_current_overall_jupr(player: dict[str, Any]) -> float | None:
    overall_rating_elo = _coerce_float(player.get("rating"))
    if overall_rating_elo is not None:
        return overall_rating_elo / 400.0
    doubles = _coerce_float(player.get("doubles_skill"))
    if doubles is not None:
        return doubles
    singles = _coerce_float(player.get("singles_skill"))
    if singles is not None:
        return singles
    return None


def _player_rating_text(player: dict[str, Any]) -> str:
    rating = _player_current_overall_jupr(player)
    if rating is None:
        return "N/A"
    return f"{rating:.3f}"


def _player_label(player: dict[str, Any]) -> str:
    display_name = _safe_text(player.get("display_name") or player.get("name") or f"Player #{player.get('id')}")
    rating_text = _player_rating_text(player)
    if rating_text == "N/A":
        return display_name
    return f"{display_name} · Rating {rating_text}"


@st.cache_data(ttl=90)
def _load_profile_confirmation_data(_supabase, club_id: str, player_id: str) -> dict[str, Any]:
    pid = int(player_id)
    total_matches = 0
    recent_matches: list[dict[str, Any]] = []
    recent_leagues: list[str] = []
    try:
        count_resp = (
            _supabase.table("matches")
            .select("id", count="exact")
            .eq("club_id", str(club_id))
            .or_(f"t1_p1.eq.{pid},t1_p2.eq.{pid},t2_p1.eq.{pid},t2_p2.eq.{pid}")
            .limit(1)
            .execute()
        )
        total_matches = int(getattr(count_resp, "count", 0) or 0)
    except Exception:
        total_matches = 0
    try:
        match_resp = (
            _supabase.table("matches")
            .select("id,date,league,score_t1,score_t2,t1_p1,t1_p2,t2_p1,t2_p2")
            .eq("club_id", str(club_id))
            .or_(f"t1_p1.eq.{pid},t1_p2.eq.{pid},t2_p1.eq.{pid},t2_p2.eq.{pid}")
            .order("date", desc=True)
            .order("id", desc=True)
            .limit(8)
            .execute()
        )
        rows = [dict(row) for row in (match_resp.data or [])]
    except Exception:
        rows = []
    for row in rows:
        team1 = {int(row.get("t1_p1") or 0), int(row.get("t1_p2") or 0)}
        team2 = {int(row.get("t2_p1") or 0), int(row.get("t2_p2") or 0)}
        score_t1 = _coerce_int(row.get("score_t1"))
        score_t2 = _coerce_int(row.get("score_t2"))
        result = "—"
        if score_t1 is not None and score_t2 is not None:
            on_team1 = pid in team1
            won = (on_team1 and score_t1 > score_t2) or ((not on_team1) and score_t2 > score_t1)
            result = "W" if won else "L"
        recent_matches.append(
            {
                "date": _safe_text(row.get("date")),
                "league": _safe_text(row.get("league")),
                "score": f"{_safe_text(row.get('score_t1'))}-{_safe_text(row.get('score_t2'))}",
                "result": result,
            }
        )
        league_name = _safe_text(row.get("league"))
        if league_name and league_name not in recent_leagues:
            recent_leagues.append(league_name)
    return {
        "total_matches": total_matches,
        "recent_matches": recent_matches[:5],
        "recent_leagues": recent_leagues[:4],
    }


@st.cache_data(ttl=90)
def _load_profile_confirmation_data(_supabase, club_id: str, player_id: str) -> dict[str, Any]:
    pid = int(player_id)
    total_matches = 0
    recent_matches: list[dict[str, Any]] = []
    recent_leagues: list[str] = []
    try:
        count_resp = (
            _supabase.table("matches")
            .select("id", count="exact")
            .eq("club_id", str(club_id))
            .or_(f"t1_p1.eq.{pid},t1_p2.eq.{pid},t2_p1.eq.{pid},t2_p2.eq.{pid}")
            .limit(1)
            .execute()
        )
        total_matches = int(getattr(count_resp, "count", 0) or 0)
    except Exception:
        total_matches = 0
    try:
        match_resp = (
            _supabase.table("matches")
            .select("id,date,league,score_t1,score_t2,t1_p1,t1_p2,t2_p1,t2_p2")
            .eq("club_id", str(club_id))
            .or_(f"t1_p1.eq.{pid},t1_p2.eq.{pid},t2_p1.eq.{pid},t2_p2.eq.{pid}")
            .order("date", desc=True)
            .order("id", desc=True)
            .limit(8)
            .execute()
        )
        rows = [dict(row) for row in (match_resp.data or [])]
    except Exception:
        rows = []
    for row in rows:
        team1 = {int(row.get("t1_p1") or 0), int(row.get("t1_p2") or 0)}
        team2 = {int(row.get("t2_p1") or 0), int(row.get("t2_p2") or 0)}
        score_t1 = _coerce_int(row.get("score_t1"))
        score_t2 = _coerce_int(row.get("score_t2"))
        result = "—"
        if score_t1 is not None and score_t2 is not None:
            on_team1 = pid in team1
            won = (on_team1 and score_t1 > score_t2) or ((not on_team1) and score_t2 > score_t1)
            result = "W" if won else "L"
        recent_matches.append(
            {
                "date": _safe_text(row.get("date")),
                "league": _safe_text(row.get("league")),
                "score": f"{_safe_text(row.get('score_t1'))}-{_safe_text(row.get('score_t2'))}",
                "result": result,
            }
        )
        league_name = _safe_text(row.get("league"))
        if league_name and league_name not in recent_leagues:
            recent_leagues.append(league_name)
    return {
        "total_matches": total_matches,
        "recent_matches": recent_matches[:5],
        "recent_leagues": recent_leagues[:4],
    }


def _family_key(day_id: str, family: str) -> str:
    return f"{day_id}::{family}"


def _normalize_name_for_match(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _normalize_email(value: Any) -> str:
    return str(value or "").strip().lower()


def _player_full_name(player: dict[str, Any]) -> str:
    first = _safe_text(player.get("first_name"))
    last = _safe_text(player.get("last_name"))
    if first or last:
        return " ".join(part for part in [first, last] if part)
    return _safe_text(player.get("display_name") or player.get("name"))


def _likely_active_player_matches(
    players: list[dict[str, Any]], *, first_name: str, last_name: str, email: str
) -> tuple[list[dict[str, Any]], str]:
    normalized_email = _normalize_email(email)
    first = _normalize_name_for_match(first_name)
    last = _normalize_name_for_match(last_name)
    target_full = _normalize_name_for_match(f"{first} {last}") if first and last else ""

    email_exact: list[dict[str, Any]] = []
    exact_name: list[dict[str, Any]] = []
    contains_name: list[dict[str, Any]] = []

    for row in players:
        if normalized_email and _normalize_email(row.get("email")) == normalized_email:
            email_exact.append(row)
            continue
        full_name = _normalize_name_for_match(_player_full_name(row))
        if not full_name or not target_full:
            continue
        if full_name == target_full:
            exact_name.append(row)
            continue
        tokens = full_name.split()
        if first in tokens and last in tokens:
            contains_name.append(row)

    if email_exact:
        match_type = "email_exact"
        matches = email_exact
    elif exact_name:
        match_type = "full_name_exact"
        matches = exact_name
    else:
        match_type = "name_likely"
        matches = contains_name

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in matches:
        pid = str(row.get("id") or "")
        key = pid or _normalize_name_for_match(_player_full_name(row))
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped[:8], match_type


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
    if _coerce_float(player.get("doubles_skill")) is None and _coerce_float(player.get("singles_skill")) is None:
        return True
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


def _wizard_key(tournament_id: Any) -> str:
    return f"registration_wizard_state_{tournament_id}"


def _init_wizard_state(tournament_id: Any) -> dict[str, Any]:
    key = _wizard_key(tournament_id)
    if key not in st.session_state:
        st.session_state[key] = {
            "current_step": 1,
            "step1": {},
            "step2": {
                "profile_mode": "new",
                "selected_player_id": "",
                "candidate_player_id": "",
                "candidate_confirmed": False,
                "rejected_likely": False,
                "search_query": "",
                "selection_source": "",
            },
            "step3": {"selected_event_ids": []},
            "step4": {"partner_details": {}},
        }
    return st.session_state[key]


def _status_badge(value: Any) -> str:
    text = _safe_text(value).lower()
    return text.replace("_", " ").title() if text else "—"


def _select_admin_tournament(ctx, supabase, *, page_key: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    club_id = _safe_text(getattr(ctx, "club_id", ""))
    tournaments = list_existing_tournaments(supabase, club_id, include_archived=False)
    if not tournaments:
        st.info("No tournaments available.")
        return None, None, [], []

    qp_tournament_id = _safe_text(st.query_params.get("tournament_id"))
    qp_slug = _safe_text(st.query_params.get("tournament"))
    selected_index = 0
    if qp_tournament_id:
        for idx, row in enumerate(tournaments):
            if str(row.get("id")) == qp_tournament_id:
                selected_index = idx
                break
    elif qp_slug:
        for idx, row in enumerate(tournaments):
            row_settings = get_registration_settings(supabase, str(row.get("id")), tournament_name=_safe_text(row.get("name")))
            if _safe_text(row_settings.get("registration_slug")) == qp_slug:
                selected_index = idx
                break

    labels = [f"{_safe_text(row.get('name'))} ({_safe_text(row.get('status') or 'DRAFT')})" for row in tournaments]
    selected_label = st.selectbox("Choose a tournament", labels, index=selected_index)
    tournament = tournaments[labels.index(selected_label)]
    tournament_id = str(tournament.get("id"))

    settings = get_registration_settings(supabase, tournament_id, tournament_name=_safe_text(tournament.get("name")))
    days = list_registration_days(supabase, tournament_id)
    event_options = list_registration_event_options(supabase, tournament_id)

    if (
        _safe_text(st.query_params.get("page")) != page_key
        or _safe_text(st.query_params.get("tournament_id")) != tournament_id
        or _safe_text(st.query_params.get("tournament")) != _safe_text(settings.get("registration_slug"))
    ):
        st.query_params["page"] = page_key
        st.query_params["tournament_id"] = tournament_id
        slug = _safe_text(settings.get("registration_slug"))
        if slug:
            st.query_params["tournament"] = slug
        st.rerun()

    return tournament, settings, days, event_options


def _render_registration_admin_roster(*, supabase, tournament: dict[str, Any], days: list[dict[str, Any]], event_options: list[dict[str, Any]]) -> None:
    tournament_id = str(tournament.get("id"))
    admin_rows = list_registration_admin_rows(supabase, tournament_id)
    day_lookup = {str(row.get("id")): row for row in days}
    event_lookup = {str(row.get("id")): row for row in event_options}

    registration_forms = len({str(row.get("registration_id")) for row in admin_rows if _safe_text(row.get("registration_id"))})
    pending = [row for row in admin_rows if _safe_text((row.get("registration") or {}).get("status")).lower() == "pending"]
    confirmed = [row for row in admin_rows if _safe_text((row.get("registration") or {}).get("status")).lower() == "confirmed"]
    needs_partner = [row for row in admin_rows if _safe_text((row.get("selection") or {}).get("partner_mode")).upper() == "NEEDS_PARTNER"]
    paid = [row for row in admin_rows if _safe_text((row.get("registration") or {}).get("payment_status")).lower() == "paid"]
    unpaid = [row for row in admin_rows if _safe_text((row.get("registration") or {}).get("payment_status")).lower() == "unpaid"]

    metrics = st.columns(7)
    for idx, (label, value) in enumerate([
        ("Registration Forms", registration_forms),
        ("Event Entries", len(admin_rows)),
        ("Confirmed Entries", len(confirmed)),
        ("Pending Entries", len(pending)),
        ("Needs Partner", len(needs_partner)),
        ("Paid", len(paid)),
        ("Unpaid", len(unpaid)),
    ]):
        metrics[idx].metric(label, value)

    filters = st.columns(5)
    status_filter = filters[0].selectbox("Status", ["All"] + ADMIN_REGISTRATION_STATUS_OPTIONS, key=f"reg_status_filter_{tournament_id}")
    payment_filter = filters[1].selectbox("Payment", ["All"] + ADMIN_PAYMENT_STATUS_OPTIONS, key=f"reg_payment_filter_{tournament_id}")
    partner_filter = filters[2].selectbox("Partner", ["All", "HAS_PARTNER", "NEEDS_PARTNER", "NONE"], key=f"reg_partner_filter_{tournament_id}")
    day_filter = filters[3].selectbox("Day", ["All"] + [str(day.get("id")) for day in days], format_func=lambda did: "All" if did == "All" else _safe_text((day_lookup.get(did) or {}).get("label") or did), key=f"reg_day_filter_{tournament_id}")
    search = filters[4].text_input("Search", key=f"reg_search_{tournament_id}")

    filtered_rows: list[dict[str, Any]] = []
    for row in admin_rows:
        reg = row.get("registration") or {}
        sel = row.get("selection") or {}
        if status_filter != "All" and _safe_text(reg.get("status")).lower() != status_filter:
            continue
        if payment_filter != "All" and _safe_text(reg.get("payment_status")).lower() != payment_filter:
            continue
        if partner_filter != "All" and _safe_text(sel.get("partner_mode")).upper() != partner_filter:
            continue
        if day_filter != "All" and _safe_text(sel.get("registration_day_id")) != day_filter:
            continue
        search_blob = " ".join([_safe_text(reg.get("display_name")), _safe_text(reg.get("email")), _safe_text(sel.get("partner_name"))]).lower()
        if search and search.lower() not in search_blob:
            continue
        filtered_rows.append(row)

    if not filtered_rows:
        st.info("No matching registration entries.")

    for row in filtered_rows:
        reg = row.get("registration") or {}
        sel = row.get("selection") or {}
        event = row.get("event") or {}
        day = row.get("day") or {}
        reg_id = _safe_text(row.get("registration_id"))
        sel_id = _safe_text(row.get("selection_id"))
        imported = registration_is_imported_to_draw(supabase, tournament_id=tournament_id, selection_id=sel_id or None, registration_id=reg_id)

        label = f"{_safe_text(reg.get('display_name') or reg.get('email'))} • {_safe_text(day.get('label'))} • {_safe_text(event.get('division_name') or event.get('label'))}"
        with st.expander(label, expanded=False):
            st.caption(f"Registrant: {_safe_text(reg.get('first_name'))} {_safe_text(reg.get('last_name'))} · {_safe_text(reg.get('email'))} · {_safe_text(reg.get('phone'))}")
            st.caption(f"Partner: {_safe_text(sel.get('partner_name')) or '—'} · {_safe_text(sel.get('partner_email')) or '—'}")
            st.caption(f"Status: {_status_badge(reg.get('status'))} · Payment: {_status_badge(reg.get('payment_status'))} · Partner mode: {_status_badge(sel.get('partner_mode'))}")
            st.caption(f"Notes: {_safe_text(sel.get('partner_note')) or _safe_text(reg.get('notes')) or '—'}")
            st.caption(f"Created: {_safe_text(reg.get('created_at')) or '—'} · Linked JUPR player: {_safe_text(reg.get('player_id')) or '—'}")
            if imported:
                st.warning("Imported into tournament_teams. Event edits and hard delete are blocked until removed from teams.")

            quick_actions = st.columns(3)
            if quick_actions[0].button("Quick Confirm", key=f"confirm_{sel_id}_{reg_id}"):
                update_admin_registration(supabase, tournament_id=tournament_id, registration_id=reg_id, payload={"status": "confirmed"})
                st.rerun()
            if quick_actions[1].button("Move to Waitlist", key=f"waitlist_{sel_id}_{reg_id}"):
                update_admin_registration(supabase, tournament_id=tournament_id, registration_id=reg_id, payload={"status": "waitlist"})
                st.rerun()
            if quick_actions[2].button("Cancel", key=f"cancel_{sel_id}_{reg_id}"):
                cancel_registration(supabase, tournament_id=tournament_id, registration_id=reg_id)
                st.rerun()

            with st.form(f"edit_reg_{sel_id}_{reg_id}"):
                c1, c2 = st.columns(2)
                first_name = c1.text_input("First name", value=_safe_text(reg.get("first_name")))
                last_name = c2.text_input("Last name", value=_safe_text(reg.get("last_name")))
                display_name = st.text_input("Display name", value=_safe_text(reg.get("display_name")))
                email = st.text_input("Email", value=_safe_text(reg.get("email")))
                phone = st.text_input("Phone", value=_safe_text(reg.get("phone")))
                reg_status = st.selectbox("Admin status", ADMIN_REGISTRATION_STATUS_OPTIONS, index=max(0, ADMIN_REGISTRATION_STATUS_OPTIONS.index(_safe_text(reg.get("status")).lower()) if _safe_text(reg.get("status")).lower() in ADMIN_REGISTRATION_STATUS_OPTIONS else 0))
                reg_payment = st.selectbox("Payment status", ADMIN_PAYMENT_STATUS_OPTIONS, index=max(0, ADMIN_PAYMENT_STATUS_OPTIONS.index(_safe_text(reg.get("payment_status")).lower()) if _safe_text(reg.get("payment_status")).lower() in ADMIN_PAYMENT_STATUS_OPTIONS else 0))
                day_ids = [str(day.get("id")) for day in days]
                event_ids = [str(event_opt.get("id")) for event_opt in event_options]
                day_id = st.selectbox("Day", day_ids, index=max(0, day_ids.index(_safe_text(sel.get("registration_day_id"))) if _safe_text(sel.get("registration_day_id")) in day_ids else 0), format_func=lambda did: _safe_text((day_lookup.get(did) or {}).get("label") or did))
                event_id = st.selectbox("Division", event_ids, index=max(0, event_ids.index(_safe_text(sel.get("event_option_id"))) if _safe_text(sel.get("event_option_id")) in event_ids else 0), format_func=lambda eid: f"{_safe_text((event_lookup.get(eid) or {}).get('event_family_label'))} / {_safe_text((event_lookup.get(eid) or {}).get('division_name') or (event_lookup.get(eid) or {}).get('label'))}")
                partner_mode = st.selectbox("Partner mode", PARTNER_MODE_OPTIONS, index=max(0, PARTNER_MODE_OPTIONS.index(_safe_text(sel.get("partner_mode")).upper()) if _safe_text(sel.get("partner_mode")).upper() in PARTNER_MODE_OPTIONS else 0))
                partner_name = st.text_input("Partner name/details", value=_safe_text(sel.get("partner_name")))
                partner_email = st.text_input("Partner email", value=_safe_text(sel.get("partner_email")))
                partner_note = st.text_area("Public partner note", value=_safe_text(sel.get("partner_note")))
                notes = st.text_area("Internal/admin notes", value=_safe_text(reg.get("notes")))
                save = st.form_submit_button("Save Changes", use_container_width=True)
            if save:
                update_admin_registration(supabase, tournament_id=tournament_id, registration_id=reg_id, payload={
                    "first_name": first_name,
                    "last_name": last_name,
                    "display_name": display_name,
                    "email": email,
                    "phone": phone,
                    "status": reg_status,
                    "payment_status": reg_payment,
                    "notes": notes,
                })
                if sel_id and not imported:
                    update_admin_registration_selection(supabase, tournament_id=tournament_id, selection_id=sel_id, payload={
                        "registration_day_id": day_id,
                        "event_option_id": event_id,
                        "partner_mode": partner_mode,
                        "partner_name": partner_name,
                        "partner_email": partner_email,
                        "partner_note": partner_note,
                        "show_on_partner_board": bool(_safe_text(sel.get("show_on_partner_board")).lower() in {"true", "1", "yes"}),
                    })
                st.rerun()

            with st.expander("Hard delete", expanded=False):
                st.caption("Prefer Cancel for normal workflow. Hard delete is permanent.")
                confirm = st.text_input("Type DELETE to hard delete", key=f"delete_confirm_{sel_id}_{reg_id}")
                if st.button("Delete registration permanently", key=f"delete_btn_{sel_id}_{reg_id}"):
                    if imported:
                        st.error("Registration already imported into tournament_teams. Remove team/draw entries first.")
                    elif confirm != "DELETE":
                        st.error("Type DELETE exactly.")
                    else:
                        delete_registration(supabase, tournament_id=tournament_id, registration_id=reg_id)
                        st.rerun()

    st.markdown("#### Players Looking for Partners")
    partner_rows = [row for row in filtered_rows if _safe_text((row.get("selection") or {}).get("partner_mode")).upper() == "NEEDS_PARTNER"]
    if not partner_rows:
        st.info("No players currently marked as NEEDS_PARTNER.")
    else:
        for row in partner_rows:
            reg = row.get("registration") or {}
            sel = row.get("selection") or {}
            st.markdown(f"- **{_safe_text(reg.get('display_name'))}** ({_safe_text(reg.get('email'))}) — {_safe_text(sel.get('partner_note')) or 'No note'}")


def render(ctx):
    mode_label = "Public" if bool(getattr(ctx, "public_mode", False)) else "Admin"
    page_shell(
        "📝 Tournament Registration",
        "Manage registration forms, player entries, approvals, partner needs, and public registration links.",
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

    admin_mode = bool(getattr(ctx, "admin_logged_in", False)) and not bool(getattr(ctx, "public_mode", False))
    qp_tournament_id = _safe_text(st.query_params.get("tournament_id"))
    qp_slug = _safe_text(st.query_params.get("tournament"))
    tournament, settings, days, event_options = (
        _select_admin_tournament(ctx, supabase, page_key="tournament_registration")
        if admin_mode
        else _select_public_tournament(ctx, supabase, page_key="tournament_registration")
    )
    if not tournament:
        st.stop()
    if qp_tournament_id and str(tournament.get("id")) != qp_tournament_id:
        st.warning("The requested tournament_id is unavailable. Showing the selected open tournament instead.")
    elif qp_slug and _safe_text(settings.get("registration_slug")) != qp_slug:
        st.warning("The requested tournament link is unavailable. Showing the selected open tournament instead.")

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
        st.link_button("View Tournament Roster", public_urls["roster"])

    if admin_mode:
        roster_tab, add_tab, links_tab = st.tabs(["Registration Roster", "Add Registration", "Public Form Preview / Links"])
        with roster_tab:
            _render_registration_admin_roster(
                supabase=supabase,
                tournament=tournament,
                days=days,
                event_options=event_options,
            )
        with add_tab:
            if not days or not event_options:
                st.warning("Configure registration days and event divisions in Tournament Setup first.")
            else:
                day_lookup = {str(row.get("id")): row for row in days}
                event_lookup = {str(row.get("id")): row for row in event_options}
                with st.form(f"admin_add_registration_{tournament.get('id')}"):
                    c1, c2 = st.columns(2)
                    first_name = c1.text_input("First name")
                    last_name = c2.text_input("Last name")
                    display_name = st.text_input("Display name")
                    email = st.text_input("Email")
                    phone = st.text_input("Phone")
                    status = st.selectbox("Admin status", ADMIN_REGISTRATION_STATUS_OPTIONS)
                    payment_status = st.selectbox("Payment status", ADMIN_PAYMENT_STATUS_OPTIONS)
                    day_id = st.selectbox("Day", [str(d.get("id")) for d in days], format_func=lambda did: _safe_text((day_lookup.get(did) or {}).get("label") or did))
                    event_id = st.selectbox("Division", [str(e.get("id")) for e in event_options], format_func=lambda eid: f"{_safe_text((event_lookup.get(eid) or {}).get('event_family_label'))} / {_safe_text((event_lookup.get(eid) or {}).get('division_name') or (event_lookup.get(eid) or {}).get('label'))}")
                    partner_mode = st.selectbox("Partner mode", PARTNER_MODE_OPTIONS)
                    partner_name = st.text_input("Partner name")
                    partner_email = st.text_input("Partner email")
                    notes = st.text_area("Notes")
                    save_add = st.form_submit_button("Save registration", use_container_width=True)
                if save_add:
                    create_admin_registration(supabase, tournament_id=str(tournament.get("id")), payload={
                        "first_name": first_name,
                        "last_name": last_name,
                        "display_name": display_name or " ".join([first_name, last_name]).strip(),
                        "email": email,
                        "phone": phone,
                        "status": status,
                        "payment_status": payment_status,
                        "notes": notes,
                        "selections": [{"registration_day_id": day_id, "event_option_id": event_id, "partner_mode": partner_mode, "partner_name": partner_name, "partner_email": partner_email}],
                    })
                    st.success("Registration created.")
                    st.rerun()
        with links_tab:
            st.code(public_urls["registration"])
            st.code(public_urls["roster"])
            st.link_button("Open Public Registration Form", public_urls["registration"])
            st.link_button("Open Public Roster", public_urls["roster"])
        return

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

    wizard = _init_wizard_state(tournament.get("id"))
    current_step = int(wizard.get("current_step") or 1)
    step1 = wizard.get("step1") or {}
    step2 = wizard.get("step2") or {}
    step3 = wizard.get("step3") or {}
    step4 = wizard.get("step4") or {}
    active_players = _load_active_players(supabase, club_id=club_id, ctx=ctx)

    st.caption(f"Step {current_step} of 4")

    if current_step == 1:
        st.markdown("### 1. Name and contact")
        c1, c2 = st.columns(2)
        with c1:
            first_name = st.text_input("First name *", value=_safe_text(step1.get("first_name")))
            email = st.text_input("Email *", value=_safe_text(step1.get("email")))
            gender = st.selectbox(
                "Gender *",
                ["", "Female", "Male", "Other", "Prefer not to say"],
                index=max(0, ["", "Female", "Male", "Other", "Prefer not to say"].index(_safe_text(step1.get("gender"))))
                if _safe_text(step1.get("gender")) in ["", "Female", "Male", "Other", "Prefer not to say"]
                else 0,
            )
        with c2:
            last_name = st.text_input("Last name *", value=_safe_text(step1.get("last_name")))
            phone = st.text_input("Phone / WhatsApp", value=_safe_text(step1.get("phone")))
            age = st.text_input("Age *", value=_safe_text(step1.get("age")))
        notes = st.text_area("Notes for tournament staff", value=_safe_text(step1.get("notes")), height=90)
        _, next_col = st.columns([4, 1])
        with next_col:
            if st.button("Next ➜", type="primary"):
                if not _safe_text(first_name) or not _safe_text(last_name):
                    st.error("First name and last name are required.")
                    st.stop()
                if not _safe_text(email):
                    st.error("Email is required.")
                    st.stop()
                if not _safe_text(age):
                    st.error("Age is required.")
                    st.stop()
                if not _safe_text(gender):
                    st.error("Gender is required.")
                    st.stop()
                wizard["step1"] = {
                    "first_name": first_name,
                    "last_name": last_name,
                    "email": email,
                    "phone": phone,
                    "gender": gender,
                    "age": age,
                    "notes": notes,
                }
                wizard["current_step"] = 2
                st.rerun()

    step1 = wizard.get("step1") or {}
    likely_matches, _match_type = _likely_active_player_matches(
        active_players,
        first_name=_safe_text(step1.get("first_name")),
        last_name=_safe_text(step1.get("last_name")),
        email=_safe_text(step1.get("email")),
    )

    if current_step == 2:
        st.markdown("### 2. Player profile")
        st.caption("If you already have a JUPR profile, we’ll use it for rating and history. If not, no problem — you can still register.")
        step2_state = dict(step2)
        selected_player_id = _safe_text(step2_state.get("selected_player_id"))
        selected_existing_player = next((row for row in active_players if str(row.get("id")) == selected_player_id), None)
        candidate_player_id = _safe_text(step2_state.get("candidate_player_id"))
        candidate_confirmed = bool(step2_state.get("candidate_confirmed"))
        rejected_likely = bool(step2_state.get("rejected_likely"))
        selection_source = _safe_text(step2_state.get("selection_source") or "")
        search_query_default = _safe_text(step2_state.get("search_query"))
        profile_mode = _safe_text(step2_state.get("profile_mode") or "")
        default_display_name = " ".join(
            part for part in [_safe_text(step1.get("first_name")), _safe_text(step1.get("last_name"))] if part
        )
        saved_display_name = _safe_text(step2_state.get("display_name")) or default_display_name

        if not profile_mode:
            profile_mode = "existing" if likely_matches else "none"
        if not selection_source:
            selection_source = "likely" if likely_matches else "none"

        if selected_existing_player and candidate_confirmed:
            st.success(f"Confirmed profile: {_player_label(selected_existing_player)}")
        else:
            if len(likely_matches) == 1 and not rejected_likely and not candidate_player_id:
                candidate_player_id = str(likely_matches[0].get("id"))
                selection_source = "likely"
            if len(likely_matches) > 1 and not candidate_player_id and not rejected_likely:
                st.caption("We found a few possible JUPR profiles. Pick one to review.")
                likely_options = {f"{_player_label(row)}": str(row.get("id")) for row in likely_matches}
                picked_likely = st.radio(
                    "Likely profiles",
                    list(likely_options.keys()),
                    key=f"wizard_likely_profile_pick_{tournament.get('id')}",
                )
                candidate_player_id = likely_options[picked_likely]
                selection_source = "likely"
            candidate_player = next((row for row in active_players if str(row.get("id")) == candidate_player_id), None)
            if candidate_player:
                profile_mode = "existing"
                st.info("We found a possible JUPR profile.")
                st.markdown("#### Suggested profile")
                st.caption(_player_label(candidate_player))
                summary = _load_profile_confirmation_data(supabase, club_id=club_id, player_id=str(candidate_player.get("id")))
                info_cols = st.columns(3)
                with info_cols[0]:
                    st.metric("Current rating", _player_rating_text(candidate_player))
                with info_cols[1]:
                    st.metric("Total matches", int(summary.get("total_matches") or 0))
                with info_cols[2]:
                    recent_leagues = summary.get("recent_leagues") or []
                    st.metric("Recent events", len(recent_leagues))
                if recent_leagues:
                    st.caption("Recent leagues/events: " + " • ".join(recent_leagues))
                recent_matches = summary.get("recent_matches") or []
                if recent_matches:
                    st.markdown("**Recent results**")
                    for row in recent_matches:
                        league_name = _safe_text(row.get("league") or "Club match")
                        st.caption(f"{_safe_text(row.get('date'))} · {league_name} · {row.get('result')} · {row.get('score')}")
                choice_cols = st.columns(2)
                with choice_cols[0]:
                    if st.button("Yes, this is me", key=f"wizard_confirm_profile_{tournament.get('id')}"):
                        selected_existing_player = candidate_player
                        selected_player_id = str(candidate_player.get("id"))
                        candidate_confirmed = True
                        rejected_likely = False
                        profile_mode = "existing"
                        wizard["step2"] = {
                            **step2_state,
                            "profile_mode": "existing",
                            "selected_player_id": selected_player_id,
                            "candidate_player_id": selected_player_id,
                            "candidate_confirmed": True,
                            "rejected_likely": False,
                            "selection_source": selection_source,
                            "search_query": search_query_default,
                        }
                        st.rerun()
                with choice_cols[1]:
                    if st.button("No, continue without this profile", key=f"wizard_reject_profile_{tournament.get('id')}"):
                        selected_existing_player = None
                        selected_player_id = ""
                        candidate_confirmed = False
                        candidate_player_id = ""
                        rejected_likely = True
                        profile_mode = "new"
                        selection_source = "create"
                        wizard["step2"] = {
                            **step2_state,
                            "profile_mode": "new",
                            "selected_player_id": "",
                            "candidate_player_id": "",
                            "candidate_confirmed": False,
                            "rejected_likely": True,
                            "selection_source": "create",
                            "search_query": search_query_default,
                            "display_name": saved_display_name,
                        }
                        st.rerun()
            if not likely_matches and not candidate_confirmed and selection_source != "search" and profile_mode != "new":
                profile_mode = "none"
                selection_source = "none"

            if (rejected_likely or not likely_matches) and not candidate_confirmed and selection_source != "search":
                st.info(
                    "We didn’t find an existing JUPR profile for this name/email.\n\n"
                    "You can continue without one. Tournament staff can review or connect your profile later."
                )
                option_cols = st.columns(2)
                with option_cols[0]:
                    if st.button("Search for my JUPR profile", key=f"wizard_search_mode_{tournament.get('id')}"):
                        profile_mode = "existing"
                        selection_source = "search"
                        wizard["step2"] = {
                            **step2_state,
                            "profile_mode": "existing",
                            "selection_source": "search",
                            "candidate_confirmed": False,
                            "selected_player_id": "",
                            "candidate_player_id": "",
                            "search_query": search_query_default,
                        }
                        st.rerun()
                with option_cols[1]:
                    if st.button("Continue without a JUPR profile", key=f"wizard_create_mode_{tournament.get('id')}"):
                        profile_mode = "new"
                        selection_source = "create"
                        rejected_likely = True
                        candidate_confirmed = False
                        candidate_player_id = ""
                        selected_player_id = ""
                        wizard["step2"] = {
                            **step2_state,
                            "profile_mode": "new",
                            "selection_source": "create",
                            "candidate_confirmed": False,
                            "candidate_player_id": "",
                            "selected_player_id": "",
                            "rejected_likely": True,
                            "display_name": saved_display_name,
                            "search_query": search_query_default,
                        }
                        st.rerun()
            if profile_mode == "existing" and not candidate_confirmed and selection_source == "search":
                st.markdown("#### Search active players")
                search_query = st.text_input(
                    "Search by name",
                    value=search_query_default,
                    key=f"wizard_profile_search_{tournament.get('id')}",
                    placeholder="Type at least 2 characters",
                )
                search_query_default = _safe_text(search_query)
                normalized_query = _normalize_name_for_match(search_query)
                search_results: list[dict[str, Any]] = []
                if len(normalized_query) >= 2:
                    for row in active_players:
                        full_name = _normalize_name_for_match(_player_full_name(row))
                        if normalized_query in full_name:
                            search_results.append(row)
                    search_results = search_results[:10]
                elif search_query:
                    st.caption("Keep typing to search.")
                if search_results:
                    result_options = {f"{_player_label(row)}": str(row.get("id")) for row in search_results}
                    picked_search = st.radio(
                        "Search results",
                        list(result_options.keys()),
                        key=f"wizard_search_pick_{tournament.get('id')}",
                    )
                    candidate_player_id = result_options[picked_search]
                elif len(normalized_query) >= 2:
                    st.info("No matching active profiles found.")
                if st.button("I don’t have a JUPR profile — continue without one", key=f"wizard_search_create_mode_{tournament.get('id')}"):
                    profile_mode = "new"
                    selection_source = "create"
                    rejected_likely = True
                    candidate_player_id = ""
                    selected_player_id = ""
                    candidate_confirmed = False
                    wizard["step2"] = {
                        **step2_state,
                        "profile_mode": "new",
                        "selected_player_id": "",
                        "candidate_player_id": "",
                        "candidate_confirmed": False,
                        "rejected_likely": True,
                        "selection_source": "create",
                        "search_query": search_query_default,
                        "display_name": saved_display_name,
                    }
                    st.rerun()

        c1, c2, c3 = st.columns([1, 1, 3])
        with c1:
            if st.button("← Back"):
                wizard["current_step"] = 1
                st.rerun()
        with c2:
            if st.button("Next ➜", type="primary"):
                next_step2: dict[str, Any] = {
                    "profile_mode": profile_mode,
                    "selected_player_id": "",
                    "candidate_player_id": candidate_player_id,
                    "candidate_confirmed": candidate_confirmed,
                    "rejected_likely": rejected_likely,
                    "search_query": search_query_default,
                    "selection_source": selection_source,
                }
                if profile_mode == "existing":
                    if not selected_existing_player or not candidate_confirmed:
                        st.error("Please confirm your JUPR profile before continuing, or choose “Continue without a JUPR profile.”")
                        st.stop()
                    next_step2["selected_player_id"] = str(selected_existing_player.get("id"))
                else:
                    new_display_name = st.session_state.get("wizard_new_display_name", saved_display_name)
                    if not _safe_text(new_display_name):
                        st.error("Display name is required.")
                        st.stop()
                    next_step2.update(
                        {
                            "display_name": _safe_text(new_display_name),
                            "dupr_id": _safe_text(st.session_state.get("wizard_new_dupr_id", step2.get("dupr_id"))),
                            "doubles_skill": _safe_text(st.session_state.get("wizard_new_doubles_skill", step2.get("doubles_skill"))),
                            "singles_skill": _safe_text(st.session_state.get("wizard_new_singles_skill", step2.get("singles_skill"))),
                        }
                    )
                wizard["step2"] = next_step2
                wizard["current_step"] = 3
                st.rerun()

        if profile_mode == "new" and selection_source == "create":
            st.markdown("#### Registration profile")
            st.caption("These details help us place you in the right division. Leave ratings blank if you’re unsure.")
            st.text_input("Display name *", value=saved_display_name, key="wizard_new_display_name")
            prefill_email = _safe_text(step1.get("email"))
            prefill_phone = _safe_text(step1.get("phone"))
            st.caption(f"Email: {prefill_email or '—'}")
            st.caption(f"Phone: {prefill_phone or '—'}")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.text_input("DUPR ID (optional)", value=_safe_text(step2.get("dupr_id")), key="wizard_new_dupr_id")
            with c2:
                st.text_input("Doubles skill (optional)", value=_safe_text(step2.get("doubles_skill")), key="wizard_new_doubles_skill")
            with c3:
                st.text_input("Singles skill (optional)", value=_safe_text(step2.get("singles_skill")), key="wizard_new_singles_skill")

    step2 = wizard.get("step2") or {}
    using_existing_player = _safe_text(step2.get("profile_mode")) == "existing"
    selected_existing_player = None
    if using_existing_player:
        selected_existing_player = next(
            (row for row in active_players if str(row.get("id")) == _safe_text(step2.get("selected_player_id"))),
            None,
        )
    if using_existing_player and selected_existing_player:
        canonical_overall_rating = _player_current_overall_jupr(selected_existing_player)
        if canonical_overall_rating is not None:
            profile_doubles = canonical_overall_rating
            profile_singles = canonical_overall_rating
        else:
            profile_doubles = _coerce_float(selected_existing_player.get("doubles_skill"))
            profile_singles = _coerce_float(selected_existing_player.get("singles_skill"))
    else:
        profile_doubles = _coerce_float(step2.get("doubles_skill"))
        profile_singles = _coerce_float(step2.get("singles_skill"))
    player_profile = {"doubles_skill": profile_doubles, "singles_skill": profile_singles}

    visible_event_options = _visible_division_options(
        selectable_event_options,
        gender=_safe_text(step1.get("gender")),
        player=player_profile,
    )
    visible_grouped_events = _group_events(days, visible_event_options)

    if current_step == 3:
        st.markdown("### 3. Select events")
        st.caption("Choose up to one division per Day + Event Family.")
        selected_ids: list[str] = step3.get("selected_event_ids") or []
        family_selection_counts: dict[str, int] = {}
        for day in days:
            day_id = str(day.get("id"))
            family_map = visible_grouped_events.get(day_id, {})
            if not family_map:
                continue
            st.markdown(f"#### {day.get('label')}")
            for family, options in family_map.items():
                st.markdown(f"**{family}**")
                selected_in_family = 0
                for event in options:
                    event_id = str(event.get("id"))
                    eligible, reason = _preview_division_eligibility(event, player_profile)
                    checked = st.checkbox(
                        _division_choice_label(event, eligible=eligible),
                        value=event_id in selected_ids,
                        key=f"wizard_event_pick_{tournament.get('id')}_{event_id}",
                    )
                    if checked:
                        selected_in_family += 1
                    help_text = _division_help(event)
                    if help_text:
                        st.caption(help_text)
                    if checked and not eligible:
                        st.warning(reason or "Not eligible based on current rating.")
                blocked_names = blocked_by_family.get((day_id, family), [])
                if blocked_names:
                    st.caption("Closed divisions: " + ", ".join(sorted(blocked_names)))
                family_selection_counts[_family_key(day_id, family)] = selected_in_family

        c1, c2, _ = st.columns([1, 1, 3])
        with c1:
            if st.button("← Back", key="step3_back"):
                wizard["current_step"] = 2
                st.rerun()
        with c2:
            if st.button("Next ➜", type="primary", key="step3_next"):
                new_selected = [
                    str(event.get("id"))
                    for event in visible_event_options
                    if bool(st.session_state.get(f"wizard_event_pick_{tournament.get('id')}_{event.get('id')}", False))
                ]
                if not new_selected:
                    st.error("Choose at least one division before continuing.")
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
                        "Choose only one division per day/event family group. Please fix: "
                        + ", ".join(sorted(set(over_selected_groups)))
                    )
                    st.stop()
                wizard["step3"] = {"selected_event_ids": new_selected}
                wizard["current_step"] = 4
                st.rerun()

    step3 = wizard.get("step3") or {}
    selected_event_ids: list[str] = step3.get("selected_event_ids") or []
    event_lookup = {str(row.get("id")): row for row in event_options}

    if current_step == 4:
        st.markdown("### 4. Partner information")
        st.caption("Partner details are only needed for selected doubles divisions.")
        profile_summary_name = _safe_text((selected_existing_player or {}).get("display_name") or (selected_existing_player or {}).get("name"))
        if using_existing_player and profile_summary_name:
            st.info(f"JUPR profile: {profile_summary_name}")
        else:
            st.info("JUPR profile: Not connected")
        partner_details: dict[str, Any] = step4.get("partner_details") or {}
        doubles_selected = [event_lookup[eid] for eid in selected_event_ids if bool((event_lookup.get(eid) or {}).get("partner_required"))]

        if not doubles_selected:
            st.info("No doubles divisions selected. You can submit now.")
        for event in doubles_selected:
            event_id = str(event.get("id"))
            existing = partner_details.get(event_id) or {}
            st.markdown(f"**{_safe_text(event.get('division_name') or event.get('label') or event_id)}**")
            mode = st.radio(
                "Partner status",
                ["HAS_PARTNER", "NEEDS_PARTNER"],
                horizontal=True,
                format_func=lambda v: "I already have a partner" if v == "HAS_PARTNER" else "I need a partner",
                index=0 if _safe_text(existing.get("partner_mode")) == "HAS_PARTNER" else 1,
                key=f"wizard_partner_mode_{event_id}",
            )
            event_payload: dict[str, Any] = {"partner_mode": mode}
            if mode == "HAS_PARTNER":
                c1, c2 = st.columns(2)
                with c1:
                    event_payload["partner_name"] = st.text_input(
                        "Partner name", value=_safe_text(existing.get("partner_name")), key=f"wizard_partner_name_{event_id}"
                    )
                    event_payload["partner_email"] = st.text_input(
                        "Partner email", value=_safe_text(existing.get("partner_email")), key=f"wizard_partner_email_{event_id}"
                    )
                    event_payload["partner_phone"] = st.text_input(
                        "Partner phone", value=_safe_text(existing.get("partner_phone")), key=f"wizard_partner_phone_{event_id}"
                    )
                with c2:
                    event_payload["partner_dupr_id"] = st.text_input(
                        "Partner DUPR ID", value=_safe_text(existing.get("partner_dupr_id")), key=f"wizard_partner_dupr_{event_id}"
                    )
                    event_payload["partner_skill"] = _coerce_float(
                        st.text_input("Partner skill", value=_safe_text(existing.get("partner_skill")), key=f"wizard_partner_skill_{event_id}")
                    )
                    event_payload["partner_age"] = _coerce_int(
                        st.text_input("Partner age", value=_safe_text(existing.get("partner_age")), key=f"wizard_partner_age_{event_id}")
                    )
            else:
                event_payload["show_on_partner_board"] = bool(
                    st.checkbox(
                        "Show me on the public partner board for this division",
                        value=bool(existing.get("show_on_partner_board", False)),
                        disabled=not bool(settings.get("partner_board_enabled", True)),
                        key=f"wizard_partner_board_optin_{event_id}",
                    )
                )
                event_payload["partner_note"] = st.text_input(
                    "Short note for partner board (optional)",
                    value=_safe_text(existing.get("partner_note")),
                    key=f"wizard_partner_note_{event_id}",
                )
            partner_details[event_id] = event_payload
        wizard["step4"] = {"partner_details": partner_details}

        c1, c2, _ = st.columns([1, 1, 3])
        with c1:
            if st.button("← Back", key="step4_back"):
                wizard["current_step"] = 3
                st.rerun()
        with c2:
            submitted = st.button("Submit registration", type="primary", key="step4_submit")
        if submitted:
            first_name = _safe_text(step1.get("first_name"))
            last_name = _safe_text(step1.get("last_name"))
            email = _safe_text(step1.get("email"))
            phone = _safe_text(step1.get("phone"))
            gender = _safe_text(step1.get("gender"))
            age = _safe_text(step1.get("age"))
            notes = _safe_text(step1.get("notes"))
            final_display_name = _safe_text(step2.get("display_name")) or " ".join(part for part in [first_name, last_name] if part)
            dupr_id = _safe_text(step2.get("dupr_id"))
            selections: list[dict[str, Any]] = []
            for event_id in selected_event_ids:
                event = event_lookup.get(event_id) or {}
                selection_row: dict[str, Any] = {
                    "id": _uid("sel"),
                    "registration_day_id": str(event.get("registration_day_id")),
                    "event_option_id": str(event_id),
                    "partner_mode": "NONE",
                }
                if bool(event.get("partner_required")):
                    saved_partner = (wizard.get("step4") or {}).get("partner_details", {}).get(event_id) or {}
                    selection_row.update(saved_partner)
                    selection_row["partner_mode"] = _safe_text(saved_partner.get("partner_mode") or "NEEDS_PARTNER")
                selections.append(selection_row)

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
                st.link_button("View Tournament Roster", public_urls["roster"])
                wizard["current_step"] = 1
                wizard["step3"] = {"selected_event_ids": []}
                wizard["step4"] = {"partner_details": {}}
            except Exception as exc:
                st.error(f"Could not save registration: {exc}")
