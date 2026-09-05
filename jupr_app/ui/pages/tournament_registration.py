from __future__ import annotations

from typing import Any
from difflib import SequenceMatcher
import json
import re
import uuid

import streamlit as st

from jupr_app.domain.tournament_registration_compiler import validate_selection_against_skill
from jupr_app.domain.notifications.smtp_mailer import get_smtp_config_status
from jupr_app.domain.notifications.tournament_registration_edit_email import send_tournament_registration_edit_email
from jupr_app.domain.tournament_registration_edit_tokens import build_registration_edit_token
from jupr_app.domain.notifications.tournament_registration_confirmation_email import (
    build_registration_confirmation_view_model,
    send_tournament_registration_confirmation_email,
)
from jupr_app.domain.tournament_partner_service import admin_confirm_partner_link, create_partner_request
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
    get_registration_by_email,
    get_registration_settings,
    get_registration_confirmation_bundle,
    is_day_enabled,
    list_event_options as list_registration_event_options,
    list_existing_tournaments,
    list_open_public_tournaments,
    list_partner_requests,
    list_partner_team_members,
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
from jupr_app.ui.public_links import build_public_url, navigate_same_tab
from jupr_app.ui.tournament_registration_confirmation_view import render_registration_confirmation_summary
from jupr_app.ui.tournament_registration_session import (
    clear_registration_wizard_for_new_start,
    get_submission_result,
    store_submission_result,
    wizard_state_key,
)


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


def _public_empty_state_message(
    *,
    registration_open: bool,
    selectable_count: int,
    hidden_draft_count: int,
) -> str | None:
    if not registration_open:
        return "Registration is closed."
    if selectable_count > 0:
        return None
    if hidden_draft_count > 0:
        return "Registration coming soon. Divisions are being finalized."
    return "No open divisions are available right now."


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


def _find_player_by_id(players: list[dict[str, Any]], player_id: Any) -> dict[str, Any] | None:
    clean_id = _safe_text(player_id)
    if not clean_id:
        return None
    return next((row for row in players if str(row.get("id")) == clean_id), None)


def _partner_search_matches(player: dict[str, Any], query: str) -> bool:
    needle = _safe_text(query).lower()
    if not needle:
        return False
    haystack = " ".join(
        _safe_text(player.get(key))
        for key in ["id", "name", "display_name", "dupr_id", "rating", "doubles_skill", "singles_skill"]
    ).lower()
    return needle in haystack


def _player_display_name(player: dict[str, Any] | None) -> str:
    row = player or {}
    return _safe_text(row.get("display_name") or row.get("name") or (f"Player #{row.get('id')}" if row.get("id") not in (None, "") else ""))


def _selection_ids_in_confirmed_teams(registration_state: dict[str, Any], event_option_id: str) -> set[str]:
    confirmed: set[str] = set()
    for roster in registration_state.get("event_rosters") or []:
        if _safe_text(roster.get("event_option_id")) != _safe_text(event_option_id):
            continue
        for entry in roster.get("entries") or []:
            if _safe_text(entry.get("status")).upper() not in {"CONFIRMED", "ADMIN_CONFIRMED"}:
                continue
            for member in entry.get("members") or []:
                selection_id = _safe_text(member.get("selection_id"))
                if selection_id:
                    confirmed.add(selection_id)
    return confirmed


def _registered_partner_target_for_player(registration_state: dict[str, Any], event_option_id: str, player_id: Any) -> dict[str, Any] | None:
    pid = _safe_text(player_id)
    if not pid:
        return None
    for roster in registration_state.get("event_rosters") or []:
        if _safe_text(roster.get("event_option_id")) != _safe_text(event_option_id):
            continue
        for entry in roster.get("entries") or []:
            for member in entry.get("members") or []:
                if _safe_text(member.get("player_id")) == pid:
                    return {
                        "target_selection_id": _safe_text(member.get("selection_id")),
                        "target_registration_id": _safe_text(member.get("registration_id")),
                        "target_player_id": pid,
                        "status": _safe_text(entry.get("status")).upper(),
                    }
    return None


def _partner_request_ready(details: dict[str, Any]) -> bool:
    return _safe_text(details.get("partner_mode")).upper() == "REQUEST_PARTNER" and bool(
        _safe_text(details.get("target_selection_id")) or _safe_text(details.get("target_player_id"))
    )


def _legacy_partner_match_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _safe_text(value).lower())


def _legacy_partner_name_match_score(legacy_name: Any, candidate_name: Any) -> float:
    legacy_key = _legacy_partner_match_key(legacy_name)
    candidate_key = _legacy_partner_match_key(candidate_name)
    if not legacy_key or not candidate_key:
        return 0.0
    if legacy_key == candidate_key:
        return 1.0
    if legacy_key in candidate_key or candidate_key in legacy_key:
        return 0.92
    return SequenceMatcher(None, legacy_key, candidate_key).ratio()


def _selection_ids_with_active_team_members(team_members: list[dict[str, Any]]) -> set[str]:
    return {
        _safe_text(row.get("selection_id"))
        for row in team_members
        if _safe_text(row.get("selection_id")) and _safe_text(row.get("status") or "ACTIVE").upper() == "ACTIVE"
    }


def _selection_ids_with_pending_partner_requests(partner_requests: list[dict[str, Any]]) -> set[str]:
    selection_ids: set[str] = set()
    for request in partner_requests:
        if _safe_text(request.get("status")).upper() != "PENDING":
            continue
        for key in ["requester_selection_id", "target_selection_id"]:
            selection_id = _safe_text(request.get(key))
            if selection_id:
                selection_ids.add(selection_id)
    return selection_ids


def _legacy_partner_reconciliation_issues(
    admin_rows: list[dict[str, Any]],
    partner_requests: list[dict[str, Any]],
    team_members: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    confirmed_selection_ids = _selection_ids_with_active_team_members(team_members)
    pending_selection_ids = _selection_ids_with_pending_partner_requests(partner_requests)
    issues: list[dict[str, Any]] = []
    for row in admin_rows:
        sel = row.get("selection") or {}
        selection_id = _safe_text(row.get("selection_id") or sel.get("id"))
        if _safe_text(sel.get("partner_mode")).upper() != "HAS_PARTNER":
            continue
        if not (_safe_text(sel.get("partner_name")) or _safe_text(sel.get("partner_email"))):
            continue
        if "admin hold: legacy partner text reviewed" in _safe_text(sel.get("partner_note")).lower():
            continue
        if selection_id in confirmed_selection_ids or selection_id in pending_selection_ids:
            continue
        issues.append(row)
    return issues


def _legacy_partner_suggestions(
    issue: dict[str, Any],
    admin_rows: list[dict[str, Any]],
    *,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    issue_reg = issue.get("registration") or {}
    issue_sel = issue.get("selection") or {}
    issue_selection_id = _safe_text(issue.get("selection_id") or issue_sel.get("id"))
    issue_registration_id = _safe_text(issue.get("registration_id") or issue_sel.get("registration_id"))
    legacy_name = _safe_text(issue_sel.get("partner_name"))
    legacy_email = _safe_text(issue_sel.get("partner_email")).lower()
    same_event: list[dict[str, Any]] = []
    same_tournament: list[dict[str, Any]] = []

    for row in admin_rows:
        reg = row.get("registration") or {}
        sel = row.get("selection") or {}
        selection_id = _safe_text(row.get("selection_id") or sel.get("id"))
        registration_id = _safe_text(row.get("registration_id") or sel.get("registration_id"))
        if not selection_id or selection_id == issue_selection_id or registration_id == issue_registration_id:
            continue
        candidate_name = _safe_text(reg.get("display_name") or " ".join(part for part in [reg.get("first_name"), reg.get("last_name")] if _safe_text(part)))
        candidate_email = _safe_text(reg.get("email")).lower()
        score = 0.0
        reason = ""
        if legacy_email and candidate_email and legacy_email == candidate_email:
            score = 1.0
            reason = "Email exact match"
        elif legacy_name:
            score = _legacy_partner_name_match_score(legacy_name, candidate_name)
            if score >= 0.84:
                reason = f"Name match ({score:.0%})"
        if score < 0.84:
            continue
        suggestion = {
            "row": row,
            "score": score,
            "reason": reason,
            "scope": "same_event" if _safe_text(sel.get("event_option_id")) == _safe_text(issue_sel.get("event_option_id")) else "same_tournament",
        }
        if suggestion["scope"] == "same_event":
            same_event.append(suggestion)
        else:
            same_tournament.append(suggestion)

    return sorted(same_event, key=lambda item: item["score"], reverse=True)[:max_results] + sorted(
        same_tournament, key=lambda item: item["score"], reverse=True
    )[: max(0, max_results - len(same_event))]


def _confirm_suggested_profile_state(step2_state: dict[str, Any], *, player_id: str, selection_source: str, search_query: str) -> dict[str, Any]:
    return {
        **step2_state,
        "profile_mode": "existing",
        "selected_player_id": _safe_text(player_id),
        "candidate_player_id": _safe_text(player_id),
        "candidate_confirmed": True,
        "rejected_likely": False,
        "selection_source": _safe_text(selection_source) or "likely",
        "search_query": _safe_text(search_query),
    }


def _can_advance_profile_step(*, profile_mode: str, selection_source: str, candidate_player_id: str, candidate_confirmed: bool) -> bool:
    if _safe_text(profile_mode) != "existing":
        return True
    if _safe_text(selection_source) == "search" and _safe_text(candidate_player_id):
        return True
    return bool(candidate_confirmed)


def _resolve_existing_profile_for_next(
    active_players: list[dict[str, Any]],
    *,
    profile_mode: str,
    selection_source: str,
    selected_player_id: str,
    candidate_player_id: str,
    candidate_confirmed: bool,
    rejected_likely: bool,
) -> tuple[dict[str, Any] | None, str, str, bool, bool]:
    selected_player_id = _safe_text(selected_player_id)
    candidate_player_id = _safe_text(candidate_player_id)
    selected_existing_player = _find_player_by_id(active_players, selected_player_id)

    if _safe_text(profile_mode) == "existing" and _safe_text(selection_source) == "search":
        search_selected_player = _find_player_by_id(active_players, candidate_player_id)
        if search_selected_player:
            selected_existing_player = search_selected_player
            selected_player_id = str(search_selected_player.get("id"))
            candidate_player_id = selected_player_id
            candidate_confirmed = True
            rejected_likely = False

    return selected_existing_player, selected_player_id, candidate_player_id, candidate_confirmed, rejected_likely


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


def _mask_email(email: str) -> str:
    text = _safe_text(email).lower()
    if "@" not in text:
        return "***"
    local, domain = text.split("@", 1)
    if len(local) <= 1:
        masked = local[:1] + "***"
    elif len(local) == 2:
        masked = local[0] + "***" + local[-1]
    else:
        masked = local[0] + "***" + local[-1]
    return f"{masked}@{domain}"


def _selected_event_ids_from_selections(selections: list[dict[str, Any]]) -> list[str]:
    return [_safe_text(row.get("event_option_id")) for row in selections if _safe_text(row.get("event_option_id"))]


def _partner_details_from_selections(selections: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    details: dict[str, dict[str, Any]] = {}
    for row in selections:
        event_id = _safe_text(row.get("event_option_id"))
        if not event_id:
            continue
        details[event_id] = {
            "partner_mode": _safe_text(row.get("partner_mode") or "NONE").upper() or "NONE",
            "partner_name": _safe_text(row.get("partner_name")),
            "partner_email": _safe_text(row.get("partner_email")),
            "partner_phone": _safe_text(row.get("partner_phone")),
            "partner_dupr_id": _safe_text(row.get("partner_dupr_id")),
            "partner_skill": row.get("partner_skill"),
            "partner_age": row.get("partner_age"),
            "selection_id": _safe_text(row.get("id")),
            "registration_id": _safe_text(row.get("registration_id")),
            "partner_request_id": _safe_text(row.get("partner_request_id")),
            "target_selection_id": _safe_text(row.get("target_selection_id")),
            "target_registration_id": _safe_text(row.get("target_registration_id")),
            "target_player_id": _safe_text(row.get("target_player_id")),
            "target_display_name_snapshot": _safe_text(row.get("target_display_name_snapshot")),
            "partner_request_source": _safe_text(row.get("partner_request_source")),
            "show_on_partner_board": bool(row.get("show_on_partner_board")),
            "partner_note": _safe_text(row.get("partner_note")),
        }
    return details


def _split_display_name(display_name: Any) -> tuple[str, str]:
    parts = _safe_text(display_name).split()
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


def _hydrate_registration_wizard_from_bundle(
    wizard: dict[str, Any],
    bundle: dict[str, Any],
    *,
    preserve_existing_progress: bool = False,
) -> dict[str, Any]:
    registration = bundle.get("registration") or {}
    selections = bundle.get("selections") or []
    registration_id = _safe_text(registration.get("id"))
    same_registration = bool(wizard.get("edit_mode")) and _safe_text(wizard.get("edit_registration_id")) == registration_id
    should_preserve_progress = preserve_existing_progress and same_registration
    current_step = int(wizard.get("current_step") or 1)

    first_name = _safe_text(registration.get("first_name"))
    last_name = _safe_text(registration.get("last_name"))
    if (not first_name or not last_name) and _safe_text(registration.get("display_name")):
        display_first, display_last = _split_display_name(registration.get("display_name"))
        first_name = first_name or display_first
        last_name = last_name or display_last

    wizard["edit_mode"] = True
    wizard["email_locked"] = True
    wizard["edit_registration_id"] = registration_id
    if not should_preserve_progress:
        wizard["current_step"] = 1

    wizard.setdefault("step1", {})
    wizard["step1"].update(
        {
            "first_name": first_name,
            "last_name": last_name,
            "email": _safe_text(registration.get("email")),
            "phone": _safe_text(registration.get("phone")),
            "gender": _safe_text(registration.get("gender")),
            "age": _safe_text(registration.get("age")),
            "notes": _safe_text(registration.get("notes")),
        }
    )
    wizard.setdefault("step2", {})
    wizard["step2"].update(
        {
            "profile_mode": "new",
            "selected_player_id": "",
            "candidate_player_id": "",
            "candidate_confirmed": False,
            "rejected_likely": False,
            "search_query": "",
            "selection_source": "",
            "display_name": _safe_text(registration.get("display_name")),
            "dupr_id": _safe_text(registration.get("dupr_id")),
            "doubles_skill": registration.get("doubles_skill"),
            "singles_skill": registration.get("singles_skill"),
        }
    )
    if not (should_preserve_progress and current_step > 1):
        wizard["step3"] = {"selected_event_ids": _selected_event_ids_from_selections(selections)}
        wizard["step4"] = {"partner_details": _partner_details_from_selections(selections)}
    else:
        wizard.setdefault("step3", {"selected_event_ids": _selected_event_ids_from_selections(selections)})
        wizard.setdefault("step4", {"partner_details": _partner_details_from_selections(selections)})
    return wizard


def _advance_step1_registration_wizard(
    wizard: dict[str, Any],
    *,
    tournament_id: str,
    first_name: Any,
    last_name: Any,
    email_for_submit: Any,
    phone: Any,
    gender: Any,
    age: Any,
    notes: Any,
    find_existing_registration,
) -> tuple[bool, str]:
    if not _safe_text(first_name) or not _safe_text(last_name) or not _safe_text(email_for_submit) or not _safe_text(age) or not _safe_text(gender):
        return False, "Please complete the highlighted required fields before continuing."
    normalized_email = _safe_text(email_for_submit).lower()
    wizard["step1"] = {
        "first_name": _safe_text(first_name),
        "last_name": _safe_text(last_name),
        "email": normalized_email,
        "phone": _safe_text(phone),
        "gender": _safe_text(gender),
        "age": _safe_text(age),
        "notes": _safe_text(notes),
    }
    if bool(wizard.get("edit_mode")):
        wizard["current_step"] = 3
        return True, ""
    existing_registration = find_existing_registration(tournament_id, normalized_email) if find_existing_registration else None
    if existing_registration:
        wizard["returning_registration_id"] = str(existing_registration.get("id") or "")
        wizard["returning_email"] = normalized_email
        wizard["returning_email_sent"] = False
        wizard["returning_email_error"] = ""
        wizard["current_step"] = 0
        return True, ""
    wizard["current_step"] = 2
    return True, ""

def _init_wizard_state(tournament_id: Any) -> dict[str, Any]:
    key = wizard_state_key(tournament_id)
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
            "edit_mode": False,
            "edit_registration_id": "",
            "returning_registration_id": "",
            "returning_email": "",
            "returning_email_sent": False,
            "returning_email_error": "",
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


def _render_legacy_partner_reconciliation_panel(
    *,
    supabase,
    tournament_id: str,
    admin_rows: list[dict[str, Any]],
    partner_requests: list[dict[str, Any]],
    team_members: list[dict[str, Any]],
    event_lookup: dict[str, dict[str, Any]],
) -> None:
    issues = _legacy_partner_reconciliation_issues(admin_rows, partner_requests, team_members)
    with st.expander(f"Legacy partner reconciliation ({len(issues)})", expanded=bool(issues)):
        st.caption(
            "Legacy partner text is visible for audit only. It is not used for public team formation unless an admin explicitly creates a request or confirms a team."
        )
        if not issues:
            st.success("No unresolved legacy partner text currently needs reconciliation.")
            return
        for issue in issues:
            reg = issue.get("registration") or {}
            sel = issue.get("selection") or {}
            event = issue.get("event") or event_lookup.get(_safe_text(sel.get("event_option_id"))) or {}
            issue_selection_id = _safe_text(issue.get("selection_id") or sel.get("id"))
            issue_registration_id = _safe_text(issue.get("registration_id") or sel.get("registration_id"))
            event_label = _safe_text(event.get("division_name") or event.get("label") or sel.get("event_option_id"))
            title = f"{_safe_text(reg.get('display_name') or reg.get('email'))} → {_safe_text(sel.get('partner_name')) or _safe_text(sel.get('partner_email'))}"
            st.markdown(f"**{title}**")
            st.caption(
                f"Division: {event_label} · Legacy partner name: {_safe_text(sel.get('partner_name')) or '—'} · "
                f"Legacy partner email: {_safe_text(sel.get('partner_email')) or '—'} · "
                f"Current mode: {_status_badge(sel.get('partner_mode'))}"
            )
            suggestions = _legacy_partner_suggestions(issue, admin_rows)
            if not suggestions:
                st.info("No conservative same-event or same-tournament match suggestions found.")
                if st.button("Mark Admin Hold / Ignore", key=f"legacy_hold_{issue_selection_id}_none"):
                    existing_note = _safe_text(sel.get("partner_note"))
                    hold_note = "Admin hold: legacy partner text reviewed."
                    update_admin_registration_selection(
                        supabase,
                        tournament_id=tournament_id,
                        selection_id=issue_selection_id,
                        payload={
                            "partner_mode": "HAS_PARTNER",
                            "partner_name": sel.get("partner_name"),
                            "partner_email": sel.get("partner_email"),
                            "partner_note": f"{existing_note}\n{hold_note}".strip() if hold_note not in existing_note else existing_note,
                        },
                    )
                    st.success("Legacy issue marked for admin hold/ignore.")
                    st.rerun()
            for suggestion in suggestions:
                target_row = suggestion.get("row") or {}
                target_reg = target_row.get("registration") or {}
                target_sel = target_row.get("selection") or {}
                target_selection_id = _safe_text(target_row.get("selection_id") or target_sel.get("id"))
                target_name = _safe_text(target_reg.get("display_name") or target_reg.get("email"))
                cols = st.columns([3, 1, 1, 1])
                with cols[0]:
                    st.caption(
                        f"Suggested match: {target_name} · {suggestion.get('reason')} · "
                        f"{str(suggestion.get('scope')).replace('_', ' ')}"
                    )
                with cols[1]:
                    if st.button("Create Partner Request", key=f"legacy_create_request_{issue_selection_id}_{target_selection_id}"):
                        create_partner_request(
                            supabase,
                            tournament_id=tournament_id,
                            event_option_id=_safe_text(sel.get("event_option_id")),
                            requester_selection_id=issue_selection_id,
                            target_selection_id=target_selection_id,
                            target_player_id=target_reg.get("player_id"),
                            target_display_name_snapshot=target_name,
                            source="LEGACY_TEXT_MATCH",
                        )
                        st.success("Partner request created for admin reconciliation.")
                        st.rerun()
                with cols[2]:
                    if st.button("Admin Confirm Team", key=f"legacy_admin_confirm_{issue_selection_id}_{target_selection_id}"):
                        admin_confirm_partner_link(
                            supabase,
                            tournament_id=tournament_id,
                            event_option_id=_safe_text(sel.get("event_option_id")),
                            selection1_id=issue_selection_id,
                            selection2_id=target_selection_id,
                            source="ADMIN_RECONCILIATION",
                        )
                        st.success("Admin-confirmed linked team created.")
                        st.rerun()
                with cols[3]:
                    if st.button("Mark Hold / Ignore", key=f"legacy_hold_{issue_selection_id}_{target_selection_id}"):
                        existing_note = _safe_text(sel.get("partner_note"))
                        hold_note = "Admin hold: legacy partner text reviewed."
                        update_admin_registration_selection(
                            supabase,
                            tournament_id=tournament_id,
                            selection_id=issue_selection_id,
                            payload={
                                "partner_mode": "HAS_PARTNER",
                                "partner_name": sel.get("partner_name"),
                                "partner_email": sel.get("partner_email"),
                                "partner_note": f"{existing_note}\n{hold_note}".strip() if hold_note not in existing_note else existing_note,
                            },
                        )
                        st.success("Legacy issue marked for admin hold/ignore.")
                        st.rerun()
            st.divider()


def _render_registration_admin_roster(*, supabase, tournament: dict[str, Any], days: list[dict[str, Any]], event_options: list[dict[str, Any]]) -> None:
    tournament_id = str(tournament.get("id"))
    admin_rows = list_registration_admin_rows(supabase, tournament_id)
    try:
        partner_requests = list_partner_requests(supabase, tournament_id)
        team_members = list_partner_team_members(supabase, tournament_id)
    except Exception:
        partner_requests = []
        team_members = []
    day_lookup = {str(row.get("id")): row for row in days}
    event_lookup = {str(row.get("id")): row for row in event_options}

    registration_forms = len({str(row.get("registration_id")) for row in admin_rows if _safe_text(row.get("registration_id"))})
    active = [
        row
        for row in admin_rows
        if _safe_text((row.get("registration") or {}).get("status")).lower() in {"confirmed", "pending"}
    ]
    needs_partner = [row for row in admin_rows if _safe_text((row.get("selection") or {}).get("partner_mode")).upper() == "NEEDS_PARTNER"]
    paid = [row for row in admin_rows if _safe_text((row.get("registration") or {}).get("payment_status")).lower() == "paid"]
    unpaid = [row for row in admin_rows if _safe_text((row.get("registration") or {}).get("payment_status")).lower() == "unpaid"]

    metrics = st.columns(6)
    for idx, (label, value) in enumerate([
        ("Registration Forms", registration_forms),
        ("Event Entries", len(admin_rows)),
        ("Active Registrations", len(active)),
        ("Needs Partner", len(needs_partner)),
        ("Paid", len(paid)),
        ("Unpaid", len(unpaid)),
    ]):
        metrics[idx].metric(label, value)

    _render_legacy_partner_reconciliation_panel(
        supabase=supabase,
        tournament_id=tournament_id,
        admin_rows=admin_rows,
        partner_requests=partner_requests,
        team_members=team_members,
        event_lookup=event_lookup,
    )

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
        effective_status = _safe_text(reg.get("status")).lower()
        if effective_status == "pending":
            effective_status = "confirmed"
        if status_filter != "All" and effective_status != status_filter:
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

            quick_actions = st.columns(2)
            if quick_actions[0].button("Move to Waitlist", key=f"waitlist_{sel_id}_{reg_id}"):
                update_admin_registration(supabase, tournament_id=tournament_id, registration_id=reg_id, payload={"status": "waitlist"})
                st.rerun()
            if quick_actions[1].button("Cancel", key=f"cancel_{sel_id}_{reg_id}"):
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
                partner_name = st.text_input("Legacy partner text — not used for team formation", value=_safe_text(sel.get("partner_name")))
                partner_email = st.text_input("Legacy partner email — not used for team formation", value=_safe_text(sel.get("partner_email")))
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



def _render_step5_confirmation_fallback(
    *,
    supabase,
    tournament_id: str,
    settings: dict[str, Any],
    registration_id: str,
    email_status: str,
) -> None:
    registration_id = _safe_text(registration_id)
    if not registration_id:
        st.error("We saved your registration, but could not load the confirmation details. Please contact tournament staff.")
        return
    try:
        bundle = get_registration_confirmation_bundle(supabase, tournament_id, registration_id)
    except Exception:
        st.error("Your registration was saved, but we could not load the confirmation summary right now. Please contact tournament staff.")
        return
    if not (bundle.get("registration") or {}):
        st.error("Your registration was saved, but we could not find the confirmation summary right now. Please contact tournament staff.")
        return

    render_registration_confirmation_summary(
        bundle=bundle,
        email_status=email_status,
        sender_status=get_smtp_config_status(),
        show_title=True,
    )
    slug = _safe_text(settings.get("registration_slug")) or _safe_text((bundle.get("settings") or {}).get("registration_slug"))
    nav_params = {"tournament_id": tournament_id, "registration_id": registration_id}
    if slug:
        nav_params["tournament"] = slug
    if email_status:
        nav_params["email_status"] = email_status
    roster_params = {"tournament_id": tournament_id}
    if slug:
        roster_params["tournament"] = slug
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("View Tournament Roster", key="fallback_view_roster"):
            navigate_same_tab(page="tournament_roster", params=roster_params, public_mode=True)
    with col2:
        if st.button("Open Confirmation Page", key="fallback_open_confirmation"):
            navigate_same_tab(page="tournament_registration_confirmation", params=nav_params, public_mode=True)
    with col3:
        if st.button("Start another registration", key="fallback_start_another"):
            clear_registration_wizard_for_new_start(tournament_id)
            navigate_same_tab(page="tournament_registration", params=roster_params, public_mode=True)
            st.rerun()

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
    current_page_key = _safe_text(st.query_params.get("page"))
    admin_page_key = (
        "tournament_registration_admin"
        if admin_mode and current_page_key == "tournament_registration_admin"
        else "tournament_registration"
    )
    qp_tournament_id = _safe_text(st.query_params.get("tournament_id"))
    qp_slug = _safe_text(st.query_params.get("tournament"))
    tournament, settings, days, event_options = (
        _select_admin_tournament(ctx, supabase, page_key=admin_page_key)
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
        if st.button("View Tournament Roster", key=f"view_tournament_roster_{tournament.get('id')}"):
            nav_params = {"tournament_id": str(tournament.get("id"))}
            slug = _safe_text(settings.get("registration_slug"))
            if slug:
                nav_params["tournament"] = slug
            navigate_same_tab(page="tournament_roster", params=nav_params, public_mode=True)

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
                    status = st.selectbox("Admin status", ADMIN_REGISTRATION_STATUS_OPTIONS, index=0)
                    payment_status = st.selectbox("Payment status", ADMIN_PAYMENT_STATUS_OPTIONS)
                    day_id = st.selectbox("Day", [str(d.get("id")) for d in days], format_func=lambda did: _safe_text((day_lookup.get(did) or {}).get("label") or did))
                    event_id = st.selectbox("Division", [str(e.get("id")) for e in event_options], format_func=lambda eid: f"{_safe_text((event_lookup.get(eid) or {}).get('event_family_label'))} / {_safe_text((event_lookup.get(eid) or {}).get('division_name') or (event_lookup.get(eid) or {}).get('label'))}")
                    partner_mode = st.selectbox("Partner mode", PARTNER_MODE_OPTIONS)
                    partner_name = st.text_input("Legacy partner text — not used for team formation")
                    partner_email = st.text_input("Legacy partner email — not used for team formation")
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
            if st.button("Open Public Registration Form", key=f"open_public_registration_{tournament.get('id')}"):
                nav_params = {"tournament_id": str(tournament.get("id"))}
                slug = _safe_text(settings.get("registration_slug"))
                if slug:
                    nav_params["tournament"] = slug
                navigate_same_tab(page="tournament_registration", params=nav_params, public_mode=True)
            if st.button("Open Public Roster", key=f"open_public_roster_{tournament.get('id')}"):
                nav_params = {"tournament_id": str(tournament.get("id"))}
                slug = _safe_text(settings.get("registration_slug"))
                if slug:
                    nav_params["tournament"] = slug
                navigate_same_tab(page="tournament_roster", params=nav_params, public_mode=True)
        return

    if settings.get("sponsor_markdown"):
        st.markdown(_safe_text(settings.get("sponsor_markdown")))
    if settings.get("rules_markdown"):
        with st.expander("Rules and registration notes", expanded=False):
            st.markdown(_safe_text(settings.get("rules_markdown")))
    if settings.get("refund_policy_markdown"):
        with st.expander("Refund policy", expanded=False):
            st.markdown(_safe_text(settings.get("refund_policy_markdown")))

    days = [row for row in days if is_day_enabled(row)]
    selectable_event_options = [row for row in event_options if public_event_option_visibility(row) == "selectable"]
    blocked_visible_options = [row for row in event_options if public_event_option_visibility(row) == "visible_blocked"]
    hidden_draft_options = [row for row in event_options if _safe_text(row.get("status") or "draft").lower() == "draft"]

    is_open, _ = registration_is_open(settings)
    empty_message = _public_empty_state_message(
        registration_open=is_open,
        selectable_count=len(selectable_event_options),
        hidden_draft_count=len(hidden_draft_options),
    )
    if empty_message:
        st.warning(empty_message)
        st.stop()
    if not days or not selectable_event_options:
        st.warning("No open divisions are available right now.")
        st.stop()

    grouped_events = _group_events(days, selectable_event_options)
    blocked_by_family: dict[tuple[str, str], list[str]] = {}
    for event in blocked_visible_options:
        day_id = str(event.get("registration_day_id"))
        family = _safe_text(event.get("event_family_label") or event.get("label") or "Event")
        division_name = _safe_text(event.get("division_name") or event.get("label") or "Division")
        blocked_by_family.setdefault((day_id, family), []).append(division_name)

    tournament_id = str(tournament.get("id"))
    wizard = _init_wizard_state(tournament.get("id"))
    submission_result = get_submission_result(tournament_id)
    if submission_result and int(wizard.get("current_step") or 1) != 5:
        wizard["current_step"] = 5
        wizard["submitted_registration_id"] = _safe_text(submission_result.get("registration_id"))
        wizard["submitted_email_status"] = _safe_text(submission_result.get("email_status"))
    current_step = int(wizard.get("current_step") or 1)
    if current_step == 5:
        registration_id = _safe_text(wizard.get("submitted_registration_id") or submission_result.get("registration_id"))
        email_status = _safe_text(wizard.get("submitted_email_status") or submission_result.get("email_status") or "sent")
        st.caption("Registration submitted")
        _render_step5_confirmation_fallback(
            supabase=supabase,
            tournament_id=tournament_id,
            settings=settings,
            registration_id=registration_id,
            email_status=email_status,
        )
        return
    step1 = wizard.get("step1") or {}
    step2 = wizard.get("step2") or {}
    step3 = wizard.get("step3") or {}
    step4 = wizard.get("step4") or {}
    active_players = _load_active_players(supabase, club_id=club_id, ctx=ctx)

    edit_mode = bool(wizard.get("edit_mode"))
    if edit_mode:
        st.info("Editing existing registration. Contact info → Events → Partner information → Confirmation.")
    st.caption("Editing existing registration" if edit_mode else f"Step {current_step} of 4")

    if current_step == 1:
        st.markdown("### 1. Name and contact")
        c1, c2 = st.columns(2)
        with c1:
            first_name = st.text_input("First name *", value=_safe_text(step1.get("first_name")))
            locked_email = _safe_text((wizard.get("step1") or {}).get("email"))
            email_widget = st.text_input(
                "Email *",
                value=locked_email,
                disabled=edit_mode,
                key=f"wizard_step1_email_{tournament_id}",
            )
            email_for_submit = locked_email if edit_mode else email_widget
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
                advanced, error = _advance_step1_registration_wizard(
                    wizard,
                    tournament_id=str(tournament.get("id")),
                    first_name=first_name,
                    last_name=last_name,
                    email_for_submit=email_for_submit,
                    phone=phone,
                    gender=gender,
                    age=age,
                    notes=notes,
                    find_existing_registration=lambda tid, email: get_registration_by_email(supabase, tid, email),
                )
                if not advanced:
                    st.error(error)
                    st.stop()
                st.rerun()

    if current_step == 0:
        masked = _mask_email(_safe_text(wizard.get("returning_email")))
        st.markdown("### You already have a registration for this tournament.")
        st.write(f"For your security, we’ll email a secure edit link to {masked}.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Email me a secure edit link", type="primary"):
                try:
                    token = build_registration_edit_token(
                        tournament_id=str(tournament.get("id")),
                        registration_id=_safe_text(wizard.get("returning_registration_id")),
                        email=_safe_text(wizard.get("returning_email")),
                    )
                    slug = _safe_text(settings.get("registration_slug"))
                    edit_url = build_public_url(page="tournament_registration_edit", params={"tournament_id": str(tournament.get("id")), "tournament": slug, "edit_token": token})
                    send_tournament_registration_edit_email(
                        tournament_name=_safe_text(tournament.get("name") or "Tournament"),
                        registered_email=_safe_text(wizard.get("returning_email")),
                        edit_url=edit_url,
                    )
                    wizard["returning_email_sent"] = True
                    wizard["returning_email_error"] = ""
                except Exception as exc:
                    wizard["returning_email_sent"] = False
                    error_text = str(exc).lower()
                    if "configuration" in error_text or "jupr_registration_edit_secret" in error_text:
                        wizard["returning_email_error"] = "Secure edit links are not configured yet. Please contact tournament staff to update your registration."
                    else:
                        wizard["returning_email_error"] = "We could not send the edit link automatically. Please contact tournament staff."
                st.rerun()
        with c2:
            if st.button("Back / use a different email"):
                wizard["current_step"] = 1
                wizard["returning_registration_id"] = ""
                wizard["returning_email"] = ""
                st.rerun()
        if wizard.get("returning_email_sent"):
            st.success(f"We sent an edit link to {masked}. Please check spam/junk.")
        if wizard.get("returning_email_error"):
            st.warning(_safe_text(wizard.get("returning_email_error")))
        return

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
        selected_existing_player = _find_player_by_id(active_players, selected_player_id)
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
            candidate_player = _find_player_by_id(active_players, candidate_player_id)
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
                        wizard["step2"] = _confirm_suggested_profile_state(
                            step2_state,
                            player_id=selected_player_id,
                            selection_source=selection_source,
                            search_query=search_query_default,
                        )
                        wizard["current_step"] = 3
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
                    st.caption("Select a profile, then click Next to continue with that JUPR profile.")
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

        next_allowed = _can_advance_profile_step(
            profile_mode=profile_mode,
            selection_source=selection_source,
            candidate_player_id=candidate_player_id,
            candidate_confirmed=candidate_confirmed,
        )
        if profile_mode == "existing" and selection_source == "likely" and candidate_player_id and not candidate_confirmed:
            st.caption("Confirm the suggested profile above, or choose to continue without it.")
        c1, c2, c3 = st.columns([1, 1, 3])
        with c1:
            if st.button("← Back"):
                wizard["current_step"] = 1
                st.rerun()
        with c2:
            next_label = "Next ➜" if next_allowed else "Confirm profile above"
            if st.button(next_label, type="primary", disabled=not next_allowed):
                (
                    selected_existing_player,
                    selected_player_id,
                    candidate_player_id,
                    candidate_confirmed,
                    rejected_likely,
                ) = _resolve_existing_profile_for_next(
                    active_players,
                    profile_mode=profile_mode,
                    selection_source=selection_source,
                    selected_player_id=selected_player_id,
                    candidate_player_id=candidate_player_id,
                    candidate_confirmed=candidate_confirmed,
                    rejected_likely=rejected_likely,
                )
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
        selected_existing_player = _find_player_by_id(active_players, step2.get("selected_player_id"))
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
        try:
            partner_registration_state = build_registration_state(supabase, tournament, settings, days, event_options)
        except Exception:
            partner_registration_state = {}
        current_player_id = _safe_text(step2.get("selected_player_id")) if using_existing_player else ""

        if not doubles_selected:
            st.info("No doubles divisions selected. You can submit now.")
        for event in doubles_selected:
            event_id = str(event.get("id"))
            existing = partner_details.get(event_id) or {}
            st.markdown(f"**{_safe_text(event.get('division_name') or event.get('label') or event_id)}**")
            mode_options = ["NEEDS_PARTNER", "REQUEST_PARTNER"]
            existing_mode = _safe_text(existing.get("partner_mode")).upper()
            if existing_mode == "HAS_PARTNER":
                existing_mode = "REQUEST_PARTNER"
            mode = st.radio(
                "Partner plan",
                mode_options,
                horizontal=True,
                format_func=lambda v: "I need a partner" if v == "NEEDS_PARTNER" else "I want to request a partner",
                index=mode_options.index(existing_mode) if existing_mode in mode_options else 0,
                key=f"wizard_partner_mode_{event_id}",
            )
            event_payload: dict[str, Any] = {"partner_mode": mode}
            if mode == "NEEDS_PARTNER":
                board_enabled = bool(event.get("show_partner_board", event.get("partner_board_enabled", True)))
                event_payload["show_on_partner_board"] = st.checkbox(
                    "Show me in Players Needing Partners for this division",
                    value=bool(existing.get("show_on_partner_board", board_enabled)),
                    disabled=not board_enabled,
                    key=f"wizard_partner_board_{event_id}",
                )
                event_payload["partner_note"] = st.text_input(
                    "Short note for potential partners (optional)",
                    value=_safe_text(existing.get("partner_note")),
                    key=f"wizard_partner_note_{event_id}",
                )
            else:
                st.caption("Choose a JUPR profile or a registered player looking for a partner. Typed names are legacy admin notes only and will not create a team.")
                existing_target_name = _safe_text(existing.get("target_display_name_snapshot"))
                if existing_target_name:
                    st.success(f"Selected partner request target: {existing_target_name}")
                search_query = st.text_input("Search JUPR/player profiles", value=_safe_text(existing.get("profile_search_query")), key=f"wizard_partner_search_{event_id}")
                matches = [row for row in active_players if _partner_search_matches(row, search_query)][:8]
                confirmed_selection_ids = _selection_ids_in_confirmed_teams(partner_registration_state, event_id)
                for player in matches:
                    pid = _safe_text(player.get("id"))
                    registered_target = _registered_partner_target_for_player(partner_registration_state, event_id, pid)
                    status = _safe_text((registered_target or {}).get("status")) or "Not registered in this division"
                    is_self = bool(current_player_id and pid == current_player_id)
                    target_confirmed = _safe_text((registered_target or {}).get("target_selection_id")) in confirmed_selection_ids
                    cols = st.columns([3, 2, 1])
                    with cols[0]:
                        st.markdown(f"**{_player_display_name(player)}**")
                        st.caption(f"Player ID: {pid or '—'} · DUPR: {_safe_text(player.get('dupr_id')) or '—'} · Rating: {_player_rating_text(player)}")
                    with cols[1]:
                        st.caption(f"Registration status: {status.replace('_', ' ').title()}")
                    with cols[2]:
                        disabled = is_self or target_confirmed
                        if st.button("Select", key=f"wizard_partner_profile_select_{event_id}_{pid}", disabled=disabled):
                            event_payload.update(
                                {
                                    "target_selection_id": _safe_text((registered_target or {}).get("target_selection_id")),
                                    "target_registration_id": _safe_text((registered_target or {}).get("target_registration_id")),
                                    "target_player_id": pid,
                                    "target_display_name_snapshot": _player_display_name(player),
                                    "partner_request_source": "PROFILE_SEARCH",
                                    "profile_search_query": search_query,
                                    "partner_skill": _player_current_overall_jupr(player),
                                    "partner_age": _coerce_int(player.get("age")),
                                }
                            )
                            partner_details[event_id] = event_payload
                            wizard["step4"] = {"partner_details": partner_details}
                            st.rerun()
                for key in ["target_selection_id", "target_registration_id", "target_player_id", "target_display_name_snapshot", "partner_request_source", "profile_search_query", "partner_skill", "partner_age"]:
                    if _safe_text(existing.get(key)) and key not in event_payload:
                        event_payload[key] = existing.get(key)
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
                    selected_partner_mode = _safe_text(saved_partner.get("partner_mode") or "NEEDS_PARTNER").upper()
                    selection_row["partner_mode"] = "HAS_PARTNER" if selected_partner_mode == "REQUEST_PARTNER" else selected_partner_mode
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

            pending_partner_requests: list[dict[str, Any]] = []
            for selection in selections:
                event = event_lookup.get(str(selection.get("event_option_id") or "")) or {}
                if not bool(event.get("partner_required")):
                    continue
                raw_details = (wizard.get("step4") or {}).get("partner_details", {}).get(str(selection.get("event_option_id"))) or {}
                if _safe_text(raw_details.get("partner_mode")).upper() == "REQUEST_PARTNER":
                    if not _partner_request_ready(raw_details):
                        st.error("For doubles events where you want to request a partner, select a JUPR profile or a registered player from the needs-partner list.")
                        st.stop()
                    if current_player_id and _safe_text(raw_details.get("target_player_id")) == current_player_id:
                        st.error("You cannot request yourself as a partner.")
                        st.stop()
                    pending_partner_requests.append({**raw_details, "requester_selection_id": selection.get("id"), "event_option_id": selection.get("event_option_id")})
                elif _safe_text(selection.get("partner_mode")).upper() != "NEEDS_PARTNER":
                    st.error("For doubles events, choose either 'I need a partner' or select a partner request target.")
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
                    if _safe_text(row.get("status")).upper() not in {"NEEDS_PARTNER", "PARTNER_MISSING", "PENDING_PARTNER_REQUEST", "LEGACY_PARTNER_UNRESOLVED"}
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
                        "email": _safe_text(wizard.get("step1", {}).get("email")) if wizard.get("edit_mode") else email,
                        "phone": phone,
                        "dupr_id": dupr_id,
                        "player_id": _coerce_int(step2.get("selected_player_id")) if using_existing_player else None,
                        "doubles_skill": submit_player.get("doubles_skill"),
                        "singles_skill": submit_player.get("singles_skill"),
                        "age": _coerce_int(age),
                        "age_bracket": None,
                        "gender": gender,
                        "notes": notes,
                        "wants_partner_board_contact": any(bool(row.get("show_on_partner_board")) for row in selections),
                        "selections": selections,
                    },
                    expected_registration_id=_safe_text(wizard.get("edit_registration_id")) if wizard.get("edit_mode") else None,
                )
                registration_id = _safe_text(result.get("registration_id"))
                created_partner_requests: list[dict[str, Any]] = []
                for request_details in pending_partner_requests:
                    created_partner_requests.append(
                        create_partner_request(
                            supabase,
                            tournament_id=str(tournament.get("id")),
                            event_option_id=_safe_text(request_details.get("event_option_id")),
                            requester_selection_id=_safe_text(request_details.get("requester_selection_id")),
                            target_selection_id=_safe_text(request_details.get("target_selection_id")) or None,
                            target_player_id=_safe_text(request_details.get("target_player_id")) or None,
                            target_display_name_snapshot=_safe_text(request_details.get("target_display_name_snapshot")) or None,
                            source=_safe_text(request_details.get("partner_request_source")) or "PROFILE_SEARCH",
                        )
                    )
                if created_partner_requests:
                    st.success(f"Created {len(created_partner_requests)} pending partner request(s). Your team is not confirmed until accepted.")
                slug = _safe_text(settings.get("registration_slug"))
                nav_params = {
                    "tournament_id": str(tournament.get("id")),
                    "registration_id": registration_id,
                }
                if slug:
                    nav_params["tournament"] = slug
                confirmation_url = build_public_url(
                    page="tournament_registration_confirmation",
                    params=nav_params,
                )
                email_status = "sent"
                try:
                    smtp_status = get_smtp_config_status()
                    view_model = build_registration_confirmation_view_model(
                        tournament=tournament,
                        registration={
                            "id": registration_id,
                            "display_name": final_display_name,
                            "email": _safe_text(wizard.get("step1", {}).get("email")) if wizard.get("edit_mode") else email,
                        },
                        selections=selections,
                        days=days,
                        event_options=event_options,
                        confirmation_url=confirmation_url,
                        sender_from_name=smtp_status.get("from_name"),
                        sender_from_email=smtp_status.get("from_email"),
                    )
                    send_result = send_tournament_registration_confirmation_email(view_model=view_model)
                    email_status = _safe_text(send_result.get("status")) or "sent"
                    if email_status == "staging_redirect":
                        email_status = "sent"
                except Exception as exc:
                    print(f"Tournament registration confirmation email failed: {exc}")
                    email_status = "failed"
                nav_params["email_status"] = "dry_run" if email_status == "dry_run" else ("failed" if email_status == "failed" else "sent")
                store_submission_result(
                    tournament_id=str(tournament.get("id")),
                    registration_id=registration_id,
                    email_status=nav_params["email_status"],
                    nav_params=nav_params,
                )
                wizard["current_step"] = 5
                wizard["submitted_registration_id"] = registration_id
                wizard["submitted_email_status"] = nav_params["email_status"]
                st.session_state["last_registration_submit_debug"] = {
                    "registration_id": registration_id,
                    "email_status": nav_params["email_status"],
                    "confirmation_page": "tournament_registration_confirmation",
                    "nav_params": dict(nav_params),
                }
                try:
                    navigate_same_tab(
                        page="tournament_registration_confirmation",
                        params=nav_params,
                        public_mode=True,
                    )
                except Exception as nav_exc:
                    st.warning("Your registration was saved, but we could not open the separate confirmation page automatically. Showing your confirmation below.")
                    st.caption(f"Navigation detail: {nav_exc}")
                _render_step5_confirmation_fallback(
                    supabase=supabase,
                    tournament_id=str(tournament.get("id")),
                    settings=settings,
                    registration_id=registration_id,
                    email_status=nav_params["email_status"],
                )
            except Exception as exc:
                st.error(f"Could not save registration: {exc}")
