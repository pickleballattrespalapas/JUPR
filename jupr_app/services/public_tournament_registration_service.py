from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Any
from urllib.parse import urlencode

from jupr_app.config import get_env_or_default
from jupr_app.domain.tournament_registration_compiler import validate_selection_against_skill
from jupr_app.domain.notifications.smtp_mailer import get_smtp_config_status
from jupr_app.domain.notifications.tournament_registration_confirmation_email import (
    PAYMENT_NOTE,
    build_registration_confirmation_view_model,
    send_tournament_registration_confirmation_email,
)
from jupr_app.domain.tournament_registration_confirmation_tokens import (
    build_registration_confirmation_token,
    verify_registration_confirmation_token,
)
from jupr_app.domain.tournament_registration_repo import (
    build_public_tournament_roster_state,
    get_public_tournament_bundle,
    get_registration_confirmation_bundle,
    get_registration_by_email,
    is_day_enabled,
    list_open_public_tournaments,
    public_event_option_visibility,
    registration_feature_available,
    registration_is_open,
    save_registration,
)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_PUBLIC_SELECTIONS = 8
_PARTNER_IDENTITY_RATING_AGE_FIELDS = (
    "partner_name",
    "partner_email",
    "partner_phone",
    "partner_dupr_id",
    "partner_skill",
    "partner_age",
    "partner_gender",
)

_WOMEN_GENDERS = {"f", "female", "woman", "women", "womens", "girl", "girls"}
_MEN_GENDERS = {"m", "male", "man", "men", "mens", "boy", "boys"}
LOGGER = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _clean_text(value: Any, *, limit: int = 240) -> str:
    text = str(value or "").replace("<", "").replace(">", "").strip()
    return text[:limit]


def _clean_email(value: Any) -> str:
    return _clean_text(value, limit=320).lower()


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_rows(response: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (response.data or [])]
    except Exception:
        return []


def _player_is_active(row: dict[str, Any]) -> bool:
    if row.get("inactive_at") not in (None, ""):
        return False
    if "active" in row and not _safe_bool(row.get("active")):
        return False
    return True


def _canonical_player_skills(row: dict[str, Any]) -> tuple[float | None, float | None]:
    """Return the same canonical registration ratings used by the Streamlit flow."""

    overall = _safe_float(row.get("rating"))
    if overall is not None:
        # JUPR stores its club rating as Elo (roughly 1,200 == 3.0 skill).
        canonical = overall / 400.0 if overall > 10 else overall
        return canonical, canonical
    return _safe_float(row.get("doubles_skill")), _safe_float(row.get("singles_skill"))


def _public_registration_player(row: dict[str, Any]) -> dict[str, Any]:
    doubles_skill, singles_skill = _canonical_player_skills(row)
    return {
        "id": str(row.get("id") or ""),
        "display_name": _clean_text(row.get("display_name") or row.get("name") or "Player", limit=160),
        "dupr_id": _clean_text(row.get("dupr_id"), limit=80),
        "doubles_skill": doubles_skill,
        "singles_skill": singles_skill,
    }


def _list_public_registration_players(supabase: Any, *, club_id: str) -> list[dict[str, Any]]:
    try:
        rows = _safe_rows(
            supabase.table("players")
            .select("*")
            .eq("club_id", str(club_id))
            .limit(2000)
            .execute()
        )
    except Exception:
        rows = []
    players = [_public_registration_player(row) for row in rows if _player_is_active(row)]
    players.sort(key=lambda row: (str(row.get("display_name") or "").lower(), str(row.get("id") or "")))
    return players


def _get_club_player(
    supabase: Any,
    *,
    club_id: str,
    player_id: Any,
    require_active: bool,
) -> dict[str, Any] | None:
    clean_id = _clean_text(player_id, limit=160)
    if not clean_id:
        return None
    try:
        rows = _safe_rows(
            supabase.table("players")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("id", clean_id)
            .limit(1)
            .execute()
        )
    except Exception:
        rows = []
    player = rows[0] if rows else None
    if not player:
        raise ValueError("The selected JUPR player profile was not found in this club.")
    if require_active and not _player_is_active(player):
        raise ValueError("The selected JUPR player profile is not active in this club.")
    return player


def _normalized_gender(value: Any) -> str:
    text = re.sub(r"[^a-z]", "", str(value or "").strip().lower())
    if text in _WOMEN_GENDERS:
        return "WOMEN"
    if text in _MEN_GENDERS:
        return "MEN"
    return "OTHER" if text else ""


def _event_label(event: dict[str, Any]) -> str:
    return _clean_text(event.get("division_name") or event.get("label") or "Division", limit=160)


def _validate_gender_eligibility(
    *,
    event: dict[str, Any],
    player_gender: Any,
    partner_mode: str,
    partner_gender: Any = None,
) -> None:
    restriction = _clean_text(event.get("gender_restriction") or "ANY", limit=40).upper()
    if restriction in {"", "ANY", "OPEN", "NONE"}:
        return

    player = _normalized_gender(player_gender)
    label = _event_label(event)
    if restriction in {"MEN", "MALE"}:
        if player != "MEN":
            raise ValueError(f"{label}: this division is limited to men's registrations.")
        if partner_mode == "HAS_PARTNER" and _normalized_gender(partner_gender) != "MEN":
            raise ValueError(f"{label}: both partners must be eligible for the men's division.")
        return
    if restriction in {"WOMEN", "FEMALE"}:
        if player != "WOMEN":
            raise ValueError(f"{label}: this division is limited to women's registrations.")
        if partner_mode == "HAS_PARTNER" and _normalized_gender(partner_gender) != "WOMEN":
            raise ValueError(f"{label}: both partners must be eligible for the women's division.")
        return
    if restriction == "MIXED":
        if player not in {"MEN", "WOMEN"}:
            raise ValueError(f"{label}: select an eligible gender for mixed doubles.")
        if partner_mode == "HAS_PARTNER":
            partner = _normalized_gender(partner_gender)
            if partner not in {"MEN", "WOMEN"}:
                raise ValueError(f"{label}: partner gender is required for mixed-doubles eligibility.")
            if partner == player:
                raise ValueError(f"{label}: mixed doubles requires one men's and one women's registrant.")


def _event_family_key(event: dict[str, Any]) -> tuple[str, str]:
    day_id = _clean_text(event.get("registration_day_id"), limit=160)
    family = " ".join(
        _clean_text(event.get("event_family_label") or event.get("label") or "Event", limit=160)
        .lower()
        .split()
    )
    return day_id, family


def _public_tournament(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "id": str(row.get("id") or ""),
        "name": _clean_text(row.get("name") or "Tournament"),
        "status": _clean_text(row.get("status")),
        "start_date": _json_safe(row.get("start_date")),
        "end_date": _json_safe(row.get("end_date")),
        "event_tags": row.get("event_tags") if isinstance(row.get("event_tags"), dict) else None,
    }


def _public_settings(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "registration_slug": _clean_text(row.get("registration_slug"), limit=120),
        "registration_status": _clean_text(row.get("registration_status") or "draft", limit=40).lower(),
        "registration_open_at": _json_safe(row.get("registration_open_at")),
        "registration_close_at": _json_safe(row.get("registration_close_at")),
        "waitlist_enabled": _safe_bool(row.get("waitlist_enabled")),
        "partner_board_enabled": _safe_bool(row.get("partner_board_enabled")),
        "rules_markdown": _clean_text(row.get("rules_markdown"), limit=4000),
        "refund_policy_markdown": _clean_text(row.get("refund_policy_markdown"), limit=4000),
        "sponsor_markdown": _clean_text(row.get("sponsor_markdown"), limit=4000),
    }


def _public_day(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "label": _clean_text(row.get("label") or "Day", limit=120),
        "event_date": _json_safe(row.get("event_date")),
        "sort_order": _safe_int(row.get("sort_order")) or 0,
        "enabled": is_day_enabled(row),
    }


def _public_event(row: dict[str, Any], *, registration_open: bool) -> dict[str, Any]:
    visibility = public_event_option_visibility(row)
    event_format = row.get("event_format_override") or row.get("event_format_default")
    scoring = row.get("scoring_override") or row.get("scoring_default")
    return {
        "id": str(row.get("id") or ""),
        "registration_day_id": str(row.get("registration_day_id") or ""),
        "label": _clean_text(row.get("label") or row.get("division_name") or "Division", limit=160),
        "event_family_label": _clean_text(row.get("event_family_label") or row.get("label") or "Event", limit=160),
        "division_name": _clean_text(row.get("division_name") or row.get("label") or "Division", limit=160),
        "event_type": _clean_text(row.get("event_type"), limit=40),
        "gender_restriction": _clean_text(row.get("gender_restriction") or "ANY", limit=40),
        "skill_label": _clean_text(row.get("skill_label"), limit=80),
        "age_label": _clean_text(row.get("age_label"), limit=80),
        "skill_mode": _clean_text(row.get("skill_mode"), limit=80),
        "age_mode": _clean_text(row.get("age_mode"), limit=80),
        "event_format": _clean_text(event_format, limit=120),
        "scoring": _clean_text(scoring, limit=120),
        "capacity_teams": _safe_int(row.get("capacity_teams")),
        "price_usd": _safe_float(row.get("price_usd")),
        "partner_required": _safe_bool(row.get("partner_required")),
        "partner_board_enabled": _safe_bool(row.get("partner_board_enabled", row.get("public_partner_board"))),
        "waitlist_enabled": _safe_bool(row.get("waitlist_enabled", True)),
        "status": _clean_text(row.get("status") or "draft", limit=40).lower(),
        "visibility": visibility,
        "selectable": bool(registration_open and visibility == "selectable"),
    }


def _open_tournament_choices(supabase: Any, *, club_id: str) -> list[dict[str, Any]]:
    choices = []
    for item in list_open_public_tournaments(supabase, str(club_id)):
        tournament = _public_tournament(item.get("tournament") or {})
        settings = _public_settings(item.get("settings") or {})
        if tournament and settings:
            choices.append({"tournament": tournament, "settings": settings})
    return choices


def _public_web_base_url(public_base_url: str | None = None) -> str:
    """Return an explicitly configured Next.js origin.

    Do not inherit ``get_public_base_url()`` here: its localhost:8501 default
    and ``JUPR_PUBLIC_BASE_URL`` compatibility value belong to the Streamlit
    surface. Confirmation email links must either target the Next.js site or
    fail visibly after the registration has been saved.
    """

    for candidate in (
        public_base_url,
        get_env_or_default("JUPR_WEB_BASE_URL"),
        get_env_or_default("STAGING_WEB_BASE_URL"),
        get_env_or_default("NEXT_PUBLIC_JUPR_WEB_BASE_URL"),
    ):
        value = str(candidate or "").strip().rstrip("/")
        if value:
            return value
    return ""


def _confirmation_page_url(
    *,
    club_slug: str,
    confirmation_token: str,
    email_status: str | None = None,
    public_base_url: str | None = None,
) -> str:
    query = {"confirmation_token": str(confirmation_token)}
    if email_status:
        query["email_status"] = str(email_status)
    return (
        f"{_public_web_base_url(public_base_url)}/clubs/{club_slug}/"
        f"tournament-registration/confirmation?{urlencode(query)}"
    )


def _roster_page_url(
    *,
    club_slug: str,
    tournament_id: str,
    registration_slug: str | None,
    public_base_url: str | None = None,
) -> str:
    query = {
        "tournament": str(registration_slug)
    } if registration_slug else {"tournament_id": str(tournament_id)}
    return (
        f"{_public_web_base_url(public_base_url)}/clubs/{club_slug}/"
        f"tournament-roster?{urlencode(query)}"
    )


def build_registration_confirmation_delivery(
    supabase: Any,
    *,
    club_id: str,
    club_slug: str,
    tournament_id: str,
    registration_id: str,
    public_base_url: str | None = None,
) -> dict[str, Any]:
    """Build signed confirmation access and attempt email after persistence.

    This function is deliberately called only after `save_registration` returns.
    A token/config/mail failure therefore cannot roll back or disguise a saved
    registration.
    """

    try:
        bundle = get_registration_confirmation_bundle(
            supabase,
            str(tournament_id),
            str(registration_id),
        )
    except Exception:
        LOGGER.exception("Tournament registration was saved but confirmation details could not be loaded")
        return {
            "confirmation_available": False,
            "confirmation_token": None,
            "email_delivery": {
                "status": "failed",
                "message": "Registration was saved, but confirmation details could not be prepared.",
            },
        }
    registration = bundle.get("registration") or {}
    tournament = bundle.get("tournament") or {}
    settings = bundle.get("settings") or {}
    if not registration or str(tournament.get("club_id") or club_id) != str(club_id):
        return {
            "confirmation_available": False,
            "confirmation_token": None,
            "email_delivery": {
                "status": "failed",
                "message": "Registration was saved, but confirmation details could not be prepared.",
            },
        }

    try:
        token = build_registration_confirmation_token(
            tournament_id=str(tournament_id),
            registration_id=str(registration_id),
            email=_clean_email(registration.get("email")),
        )
    except Exception:
        LOGGER.exception("Unable to create a tournament registration confirmation token")
        return {
            "confirmation_available": False,
            "confirmation_token": None,
            "email_delivery": {
                "status": "failed",
                "message": "Registration was saved, but secure confirmation access is not configured.",
            },
        }

    web_base = _public_web_base_url(public_base_url)
    if not web_base:
        return {
            "confirmation_available": True,
            "confirmation_token": token,
            "email_delivery": {
                "status": "failed",
                "message": "Registration was saved, but the confirmation email link is not configured.",
            },
        }

    try:
        smtp_status = get_smtp_config_status()
        confirmation_url = _confirmation_page_url(
            club_slug=str(club_slug),
            confirmation_token=token,
            public_base_url=web_base,
        )
        roster_url = _roster_page_url(
            club_slug=str(club_slug),
            tournament_id=str(tournament_id),
            registration_slug=_clean_text(settings.get("registration_slug"), limit=120) or None,
            public_base_url=web_base,
        )
        view_model = build_registration_confirmation_view_model(
            tournament=tournament,
            registration=registration,
            selections=bundle.get("selections") or [],
            days=bundle.get("days") or [],
            event_options=bundle.get("event_options") or [],
            confirmation_url=confirmation_url,
            roster_url=roster_url,
            sender_from_name=smtp_status.get("from_name"),
            sender_from_email=smtp_status.get("from_email"),
        )
        send_result = send_tournament_registration_confirmation_email(
            view_model=view_model
        )
        status = _clean_text(send_result.get("status"), limit=40) or "sent"
        message = {
            "dry_run": "Registration was saved; confirmation email was safely dry-run.",
            "staging_redirect": "Registration was saved; confirmation email was sent to the staging redirect.",
            "sent": "Registration was saved and the confirmation email was sent.",
        }.get(status, "Registration was saved and confirmation delivery was accepted.")
    except Exception:
        LOGGER.exception("Tournament registration was saved but confirmation email failed")
        status = "failed"
        message = "Registration was saved, but the confirmation email could not be sent."

    return {
        "confirmation_available": True,
        "confirmation_token": token,
        "email_delivery": {"status": status, "message": message},
    }


def build_public_tournament_registration_page(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str | None = None,
    registration_slug: str | None = None,
) -> dict[str, Any]:
    available, detail = registration_feature_available(supabase)
    if not available:
        return {
            "available": False,
            "setup_error": detail,
            "tournaments": [],
            "tournament": None,
            "settings": None,
            "registration_open": False,
            "registration_closed_reason": "Registration is not configured.",
            "days": [],
            "events": [],
            "players": [],
            "roster_summary": None,
        }

    slug = _clean_text(registration_slug, limit=120)
    tid = _clean_text(tournament_id, limit=120)
    open_choices = _open_tournament_choices(supabase, club_id=str(club_id))
    if not slug and not tid and open_choices:
        tid = str(open_choices[0]["tournament"].get("id") or "")

    tournament, settings, days_raw, events_raw = get_public_tournament_bundle(
        supabase,
        club_id=str(club_id),
        tournament_id=tid or None,
        registration_slug=slug or None,
    )
    if not tournament or not settings:
        return {
            "available": True,
            "setup_error": None,
            "tournaments": open_choices,
            "tournament": None,
            "settings": None,
            "registration_open": False,
            "registration_closed_reason": "No open tournament registration was found.",
            "days": [],
            "events": [],
            "players": [],
            "roster_summary": None,
        }

    registration_open, closed_reason = registration_is_open(settings)
    public_days = [_public_day(row) for row in days_raw if is_day_enabled(row)]
    public_day_ids = {row["id"] for row in public_days}
    public_events = [
        _public_event(row, registration_open=registration_open)
        for row in events_raw
        if str(row.get("registration_day_id") or "") in public_day_ids and public_event_option_visibility(row) != "hidden"
    ]
    public_events.sort(key=lambda item: (item.get("registration_day_id") or "", str(item.get("event_family_label") or ""), str(item.get("division_name") or "")))

    roster_state = build_public_tournament_roster_state(supabase, tournament, settings, days_raw, events_raw)
    return {
        "available": True,
        "setup_error": None,
        "tournaments": open_choices,
        "tournament": _public_tournament(tournament),
        "settings": _public_settings(settings),
        "registration_open": bool(registration_open),
        "registration_closed_reason": closed_reason,
        "days": public_days,
        "events": public_events,
        # Public intake does not establish player identity, so it must not expose
        # the club's player directory. A token-gated edit may add only its single
        # already-linked player to this otherwise-empty collection.
        "players": [],
        "roster_summary": roster_state.get("summary") if isinstance(roster_state, dict) else None,
    }


def _validate_submit_payload(payload: dict[str, Any]) -> None:
    if _clean_text(payload.get("website")):
        raise ValueError("Unable to submit registration.")
    if not _safe_bool(payload.get("terms_accepted")):
        raise ValueError("Please confirm the tournament policies before submitting.")
    email = _clean_email(payload.get("email"))
    if not email or not _EMAIL_RE.match(email):
        raise ValueError("A valid email is required.")
    display_name = _clean_text(payload.get("display_name") or " ".join(part for part in [payload.get("first_name"), payload.get("last_name")] if _clean_text(part)))
    if not display_name:
        raise ValueError("Player name is required.")
    selections = payload.get("selections") or []
    if not isinstance(selections, list) or not selections:
        raise ValueError("Select at least one event.")
    if len(selections) > MAX_PUBLIC_SELECTIONS:
        raise ValueError(f"Select no more than {MAX_PUBLIC_SELECTIONS} events.")


def _clean_selection(selection: dict[str, Any]) -> dict[str, Any]:
    partner_mode = _clean_text(selection.get("partner_mode") or "NONE", limit=40).upper()
    if partner_mode not in {"NONE", "HAS_PARTNER", "NEEDS_PARTNER"}:
        partner_mode = "NONE"
    return {
        "event_option_id": _clean_text(selection.get("event_option_id"), limit=160),
        "registration_day_id": _clean_text(selection.get("registration_day_id"), limit=160),
        "partner_mode": partner_mode,
        "partner_name": _clean_text(selection.get("partner_name"), limit=160),
        "partner_email": _clean_email(selection.get("partner_email")),
        "partner_phone": _clean_text(selection.get("partner_phone"), limit=60),
        "partner_dupr_id": _clean_text(selection.get("partner_dupr_id"), limit=80),
        "partner_skill": _safe_float(selection.get("partner_skill")),
        "partner_age": _safe_int(selection.get("partner_age")),
        # Transient validation-only field. The established staging schema does not
        # persist partner gender, so edits re-resolve it from a registered partner
        # or ask for it again when a restricted division needs it.
        "partner_gender": _clean_text(selection.get("partner_gender"), limit=40),
        "partner_note": _clean_text(selection.get("partner_note"), limit=500),
        "show_on_partner_board": _safe_bool(selection.get("show_on_partner_board")),
    }


def _validated_rating(value: Any, *, label: str) -> float | None:
    rating = _safe_float(value)
    if rating is not None and not 0.0 <= rating <= 7.0:
        raise ValueError(f"{label} must be between 0 and 7.")
    return rating


def _validated_age(value: Any, *, label: str) -> int | None:
    age = _safe_int(value)
    if age is not None and not 1 <= age <= 120:
        raise ValueError(f"{label} must be between 1 and 120.")
    return age


def build_tournament_registration_player_profile(
    supabase: Any,
    *,
    club_id: str,
    registration: dict[str, Any],
    require_active_link: bool = False,
) -> dict[str, Any]:
    """Build the canonical eligibility profile for an existing registration."""

    player_id = registration.get("player_id")
    linked_player = (
        _get_club_player(
            supabase,
            club_id=str(club_id),
            player_id=player_id,
            require_active=require_active_link,
        )
        if player_id not in (None, "")
        else None
    )
    doubles_skill = _validated_rating(registration.get("doubles_skill"), label="Doubles skill")
    singles_skill = _validated_rating(registration.get("singles_skill"), label="Singles skill")
    gender = _clean_text(registration.get("gender"), limit=40)
    age = _validated_age(registration.get("age"), label="Age")
    if linked_player:
        canonical_doubles, canonical_singles = _canonical_player_skills(linked_player)
        doubles_skill = canonical_doubles if canonical_doubles is not None else doubles_skill
        singles_skill = canonical_singles if canonical_singles is not None else singles_skill
        gender = _clean_text(linked_player.get("gender") or gender, limit=40)
        if linked_player.get("age") not in (None, ""):
            age = _validated_age(linked_player.get("age"), label="Age")

    return {
        "email": _clean_email(registration.get("email")),
        "player_id": linked_player.get("id") if linked_player else player_id,
        "doubles_skill": doubles_skill,
        "singles_skill": singles_skill,
        "gender": gender,
        "age": age,
    }


def _registered_partner_profile(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    partner_email: str,
    primary_registration_id: str | None,
    primary_player_id: Any,
) -> dict[str, Any] | None:
    registration = get_registration_by_email(supabase, str(tournament_id), partner_email)
    if not registration:
        return None
    if primary_registration_id and str(registration.get("id") or "") == str(primary_registration_id):
        raise ValueError("A player cannot register themselves as their own partner.")
    partner_player_id = registration.get("player_id")
    if primary_player_id not in (None, "") and partner_player_id not in (None, "") and str(partner_player_id) == str(primary_player_id):
        raise ValueError("A player cannot register themselves as their own partner.")

    profile = dict(registration)
    if partner_player_id not in (None, ""):
        linked = _get_club_player(
            supabase,
            club_id=str(club_id),
            player_id=partner_player_id,
            require_active=False,
        )
        if linked:
            doubles_skill, singles_skill = _canonical_player_skills(linked)
            profile["doubles_skill"] = doubles_skill if doubles_skill is not None else profile.get("doubles_skill")
            profile["singles_skill"] = singles_skill if singles_skill is not None else profile.get("singles_skill")
            profile["gender"] = linked.get("gender") or profile.get("gender")
            profile["age"] = linked.get("age") if linked.get("age") not in (None, "") else profile.get("age")
    return profile


def validate_and_clean_tournament_selection(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    event: dict[str, Any],
    raw_selection: dict[str, Any],
    player_profile: dict[str, Any],
    settings: dict[str, Any],
    primary_registration_id: str | None = None,
) -> dict[str, Any]:
    """Validate and canonicalize one tournament event selection.

    The caller remains responsible for deciding whether ``event`` is currently
    selectable and for enforcing cross-selection day/family uniqueness.
    """

    if not isinstance(raw_selection, dict):
        raise ValueError("Each event selection must be an object.")
    raw_mode = _clean_text(raw_selection.get("partner_mode") or "NONE", limit=40).upper()
    if raw_mode not in {"NONE", "HAS_PARTNER", "NEEDS_PARTNER"}:
        raise ValueError("Invalid partner status in event selection.")

    clean_selection = _clean_selection(raw_selection)
    event_option_id = str(clean_selection.get("event_option_id") or "").strip()
    if not event_option_id:
        raise ValueError("Each event selection must identify a division.")

    partner_required = _safe_bool(event.get("partner_required"))
    event_type = _clean_text(event.get("event_type"), limit=40).upper()
    singles_event = event_type == "SINGLES"
    partner_mode = str(clean_selection.get("partner_mode") or "NONE")
    if partner_required and partner_mode not in {"HAS_PARTNER", "NEEDS_PARTNER"}:
        raise ValueError(f"{_event_label(event)}: choose whether you have or need a partner.")
    if singles_event and partner_mode != "NONE":
        raise ValueError(f"{_event_label(event)} does not accept partner information.")

    partner_profile: dict[str, Any] | None = None
    partner_gender = clean_selection.get("partner_gender")
    if partner_mode == "HAS_PARTNER":
        partner_name = _clean_text(clean_selection.get("partner_name"), limit=160)
        partner_email = _clean_email(clean_selection.get("partner_email"))
        if not partner_name:
            raise ValueError(f"{_event_label(event)}: partner name is required.")
        if not partner_email or not _EMAIL_RE.match(partner_email):
            raise ValueError(f"{_event_label(event)}: a valid partner email is required.")
        if partner_email == _clean_email(player_profile.get("email")):
            raise ValueError("A player cannot register themselves as their own partner.")

        registered_partner = _registered_partner_profile(
            supabase,
            club_id=str(club_id),
            tournament_id=str(tournament_id),
            partner_email=partner_email,
            primary_registration_id=primary_registration_id,
            primary_player_id=player_profile.get("player_id"),
        )
        if registered_partner:
            partner_profile = {
                "doubles_skill": _safe_float(registered_partner.get("doubles_skill")),
                "singles_skill": _safe_float(registered_partner.get("singles_skill")),
            }
            partner_gender = registered_partner.get("gender") or partner_gender
            clean_selection["partner_skill"] = (
                partner_profile.get("doubles_skill")
                if partner_profile.get("doubles_skill") is not None
                else partner_profile.get("singles_skill")
            )
            clean_selection["partner_age"] = _safe_int(registered_partner.get("age"))
        else:
            partner_skill = _validated_rating(clean_selection.get("partner_skill"), label="Partner skill")
            partner_profile = {"doubles_skill": partner_skill, "singles_skill": partner_skill}
        clean_selection["show_on_partner_board"] = False
    elif partner_mode == "NEEDS_PARTNER":
        if _safe_bool(clean_selection.get("show_on_partner_board")):
            if not _safe_bool(settings.get("partner_board_enabled")) or not _safe_bool(
                event.get("partner_board_enabled")
            ):
                raise ValueError(f"{_event_label(event)}: the public partner board is not enabled.")
        for key in _PARTNER_IDENTITY_RATING_AGE_FIELDS:
            clean_selection[key] = None if key in {"partner_skill", "partner_age"} else ""
        partner_gender = ""
    else:
        clean_selection["show_on_partner_board"] = False
        for key in _PARTNER_IDENTITY_RATING_AGE_FIELDS:
            clean_selection[key] = None if key in {"partner_skill", "partner_age"} else ""
        partner_gender = ""

    _validate_gender_eligibility(
        event=event,
        player_gender=player_profile.get("gender"),
        partner_mode=partner_mode,
        partner_gender=partner_gender,
    )
    eligible, message = validate_selection_against_skill(
        event=event,
        selection=clean_selection,
        player=player_profile,
        partner=partner_profile,
        allow_missing_partner_for_preview=False,
    )
    if not eligible:
        raise ValueError(f"{_event_label(event)}: {message or 'Skill eligibility requirements were not met.'}")

    clean_selection["registration_day_id"] = str(event.get("registration_day_id") or "")
    # This field is intentionally validation-only and is not part of the
    # established tournament_registration_selections schema.
    clean_selection.pop("partner_gender", None)
    return clean_selection


def _validate_and_clean_selections(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    page: dict[str, Any],
    payload: dict[str, Any],
    player_profile: dict[str, Any],
    primary_registration_id: str | None = None,
    existing_event_options: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    selectable = {str(event.get("id") or ""): event for event in (page.get("events") or []) if event.get("selectable")}
    allowed_existing = {str(key): value for key, value in (existing_event_options or {}).items() if value}
    settings = page.get("settings") or {}
    selections: list[dict[str, Any]] = []
    seen_events: set[str] = set()
    seen_families: set[tuple[str, str]] = set()

    for raw_selection in payload.get("selections") or []:
        if not isinstance(raw_selection, dict):
            raise ValueError("Each event selection must be an object.")
        raw_mode = _clean_text(raw_selection.get("partner_mode") or "NONE", limit=40).upper()
        if raw_mode not in {"NONE", "HAS_PARTNER", "NEEDS_PARTNER"}:
            raise ValueError("Invalid partner status in event selection.")
        event_option_id = _clean_text(raw_selection.get("event_option_id"), limit=160)
        if not event_option_id:
            raise ValueError("Each event selection must identify a division.")
        if event_option_id in seen_events:
            raise ValueError("The same division cannot be selected more than once.")

        event = selectable.get(event_option_id) or allowed_existing.get(event_option_id)
        if not event:
            raise ValueError("One or more selected events is no longer open for registration.")
        family_key = _event_family_key(event)
        if family_key in seen_families:
            family = _clean_text(event.get("event_family_label") or event.get("label") or "Event", limit=160)
            raise ValueError(f"Choose only one division for {family} on the same registration day.")

        clean_selection = validate_and_clean_tournament_selection(
            supabase,
            club_id=str(club_id),
            tournament_id=str(tournament_id),
            event=event,
            raw_selection=raw_selection,
            player_profile=player_profile,
            settings=settings,
            primary_registration_id=primary_registration_id,
        )
        selections.append(clean_selection)
        seen_events.add(event_option_id)
        seen_families.add(family_key)

    if not selections:
        raise ValueError("Select at least one open event.")
    return selections


def build_validated_public_registration_save_payload(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    page: dict[str, Any],
    payload: dict[str, Any],
    locked_registration: dict[str, Any] | None = None,
    existing_event_options: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    locked = locked_registration or None
    validation_payload = dict(payload)
    if locked:
        validation_payload["email"] = _clean_email(locked.get("email"))
    _validate_submit_payload(validation_payload)

    requested_player_id = payload.get("player_id")
    locked_player_id = locked.get("player_id") if locked else None
    if locked and requested_player_id not in (None, "") and str(requested_player_id) != str(locked_player_id or ""):
        raise ValueError("The linked JUPR player profile cannot be changed from an edit link.")
    # A new public submission has no authenticated player session or signed proof
    # of profile ownership. Treat its player_id as an untrusted suggestion: do not
    # persist it and do not use the selected profile's rating or DUPR identifier.
    # A token-gated edit may only preserve an already-established locked link.
    player_id = locked_player_id if locked else None
    linked_player = _get_club_player(
        supabase,
        club_id=str(club_id),
        player_id=player_id,
        require_active=not bool(locked),
    ) if player_id not in (None, "") else None
    doubles_skill = _validated_rating(payload.get("doubles_skill"), label="Doubles skill")
    singles_skill = _validated_rating(payload.get("singles_skill"), label="Singles skill")
    if linked_player:
        canonical_doubles, canonical_singles = _canonical_player_skills(linked_player)
        doubles_skill = canonical_doubles if canonical_doubles is not None else doubles_skill
        singles_skill = canonical_singles if canonical_singles is not None else singles_skill

    player_profile = {
        "email": _clean_email(locked.get("email") if locked else payload.get("email")),
        "player_id": linked_player.get("id") if linked_player else None,
        "doubles_skill": doubles_skill,
        "singles_skill": singles_skill,
        "gender": _clean_text(payload.get("gender"), limit=40),
        "age": _validated_age(payload.get("age"), label="Age"),
    }
    selections = _validate_and_clean_selections(
        supabase,
        club_id=str(club_id),
        tournament_id=str(tournament_id),
        page=page,
        payload=payload,
        player_profile=player_profile,
        primary_registration_id=str(locked.get("id") or "") if locked else None,
        existing_event_options=existing_event_options,
    )

    save_payload = {
        "first_name": _clean_text(payload.get("first_name"), limit=80),
        "last_name": _clean_text(payload.get("last_name"), limit=80),
        "display_name": _clean_text(payload.get("display_name"), limit=160),
        "email": player_profile["email"],
        "phone": _clean_text(payload.get("phone"), limit=60),
        "player_id": linked_player.get("id") if linked_player else None,
        "dupr_id": _clean_text(payload.get("dupr_id"), limit=80) or _clean_text((linked_player or {}).get("dupr_id"), limit=80),
        "doubles_skill": doubles_skill,
        "singles_skill": singles_skill,
        "age": player_profile["age"],
        "gender": player_profile["gender"],
        "notes": _clean_text(payload.get("notes"), limit=800),
        "wants_partner_board_contact": _safe_bool(payload.get("wants_partner_board_contact")),
        "selections": selections,
    }
    if locked:
        save_payload["payment_status"] = locked.get("payment_status") or "unpaid"
        save_payload["status"] = locked.get("status") or "confirmed"
    return save_payload


def submit_public_tournament_registration(
    supabase: Any,
    *,
    club_id: str,
    club_slug: str | None = None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    page = build_public_tournament_registration_page(
        supabase,
        club_id=str(club_id),
        tournament_id=_clean_text(payload.get("tournament_id"), limit=120) or None,
        registration_slug=_clean_text(payload.get("registration_slug"), limit=120) or None,
    )
    if not page.get("available"):
        raise ValueError("Tournament registration is not configured.")
    if not page.get("registration_open"):
        raise ValueError(str(page.get("registration_closed_reason") or "Registration is not open."))
    tournament = page.get("tournament") or {}
    tournament_id = str(tournament.get("id") or "").strip()
    if not tournament_id:
        raise ValueError("Tournament registration was not found.")
    save_payload = build_validated_public_registration_save_payload(
        supabase,
        club_id=str(club_id),
        tournament_id=tournament_id,
        page=page,
        payload=payload,
    )
    result = save_registration(supabase, tournament_id=tournament_id, payload=save_payload)
    delivery = build_registration_confirmation_delivery(
        supabase,
        club_id=str(club_id),
        club_slug=str(club_slug or club_id),
        tournament_id=tournament_id,
        registration_id=str(result.get("registration_id") or ""),
    )
    return {
        "ok": True,
        "tournament": tournament,
        "settings": page.get("settings"),
        "registration_id": result.get("registration_id"),
        "submitted_at": result.get("submitted_at"),
        "selection_count": result.get("selection_count"),
        **delivery,
    }


def build_public_tournament_registration_confirmation(
    supabase: Any,
    *,
    club_id: str,
    confirmation_token: str,
) -> dict[str, Any] | None:
    verified = verify_registration_confirmation_token(confirmation_token)
    tid = str(verified.get("tournament_id") or "").strip()
    registration_id = str(verified.get("registration_id") or "").strip()
    if not tid:
        return None
    bundle = get_registration_confirmation_bundle(supabase, tid, str(registration_id))
    registration = bundle.get("registration") or None
    if not registration:
        return None
    if str((bundle.get("tournament") or {}).get("club_id") or club_id) != str(club_id):
        return None
    verify_registration_confirmation_token(
        confirmation_token,
        expected_tournament_id=tid,
        expected_registration_id=registration_id,
        expected_email=_clean_email(registration.get("email")),
    )
    event_lookup = {str(row.get("id")): row for row in (bundle.get("event_options") or [])}
    day_lookup = {str(row.get("id")): row for row in (bundle.get("days") or [])}
    selections = []
    for selection in bundle.get("selections") or []:
        event = event_lookup.get(str(selection.get("event_option_id") or "")) or {}
        day = day_lookup.get(str(selection.get("registration_day_id") or "")) or {}
        selections.append(
            {
                "event_label": _clean_text(event.get("division_name") or event.get("label") or "Division"),
                "event_family_label": _clean_text(event.get("event_family_label") or event.get("label") or "Event"),
                "day_label": _clean_text(day.get("label") or "Day"),
                "event_date": _json_safe(day.get("event_date")),
                "skill_label": _clean_text(event.get("skill_label"), limit=80),
                "age_label": _clean_text(event.get("age_label"), limit=80),
                "price_usd": _safe_float(event.get("price_usd")) or 0,
                "partner_mode": _clean_text(selection.get("partner_mode")),
                "partner_name": _clean_text(selection.get("partner_name")),
                "show_on_partner_board": _safe_bool(selection.get("show_on_partner_board")),
            }
        )
    sender_status = get_smtp_config_status()
    return {
        "tournament": _public_tournament(bundle.get("tournament") or {}),
        "settings": _public_settings(bundle.get("settings") or {}),
        "registration": {
            "display_name": _clean_text(registration.get("display_name") or "Player"),
            "status": _clean_text(registration.get("status")),
            "payment_status": _clean_text(registration.get("payment_status")),
            "submitted_at": _json_safe(registration.get("submitted_at")),
        },
        "selections": selections,
        "total_price_usd": float(bundle.get("total_price_usd") or 0),
        "payment_note": PAYMENT_NOTE,
        "confirmation_expires_at": verified.get("exp"),
        "notification_sender": {
            "from_name": _clean_text(sender_status.get("from_name"), limit=120),
            "from_email": _clean_email(sender_status.get("from_email")),
        },
    }
