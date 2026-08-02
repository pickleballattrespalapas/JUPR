from __future__ import annotations

from datetime import date, datetime
from typing import Any

from jupr_app.domain.tournament_registration_repo import (
    build_public_tournament_roster_state,
    get_public_tournament_bundle,
    is_day_enabled,
    list_open_public_tournaments,
    public_event_option_visibility,
    registration_feature_available,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _clean_text(value: Any, *, limit: int = 240) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


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
        "weather_policy_markdown": _clean_text(row.get("weather_policy_markdown"), limit=4000),
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


def _public_event(row: dict[str, Any]) -> dict[str, Any]:
    event_format = row.get("event_format_override") or row.get("event_format_default")
    scoring = row.get("scoring_override") or row.get("scoring_default")
    return {
        "id": str(row.get("id") or ""),
        "registration_day_id": str(row.get("registration_day_id") or ""),
        "scheduled_day_ids": [
            _clean_text(value, limit=160)
            for value in (
                row.get("scheduled_day_ids")
                if isinstance(row.get("scheduled_day_ids"), list)
                else []
            )
            if _clean_text(value, limit=160)
        ]
        or [str(row.get("registration_day_id") or "")],
        "label": _clean_text(row.get("label") or row.get("division_name") or "Division", limit=160),
        "event_family_label": _clean_text(row.get("event_family_label") or row.get("label") or "Event", limit=160),
        "division_name": _clean_text(row.get("division_name") or row.get("label") or "Division", limit=160),
        "event_type": _clean_text(row.get("event_type"), limit=40),
        "event_format": _clean_text(event_format, limit=120),
        "scoring": _clean_text(scoring, limit=120),
        "capacity_teams": _safe_int(row.get("capacity_teams")),
        "partner_required": _safe_bool(row.get("partner_required")),
        "partner_board_enabled": _safe_bool(row.get("partner_board_enabled", row.get("public_partner_board"))),
        "status": _clean_text(row.get("status") or "draft", limit=40).lower(),
        "visibility": public_event_option_visibility(row),
    }


def _open_tournament_choices(supabase: Any, *, club_id: str) -> list[dict[str, Any]]:
    choices = []
    for item in list_open_public_tournaments(supabase, str(club_id)):
        tournament = _public_tournament(item.get("tournament") or {})
        settings = _public_settings(item.get("settings") or {})
        if tournament and settings:
            choices.append({"tournament": tournament, "settings": settings})
    return choices


def _empty_payload(*, available: bool, setup_error: str | None, tournaments: list[dict[str, Any]] | None = None, reason: str | None = None) -> dict[str, Any]:
    return {
        "available": bool(available),
        "setup_error": setup_error,
        "tournaments": list(tournaments or []),
        "tournament": None,
        "settings": None,
        "days": [],
        "events": [],
        "roster": None,
        "summary": None,
        "empty_reason": reason,
    }


def build_public_tournament_roster_page(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str | None = None,
    registration_slug: str | None = None,
) -> dict[str, Any]:
    available, detail = registration_feature_available(supabase)
    if not available:
        return _empty_payload(available=False, setup_error=detail, reason="Tournament registration is not configured.")

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
        return _empty_payload(
            available=True,
            setup_error=None,
            tournaments=open_choices,
            reason="No published tournament roster was found.",
        )

    public_days = [_public_day(row) for row in days_raw if is_day_enabled(row)]
    public_day_ids = {row["id"] for row in public_days}
    public_events = [
        _public_event(row)
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
        "days": public_days,
        "events": public_events,
        "roster": roster_state,
        "summary": roster_state.get("summary") if isinstance(roster_state, dict) else None,
        "empty_reason": None,
    }
