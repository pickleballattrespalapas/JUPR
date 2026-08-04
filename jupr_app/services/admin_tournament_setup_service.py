from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.tournament_admin_operations import stable_tournament_admin_fingerprint
from jupr_app.domain.tournament_registration_repo import (
    REGISTRATION_STATUS_OPTIONS,
    analyze_registration_publish_impact,
    clear_builder_draft,
    count_tournament_registrations,
    get_builder_draft,
    get_registration_settings,
    get_tournament_record,
    list_event_options,
    list_existing_tournaments,
    list_registration_days,
    publish_registration_configuration,
    save_builder_draft,
    upsert_registration_settings,
)
from jupr_app.services.admin_tournament_shell_create_service import CONFIRM_CREATE

TRUTHY = {"1", "true", "yes", "y", "on"}
CONFIRM_SETTINGS = "SAVE SETUP"
CONFIRM_DRAFT = "SAVE SETUP DRAFT"
CONFIRM_PUBLISH = "PUBLISH SETUP"


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY


def is_admin_tournament_setup_enabled() -> bool:
    return _truthy_env("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS")


def _clean(value: Any, *, limit: int = 500) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    return str(value).strip().lower() in TRUTHY


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _audit(
    supabase: Any,
    *,
    club_id: str,
    actor_email: str,
    actor_role: str,
    action_type: str,
    entity_id: str,
    before_json: Any,
    after_json: Any,
    source: str,
    flagged: bool = True,
) -> list[str]:
    result = write_admin_activity_log(
        supabase,
        build_activity_payload(
            club_id=str(club_id),
            actor_email=str(actor_email or ""),
            actor_role=str(actor_role or ""),
            action_type=action_type,
            entity_type="tournament_setup",
            entity_id=str(entity_id),
            before_json=before_json,
            after_json={"source_client": "fastapi/nextjs", "source_page": source, "value": after_json},
            source_page=source,
            flagged_for_review=bool(flagged),
        ),
    )
    if not result.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")
    return [result.warning] if result.warning else []


def _assert_enabled() -> None:
    if not is_admin_tournament_setup_enabled():
        raise PermissionError("Next Tournament Setup is disabled.")


def _get_tournament_for_club(supabase: Any, *, club_id: str, tournament_id: str) -> dict[str, Any]:
    tournament = get_tournament_record(supabase, str(tournament_id))
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    return tournament


def _settings_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "tournament_id": row.get("tournament_id"),
        "registration_slug": row.get("registration_slug"),
        "locale": row.get("locale"),
        "registration_status": row.get("registration_status"),
        "registration_open_at": row.get("registration_open_at"),
        "registration_close_at": row.get("registration_close_at"),
        "waitlist_enabled": row.get("waitlist_enabled"),
        "partner_board_enabled": row.get("partner_board_enabled"),
        "rules_markdown": row.get("rules_markdown"),
        "refund_policy_markdown": row.get("refund_policy_markdown"),
        "weather_policy_markdown": row.get("weather_policy_markdown"),
        "sponsor_markdown": row.get("sponsor_markdown"),
        "location_name": row.get("location_name"),
        "timezone": row.get("timezone"),
        "sponsors_json": list(row.get("sponsors_json") or []),
        "updated_at": row.get("updated_at"),
    }


def _event_option_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "tournament_id": row.get("tournament_id"),
        "registration_day_id": row.get("registration_day_id"),
        "scheduled_day_ids": list(row.get("scheduled_day_ids") or []),
        "event_family_label": row.get("event_family_label"),
        "division_name": row.get("division_name"),
        "event_type": row.get("event_type"),
        "gender_restriction": row.get("gender_restriction"),
        "event_format_default": row.get("event_format_default"),
        "scoring_default": row.get("scoring_default"),
        "skill_label": row.get("skill_label"),
        "skill_mode": row.get("skill_mode"),
        "age_label": row.get("age_label"),
        "age_mode": row.get("age_mode"),
        "age_rules": row.get("age_rules"),
        "capacity_teams": row.get("capacity_teams"),
        "price_usd": row.get("price_usd"),
        "waitlist_enabled": row.get("waitlist_enabled"),
        "partner_board_enabled": row.get("partner_board_enabled"),
        "status": row.get("status"),
        "enabled": row.get("enabled"),
        "sort_order": row.get("sort_order"),
    }


def _day_payload(row: dict[str, Any]) -> dict[str, Any]:
    court_labels = row.get("court_labels")
    if not isinstance(court_labels, list):
        court_labels = []
    return {
        "id": row.get("id"),
        "tournament_id": row.get("tournament_id"),
        "label": row.get("label"),
        "event_date": row.get("event_date") or row.get("date") or row.get("start_date"),
        "date": row.get("event_date") or row.get("date") or row.get("start_date"),
        "court_count": row.get("court_count"),
        "court_labels": list(court_labels),
        "court_open_time": row.get("court_open_time"),
        "court_close_time": row.get("court_close_time"),
        "court_notes": row.get("court_notes"),
        "enabled": row.get("enabled"),
        "sort_order": row.get("sort_order"),
    }


def _setup_state_fingerprint(
    *,
    tournament: dict[str, Any],
    settings: dict[str, Any],
    days: list[dict[str, Any]],
    events: list[dict[str, Any]],
    draft: dict[str, Any] | None,
) -> str:
    stable_settings = _settings_payload(settings)
    stable_settings.pop("id", None)
    return stable_tournament_admin_fingerprint(
        {
            "tournament": {
                "id": tournament.get("id"),
                "status": tournament.get("status"),
                "updated_at": tournament.get("updated_at"),
            },
            "settings": stable_settings,
            "days": days,
            "event_options": events,
            "builder_draft": draft or {},
        }
    )


def _template_id(tournament_id: str, kind: str, label: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"jupr:tournament-setup:{tournament_id}:{kind}:{label}"))


def build_admin_tournament_setup_templates(
    *,
    tournament: dict[str, Any],
    days: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return Python-authoritative builder templates for the Next renderer."""

    tournament_id = str(tournament.get("id") or "").strip()
    template_days = [dict(row) for row in (days or [])]
    if not template_days:
        fallback_date = str(tournament.get("start_date") or "").strip() or None
        template_days = [
            {
                "id": _template_id(tournament_id, "day", "day-1"),
                "tournament_id": tournament_id,
                "label": "Day 1",
                "event_date": fallback_date,
                "date": fallback_date,
                "enabled": True,
                "sort_order": 1,
            }
        ]
    day_id = str(template_days[0].get("id") or "")
    definitions = [
        ("Men's Doubles", "GENDER_DOUBLES", "MEN", True),
        ("Women's Doubles", "GENDER_DOUBLES", "WOMEN", True),
        ("Mixed Doubles", "MIXED_DOUBLES", "MIXED", True),
        ("Men's Singles", "SINGLES", "MEN", False),
        ("Women's Singles", "SINGLES", "WOMEN", False),
    ]
    event_options: list[dict[str, Any]] = []
    for index, (family, event_type, gender, partner_board) in enumerate(definitions, start=1):
        event_options.append(
            {
                "id": _template_id(tournament_id, "event", family),
                "tournament_id": tournament_id,
                "registration_day_id": day_id,
                "event_family_label": family,
                "division_name": f"{family} Open",
                "event_type": event_type,
                "gender_restriction": gender,
                "event_format_default": "ROUND_ROBIN_PLUS_PLAYOFF",
                "scoring_default": "GAME_TO_15",
                "skill_label": "Open",
                "skill_mode": "OPEN",
                "age_mode": "ALL_AGES",
                "age_label": "All ages",
                "age_rules": "{}",
                "capacity_teams": 16,
                "price_usd": 0,
                "waitlist_enabled": True,
                "partner_board_enabled": partner_board,
                "status": "open",
                "enabled": True,
                "sort_order": index,
            }
        )
    return [
        {
            "key": "standard_doubles_singles",
            "label": "Standard doubles and singles",
            "description": "Five standard open divisions on the first tournament day.",
            "days": template_days,
            "event_families": [],
            "event_options": event_options,
        }
    ]


def build_admin_tournament_setup_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    from jupr_app.services.admin_tournament_guarded_operation import tournament_admin_mutation_status

    if not is_admin_tournament_setup_enabled():
        return {
            "enabled": False,
            "status": "guarded_off",
            "warnings": ["Enable JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS to use Tournament Setup in Next."],
            "streamlit_fallback_url": os.getenv("JUPR_STREAMLIT_FALLBACK_URL", "").strip() or "https://juprtrespalapas.streamlit.app",
            "mutation_runtime": tournament_admin_mutation_status(),
        }
    count = None
    if supabase is not None:
        try:
            count = len(list_existing_tournaments(supabase, str(club_id), include_archived=True))
        except Exception:
            count = None
    return {
        "enabled": True,
        "status": "ready_for_tournament_setup_manager",
        "tournament_count": count,
        "confirmation_text": {
            "create": CONFIRM_CREATE,
            "settings": CONFIRM_SETTINGS,
            "draft": CONFIRM_DRAFT,
            "publish": CONFIRM_PUBLISH,
        },
        "warnings": [],
        "streamlit_fallback_url": os.getenv("JUPR_STREAMLIT_FALLBACK_URL", "").strip() or "https://juprtrespalapas.streamlit.app",
        "mutation_runtime": tournament_admin_mutation_status(),
    }


def list_admin_tournament_setup_tournaments(supabase: Any, *, club_id: str, include_archived: bool = True) -> dict[str, Any]:
    _assert_enabled()
    rows = list_existing_tournaments(supabase, str(club_id), include_archived=include_archived)
    tournaments: list[dict[str, Any]] = []
    for row in rows:
        tournament_id = _clean(row.get("id"), limit=120)
        if not tournament_id:
            continue
        settings = get_registration_settings(supabase, tournament_id, tournament_name=row.get("name"))
        days = list_registration_days(supabase, tournament_id)
        events = list_event_options(supabase, tournament_id)
        tournaments.append(
            {
                "id": tournament_id,
                "name": row.get("name") or tournament_id,
                "status": row.get("status"),
                "start_date": row.get("start_date"),
                "end_date": row.get("end_date"),
                "registration_status": settings.get("registration_status"),
                "registration_slug": settings.get("registration_slug"),
                "day_count": len(days),
                "event_option_count": len(events),
                "registration_count": count_tournament_registrations(supabase, tournament_id),
            }
        )
    return {"ok": True, "mode": "tournament_setup_list", "tournaments": tournaments, "count": len(tournaments)}


def get_admin_tournament_setup_detail(supabase: Any, *, club_id: str, tournament_id: str) -> dict[str, Any]:
    _assert_enabled()
    tournament = _get_tournament_for_club(supabase, club_id=str(club_id), tournament_id=str(tournament_id))
    settings = get_registration_settings(supabase, str(tournament_id), tournament_name=tournament.get("name"))
    days = [_day_payload(row) for row in list_registration_days(supabase, str(tournament_id))]
    events = [_event_option_payload(row) for row in list_event_options(supabase, str(tournament_id))]
    draft = get_builder_draft(supabase, str(tournament_id)) or {"days": days, "event_families": [], "divisions": events}
    draft_days = list(draft.get("days") or days)
    draft_events = list(draft.get("event_options") or draft.get("divisions") or events)
    state_fingerprint = _setup_state_fingerprint(
        tournament=tournament,
        settings=settings,
        days=days,
        events=events,
        draft=draft,
    )
    impact = None
    impact_warning = None
    try:
        impact = analyze_registration_publish_impact(supabase, tournament_id=str(tournament_id), days=draft_days, event_options=draft_events)
    except Exception as exc:  # noqa: BLE001 - setup detail should still render
        impact_warning = str(exc)
    return {
        "ok": True,
        "mode": "tournament_setup_detail",
        "tournament": tournament,
        "settings": _settings_payload(settings),
        "days": days,
        "event_options": events,
        "builder_draft": draft,
        "publish_impact": impact,
        "publish_impact_warning": impact_warning,
        "registration_count": count_tournament_registrations(supabase, str(tournament_id)),
        "state_fingerprint": state_fingerprint,
        "templates": build_admin_tournament_setup_templates(tournament=tournament, days=days),
    }


def review_admin_tournament_setup_impact(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    days: list[dict[str, Any]],
    event_options: list[dict[str, Any]],
    basics: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
    expected_state_fingerprint: str,
) -> dict[str, Any]:
    """Analyze a draft without writing any table."""

    detail = get_admin_tournament_setup_detail(
        supabase,
        club_id=str(club_id),
        tournament_id=str(tournament_id),
    )
    if str(expected_state_fingerprint or "").strip() != str(detail.get("state_fingerprint") or ""):
        from jupr_app.services.admin_tournament_guarded_operation import StaleTournamentAdminStateError

        raise StaleTournamentAdminStateError(
            "Tournament setup changed after it was loaded. Reload before reviewing publish impact."
        )
    normalized_days = list(days or [])
    normalized_events = list(event_options or [])
    impact = analyze_registration_publish_impact(
        supabase,
        tournament_id=str(tournament_id),
        days=normalized_days,
        event_options=normalized_events,
    )
    impact_fingerprint = stable_tournament_admin_fingerprint(
        {
            "tournament_id": str(tournament_id),
            "state_fingerprint": detail["state_fingerprint"],
            "days": normalized_days,
            "event_options": normalized_events,
            "basics": dict(basics or {}),
            "settings": dict(settings or {}),
            "impact": impact,
        }
    )
    return {
        "ok": True,
        "mode": "tournament_setup_impact_review",
        "dry_run": True,
        "write_count": 0,
        "state_fingerprint": detail["state_fingerprint"],
        "impact_fingerprint": impact_fingerprint,
        "publish_impact": impact,
    }


def update_admin_tournament_setup_settings(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    patch: dict[str, Any],
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_tournament_setup_settings",
    dry_run: bool = False,
) -> dict[str, Any]:
    _assert_enabled()
    if _clean(confirmation_text, limit=80).upper() != CONFIRM_SETTINGS:
        raise ValueError(f"Type {CONFIRM_SETTINGS} to save tournament setup settings.")
    tournament = _get_tournament_for_club(supabase, club_id=str(club_id), tournament_id=str(tournament_id))
    before = get_registration_settings(supabase, str(tournament_id), tournament_name=tournament.get("name"))
    allowed = {
        "id",
        "tournament_id",
        "registration_slug",
        "locale",
        "registration_status",
        "registration_open_at",
        "registration_close_at",
        "waitlist_enabled",
        "partner_board_enabled",
        "rules_markdown",
        "refund_policy_markdown",
        "sponsor_markdown",
        "location_name",
        "timezone",
        "sponsors_json",
    }
    payload = {key: patch.get(key) for key in allowed if key in patch}
    payload["id"] = before.get("id") or payload.get("id")
    payload["tournament_id"] = str(tournament_id)
    if payload.get("registration_status") not in (None, ""):
        status = _clean(payload.get("registration_status"), limit=40).lower()
        if status not in REGISTRATION_STATUS_OPTIONS:
            raise ValueError("Invalid registration_status")
        payload["registration_status"] = status
    for key in ("waitlist_enabled", "partner_board_enabled"):
        if key in payload:
            payload[key] = _bool(payload.get(key), default=True)
    if dry_run:
        return {"ok": True, "mode": "tournament_setup_settings_preflight", "dry_run": True, "write_count": 0, "patch": payload}
    updated = upsert_registration_settings(supabase, payload)
    warnings = _audit(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="tournament_setup_settings_update",
        entity_id=str(tournament_id),
        before_json={"settings": _settings_payload(before)},
        after_json={"settings": _settings_payload(updated)},
        source=source,
    )
    return {"ok": True, "mode": "tournament_setup_settings_update", "settings": _settings_payload(updated), "warnings": warnings}


def save_admin_tournament_setup_draft(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    days: list[dict[str, Any]],
    event_families: list[dict[str, Any]],
    event_options: list[dict[str, Any]],
    basics: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
    saved_step: str | None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_tournament_setup_draft",
    dry_run: bool = False,
) -> dict[str, Any]:
    _assert_enabled()
    if _clean(confirmation_text, limit=80).upper() != CONFIRM_DRAFT:
        raise ValueError(f"Type {CONFIRM_DRAFT} to save tournament setup draft.")
    _get_tournament_for_club(supabase, club_id=str(club_id), tournament_id=str(tournament_id))
    before = get_builder_draft(supabase, str(tournament_id))
    if dry_run:
        return {
            "ok": True,
            "mode": "tournament_setup_draft_preflight",
            "dry_run": True,
            "write_count": 0,
            "day_count": len(days or []),
            "event_family_count": len(event_families or []),
            "event_option_count": len(event_options or []),
            "basics_saved": bool(basics),
            "settings_saved": bool(settings),
        }
    draft = save_builder_draft(
        supabase,
        tournament_id=str(tournament_id),
        days=list(days or []),
        event_families=list(event_families or []),
        divisions=list(event_options or []),
        basics=dict(basics or {}),
        settings=dict(settings or {}),
        saved_step=saved_step,
    )
    warnings = _audit(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="tournament_setup_draft_save",
        entity_id=str(tournament_id),
        before_json={"builder_draft": before},
        after_json={"builder_draft": draft},
        source=source,
        flagged=False,
    )
    return {"ok": True, "mode": "tournament_setup_draft_save", "builder_draft": draft, "warnings": warnings}


def publish_admin_tournament_setup(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    days: list[dict[str, Any]],
    event_options: list[dict[str, Any]],
    basics: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    expected_state_fingerprint: str | None = None,
    reviewed_impact_fingerprint: str | None = None,
    source: str = "next_tournament_setup_publish",
    dry_run: bool = False,
) -> dict[str, Any]:
    _assert_enabled()
    if _clean(confirmation_text, limit=80).upper() != CONFIRM_PUBLISH:
        raise ValueError(f"Type {CONFIRM_PUBLISH} to publish tournament setup.")
    _get_tournament_for_club(supabase, club_id=str(club_id), tournament_id=str(tournament_id))
    impact = analyze_registration_publish_impact(supabase, tournament_id=str(tournament_id), days=list(days or []), event_options=list(event_options or []))
    if reviewed_impact_fingerprint:
        detail = get_admin_tournament_setup_detail(
            supabase,
            club_id=str(club_id),
            tournament_id=str(tournament_id),
        )
        if str(expected_state_fingerprint or "") != str(detail.get("state_fingerprint") or ""):
            from jupr_app.services.admin_tournament_guarded_operation import StaleTournamentAdminStateError

            raise StaleTournamentAdminStateError("Tournament setup changed after impact review. Reload and review again.")
        expected_impact_fingerprint = stable_tournament_admin_fingerprint(
            {
                "tournament_id": str(tournament_id),
                "state_fingerprint": detail["state_fingerprint"],
                "days": list(days or []),
                "event_options": list(event_options or []),
                "basics": dict(basics or {}),
                "settings": dict(settings or {}),
                "impact": impact,
            }
        )
        if str(reviewed_impact_fingerprint) != expected_impact_fingerprint:
            raise ValueError("Publish payload does not match the last reviewed impact. Review impact again before publishing.")
    if impact.get("blocked"):
        raise ValueError("Publish blocked due to destructive changes: " + " | ".join(str(x) for x in impact.get("blocked") or []))
    if dry_run:
        return {
            "ok": True,
            "mode": "tournament_setup_publish_preflight",
            "dry_run": True,
            "write_count": 0,
            "publish_impact": impact,
            "basics": dict(basics or {}),
            "settings": dict(settings or {}),
        }

    clean_basics = dict(basics or {})
    tournament_patch: dict[str, Any] = {}
    for source_key, target_key in (("name", "name"), ("start_date", "start_date"), ("end_date", "end_date")):
        if source_key in clean_basics:
            tournament_patch[target_key] = clean_basics.get(source_key)
    if tournament_patch:
        tournament_patch["updated_at"] = datetime.now(timezone.utc).isoformat()
        response = (
            supabase.table("tournaments")
            .update(tournament_patch)
            .eq("id", str(tournament_id))
            .eq("club_id", str(club_id))
            .execute()
        )
        if not _safe_rows(response):
            raise RuntimeError("Tournament basics were not published.")

    clean_settings = dict(settings or {})
    for basics_key in ("location_name", "timezone", "sponsors_json"):
        if basics_key in clean_basics:
            clean_settings[basics_key] = clean_basics.get(basics_key)
    # Registration status is deliberately controlled by the separate Open/Close
    # Registration action on the final review page.
    clean_settings.pop("registration_status", None)
    if clean_settings:
        before_settings = get_registration_settings(
            supabase, str(tournament_id)
        )
        clean_settings["id"] = before_settings.get("id")
        clean_settings["tournament_id"] = str(tournament_id)
        upsert_registration_settings(supabase, clean_settings)

    result = publish_registration_configuration(supabase, tournament_id=str(tournament_id), days=list(days or []), event_options=list(event_options or []))
    clear_builder_draft(supabase, str(tournament_id))
    warnings = _audit(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="tournament_setup_publish",
        entity_id=str(tournament_id),
        before_json={"impact": impact},
        after_json={
            "result": result,
            "day_count": len(days or []),
            "event_option_count": len(event_options or []),
            "basics": dict(basics or {}),
            "settings": dict(settings or {}),
        },
        source=source,
    )
    return {
        "ok": True,
        "mode": "tournament_setup_publish",
        "publish_result": result,
        "publish_impact": impact,
        "days": [_day_payload(row) for row in list_registration_days(supabase, str(tournament_id))],
        "event_options": [_event_option_payload(row) for row in list_event_options(supabase, str(tournament_id))],
        "warnings": [*warnings, *list(result.get("warnings") or [])],
    }
