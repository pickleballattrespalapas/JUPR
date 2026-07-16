from __future__ import annotations

import os
from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.tournament_registration_repo import (
    REGISTRATION_STATUS_OPTIONS,
    analyze_registration_publish_impact,
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
        "sponsor_markdown": row.get("sponsor_markdown"),
        "updated_at": row.get("updated_at"),
    }


def _event_option_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "tournament_id": row.get("tournament_id"),
        "registration_day_id": row.get("registration_day_id"),
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
        "notes": row.get("notes"),
    }


def _day_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "tournament_id": row.get("tournament_id"),
        "label": row.get("label"),
        "date": row.get("date"),
        "start_date": row.get("start_date"),
        "end_date": row.get("end_date"),
        "enabled": row.get("enabled"),
        "sort_order": row.get("sort_order"),
    }


def build_admin_tournament_setup_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    if not is_admin_tournament_setup_enabled():
        return {
            "enabled": False,
            "status": "guarded_off",
            "warnings": ["Enable JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS to use Tournament Setup in Next."],
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
        "confirmation_text": {"settings": CONFIRM_SETTINGS, "draft": CONFIRM_DRAFT, "publish": CONFIRM_PUBLISH},
        "warnings": [],
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
    saved_step: str | None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_tournament_setup_draft",
) -> dict[str, Any]:
    _assert_enabled()
    if _clean(confirmation_text, limit=80).upper() != CONFIRM_DRAFT:
        raise ValueError(f"Type {CONFIRM_DRAFT} to save tournament setup draft.")
    _get_tournament_for_club(supabase, club_id=str(club_id), tournament_id=str(tournament_id))
    before = get_builder_draft(supabase, str(tournament_id))
    draft = save_builder_draft(
        supabase,
        tournament_id=str(tournament_id),
        days=list(days or []),
        event_families=list(event_families or []),
        divisions=list(event_options or []),
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
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_tournament_setup_publish",
) -> dict[str, Any]:
    _assert_enabled()
    if _clean(confirmation_text, limit=80).upper() != CONFIRM_PUBLISH:
        raise ValueError(f"Type {CONFIRM_PUBLISH} to publish tournament setup.")
    _get_tournament_for_club(supabase, club_id=str(club_id), tournament_id=str(tournament_id))
    impact = analyze_registration_publish_impact(supabase, tournament_id=str(tournament_id), days=list(days or []), event_options=list(event_options or []))
    if impact.get("blocked"):
        raise ValueError("Publish blocked due to destructive changes: " + " | ".join(str(x) for x in impact.get("blocked") or []))
    result = publish_registration_configuration(supabase, tournament_id=str(tournament_id), days=list(days or []), event_options=list(event_options or []))
    warnings = _audit(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="tournament_setup_publish",
        entity_id=str(tournament_id),
        before_json={"impact": impact},
        after_json={"result": result, "day_count": len(days or []), "event_option_count": len(event_options or [])},
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
