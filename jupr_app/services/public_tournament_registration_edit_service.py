from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

from jupr_app.config import get_env_or_default, get_explicit_registration_edit_token_secret, get_public_base_url
from jupr_app.domain.notifications.tournament_registration_edit_email import send_tournament_registration_edit_email
from jupr_app.domain.tournament_registration_edit_tokens import build_registration_edit_token, verify_registration_edit_token
from jupr_app.domain.tournament_registration_repo import (
    TournamentRegistrationEditConflictError,
    TournamentRegistrationImportedDrawError,
    TournamentRegistrationRelationshipLockedError,
    get_registration_by_email,
    get_registration_confirmation_bundle,
    registration_has_imported_draw_selection,
    save_registration,
)
from jupr_app.services.public_tournament_registration_service import (
    _clean_email,
    _clean_text,
    _get_club_player,
    _public_day,
    _public_event,
    _public_registration_player,
    _safe_bool,
    _safe_float,
    _safe_int,
    build_registration_confirmation_delivery,
    build_validated_public_registration_save_payload,
    build_public_tournament_registration_page,
)
from jupr_app.services.public_tournament_commerce_service import (
    build_public_tournament_commerce_catalog,
    build_public_tournament_commerce_order,
    is_tournament_commerce_enabled,
    prepare_public_registration_commerce_transaction,
    require_tournament_commerce_mutation_runtime,
)
from jupr_app.services.admin_guarded_write_service import (
    GuardedWriteRecoveryRequired,
    begin_guarded_operation,
    update_guarded_operation,
)


class PublicRegistrationEditUnavailableError(RuntimeError):
    """Raised when the public edit surface fails a configuration preflight."""


def _registration_public_payload(registration: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(registration.get("id") or ""),
        "first_name": _clean_text(registration.get("first_name"), limit=80),
        "last_name": _clean_text(registration.get("last_name"), limit=80),
        "display_name": _clean_text(registration.get("display_name"), limit=160),
        "email": _clean_email(registration.get("email")),
        "phone": _clean_text(registration.get("phone"), limit=60),
        "player_id": registration.get("player_id"),
        "dupr_id": _clean_text(registration.get("dupr_id"), limit=80),
        "doubles_skill": _safe_float(registration.get("doubles_skill")),
        "singles_skill": _safe_float(registration.get("singles_skill")),
        "age": _safe_int(registration.get("age")),
        "gender": _clean_text(registration.get("gender"), limit=40),
        "notes": _clean_text(registration.get("notes"), limit=800),
        "wants_partner_board_contact": _safe_bool(registration.get("wants_partner_board_contact")),
        "status": _clean_text(registration.get("status"), limit=40),
        "payment_status": _clean_text(registration.get("payment_status"), limit=40),
        "submitted_at": registration.get("submitted_at"),
        "updated_at": registration.get("updated_at"),
    }


def _selection_public_payload(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(selection.get("id") or ""),
        "event_option_id": str(selection.get("event_option_id") or ""),
        "registration_day_id": str(selection.get("registration_day_id") or ""),
        "partner_mode": _clean_text(selection.get("partner_mode") or "NONE", limit=40).upper(),
        "partner_name": _clean_text(selection.get("partner_name"), limit=160),
        "partner_email": _clean_email(selection.get("partner_email")),
        "partner_phone": _clean_text(selection.get("partner_phone"), limit=60),
        "partner_dupr_id": _clean_text(selection.get("partner_dupr_id"), limit=80),
        "partner_skill": _safe_float(selection.get("partner_skill")),
        "partner_age": _safe_int(selection.get("partner_age")),
        "partner_gender": _clean_text(selection.get("partner_gender"), limit=40),
        "partner_note": _clean_text(selection.get("partner_note"), limit=500),
        "show_on_partner_board": _safe_bool(selection.get("show_on_partner_board")),
        "updated_at": selection.get("updated_at"),
    }


def _stable_edit_secret() -> str:
    try:
        return get_explicit_registration_edit_token_secret()
    except ValueError as exc:
        raise PublicRegistrationEditUnavailableError(
            "Registration changes are temporarily unavailable. Please try again later."
        ) from exc


def _verified_bundle(
    supabase: Any,
    *,
    club_id: str,
    edit_token: str,
    tournament_id: str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    secret = _stable_edit_secret()
    expected_tournament_id = _clean_text(tournament_id, limit=120) or None
    verified = verify_registration_edit_token(
        edit_token,
        expected_tournament_id=expected_tournament_id,
        secret=secret,
    )
    tid = str(verified.get("tournament_id") or "").strip()
    registration_id = str(verified.get("registration_id") or "").strip()
    if not tid or not registration_id:
        raise ValueError("Invalid registration edit link.")
    bundle = get_registration_confirmation_bundle(supabase, tid, registration_id)
    registration = bundle.get("registration") or None
    tournament = bundle.get("tournament") or {}
    if not registration:
        raise ValueError("Registration was not found.")
    if str(tournament.get("club_id") or club_id) != str(club_id):
        raise ValueError("Registration edit link is for a different club.")
    verify_registration_edit_token(
        edit_token,
        expected_tournament_id=tid,
        expected_registration_id=registration_id,
        expected_email=_clean_email(registration.get("email")),
        secret=secret,
    )
    return verified, bundle


def _public_web_base_url(public_base_url: str | None = None) -> str:
    for candidate in (
        public_base_url,
        get_env_or_default("JUPR_WEB_BASE_URL"),
        get_env_or_default("STAGING_WEB_BASE_URL"),
        get_env_or_default("NEXT_PUBLIC_JUPR_WEB_BASE_URL"),
        get_env_or_default("JUPR_PUBLIC_BASE_URL"),
        get_public_base_url(),
    ):
        value = str(candidate or "").strip().rstrip("/")
        if value:
            return value
    return ""


def _edit_url(*, club_slug: str, edit_token: str, tournament_id: str, registration_slug: str | None, public_base_url: str | None = None) -> str:
    query: dict[str, str] = {"edit_token": str(edit_token)}
    if _clean_text(registration_slug, limit=120):
        query["tournament"] = _clean_text(registration_slug, limit=120)
    else:
        query["tournament_id"] = str(tournament_id)
    return f"{_public_web_base_url(public_base_url)}/clubs/{club_slug}/tournament-registration/edit?{urlencode(query)}"


def _generic_edit_link_response() -> dict[str, Any]:
    return {
        "ok": True,
        "mode": "registration_edit_link_request",
        "accepted": True,
        "message": "If that email matches a registration, we’ll send the edit link there.",
    }


def _require_token_bound_registration_slug(bundle: dict[str, Any], registration_slug: Any) -> None:
    """Reject routing hints that do not belong to the token-bound tournament."""

    supplied_slug = _clean_text(registration_slug, limit=120)
    expected_slug = _clean_text((bundle.get("settings") or {}).get("registration_slug"), limit=120)
    if supplied_slug and supplied_slug != expected_slug:
        raise ValueError("Registration edit link is for a different tournament.")


def _event_family_key(event: dict[str, Any]) -> tuple[str, str]:
    day_id = str(event.get("registration_day_id") or "").strip()
    family = _clean_text(
        event.get("event_family_label") or event.get("label") or "Event",
        limit=160,
    )
    return day_id, re.sub(r"\s+", " ", family).strip().lower()


def _versioned_edit_selections(
    *,
    bundle: dict[str, Any],
    payload: dict[str, Any],
    validated_selections: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    existing = [dict(row) for row in (bundle.get("selections") or [])]
    expected_raw = payload.get("expected_selection_versions")
    if not isinstance(expected_raw, list):
        raise TournamentRegistrationEditConflictError(
            "Registration changed after it was loaded. Refresh the edit link and try again."
        )
    expected_versions = [
        {
            "id": str(row.get("id") or "").strip(),
            "updated_at": str(row.get("updated_at") or "").strip(),
        }
        for row in expected_raw
        if isinstance(row, dict)
    ]
    current_versions = {
        str(row.get("id") or "").strip(): str(row.get("updated_at") or "").strip()
        for row in existing
    }
    supplied_versions = {row["id"]: row["updated_at"] for row in expected_versions if row["id"]}
    if supplied_versions != current_versions or len(expected_versions) != len(current_versions):
        raise TournamentRegistrationEditConflictError(
            "Registration changed after it was loaded. Refresh the edit link and try again."
        )

    events = {str(row.get("id") or ""): row for row in (bundle.get("event_options") or [])}
    existing_by_id = {str(row.get("id") or ""): row for row in existing}
    existing_by_event = {str(row.get("event_option_id") or ""): row for row in existing}
    existing_by_family = {
        _event_family_key(events.get(str(row.get("event_option_id") or "")) or {}): row
        for row in existing
    }
    raw_by_event = {
        str(row.get("event_option_id") or ""): row
        for row in (payload.get("selections") or [])
        if isinstance(row, dict)
    }
    used_ids: set[str] = set()
    versioned: list[dict[str, Any]] = []
    for selection in validated_selections:
        event_id = str(selection.get("event_option_id") or "")
        raw = raw_by_event.get(event_id) or {}
        requested_id = str(raw.get("id") or "").strip()
        candidate = existing_by_id.get(requested_id) if requested_id else None
        target_family = _event_family_key(events.get(event_id) or {})
        if candidate:
            candidate_event = events.get(str(candidate.get("event_option_id") or "")) or {}
            if _event_family_key(candidate_event) != target_family:
                raise ValueError("The selected event doesn’t match this registration entry.")
        elif requested_id:
            raise TournamentRegistrationEditConflictError(
                "Registration changed after it was loaded. Refresh the edit link and try again."
            )
        if candidate is None:
            candidate = existing_by_event.get(event_id) or existing_by_family.get(target_family)
        selection_id = str((candidate or {}).get("id") or f"sel_{uuid.uuid4().hex}")
        if selection_id in used_ids:
            raise ValueError("The same registration selection cannot be used more than once.")
        used_ids.add(selection_id)
        versioned.append({**selection, "id": selection_id})
    return versioned, expected_versions


def request_public_tournament_registration_edit_link(
    supabase: Any,
    *,
    club_id: str,
    club_slug: str,
    email: str,
    tournament_id: str | None = None,
    registration_slug: str | None = None,
    idempotency_key: str = "legacy-edit-link-request",
    website: str | None = None,
    public_base_url: str | None = None,
) -> dict[str, Any]:
    if _clean_text(website, limit=200):
        return _generic_edit_link_response()
    secret = _stable_edit_secret()
    clean_email = _clean_email(email)
    if not clean_email or "@" not in clean_email:
        raise ValueError("A valid email is required.")
    page = build_public_tournament_registration_page(
        supabase,
        club_id=str(club_id),
        tournament_id=_clean_text(tournament_id, limit=120) or None,
        registration_slug=_clean_text(registration_slug, limit=120) or None,
    )
    if not page.get("available"):
        raise ValueError(str(page.get("setup_error") or "Tournament registration is not configured."))
    tournament = page.get("tournament") or {}
    settings = page.get("settings") or {}
    tid = str(tournament.get("id") or "").strip()
    if not tid:
        raise ValueError("Tournament registration was not found.")
    registration = get_registration_by_email(supabase, tid, clean_email)
    if not registration:
        return _generic_edit_link_response()
    registration_id = str(registration.get("id") or "").strip()
    if registration_has_imported_draw_selection(
        supabase,
        tournament_id=tid,
        registration_id=registration_id,
    ):
        return _generic_edit_link_response()
    # Dedupe by recipient/tournament in a short server-side bucket as well as by
    # the caller's request. A user can double-click, retry after a dropped
    # response, or generate a new browser key without triggering another email.
    # The opaque operation key never exposes the recipient address.
    bucket = int(datetime.now(timezone.utc).timestamp() // (15 * 60))
    operation_scope = "\x1f".join((str(club_id), tid, registration_id, clean_email, str(bucket)))
    delivery_operation_key = "editlink:" + hashlib.sha256(operation_scope.encode("utf-8")).hexdigest()
    try:
        operation, idempotent = begin_guarded_operation(
            supabase,
            club_id=str(club_id),
            workflow="public_tournament_edit_link_delivery",
            action="send_tournament_registration_edit_link",
            operation_key=delivery_operation_key,
            request_payload={
                "tournament_id": tid,
                "registration_id": registration_id,
                "recipient_fingerprint": hashlib.sha256(clean_email.encode("utf-8")).hexdigest(),
                "delivery_bucket": bucket,
            },
            actor_email="public-registration@system.invalid",
            actor_role="public_registration",
            source="next_public_tournament_registration_edit_link",
            before_json={"client_request_key_fingerprint": hashlib.sha256(str(idempotency_key).encode("utf-8")).hexdigest()},
        )
        if idempotent:
            return _generic_edit_link_response()
    except (GuardedWriteRecoveryRequired, RuntimeError, ValueError):
        # Preserve the anti-enumeration response and suppress a second delivery
        # whenever durable delivery state cannot prove that sending is safe.
        return _generic_edit_link_response()
    token = build_registration_edit_token(
        tournament_id=tid,
        registration_id=registration_id,
        email=clean_email,
        secret=secret,
    )
    try:
        send_tournament_registration_edit_email(
            tournament_name=_clean_text(tournament.get("name") or "Tournament"),
            registered_email=clean_email,
            edit_url=_edit_url(
                club_slug=str(club_slug),
                edit_token=token,
                tournament_id=tid,
                registration_slug=_clean_text(settings.get("registration_slug"), limit=120) or _clean_text(registration_slug, limit=120) or None,
                public_base_url=public_base_url,
            ),
        )
    except Exception as exc:
        try:
            update_guarded_operation(
                supabase,
                operation_id=operation.get("id"),
                operation_key=delivery_operation_key,
                status="recovery_required",
                error_text=f"Edit-link provider outcome is uncertain: {exc.__class__.__name__}",
            )
        except GuardedWriteRecoveryRequired:
            pass
        return _generic_edit_link_response()
    try:
        update_guarded_operation(
            supabase,
            operation_id=operation.get("id"),
            operation_key=delivery_operation_key,
            status="completed",
            result_json=_generic_edit_link_response(),
            after_json={"delivery": "accepted", "delivery_bucket": bucket},
        )
    except GuardedWriteRecoveryRequired:
        # The provider may already have accepted the message. Never send again
        # just because the response/receipt write was interrupted.
        return _generic_edit_link_response()
    return _generic_edit_link_response()


def build_public_tournament_registration_edit_page(
    supabase: Any,
    *,
    club_id: str,
    edit_token: str,
    tournament_id: str | None = None,
    registration_slug: str | None = None,
) -> dict[str, Any]:
    verified, bundle = _verified_bundle(
        supabase,
        club_id=str(club_id),
        edit_token=edit_token,
        tournament_id=tournament_id,
    )
    tid = str(verified.get("tournament_id") or "")
    _require_token_bound_registration_slug(bundle, registration_slug)
    page = build_public_tournament_registration_page(
        supabase,
        club_id=str(club_id),
        tournament_id=tid,
        registration_slug=None,
    )
    if not page.get("available"):
        raise ValueError(str(page.get("setup_error") or "Tournament registration is not configured."))
    if not page.get("tournament"):
        raise ValueError("Tournament registration was not found.")

    registration = bundle.get("registration") or {}
    if registration_has_imported_draw_selection(
        supabase,
        tournament_id=tid,
        registration_id=str(registration.get("id") or ""),
    ):
        raise TournamentRegistrationImportedDrawError(
            "This registration is already imported into a draw and can no longer be edited publicly."
        )
    linked_player_id = registration.get("player_id")
    linked_player = (
        _get_club_player(
            supabase,
            club_id=str(club_id),
            player_id=linked_player_id,
            require_active=False,
        )
        if linked_player_id not in (None, "")
        else None
    )
    page["players"] = [_public_registration_player(linked_player)] if linked_player else []
    if is_tournament_commerce_enabled():
        page["commerce"] = build_public_tournament_commerce_catalog(
            supabase,
            club_id=str(club_id),
            tournament_id=tid,
            registration_id=str(registration.get("id") or ""),
            token_bound_edit=True,
        )
        page["commerce_order"] = build_public_tournament_commerce_order(
            supabase,
            club_id=str(club_id),
            tournament_id=tid,
            registration_id=str(registration.get("id") or ""),
        )

    # Existing selections remain visible and may be preserved even when an
    # organizer safely closes or disables that division after registration.
    # They are not made selectable for anyone else.
    selected_event_ids = {
        str(selection.get("event_option_id") or "")
        for selection in (bundle.get("selections") or [])
        if str(selection.get("event_option_id") or "")
    }
    page_event_ids = {str(event.get("id") or "") for event in (page.get("events") or [])}
    missing_events = [
        event
        for event in (bundle.get("event_options") or [])
        if str(event.get("id") or "") in selected_event_ids and str(event.get("id") or "") not in page_event_ids
    ]
    if missing_events:
        page["events"] = [
            *(page.get("events") or []),
            *[_public_event(event, registration_open=bool(page.get("registration_open"))) for event in missing_events],
        ]
    selected_day_ids = {
        str(event.get("registration_day_id") or "")
        for event in (bundle.get("event_options") or [])
        if str(event.get("id") or "") in selected_event_ids
    }
    page_day_ids = {str(day.get("id") or "") for day in (page.get("days") or [])}
    missing_days = [
        day
        for day in (bundle.get("days") or [])
        if str(day.get("id") or "") in selected_day_ids and str(day.get("id") or "") not in page_day_ids
    ]
    if missing_days:
        page["days"] = [*(page.get("days") or []), *[_public_day(day) for day in missing_days]]
    return {
        **page,
        "edit_mode": True,
        "edit_token_valid": True,
        "edit_token_expires_at": verified.get("exp"),
        "registration": _registration_public_payload(registration),
        "selections": [_selection_public_payload(selection) for selection in (bundle.get("selections") or [])],
        "total_price_usd": float(bundle.get("total_price_usd") or 0),
    }


def submit_public_tournament_registration_edit(
    supabase: Any,
    *,
    club_id: str,
    club_slug: str | None = None,
    edit_token: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    verified, bundle = _verified_bundle(
        supabase,
        club_id=str(club_id),
        edit_token=edit_token,
        tournament_id=_clean_text(payload.get("tournament_id"), limit=120) or None,
    )
    tournament = bundle.get("tournament") or {}
    settings = bundle.get("settings") or {}
    registration = bundle.get("registration") or {}
    tournament_id = str(verified.get("tournament_id") or "").strip()
    registration_id = str(verified.get("registration_id") or "").strip()
    _require_token_bound_registration_slug(bundle, payload.get("registration_slug"))
    if registration_has_imported_draw_selection(
        supabase,
        tournament_id=tournament_id,
        registration_id=registration_id,
    ):
        raise TournamentRegistrationImportedDrawError(
            "This registration is already imported into a draw and can no longer be edited publicly."
        )
    expected_updated_at = _clean_text(payload.get("expected_updated_at"), limit=80)
    current_updated_at = str(registration.get("updated_at") or "").strip()
    if not expected_updated_at or expected_updated_at != current_updated_at:
        raise TournamentRegistrationEditConflictError(
            "Registration changed after it was loaded. Refresh the edit link and try again."
        )
    page = build_public_tournament_registration_page(
        supabase,
        club_id=str(club_id),
        tournament_id=tournament_id,
        registration_slug=None,
    )
    if not page.get("registration_open"):
        raise ValueError(str(page.get("registration_closed_reason") or "Registration is not open."))
    existing_selected_ids = {
        str(selection.get("event_option_id") or "")
        for selection in (bundle.get("selections") or [])
        if str(selection.get("event_option_id") or "")
    }
    existing_event_options = {
        str(event.get("id") or ""): _public_event(event, registration_open=bool(page.get("registration_open")))
        for event in (bundle.get("event_options") or [])
        if str(event.get("id") or "") in existing_selected_ids
    }
    save_payload = build_validated_public_registration_save_payload(
        supabase,
        club_id=str(club_id),
        tournament_id=tournament_id,
        page=page,
        payload=payload,
        locked_registration=registration,
        existing_event_options=existing_event_options,
    )
    # The public edit form does not expose the derived/organizer-managed age
    # bracket, so retain it instead of clearing it on every edit.
    save_payload["age_bracket"] = registration.get("age_bracket")
    versioned_selections, expected_selection_versions = _versioned_edit_selections(
        bundle=bundle,
        payload=payload,
        validated_selections=list(save_payload.get("selections") or []),
    )
    save_payload["selections"] = versioned_selections
    commerce_transaction = None
    if (
        isinstance(page.get("commerce"), dict)
        and page["commerce"].get("available")
    ):
        require_tournament_commerce_mutation_runtime(
            actor_type="PUBLIC_REGISTRANT"
        )
        commerce_transaction = (
            prepare_public_registration_commerce_transaction(
                supabase,
                club_id=str(club_id),
                tournament_id=tournament_id,
                registration_id=registration_id,
                registration_email=_clean_email(registration.get("email")),
                event_option_ids=[
                    str(row.get("event_option_id") or "")
                    for row in versioned_selections
                    if row.get("event_option_id")
                ],
                commerce=payload.get("commerce"),
                edit_mode=True,
            )
        )
        if commerce_transaction is not None:
            commerce_transaction.update(
                {
                    "club_id": str(club_id),
                    "source": "next_public_tournament_registration_edit",
                }
            )
    result = save_registration(
        supabase,
        tournament_id=tournament_id,
        payload=save_payload,
        expected_registration_id=registration_id,
        allow_existing_unselectable_event_ids=existing_selected_ids,
        expected_updated_at=expected_updated_at,
        expected_selection_versions=expected_selection_versions,
        atomic_edit=True,
        commerce_transaction=commerce_transaction,
    )
    delivery = build_registration_confirmation_delivery(
        supabase,
        club_id=str(club_id),
        club_slug=str(club_slug or club_id),
        tournament_id=tournament_id,
        registration_id=str(result.get("registration_id") or ""),
        send_email=not bool(result.get("idempotent_replay")),
    )
    delivery_status = str(
        (delivery.get("email_delivery") or {}).get("status") or "unknown"
    ).strip().lower()
    if delivery_status not in {
        "sent",
        "staging_redirect",
        "dry_run",
        "failed",
        "already_completed",
    }:
        delivery_status = "unknown"
    confirmation_delivery = {
        "status": delivery_status,
        "delivered": delivery_status in {"sent", "staging_redirect"},
    }
    return {
        "ok": True,
        "mode": "registration_edit",
        "tournament": {
            "id": str(tournament.get("id") or tournament_id),
            "name": _clean_text(tournament.get("name") or "Tournament"),
            "status": _clean_text(tournament.get("status")),
            "start_date": tournament.get("start_date"),
            "end_date": tournament.get("end_date"),
        },
        "settings": {
            "registration_slug": _clean_text(settings.get("registration_slug"), limit=120),
            "registration_status": _clean_text(settings.get("registration_status") or "draft", limit=40).lower(),
        },
        "registration_id": result.get("registration_id"),
        "submitted_at": result.get("submitted_at"),
        "updated_at": result.get("updated_at"),
        "selection_count": result.get("selection_count"),
        "commerce_order": result.get("commerce_order"),
        "confirmation_delivery": confirmation_delivery,
        **delivery,
    }
