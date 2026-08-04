"""Manager-only tournament commerce catalog and order operations."""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Any, Mapping

from jupr_app.domain.admin_activity_log import (
    build_activity_payload,
    write_admin_activity_log,
)
from jupr_app.domain.tournament_commerce import (
    TournamentCommerceValidationError,
    normalize_tournament_commerce_catalog,
    quote_tournament_commerce,
    stable_fingerprint,
    tournament_commerce_catalog_payload,
)
from jupr_app.services.public_tournament_commerce_service import (
    TournamentCommerceConflictError,
    TournamentCommerceUnavailableError,
    _batched_rows,
    _canonical_uuid,
    _clean_text,
    _current_order,
    _execute,
    _load_catalog,
    _paged_rows,
    _registration_position,
    _registration_rows,
    _raise_rpc_error,
    _rpc_result,
    is_tournament_commerce_enabled,
    require_tournament_commerce_mutation_runtime,
    tournament_commerce_runtime_status,
)


FULFILLMENT_CORRECTION_NOTE_MIN_LENGTH = 8


class TournamentCommerceRecoveryRequiredError(RuntimeError):
    """The domain RPC completed but its secondary admin audit is unavailable."""


def build_admin_tournament_commerce_status(
    supabase: Any | None = None, *, club_id: str | None = None
) -> dict[str, Any]:
    runtime = tournament_commerce_runtime_status()
    storage_ready: bool | None = None
    if supabase is not None and club_id:
        try:
            _execute(
                supabase.table("tournament_commerce_catalog_state")
                .select("tournament_id")
                .eq("club_id", str(club_id))
                .limit(1),
                label="status",
            )
            storage_ready = True
        except TournamentCommerceUnavailableError:
            storage_ready = False
    return {
        "available": is_tournament_commerce_enabled(),
        "offline_payment_only": True,
        "storage_ready": storage_ready,
        "runtime": runtime,
        "capabilities": {
            "catalog": True,
            "inventory": True,
            "bundles": True,
            "date_window_giveaways": True,
            "first_n_registrants": True,
            "first_n_claims": True,
            "payment_tracking": True,
            "fulfillment": True,
            "recovery_inspector": True,
        },
    }


def list_admin_tournament_commerce_tournaments(
    supabase: Any, *, club_id: str, include_archived: bool = True
) -> dict[str, Any]:
    rows = _paged_rows(
        supabase,
        "tournaments",
        equals={"club_id": str(club_id)},
    )
    if not include_archived:
        rows = [
            row
            for row in rows
            if str(row.get("status") or "").upper() != "ARCHIVED"
        ]
    rows.sort(
        key=lambda row: (
            str(row.get("start_date") or ""),
            str(row.get("name") or "").lower(),
            str(row.get("id") or ""),
        ),
        reverse=True,
    )
    return {
        "ok": True,
        "tournaments": [
            {
                "id": str(row.get("id") or ""),
                "name": _clean_text(row.get("name") or "Tournament", limit=180),
                "status": _clean_text(row.get("status"), limit=40),
                "start_date": row.get("start_date"),
                "end_date": row.get("end_date"),
            }
            for row in rows
        ],
    }


def _latest_rows(
    supabase: Any,
    table: str,
    *,
    club_id: str,
    tournament_id: str,
    limit: int = 200,
) -> list[dict[str, Any]]:
    return _execute(
        supabase.table(table)
        .select("*")
        .eq("club_id", str(club_id))
        .eq("tournament_id", str(tournament_id))
        .order("created_at", desc=True)
        .limit(limit),
        label=table.replace("_", " "),
    )


def get_admin_tournament_commerce_detail(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
) -> dict[str, Any]:
    tournament_rows = _execute(
        supabase.table("tournaments")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("id", str(tournament_id))
        .limit(1),
        label="tournament",
    )
    if not tournament_rows:
        raise TournamentCommerceValidationError("Tournament was not found.")
    tournament = tournament_rows[0]
    state, raw_catalog = _load_catalog(
        supabase,
        club_id=str(club_id),
        tournament_id=str(tournament_id),
    )
    if state:
        catalog = normalize_tournament_commerce_catalog(raw_catalog)
    else:
        catalog = {
            "currency": "USD",
            "catalog_revision": 0,
            "catalog_fingerprint": "",
            "items": [],
            "variants": [],
            "bundles": [],
            "bundle_components": [],
            "promotions": [],
            "event_options": raw_catalog.get("event_options") or [],
        }

    orders = _paged_rows(
        supabase,
        "tournament_commerce_orders",
        equals={
            "club_id": str(club_id),
            "tournament_id": str(tournament_id),
        },
    )
    order_ids = [str(row["id"]) for row in orders if row.get("id")]
    registration_ids = [
        str(row["registration_id"])
        for row in orders
        if row.get("registration_id")
    ]
    registrations = _batched_rows(
        supabase,
        "tournament_registrations",
        filter_column="id",
        values=registration_ids,
        equals={"tournament_id": str(tournament_id)},
    )
    registrations_by_id = {
        str(row["id"]): row for row in registrations if row.get("id")
    }
    revisions = _batched_rows(
        supabase,
        "tournament_commerce_order_revisions",
        filter_column="order_id",
        values=order_ids,
    )
    current_revision_ids = {
        str(revision["id"])
        for order in orders
        for revision in revisions
        if str(revision.get("order_id") or "") == str(order.get("id") or "")
        and int(revision.get("revision") or 0)
        == int(order.get("current_revision") or 0)
        and revision.get("id")
    }
    lines = [
        row
        for row in _batched_rows(
            supabase,
            "tournament_commerce_order_lines",
            filter_column="order_id",
            values=order_ids,
        )
        if bool(row.get("active"))
        and str(row.get("revision_id") or "") in current_revision_ids
    ]
    active_line_ids = {
        str(row["id"]) for row in lines if row.get("id")
    }
    fulfillment = [
        row
        for row in _paged_rows(
            supabase,
            "tournament_commerce_fulfillment",
            equals={"tournament_id": str(tournament_id)},
        )
        if str(row.get("order_id") or "") in order_ids
        and str(row.get("order_line_id") or "") in active_line_ids
    ]
    for order in orders:
        registration = registrations_by_id.get(
            str(order.get("registration_id") or "")
        ) or {}
        order["registration"] = {
            "id": str(registration.get("id") or ""),
            "display_name": _clean_text(
                registration.get("display_name") or "Registrant", limit=180
            ),
            "email": _clean_text(registration.get("email"), limit=320),
            "status": _clean_text(registration.get("status"), limit=40),
        }
        order["lines"] = [
            row
            for row in lines
            if str(row.get("order_id") or "") == str(order.get("id") or "")
        ]
        order["fulfillment"] = [
            row
            for row in fulfillment
            if str(row.get("order_id") or "") == str(order.get("id") or "")
        ]
    fulfillment_by_order = {
        str(order["id"]): order.get("registration") or {}
        for order in orders
        if order.get("id")
    }
    for row in fulfillment:
        row["registration"] = fulfillment_by_order.get(
            str(row.get("order_id") or ""), {}
        )

    operations = _latest_rows(
        supabase,
        "tournament_commerce_operations",
        club_id=str(club_id),
        tournament_id=str(tournament_id),
    )
    audit = _latest_rows(
        supabase,
        "tournament_commerce_audit_log",
        club_id=str(club_id),
        tournament_id=str(tournament_id),
    )
    return {
        "ok": True,
        "tournament": {
            "id": str(tournament.get("id") or ""),
            "name": _clean_text(tournament.get("name") or "Tournament", limit=180),
            "status": _clean_text(tournament.get("status"), limit=40),
            "start_date": tournament.get("start_date"),
            "end_date": tournament.get("end_date"),
        },
        "catalog": {
            "currency": "USD",
            "catalog_revision": int(
                (state or {}).get("catalog_revision") or 0
            ),
            "catalog_fingerprint": str(
                (state or {}).get("catalog_fingerprint") or ""
            ),
            "items": catalog.get("items") or [],
            "variants": catalog.get("variants") or [],
            "bundles": catalog.get("bundles") or [],
            "bundle_components": catalog.get("bundle_components") or [],
            "promotions": catalog.get("promotions") or [],
            "event_options": catalog.get("event_options") or [],
        },
        "orders": orders,
        "fulfillment": fulfillment,
        "operations": operations,
        "audit": audit,
        "runtime": tournament_commerce_runtime_status(),
        "offline_payment_only": True,
    }


def _shared_admin_audit_present(
    supabase: Any,
    *,
    club_id: str,
    operation_id: Any,
) -> bool | None:
    clean_operation_id = _clean_text(operation_id, limit=80)
    if not clean_operation_id:
        return None
    try:
        response = (
            supabase.table("admin_activity_log")
            .select("id")
            .eq("club_id", str(club_id))
            .eq("entity_type", "tournament_commerce")
            .filter("after_json->>operation_id", "eq", clean_operation_id)
            .limit(1)
            .execute()
        )
    except Exception:
        return None
    return bool(getattr(response, "data", None))


def _audit_admin_result(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    actor_email: str,
    actor_role: str,
    action: str,
    result: dict[str, Any],
    source: str,
) -> None:
    operation_id = result.get("operation_id")
    if _shared_admin_audit_present(
        supabase, club_id=str(club_id), operation_id=operation_id
    ):
        return
    payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email),
        actor_role=str(actor_role),
        action_type=f"admin_tournament_commerce_{action.lower()}",
        entity_type="tournament_commerce",
        entity_id=str(tournament_id),
        after_json={
            "operation_id": operation_id,
            "idempotent_replay": bool(result.get("idempotent_replay")),
            "result": result,
        },
        source_page=str(source),
        flagged_for_review=False,
    )
    audit_result = write_admin_activity_log(supabase, payload)
    if not audit_result.ok:
        raise TournamentCommerceRecoveryRequiredError(
            "The commerce change completed, but its shared admin audit needs "
            "recovery. Retry with the exact same idempotency key."
        )


def replace_admin_tournament_commerce_catalog(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    expected_catalog_fingerprint: str,
    catalog: Mapping[str, Any],
    idempotency_key: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str = "SAVE",
    source: str = "next_tournament_commerce_admin",
) -> dict[str, Any]:
    if _clean_text(confirmation_text, limit=40).upper() != "SAVE":
        raise TournamentCommerceValidationError(
            "Type SAVE to confirm the tournament extras catalog."
        )
    require_tournament_commerce_mutation_runtime(actor_type="ADMIN")
    payload = tournament_commerce_catalog_payload(catalog)
    request = {
        "action": "CATALOG_REPLACE",
        "expected_catalog_fingerprint": _clean_text(
            expected_catalog_fingerprint, limit=128
        ),
        "catalog": payload,
    }
    try:
        response = supabase.rpc(
            "server_replace_tournament_commerce_catalog",
            {
                "p_club_id": str(club_id),
                "p_tournament_id": str(tournament_id),
                "p_expected_catalog_fingerprint": request[
                    "expected_catalog_fingerprint"
                ],
                "p_catalog": payload,
                "p_idempotency_key": _canonical_uuid(
                    idempotency_key, field="idempotency_key"
                ),
                "p_request_fingerprint": stable_fingerprint(request),
                "p_actor_label": _clean_text(actor_email, limit=160),
                "p_source": _clean_text(source, limit=160),
            },
        ).execute()
    except Exception as exc:
        _raise_rpc_error(exc)
    result = _rpc_result(response)
    _audit_admin_result(
        supabase,
        club_id=str(club_id),
        tournament_id=str(tournament_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action="catalog_replace",
        result=result,
        source=source,
    )
    return result


def _admin_order_rpc(
    supabase: Any,
    *,
    rpc_name: str,
    rpc_payload: dict[str, Any],
    request: dict[str, Any],
    club_id: str,
    tournament_id: str,
    actor_email: str,
    actor_role: str,
    action: str,
    source: str,
) -> dict[str, Any]:
    require_tournament_commerce_mutation_runtime(actor_type="ADMIN")
    params = {
        **rpc_payload,
        "p_request_fingerprint": stable_fingerprint(request),
        "p_actor_label": _clean_text(actor_email, limit=160),
        "p_source": _clean_text(source, limit=160),
    }
    try:
        response = supabase.rpc(rpc_name, params).execute()
    except Exception as exc:
        _raise_rpc_error(exc)
    result = _rpc_result(response)
    _audit_admin_result(
        supabase,
        club_id=str(club_id),
        tournament_id=str(tournament_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action=action,
        result=result,
        source=source,
    )
    return result



def _registration_event_option_ids(
    supabase: Any,
    *,
    tournament_id: str,
    registration_id: str,
) -> list[str]:
    rows = _execute(
        supabase.table("tournament_registration_selections")
        .select("event_option_id")
        .eq("tournament_id", str(tournament_id))
        .eq("registration_id", str(registration_id)),
        label="registration event entries",
    )
    return sorted(
        {
            str(row.get("event_option_id") or "").strip()
            for row in rows
            if str(row.get("event_option_id") or "").strip()
        }
    )


def quote_admin_tournament_commerce_order(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    registration_id: str,
    item_selections: list[dict[str, Any]],
) -> dict[str, Any]:
    current_order = _current_order(
        supabase,
        tournament_id=str(tournament_id),
        registration_id=str(registration_id),
    )
    state, raw_catalog = _load_catalog(
        supabase,
        club_id=str(club_id),
        tournament_id=str(tournament_id),
        exclude_order_id=str((current_order or {}).get("id") or "") or None,
    )
    if not state:
        raise TournamentCommerceUnavailableError(
            "Tournament extras are not configured."
        )
    registrations = _registration_rows(
        supabase, tournament_id=str(tournament_id)
    )
    registration = next(
        (
            row
            for row in registrations
            if str(row.get("id") or "") == str(registration_id)
        ),
        None,
    )
    if not registration:
        raise TournamentCommerceValidationError("Registration was not found.")
    submitted_at: datetime | None = None
    if registration.get("submitted_at"):
        try:
            submitted_at = datetime.fromisoformat(
                str(registration["submitted_at"]).replace("Z", "+00:00")
            )
        except ValueError:
            submitted_at = None
    if submitted_at is None:
        submitted_at = datetime.now(timezone.utc)
    promotion_usage = {
        str(row.get("id")): int(row.get("used_claims") or 0)
        for row in raw_catalog.get("promotions") or []
    }
    event_option_ids = _registration_event_option_ids(
        supabase,
        tournament_id=str(tournament_id),
        registration_id=str(registration_id),
    )
    quote = quote_tournament_commerce(
        raw_catalog,
        {
            "event_option_ids": event_option_ids,
            "item_selections": list(item_selections or []),
        },
        registration_submitted_at=submitted_at,
        registrant_position=_registration_position(
            registrations, registration_id=str(registration_id)
        ),
        promotion_usage=promotion_usage,
    )
    return {
        "ok": True,
        "mode": "admin_tournament_commerce_order_quote",
        "quote": quote,
        "current_order": (
            {
                "status": current_order.get("status"),
                "payment_status": current_order.get("payment_status"),
                "updated_at": current_order.get("updated_at"),
                "quote_fingerprint": current_order.get("quote_fingerprint"),
            }
            if current_order
            else None
        ),
    }


def replace_admin_tournament_commerce_order(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    registration_id: str,
    item_selections: list[dict[str, Any]],
    expected_quote_fingerprint: str,
    expected_order_updated_at: str | None,
    idempotency_key: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str = "SAVE EXTRAS",
    source: str = "next_tournament_registration_detail",
) -> dict[str, Any]:
    if _clean_text(confirmation_text, limit=40).upper() != "SAVE EXTRAS":
        raise TournamentCommerceValidationError(
            "Type SAVE EXTRAS to confirm the registrant extras update."
        )
    require_tournament_commerce_mutation_runtime(actor_type="ADMIN")
    current = quote_admin_tournament_commerce_order(
        supabase,
        club_id=str(club_id),
        tournament_id=str(tournament_id),
        registration_id=str(registration_id),
        item_selections=list(item_selections or []),
    )
    quote = dict(current["quote"])
    expected_quote = _clean_text(expected_quote_fingerprint, limit=128)
    if not expected_quote or quote.get("quote_fingerprint") != expected_quote:
        raise TournamentCommerceConflictError(
            "Tournament extras or pricing changed. Review the updated total before saving."
        )
    current_order = current.get("current_order") or {}
    if current_order:
        if (
            not expected_order_updated_at
            or str(expected_order_updated_at)
            != str(current_order.get("updated_at") or "")
        ):
            raise TournamentCommerceConflictError(
                "Tournament extras changed after this registration was loaded. Refresh and review again."
            )
    elif expected_order_updated_at:
        raise TournamentCommerceConflictError(
            "The registrant extras order no longer matches the loaded state. Refresh and review again."
        )
    request_id = _canonical_uuid(idempotency_key, field="idempotency_key")
    try:
        response = supabase.rpc(
            "server_apply_tournament_commerce_order",
            {
                "p_club_id": str(club_id),
                "p_tournament_id": str(tournament_id),
                "p_registration_id": str(registration_id),
                "p_expected_order_updated_at": expected_order_updated_at,
                "p_quote_snapshot": quote,
                "p_idempotency_key": request_id,
                "p_request_fingerprint": quote["request_fingerprint"],
                "p_actor_type": "ADMIN",
                "p_actor_label": _clean_text(actor_email, limit=160),
                "p_source": _clean_text(source, limit=160),
            },
        ).execute()
    except Exception as exc:
        _raise_rpc_error(exc)
    result = {**_rpc_result(response), "quote": quote}
    _audit_admin_result(
        supabase,
        club_id=str(club_id),
        tournament_id=str(tournament_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action="order_replace",
        result=result,
        source=source,
    )
    return result

def cancel_admin_tournament_commerce_order(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    registration_id: str,
    expected_order_updated_at: str,
    reason: str,
    confirmation_text: str,
    idempotency_key: str,
    actor_email: str,
    actor_role: str,
    source: str = "next_tournament_commerce_admin",
) -> dict[str, Any]:
    if _clean_text(confirmation_text, limit=40).upper() != "CANCEL":
        raise TournamentCommerceValidationError(
            "Type CANCEL to confirm cancelling this extras order."
        )
    request = {
        "action": "ORDER_CANCEL",
        "registration_id": str(registration_id),
        "expected_order_updated_at": str(expected_order_updated_at),
        "reason": _clean_text(reason, limit=500),
        "confirmation_text": "CANCEL",
    }
    return _admin_order_rpc(
        supabase,
        rpc_name="server_cancel_tournament_commerce_order",
        rpc_payload={
            "p_club_id": str(club_id),
            "p_tournament_id": str(tournament_id),
            "p_registration_id": str(registration_id),
            "p_expected_order_updated_at": str(expected_order_updated_at),
            "p_reason": request["reason"],
            "p_idempotency_key": _canonical_uuid(
                idempotency_key, field="idempotency_key"
            ),
            "p_actor_type": "ADMIN",
        },
        request=request,
        club_id=club_id,
        tournament_id=tournament_id,
        actor_email=actor_email,
        actor_role=actor_role,
        action="order_cancel",
        source=source,
    )


def update_admin_tournament_commerce_payment(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    registration_id: str,
    payment_status: str,
    expected_order_updated_at: str,
    idempotency_key: str,
    actor_email: str,
    actor_role: str,
    source: str = "next_tournament_commerce_admin",
) -> dict[str, Any]:
    clean_status = _clean_text(payment_status, limit=40).upper()
    if clean_status not in {"UNPAID", "PAID", "WAIVED", "REFUNDED"}:
        raise TournamentCommerceValidationError(
            "Payment status must be unpaid, paid, waived, or refunded."
        )
    request = {
        "action": "PAYMENT_UPDATE",
        "registration_id": str(registration_id),
        "payment_status": clean_status,
        "expected_order_updated_at": str(expected_order_updated_at),
    }
    return _admin_order_rpc(
        supabase,
        rpc_name="server_update_tournament_commerce_payment",
        rpc_payload={
            "p_club_id": str(club_id),
            "p_tournament_id": str(tournament_id),
            "p_registration_id": str(registration_id),
            "p_payment_status": clean_status,
            "p_expected_order_updated_at": str(expected_order_updated_at),
            "p_idempotency_key": _canonical_uuid(
                idempotency_key, field="idempotency_key"
            ),
        },
        request=request,
        club_id=club_id,
        tournament_id=tournament_id,
        actor_email=actor_email,
        actor_role=actor_role,
        action="payment_update",
        source=source,
    )


def update_admin_tournament_commerce_fulfillment(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    fulfillment_id: str,
    status: str,
    notes: str,
    expected_updated_at: str,
    idempotency_key: str,
    actor_email: str,
    actor_role: str,
    source: str = "next_tournament_commerce_admin",
) -> dict[str, Any]:
    clean_status = _clean_text(status, limit=40).upper()
    if clean_status not in {"PENDING", "READY", "FULFILLED", "CANCELLED"}:
        raise TournamentCommerceValidationError(
            "Fulfillment status is invalid."
        )
    clean_notes = _clean_text(notes, limit=2000)
    fulfillment_uuid = _canonical_uuid(
        fulfillment_id, field="fulfillment_id"
    )
    if clean_status != "FULFILLED":
        current_rows = _execute(
            supabase.table("tournament_commerce_fulfillment")
            .select("id,status")
            .eq("tournament_id", str(tournament_id))
            .eq("id", fulfillment_uuid)
            .limit(1),
            label="fulfillment preflight",
        )
        current_status = (
            str(current_rows[0].get("status") or "").strip().upper()
            if current_rows
            else ""
        )
        if (
            current_status == "FULFILLED"
            and len(clean_notes) < FULFILLMENT_CORRECTION_NOTE_MIN_LENGTH
        ):
            raise TournamentCommerceValidationError(
                "Add a correction note of at least 8 characters before "
                "changing a fulfilled item."
            )
    request = {
        "action": "FULFILLMENT_UPDATE",
        "fulfillment_id": fulfillment_uuid,
        "status": clean_status,
        "notes": clean_notes,
        "expected_updated_at": str(expected_updated_at),
    }
    return _admin_order_rpc(
        supabase,
        rpc_name="server_update_tournament_commerce_fulfillment",
        rpc_payload={
            "p_club_id": str(club_id),
            "p_tournament_id": str(tournament_id),
            "p_fulfillment_id": fulfillment_uuid,
            "p_status": clean_status,
            "p_notes": clean_notes,
            "p_expected_updated_at": str(expected_updated_at),
            "p_idempotency_key": _canonical_uuid(
                idempotency_key, field="idempotency_key"
            ),
        },
        request=request,
        club_id=club_id,
        tournament_id=tournament_id,
        actor_email=actor_email,
        actor_role=actor_role,
        action="fulfillment_update",
        source=source,
    )


def build_admin_tournament_commerce_fulfillment_export(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
) -> tuple[bytes, str]:
    detail = get_admin_tournament_commerce_detail(
        supabase,
        club_id=str(club_id),
        tournament_id=str(tournament_id),
    )
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "registration_name",
            "registration_email",
            "item",
            "option",
            "sku",
            "quantity",
            "status",
            "instructions",
            "notes",
        ],
    )
    writer.writeheader()
    for row in detail.get("fulfillment") or []:
        registration = row.get("registration") or {}
        writer.writerow(
            {
                "registration_name": registration.get("display_name") or "",
                "registration_email": registration.get("email") or "",
                "item": row.get("label_snapshot") or "",
                "option": row.get("option_snapshot") or "",
                "sku": row.get("sku_snapshot") or "",
                "quantity": int(row.get("quantity") or 0),
                "status": row.get("status") or "",
                "instructions": row.get("instructions_snapshot") or "",
                "notes": row.get("notes") or "",
            }
        )
    name = f"tournament-commerce-fulfillment-{tournament_id}.csv"
    return output.getvalue().encode("utf-8"), name


def inspect_admin_tournament_commerce_operation(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    operation_id: str,
) -> dict[str, Any]:
    operation_uuid = _canonical_uuid(operation_id, field="operation_id")
    rows = _execute(
        supabase.table("tournament_commerce_operations")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("tournament_id", str(tournament_id))
        .eq("id", operation_uuid)
        .limit(1),
        label="operation recovery",
    )
    if not rows:
        raise TournamentCommerceValidationError(
            "Tournament commerce operation was not found."
        )
    operation = rows[0]
    internal_audit = _execute(
        supabase.table("tournament_commerce_audit_log")
        .select("id,action,created_at")
        .eq("club_id", str(club_id))
        .eq("tournament_id", str(tournament_id))
        .eq("operation_id", operation_uuid)
        .limit(20),
        label="operation audit",
    )
    shared_audit = _shared_admin_audit_present(
        supabase, club_id=str(club_id), operation_id=operation_uuid
    )
    status = str(operation.get("status") or "").upper()
    authoritative_complete = status == "COMPLETED"
    if authoritative_complete and shared_audit is True:
        recovery_state = "complete"
    elif authoritative_complete:
        recovery_state = "shared_audit_retry"
    elif status == "RECOVERY_REQUIRED":
        recovery_state = "inspect_before_retry"
    else:
        recovery_state = "in_progress_or_interrupted"
    return {
        "ok": True,
        "operation": operation,
        "internal_audit": internal_audit,
        "shared_admin_audit_present": shared_audit,
        "authoritative_mutation_complete": authoritative_complete,
        "recovery_state": recovery_state,
        "safe_retry": authoritative_complete,
        "retry_mode": (
            "same_idempotency_key"
            if authoritative_complete
            else "inspect_authoritative_state"
        ),
    }


__all__ = [
    "FULFILLMENT_CORRECTION_NOTE_MIN_LENGTH",
    "TournamentCommerceConflictError",
    "TournamentCommerceRecoveryRequiredError",
    "TournamentCommerceUnavailableError",
    "build_admin_tournament_commerce_fulfillment_export",
    "build_admin_tournament_commerce_status",
    "cancel_admin_tournament_commerce_order",
    "get_admin_tournament_commerce_detail",
    "inspect_admin_tournament_commerce_operation",
    "list_admin_tournament_commerce_tournaments",
    "replace_admin_tournament_commerce_catalog",
    "quote_admin_tournament_commerce_order",
    "replace_admin_tournament_commerce_order",
    "update_admin_tournament_commerce_fulfillment",
    "update_admin_tournament_commerce_payment",
]
