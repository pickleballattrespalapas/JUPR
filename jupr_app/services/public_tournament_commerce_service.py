"""Server-owned tournament extras, bundles, pricing, and recovery helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any, Mapping
import uuid

from jupr_app.domain.tournament_commerce import (
    TournamentCommerceValidationError,
    normalize_tournament_commerce_catalog,
    quote_tournament_commerce,
    stable_fingerprint,
)
from jupr_app.services.staging_write_guard import staging_write_wave_allows


FEATURE_FLAG = "JUPR_ENABLE_TOURNAMENT_COMMERCE"
MUTATION_FLAG = "JUPR_ENABLE_STAGING_TOURNAMENT_COMMERCE_WRITES"
ADMIN_WRITE_WAVE = "tournament-commerce-admin"
PUBLIC_WRITE_WAVE = "public-intake-auth"
PAGE_SIZE = 500
IN_FILTER_BATCH_SIZE = 100
TRUTHY = {"1", "true", "yes", "y", "on"}
ORDER_OPERATION_NAMESPACE = uuid.UUID("34bf1d95-8c52-4504-bec3-80215e97aba5")


class TournamentCommerceUnavailableError(RuntimeError):
    """Commerce storage or recovery evidence is not available."""


class TournamentCommerceConflictError(ValueError):
    """The reviewed state changed and the request must be reviewed again."""


class TournamentCommerceQuoteChangedError(TournamentCommerceConflictError):
    """The server produced a different quote than the browser reviewed."""

    def __init__(self, message: str, *, current_quote: dict[str, Any]):
        super().__init__(message)
        self.current_quote = current_quote


def _truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY


def is_tournament_commerce_enabled() -> bool:
    return _truthy(FEATURE_FLAG)


def tournament_commerce_runtime_status() -> dict[str, Any]:
    environment = os.getenv("JUPR_ENV", "").strip().lower() or "local"
    wave = os.getenv("JUPR_STAGING_WRITE_WAVE", "").strip() or "none"
    return {
        "enabled": is_tournament_commerce_enabled(),
        "environment": environment,
        "staging_only": True,
        "service_role_ready": bool(
            os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        ),
        "mutation_flag": {
            "name": MUTATION_FLAG,
            "enabled": _truthy(MUTATION_FLAG),
        },
        "write_wave": wave,
        "admin_write_ready": (
            environment == "staging"
            and staging_write_wave_allows(ADMIN_WRITE_WAVE)
            and _truthy(MUTATION_FLAG)
        ),
        "public_registration_write_ready": (
            environment == "staging"
            and staging_write_wave_allows(PUBLIC_WRITE_WAVE)
            and _truthy(MUTATION_FLAG)
        ),
        "offline_payment_only": True,
    }


def require_tournament_commerce_mutation_runtime(*, actor_type: str) -> None:
    """Refuse production and require the exact isolated staging write wave."""

    environment = os.getenv("JUPR_ENV", "").strip().lower()
    if environment == "production":
        raise PermissionError(
            "Tournament commerce mutations are staging-only until manual "
            "acceptance is complete."
        )
    if environment != "staging":
        return
    expected_wave = (
        PUBLIC_WRITE_WAVE
        if str(actor_type).upper() == "PUBLIC_REGISTRANT"
        else ADMIN_WRITE_WAVE
    )
    if (
        not staging_write_wave_allows(expected_wave)
        or not _truthy(MUTATION_FLAG)
    ):
        raise PermissionError(
            "Tournament commerce writes are disabled. Use permanent-open staging "
            f"or the approved {expected_wave} wave."
        )
    if not os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip():
        raise TournamentCommerceUnavailableError(
            "Tournament commerce staging writes require the server-only "
            "Supabase service credential."
        )


def _clean_text(value: Any, *, limit: int = 500) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _safe_rows(response: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (response.data or [])]
    except Exception:
        return []


def _first(response: Any) -> dict[str, Any] | None:
    rows = _safe_rows(response)
    return rows[0] if rows else None


def _execute(query: Any, *, label: str) -> list[dict[str, Any]]:
    try:
        return _safe_rows(query.execute())
    except Exception as exc:
        raise TournamentCommerceUnavailableError(
            f"Tournament commerce {label} is unavailable."
        ) from exc


def _canonical_uuid(value: Any, *, field: str) -> str:
    text = _clean_text(value, limit=80)
    try:
        parsed = uuid.UUID(text)
    except (TypeError, ValueError, AttributeError) as exc:
        raise TournamentCommerceValidationError(
            f"{field} must be a valid UUID."
        ) from exc
    if str(parsed) != text.lower():
        raise TournamentCommerceValidationError(
            f"{field} must use canonical UUID format."
        )
    return str(parsed)


def _public_registration_window_is_open(settings: Mapping[str, Any]) -> bool:
    if _clean_text(
        settings.get("registration_status"), limit=40
    ).lower() != "open":
        return False
    now = datetime.now(timezone.utc)
    for field, is_open_bound in (
        ("registration_open_at", True),
        ("registration_close_at", False),
    ):
        raw = settings.get(field)
        if raw in (None, ""):
            continue
        try:
            bound = datetime.fromisoformat(
                str(raw).strip().replace("Z", "+00:00")
            )
            if bound.tzinfo is None:
                return False
            bound = bound.astimezone(timezone.utc)
        except (TypeError, ValueError):
            return False
        if is_open_bound and now < bound:
            return False
        if not is_open_bound and now > bound:
            return False
    return True


def _require_public_tournament_registration_visibility(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
) -> str:
    """Authorize standalone public catalog reads before commerce data access."""

    canonical_tournament_id = _canonical_uuid(
        tournament_id, field="tournament_id"
    )
    tournaments = _execute(
        supabase.table("tournaments")
        .select("id,club_id,status")
        .eq("id", canonical_tournament_id)
        .eq("club_id", str(club_id))
        .limit(1),
        label="public tournament",
    )
    tournament = tournaments[0] if tournaments else None
    if (
        not tournament
        or _clean_text(tournament.get("status"), limit=40).upper()
        == "ARCHIVED"
    ):
        raise PermissionError(
            "Tournament extras are unavailable for this registration."
        )

    settings_rows = _execute(
        supabase.table("tournament_registration_settings")
        .select(
            "tournament_id,registration_status,"
            "registration_open_at,registration_close_at"
        )
        .eq("tournament_id", canonical_tournament_id)
        .limit(1),
        label="public registration settings",
    )
    settings = settings_rows[0] if settings_rows else None
    if not settings or not _public_registration_window_is_open(settings):
        raise PermissionError(
            "Tournament extras are unavailable for this registration."
        )
    return canonical_tournament_id


def _paged_rows(
    supabase: Any,
    table: str,
    *,
    columns: str = "*",
    equals: Mapping[str, Any] | None = None,
    in_filter: tuple[str, list[str]] | None = None,
    order_column: str = "id",
) -> list[dict[str, Any]]:
    """Read every scoped row without relying on PostgREST's default row cap."""

    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        query = supabase.table(table).select(columns)
        for key, value in (equals or {}).items():
            query = query.eq(key, value)
        if in_filter and in_filter[1]:
            query = query.in_(in_filter[0], in_filter[1])
        query = query.order(order_column).limit(PAGE_SIZE)
        if cursor is not None:
            query = query.gt(order_column, cursor)
        page = _execute(query, label=table.replace("_", " "))
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        next_cursor = _clean_text(page[-1].get(order_column), limit=200)
        if not next_cursor or next_cursor == cursor:
            raise TournamentCommerceUnavailableError(
                f"Tournament commerce {table.replace('_', ' ')} pagination "
                "could not advance safely."
            )
        cursor = next_cursor
    return rows


def _batched_rows(
    supabase: Any,
    table: str,
    *,
    filter_column: str,
    values: list[str],
    equals: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(values), IN_FILTER_BATCH_SIZE):
        batch = values[start : start + IN_FILTER_BATCH_SIZE]
        rows.extend(
            _paged_rows(
                supabase,
                table,
                equals=equals,
                in_filter=(filter_column, batch),
            )
        )
    return rows


def _registration_rows(
    supabase: Any, *, tournament_id: str
) -> list[dict[str, Any]]:
    rows = _paged_rows(
        supabase,
        "tournament_registrations",
        columns="id,tournament_id,status,submitted_at,updated_at,email",
        equals={"tournament_id": str(tournament_id)},
    )
    return [
        row
        for row in rows
        if str(row.get("status") or "confirmed").strip().lower()
        != "cancelled"
    ]


def _registration_position(
    rows: list[dict[str, Any]], *, registration_id: str | None
) -> int:
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("submitted_at") or ""),
            str(row.get("id") or ""),
        ),
    )
    if registration_id:
        for index, row in enumerate(ordered, start=1):
            if str(row.get("id") or "") == str(registration_id):
                return index
    return len(ordered) + 1


def _current_order(
    supabase: Any, *, tournament_id: str, registration_id: str | None
) -> dict[str, Any] | None:
    if not registration_id:
        return None
    rows = _execute(
        supabase.table("tournament_commerce_orders")
        .select("*")
        .eq("tournament_id", str(tournament_id))
        .eq("registration_id", str(registration_id))
        .limit(1),
        label="order",
    )
    return rows[0] if rows else None


def _load_catalog(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    exclude_order_id: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    state_rows = _execute(
        supabase.table("tournament_commerce_catalog_state")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("tournament_id", str(tournament_id))
        .limit(1),
        label="catalog",
    )
    state = state_rows[0] if state_rows else None
    if not state:
        return None, {
            "currency": "USD",
            "items": [],
            "variants": [],
            "bundles": [],
            "bundle_components": [],
            "promotions": [],
            "event_options": [],
        }

    scoped = {"club_id": str(club_id), "tournament_id": str(tournament_id)}
    items = _paged_rows(
        supabase, "tournament_commerce_items", equals=scoped
    )
    bundles = _paged_rows(
        supabase, "tournament_commerce_bundles", equals=scoped
    )
    promotions = _paged_rows(
        supabase, "tournament_commerce_promotions", equals=scoped
    )
    item_ids = [str(row["id"]) for row in items if row.get("id")]
    bundle_ids = [str(row["id"]) for row in bundles if row.get("id")]
    variants = _batched_rows(
        supabase,
        "tournament_commerce_item_variants",
        filter_column="item_id",
        values=item_ids,
    )
    components = _batched_rows(
        supabase,
        "tournament_commerce_bundle_components",
        filter_column="bundle_id",
        values=bundle_ids,
    )
    events = _paged_rows(
        supabase,
        "tournament_event_options",
        equals={"tournament_id": str(tournament_id)},
    )
    reservations = _paged_rows(
        supabase,
        "tournament_commerce_inventory_reservations",
        equals={"tournament_id": str(tournament_id), "status": "ACTIVE"},
    )
    if exclude_order_id:
        reservations = [
            row
            for row in reservations
            if str(row.get("order_id") or "") != str(exclude_order_id)
        ]
    reserved_by_item: dict[str, int] = {}
    reserved_by_variant: dict[str, int] = {}
    for row in reservations:
        quantity = int(row.get("quantity") or 0)
        item_id = str(row.get("item_id") or "")
        variant_id = str(row.get("variant_id") or "")
        reserved_by_item[item_id] = reserved_by_item.get(item_id, 0) + quantity
        reserved_by_variant[variant_id] = (
            reserved_by_variant.get(variant_id, 0) + quantity
        )

    for item in items:
        limit = item.get("inventory_limit")
        item["remaining"] = (
            max(0, int(limit) - reserved_by_item.get(str(item["id"]), 0))
            if limit is not None
            else None
        )
    item_by_id = {str(row["id"]): row for row in items}
    for variant in variants:
        limit = variant.get("inventory_limit")
        item = item_by_id.get(str(variant.get("item_id") or "")) or {}
        own_remaining = (
            max(
                0,
                int(limit)
                - reserved_by_variant.get(str(variant["id"]), 0),
            )
            if limit is not None
            else None
        )
        item_remaining = item.get("remaining")
        if own_remaining is None:
            variant["remaining"] = item_remaining
        elif item_remaining is None:
            variant["remaining"] = own_remaining
        else:
            variant["remaining"] = min(
                int(own_remaining), int(item_remaining)
            )

    promotion_ids = [
        str(row["id"]) for row in promotions if row.get("id")
    ]
    claims = _batched_rows(
        supabase,
        "tournament_commerce_promotion_claims",
        filter_column="promotion_id",
        values=promotion_ids,
        equals={"status": "ACTIVE"},
    )
    if exclude_order_id:
        claims = [
            row
            for row in claims
            if str(row.get("order_id") or "") != str(exclude_order_id)
        ]
    claimed: dict[str, int] = {}
    for row in claims:
        promotion_id = str(row.get("promotion_id") or "")
        claimed[promotion_id] = claimed.get(promotion_id, 0) + int(
            row.get("quantity") or 0
        )
    for promotion in promotions:
        promotion["used_claims"] = claimed.get(str(promotion["id"]), 0)

    return state, {
        "currency": str(state.get("currency") or "USD"),
        "catalog_revision": int(state.get("catalog_revision") or 0),
        "catalog_fingerprint": str(
            state.get("catalog_fingerprint") or ""
        ),
        "items": items,
        "variants": variants,
        "bundles": bundles,
        "bundle_components": components,
        "promotions": promotions,
        "event_options": events,
    }


def build_public_tournament_commerce_catalog(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    registration_id: str | None = None,
    token_bound_edit: bool = False,
) -> dict[str, Any]:
    if not is_tournament_commerce_enabled():
        return {
            "available": False,
            "tournament_id": str(tournament_id),
            "reason": "Tournament extras are not enabled.",
            "currency": "USD",
            "offline_payment": True,
            "items": [],
            "variants": [],
            "bundles": [],
            "bundle_components": [],
            "promotions": [],
            "runtime": tournament_commerce_runtime_status(),
        }
    tournament_id = (
        _canonical_uuid(tournament_id, field="tournament_id")
        if token_bound_edit
        else _require_public_tournament_registration_visibility(
            supabase,
            club_id=str(club_id),
            tournament_id=str(tournament_id),
        )
    )
    current_order = _current_order(
        supabase,
        tournament_id=tournament_id,
        registration_id=registration_id,
    )
    state, raw = _load_catalog(
        supabase,
        club_id=str(club_id),
        tournament_id=tournament_id,
        exclude_order_id=str((current_order or {}).get("id") or "") or None,
    )
    if not state:
        return {
            "available": False,
            "tournament_id": str(tournament_id),
            "reason": "Tournament extras are not configured.",
            "currency": "USD",
            "offline_payment": True,
            "items": [],
            "variants": [],
            "bundles": [],
            "bundle_components": [],
            "promotions": [],
            "runtime": tournament_commerce_runtime_status(),
        }
    catalog = normalize_tournament_commerce_catalog(raw)
    public_items = [
        {
            key: row.get(key)
            for key in (
                "id",
                "name",
                "description",
                "kind",
                "status",
                "base_price_minor",
                "inventory_limit",
                "remaining",
                "max_per_registration",
                "available_from",
                "available_until",
                "requires_fulfillment",
                "fulfillment_instructions",
                "sort_order",
            )
        }
        for row in catalog["items"]
        if row["status"] == "ACTIVE"
    ]
    public_item_ids = {str(row["id"]) for row in public_items}
    public_variants = [
        {
            key: row.get(key)
            for key in (
                "id",
                "item_id",
                "name",
                "sku",
                "status",
                "price_delta_minor",
                "price_minor",
                "inventory_limit",
                "remaining",
                "sort_order",
            )
        }
        for row in catalog["variants"]
        if row["status"] == "ACTIVE"
        and str(row["item_id"]) in public_item_ids
    ]
    public_bundles = [
        {
            key: row.get(key)
            for key in (
                "id",
                "name",
                "description",
                "status",
                "price_minor",
                "regular_minor",
                "savings_minor",
                "max_per_registration",
                "available_from",
                "available_until",
                "sort_order",
                "components",
            )
        }
        for row in catalog["bundles"]
        if row["status"] == "ACTIVE"
    ]
    public_bundle_ids = {str(row["id"]) for row in public_bundles}
    item_by_id = {
        str(row["id"]): row for row in catalog["items"] if row.get("id")
    }
    variant_by_id = {
        str(row["id"]): row
        for row in catalog["variants"]
        if row.get("id")
    }
    event_by_id = {
        str(row["id"]): row
        for row in catalog["event_options"]
        if row.get("id")
    }
    public_bundle_components: list[dict[str, Any]] = []
    for row in catalog["bundle_components"]:
        if str(row["bundle_id"]) not in public_bundle_ids:
            continue
        component = dict(row)
        if component.get("component_type") == "EVENT_OPTION":
            event = event_by_id.get(str(component.get("event_option_id") or ""))
            if event:
                component["label"] = event.get("label")
                component["option_label"] = event.get("option_label")
        else:
            item = item_by_id.get(str(component.get("item_id") or ""))
            variant = variant_by_id.get(str(component.get("variant_id") or ""))
            if item:
                component["label"] = item.get("name")
                component["requires_fulfillment"] = bool(
                    item.get("requires_fulfillment")
                )
                component["fulfillment_instructions"] = item.get(
                    "fulfillment_instructions"
                )
            if variant:
                component["option_label"] = variant.get("name")
                component["sku"] = variant.get("sku")
        public_bundle_components.append(component)
    return {
        "available": True,
        "tournament_id": str(tournament_id),
        "currency": "USD",
        "offline_payment": True,
        "catalog_revision": int((state or {}).get("catalog_revision") or 0),
        "catalog_fingerprint": catalog["catalog_fingerprint"],
        "items": public_items,
        "variants": public_variants,
        "bundles": public_bundles,
        "bundle_components": public_bundle_components,
        "promotions": [
            {
                key: row.get(key)
                for key in (
                    "id",
                    "name",
                    "promotion_type",
                    "target_type",
                    "item_id",
                    "variant_id",
                    "bundle_id",
                    "starts_at",
                    "ends_at",
                    "giveaway_limit",
                    "per_registration_limit",
                    "status",
                )
            }
            for row in catalog["promotions"]
            if row["status"] == "ACTIVE"
        ],
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
        "runtime": tournament_commerce_runtime_status(),
    }


def quote_public_tournament_commerce(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    request: Mapping[str, Any],
    registration_id: str | None = None,
    quote_at: datetime | None = None,
) -> dict[str, Any]:
    if not is_tournament_commerce_enabled():
        raise PermissionError("Tournament extras are not enabled.")
    if _clean_text(request.get("website"), limit=200):
        raise TournamentCommerceValidationError("Unable to quote this request.")
    tournament_id = _require_public_tournament_registration_visibility(
        supabase,
        club_id=str(club_id),
        tournament_id=str(tournament_id),
    )
    current_order = _current_order(
        supabase,
        tournament_id=tournament_id,
        registration_id=registration_id,
    )
    state, raw_catalog = _load_catalog(
        supabase,
        club_id=str(club_id),
        tournament_id=tournament_id,
        exclude_order_id=str((current_order or {}).get("id") or "") or None,
    )
    if not state:
        raise TournamentCommerceUnavailableError(
            "Tournament extras are not configured."
        )
    registrations = _registration_rows(
        supabase, tournament_id=tournament_id
    )
    registration = next(
        (
            row
            for row in registrations
            if str(row.get("id") or "") == str(registration_id or "")
        ),
        None,
    )
    submitted_at: datetime | None = None
    if registration and registration.get("submitted_at"):
        text = str(registration["submitted_at"]).replace("Z", "+00:00")
        try:
            submitted_at = datetime.fromisoformat(text)
        except ValueError:
            submitted_at = None
    if submitted_at is None:
        submitted_at = quote_at or datetime.now(timezone.utc)
    promotion_usage = {
        str(row.get("id")): int(row.get("used_claims") or 0)
        for row in raw_catalog.get("promotions") or []
    }
    quote = quote_tournament_commerce(
        raw_catalog,
        {
            "event_option_ids": list(request.get("event_option_ids") or []),
            "item_selections": list(request.get("item_selections") or []),
        },
        quote_at=quote_at,
        registration_submitted_at=submitted_at,
        registrant_position=_registration_position(
            registrations, registration_id=registration_id
        ),
        promotion_usage=promotion_usage,
    )
    return {
        "ok": True,
        "mode": "tournament_commerce_quote",
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


def _public_actor_label(email: Any) -> str:
    clean = _clean_text(email, limit=320).lower()
    if "@" not in clean:
        return "public registrant"
    local, domain = clean.split("@", 1)
    return f"{local[:1]}***@{domain}"[:160]


def _rpc_result(response: Any) -> dict[str, Any]:
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        return dict(data)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return dict(data[0])
    raise TournamentCommerceUnavailableError(
        "Tournament commerce mutation returned no recovery evidence."
    )


def _raise_rpc_error(exc: Exception) -> None:
    detail = str(exc)
    if (
        "JUPR_TOURNAMENT_COMMERCE_FULFILLED_CORRECTION_NOTE_REQUIRED"
        in detail
    ):
        raise TournamentCommerceValidationError(
            "Add a correction note of at least 8 characters before changing "
            "a fulfilled item."
        ) from exc
    if any(
        marker in detail
        for marker in (
            "JUPR_TOURNAMENT_COMMERCE_ORDER_STALE",
            "JUPR_TOURNAMENT_COMMERCE_QUOTE_STALE",
            "JUPR_TOURNAMENT_COMMERCE_INVENTORY_UNAVAILABLE",
            "JUPR_TOURNAMENT_COMMERCE_CLAIM_GIVEAWAY_FULL",
            "JUPR_TOURNAMENT_COMMERCE_REGISTRANT_GIVEAWAY_FULL",
            "JUPR_TOURNAMENT_COMMERCE_FULFILLED_ORDER_LOCKED",
            "JUPR_TOURNAMENT_COMMERCE_IDEMPOTENCY_CONFLICT",
        )
    ):
        raise TournamentCommerceConflictError(
            "Tournament extras changed while this request was being saved. "
            "Reload the current registration before retrying."
        ) from exc
    if "JUPR_TOURNAMENT_COMMERCE_" in detail:
        raise TournamentCommerceValidationError(
            "Tournament extras could not be saved because the reviewed "
            "selection is no longer valid."
        ) from exc
    raise TournamentCommerceUnavailableError(
        "Tournament extras could not be saved. No retry was attempted."
    ) from exc


def _derived_operation_keys(
    idempotency_key: str, *, mode: str
) -> tuple[str, str]:
    canonical = _canonical_uuid(idempotency_key, field="idempotency_key")
    return (
        str(
            uuid.uuid5(
                ORDER_OPERATION_NAMESPACE,
                f"{mode}:registration:{canonical}",
            )
        ),
        str(
            uuid.uuid5(
                ORDER_OPERATION_NAMESPACE,
                f"{mode}:commerce-order:{canonical}",
            )
        ),
    )


def prepare_public_registration_commerce_transaction(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    registration_id: str,
    registration_email: str,
    event_option_ids: list[str],
    commerce: Mapping[str, Any] | None,
    edit_mode: bool = False,
) -> dict[str, Any] | None:
    """Requote and prepare server-only evidence for an atomic registration RPC."""

    if not is_tournament_commerce_enabled():
        return None
    selection = dict(commerce or {})
    expected_fingerprint = _clean_text(
        selection.get("expected_quote_fingerprint"), limit=128
    )
    idempotency_key = _canonical_uuid(
        selection.get("idempotency_key"), field="idempotency_key"
    )
    operation_key, order_key = _derived_operation_keys(
        idempotency_key, mode="edit" if edit_mode else "create"
    )
    replay_rows = _execute(
        supabase.table("tournament_commerce_operations")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("tournament_id", str(tournament_id))
        .eq("idempotency_key", operation_key)
        .limit(1),
        label="registration recovery",
    )
    replay = replay_rows[0] if replay_rows else None
    if replay and str(replay.get("status") or "").upper() == "COMPLETED":
        order_id = str(replay.get("order_id") or "")
        revision_rows = (
            _execute(
                supabase.table("tournament_commerce_order_revisions")
                .select("*")
                .eq("order_id", order_id)
                .order("revision", desc=True)
                .limit(1),
                label="registration recovery quote",
            )
            if order_id
            else []
        )
        quote = (
            dict(revision_rows[0].get("price_snapshot") or {})
            if revision_rows
            and isinstance(revision_rows[0].get("price_snapshot"), dict)
            else {}
        )
        if (
            not quote
            or quote.get("quote_fingerprint") != expected_fingerprint
        ):
            raise TournamentCommerceConflictError(
                "The completed registration retry does not match the "
                "reviewed tournament extras."
            )
        return {
            "quote": quote,
            "operation_idempotency_key": operation_key,
            "order_idempotency_key": order_key,
            "request_fingerprint": str(
                replay.get("request_fingerprint") or ""
            ),
            "expected_order_updated_at": selection.get(
                "expected_order_updated_at"
            ),
            "actor_label": _public_actor_label(registration_email),
            "completed_replay": True,
        }
    current = quote_public_tournament_commerce(
        supabase,
        club_id=str(club_id),
        tournament_id=str(tournament_id),
        registration_id=str(registration_id) if edit_mode else None,
        request={
            "event_option_ids": list(event_option_ids),
            "item_selections": list(selection.get("item_selections") or []),
        },
    )
    quote = dict(current["quote"])
    if (
        not expected_fingerprint
        or quote["quote_fingerprint"] != expected_fingerprint
    ):
        raise TournamentCommerceQuoteChangedError(
            "Tournament extras or pricing changed. Review the updated total "
            "before submitting again.",
            current_quote=quote,
        )
    current_order = current.get("current_order") or {}
    expected_order_updated_at = selection.get("expected_order_updated_at")
    if edit_mode and current_order:
        if (
            not expected_order_updated_at
            or str(expected_order_updated_at)
            != str(current_order.get("updated_at") or "")
        ):
            raise TournamentCommerceConflictError(
                "Tournament extras changed after this registration was "
                "loaded. Refresh before retrying."
            )
    request_fingerprint = stable_fingerprint(
        {
            "mode": "REGISTRATION_EDIT_WITH_COMMERCE"
            if edit_mode
            else "REGISTRATION_CREATE_WITH_COMMERCE",
            "club_id": str(club_id),
            "tournament_id": str(tournament_id),
            "registration_id": str(registration_id),
            "quote_fingerprint": quote["quote_fingerprint"],
        }
    )
    return {
        "quote": quote,
        "operation_idempotency_key": operation_key,
        "order_idempotency_key": order_key,
        "request_fingerprint": request_fingerprint,
        "expected_order_updated_at": expected_order_updated_at,
        "actor_label": _public_actor_label(registration_email),
    }


def apply_public_tournament_commerce_order(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    registration_id: str,
    registration_email: str,
    item_selections: list[dict[str, Any]],
    event_option_ids: list[str],
    expected_quote_fingerprint: str,
    idempotency_key: str,
    expected_order_updated_at: str | None = None,
    source: str = "next_public_tournament_registration",
) -> dict[str, Any]:
    require_tournament_commerce_mutation_runtime(
        actor_type="PUBLIC_REGISTRANT"
    )
    current = quote_public_tournament_commerce(
        supabase,
        club_id=str(club_id),
        tournament_id=str(tournament_id),
        registration_id=str(registration_id),
        request={
            "event_option_ids": event_option_ids,
            "item_selections": item_selections,
        },
    )
    quote = dict(current["quote"])
    if (
        not _clean_text(expected_quote_fingerprint, limit=128)
        or quote["quote_fingerprint"]
        != _clean_text(expected_quote_fingerprint, limit=128)
    ):
        raise TournamentCommerceQuoteChangedError(
            "Tournament extras or pricing changed. Review the updated total "
            "before submitting again.",
            current_quote=quote,
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
                "p_actor_type": "PUBLIC_REGISTRANT",
                "p_actor_label": _public_actor_label(registration_email),
                "p_source": _clean_text(source, limit=160),
            },
        ).execute()
    except Exception as exc:
        _raise_rpc_error(exc)
    return {**_rpc_result(response), "quote": quote}


def cancel_public_tournament_commerce_order(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    registration_id: str,
    registration_email: str,
    expected_order_updated_at: str,
    idempotency_key: str,
    reason: str = "Registration cancelled by registrant",
    source: str = "next_public_tournament_registration_edit",
) -> dict[str, Any]:
    require_tournament_commerce_mutation_runtime(
        actor_type="PUBLIC_REGISTRANT"
    )
    request = {
        "action": "ORDER_CANCEL",
        "registration_id": str(registration_id),
        "expected_order_updated_at": str(expected_order_updated_at),
        "reason": _clean_text(reason, limit=500),
    }
    try:
        response = supabase.rpc(
            "server_cancel_tournament_commerce_order",
            {
                "p_club_id": str(club_id),
                "p_tournament_id": str(tournament_id),
                "p_registration_id": str(registration_id),
                "p_expected_order_updated_at": str(
                    expected_order_updated_at
                ),
                "p_reason": request["reason"],
                "p_idempotency_key": _canonical_uuid(
                    idempotency_key, field="idempotency_key"
                ),
                "p_actor_type": "PUBLIC_REGISTRANT",
                "p_actor_label": _public_actor_label(registration_email),
                "p_source": _clean_text(source, limit=160),
                "p_request_fingerprint": stable_fingerprint(request),
            },
        ).execute()
    except Exception as exc:
        _raise_rpc_error(exc)
    return _rpc_result(response)


def build_public_tournament_commerce_order(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    registration_id: str,
) -> dict[str, Any] | None:
    order = _current_order(
        supabase,
        tournament_id=str(tournament_id),
        registration_id=str(registration_id),
    )
    if not order or str(order.get("club_id") or club_id) != str(club_id):
        return None
    revisions = _execute(
        supabase.table("tournament_commerce_order_revisions")
        .select("*")
        .eq("order_id", str(order["id"]))
        .eq("revision", int(order.get("current_revision") or 0))
        .limit(1),
        label="order revision",
    )
    revision = revisions[0] if revisions else None
    quote = (
        dict(revision.get("price_snapshot") or {})
        if revision and isinstance(revision.get("price_snapshot"), dict)
        else {}
    )
    return {
        "id": str(order.get("id") or ""),
        "status": str(order.get("status") or ""),
        "payment_status": str(order.get("payment_status") or "UNPAID"),
        "currency": "USD",
        "current_revision": int(order.get("current_revision") or 0),
        "list_subtotal_minor": int(order.get("list_subtotal_minor") or 0),
        "discount_minor": int(order.get("discount_minor") or 0),
        "total_minor": int(order.get("total_minor") or 0),
        "updated_at": order.get("updated_at"),
        "quote": quote,
        "offline_payment": True,
    }


__all__ = [
    "ADMIN_WRITE_WAVE",
    "FEATURE_FLAG",
    "MUTATION_FLAG",
    "PUBLIC_WRITE_WAVE",
    "TournamentCommerceConflictError",
    "TournamentCommerceQuoteChangedError",
    "TournamentCommerceUnavailableError",
    "apply_public_tournament_commerce_order",
    "build_public_tournament_commerce_catalog",
    "build_public_tournament_commerce_order",
    "cancel_public_tournament_commerce_order",
    "is_tournament_commerce_enabled",
    "prepare_public_registration_commerce_transaction",
    "quote_public_tournament_commerce",
    "require_tournament_commerce_mutation_runtime",
    "tournament_commerce_runtime_status",
]
