"""Pure tournament-commerce rules.

Money is represented as integer minor units. Bundles and free offers are
calculated from server-owned catalog rows, with deterministic allocation and
complete immutable snapshots for the database transaction layer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from functools import lru_cache
import hashlib
import json
from typing import Any, Iterable, Mapping


ITEM_KINDS = frozenset(
    {"MERCHANDISE", "LODGING", "MEAL", "DRINK_PACK", "OTHER"}
)
CATALOG_STATUSES = frozenset({"DRAFT", "ACTIVE", "ARCHIVED"})
PROMOTION_TYPES = frozenset(
    {"DATE_WINDOW_FREE", "FIRST_N_REGISTRANTS", "FIRST_N_CLAIMS"}
)
PROMOTION_TARGET_TYPES = frozenset({"ITEM", "ITEM_VARIANT", "BUNDLE"})
COMPONENT_TYPES = frozenset({"EVENT_OPTION", "ITEM_VARIANT"})
MAX_CART_ITEM_QUANTITY = 20
MAX_CART_DISTINCT_ITEMS = 50
MAX_BUNDLE_CANDIDATES = 20
MAX_EVENT_SELECTIONS = 20


class TournamentCommerceValidationError(ValueError):
    """Catalog or cart data cannot produce a safe quote."""


@dataclass(frozen=True)
class _BundleCandidate:
    bundle: dict[str, Any]
    components: tuple[tuple[str, int], ...]
    regular_minor: int
    price_minor: int
    savings_minor: int


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Unsupported canonical JSON value: {type(value)!r}")


def stable_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clean_text(value: Any, *, limit: int = 500) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _upper(value: Any) -> str:
    return _clean_text(value, limit=80).upper()


def _required_id(value: Any, *, field: str) -> str:
    clean = _clean_text(value, limit=160)
    if not clean:
        raise TournamentCommerceValidationError(f"{field} is required.")
    return clean


def _safe_int(
    value: Any,
    *,
    field: str,
    minimum: int = 0,
    maximum: int | None = None,
    allow_none: bool = False,
) -> int | None:
    if value in (None, "") and allow_none:
        return None
    if isinstance(value, bool):
        raise TournamentCommerceValidationError(f"{field} must be a whole number.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TournamentCommerceValidationError(
            f"{field} must be a whole number."
        ) from exc
    if parsed < minimum or (maximum is not None and parsed > maximum):
        suffix = (
            f" between {minimum} and {maximum}"
            if maximum is not None
            else f" at least {minimum}"
        )
        raise TournamentCommerceValidationError(f"{field} must be{suffix}.")
    return parsed


def _minor_value(
    row: Mapping[str, Any],
    *,
    minor_key: str,
    major_key: str,
    field: str,
    allow_negative: bool = False,
) -> int:
    raw_minor = row.get(minor_key)
    if raw_minor not in (None, ""):
        parsed = _safe_int(
            raw_minor,
            field=field,
            minimum=-10_000_000 if allow_negative else 0,
            maximum=10_000_000,
        )
        return int(parsed or 0)
    raw_major = row.get(major_key)
    if raw_major in (None, ""):
        return 0
    try:
        amount = Decimal(str(raw_major)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
    except (InvalidOperation, ValueError) as exc:
        raise TournamentCommerceValidationError(
            f"{field} must be a valid amount."
        ) from exc
    if not amount.is_finite() or (amount < 0 and not allow_negative):
        raise TournamentCommerceValidationError(
            f"{field} must be a valid non-negative amount."
        )
    minor = int(amount * 100)
    if abs(minor) > 10_000_000:
        raise TournamentCommerceValidationError(f"{field} is too large.")
    return minor


def _datetime(value: Any, *, field: str) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise TournamentCommerceValidationError(
                f"{field} must be a valid date and time."
            ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(timezone.utc).isoformat() if value else None


def _available_at(
    row: Mapping[str, Any],
    *,
    at: datetime,
) -> bool:
    start = row.get("_available_from")
    end = row.get("_available_until")
    return not (
        (isinstance(start, datetime) and at < start)
        or (isinstance(end, datetime) and at > end)
    )


def _unique_ids(rows: Iterable[Mapping[str, Any]], *, label: str) -> None:
    seen: set[str] = set()
    for row in rows:
        row_id = str(row["id"])
        if row_id in seen:
            raise TournamentCommerceValidationError(
                f"Duplicate {label} id: {row_id}."
            )
        seen.add(row_id)


def _catalog_fingerprint_payload(catalog: Mapping[str, Any]) -> dict[str, Any]:
    dynamic = {
        "catalog_fingerprint",
        "catalog_revision",
        "remaining",
        "reserved",
        "claimed",
        "used_claims",
        "_available_from",
        "_available_until",
    }

    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: clean(item)
                for key, item in sorted(value.items())
                if key not in dynamic and not key.startswith("_")
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, datetime):
            return _iso(value)
        return value

    return clean(
        {
            "currency": catalog.get("currency") or "USD",
            "items": catalog.get("items") or [],
            "variants": catalog.get("variants") or [],
            "bundles": catalog.get("bundles") or [],
            "bundle_components": catalog.get("bundle_components") or [],
            "promotions": catalog.get("promotions") or [],
            "event_options": catalog.get("event_options") or [],
        }
    )


def normalize_tournament_commerce_catalog(
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and canonicalize a catalog returned by the server repository."""

    if not isinstance(catalog, Mapping):
        raise TournamentCommerceValidationError("Commerce catalog is invalid.")
    currency = _upper(catalog.get("currency") or "USD")
    if currency != "USD":
        raise TournamentCommerceValidationError(
            "Only USD is supported for offline tournament payment."
        )

    raw_items = [dict(row) for row in (catalog.get("items") or [])]
    raw_variants = [dict(row) for row in (catalog.get("variants") or [])]
    raw_bundles = [dict(row) for row in (catalog.get("bundles") or [])]
    raw_components = [
        dict(row) for row in (catalog.get("bundle_components") or [])
    ]
    raw_promotions = [dict(row) for row in (catalog.get("promotions") or [])]
    raw_events = [dict(row) for row in (catalog.get("event_options") or [])]

    items: list[dict[str, Any]] = []
    for row in raw_items:
        item_id = _required_id(row.get("id"), field="Item id")
        name = _clean_text(row.get("name"), limit=160)
        if not name:
            raise TournamentCommerceValidationError("Item name is required.")
        kind = _upper(row.get("kind") or "OTHER")
        status = _upper(row.get("status") or "DRAFT")
        if kind not in ITEM_KINDS:
            raise TournamentCommerceValidationError(
                f"{name}: unsupported item type {kind}."
            )
        if status not in CATALOG_STATUSES:
            raise TournamentCommerceValidationError(
                f"{name}: unsupported item status {status}."
            )
        start = _datetime(
            row.get("available_from"), field=f"{name} available-from date"
        )
        end = _datetime(
            row.get("available_until"), field=f"{name} available-until date"
        )
        if start and end and end < start:
            raise TournamentCommerceValidationError(
                f"{name}: availability end must follow its start."
            )
        items.append(
            {
                "id": item_id,
                "name": name,
                "description": _clean_text(row.get("description"), limit=2000),
                "kind": kind,
                "status": status,
                "base_price_minor": _minor_value(
                    row,
                    minor_key="base_price_minor",
                    major_key="base_price_usd",
                    field=f"{name} base price",
                ),
                "inventory_limit": _safe_int(
                    row.get("inventory_limit"),
                    field=f"{name} inventory",
                    minimum=0,
                    allow_none=True,
                ),
                "remaining": _safe_int(
                    row.get("remaining"),
                    field=f"{name} remaining inventory",
                    minimum=0,
                    allow_none=True,
                ),
                "max_per_registration": _safe_int(
                    row.get("max_per_registration", 1),
                    field=f"{name} maximum per registration",
                    minimum=1,
                    maximum=MAX_CART_ITEM_QUANTITY,
                ),
                "available_from": _iso(start),
                "available_until": _iso(end),
                "_available_from": start,
                "_available_until": end,
                "requires_fulfillment": bool(
                    row.get("requires_fulfillment", True)
                ),
                "fulfillment_instructions": _clean_text(
                    row.get("fulfillment_instructions"), limit=2000
                ),
                "sort_order": _safe_int(
                    row.get("sort_order", 0),
                    field=f"{name} sort order",
                    minimum=0,
                ),
            }
        )
    _unique_ids(items, label="item")
    item_by_id = {row["id"]: row for row in items}

    variants: list[dict[str, Any]] = []
    for row in raw_variants:
        variant_id = _required_id(row.get("id"), field="Item option id")
        item_id = _required_id(row.get("item_id"), field="Item option item id")
        item = item_by_id.get(item_id)
        if not item:
            raise TournamentCommerceValidationError(
                f"Item option {variant_id} references an unknown item."
            )
        name = _clean_text(row.get("name"), limit=160)
        if not name:
            raise TournamentCommerceValidationError(
                f"{item['name']}: option name is required."
            )
        status = _upper(row.get("status") or "ACTIVE")
        if status not in CATALOG_STATUSES:
            raise TournamentCommerceValidationError(
                f"{item['name']} — {name}: unsupported status {status}."
            )
        delta = _minor_value(
            row,
            minor_key="price_delta_minor",
            major_key="price_delta_usd",
            field=f"{item['name']} — {name} price adjustment",
            allow_negative=True,
        )
        price = int(item["base_price_minor"]) + delta
        if price < 0:
            raise TournamentCommerceValidationError(
                f"{item['name']} — {name}: final price cannot be negative."
            )
        variants.append(
            {
                "id": variant_id,
                "item_id": item_id,
                "name": name,
                "sku": _clean_text(row.get("sku"), limit=120),
                "status": status,
                "price_delta_minor": delta,
                "price_minor": price,
                "inventory_limit": _safe_int(
                    row.get("inventory_limit"),
                    field=f"{item['name']} — {name} inventory",
                    minimum=0,
                    allow_none=True,
                ),
                "remaining": _safe_int(
                    row.get("remaining"),
                    field=f"{item['name']} — {name} remaining inventory",
                    minimum=0,
                    allow_none=True,
                ),
                "sort_order": _safe_int(
                    row.get("sort_order", 0),
                    field=f"{item['name']} — {name} sort order",
                    minimum=0,
                ),
            }
        )
    _unique_ids(variants, label="item option")
    variant_by_id = {row["id"]: row for row in variants}

    event_options: list[dict[str, Any]] = []
    for row in raw_events:
        event_id = _required_id(row.get("id"), field="Event option id")
        label = (
            _clean_text(row.get("division_name"), limit=200)
            or _clean_text(row.get("label"), limit=200)
            or _clean_text(row.get("event_family_label"), limit=200)
            or "Tournament event"
        )
        event_options.append(
            {
                "id": event_id,
                "label": label,
                "option_label": _clean_text(
                    row.get("day_label") or row.get("event_date"), limit=200
                )
                or None,
                "price_minor": _minor_value(
                    row,
                    minor_key="price_minor",
                    major_key="price_usd",
                    field=f"{label} price",
                ),
                "enabled": bool(row.get("enabled", True)),
                "status": _clean_text(row.get("status"), limit=40).lower(),
            }
        )
    _unique_ids(event_options, label="event option")
    event_by_id = {row["id"]: row for row in event_options}

    bundles: list[dict[str, Any]] = []
    for row in raw_bundles:
        bundle_id = _required_id(row.get("id"), field="Bundle id")
        name = _clean_text(row.get("name"), limit=160)
        if not name:
            raise TournamentCommerceValidationError("Bundle name is required.")
        status = _upper(row.get("status") or "DRAFT")
        if status not in CATALOG_STATUSES:
            raise TournamentCommerceValidationError(
                f"{name}: unsupported bundle status {status}."
            )
        start = _datetime(
            row.get("available_from"), field=f"{name} available-from date"
        )
        end = _datetime(
            row.get("available_until"), field=f"{name} available-until date"
        )
        if start and end and end < start:
            raise TournamentCommerceValidationError(
                f"{name}: availability end must follow its start."
            )
        bundles.append(
            {
                "id": bundle_id,
                "name": name,
                "description": _clean_text(row.get("description"), limit=2000),
                "status": status,
                "price_minor": _minor_value(
                    row,
                    minor_key="price_minor",
                    major_key="price_usd",
                    field=f"{name} price",
                ),
                "max_per_registration": _safe_int(
                    row.get("max_per_registration", 1),
                    field=f"{name} maximum per registration",
                    minimum=1,
                    maximum=MAX_CART_ITEM_QUANTITY,
                ),
                "available_from": _iso(start),
                "available_until": _iso(end),
                "_available_from": start,
                "_available_until": end,
                "sort_order": _safe_int(
                    row.get("sort_order", 0),
                    field=f"{name} sort order",
                    minimum=0,
                ),
            }
        )
    _unique_ids(bundles, label="bundle")
    bundle_by_id = {row["id"]: row for row in bundles}

    components: list[dict[str, Any]] = []
    component_keys: set[tuple[str, str, str]] = set()
    for position, row in enumerate(raw_components):
        bundle_id = _required_id(
            row.get("bundle_id"), field="Bundle component bundle id"
        )
        bundle = bundle_by_id.get(bundle_id)
        if not bundle:
            raise TournamentCommerceValidationError(
                f"Bundle component references an unknown bundle: {bundle_id}."
            )
        component_type = _upper(row.get("component_type"))
        if component_type not in COMPONENT_TYPES:
            raise TournamentCommerceValidationError(
                f"{bundle['name']}: unsupported component type."
            )
        event_option_id = _clean_text(row.get("event_option_id"), limit=160) or None
        item_id = _clean_text(row.get("item_id"), limit=160) or None
        variant_id = _clean_text(row.get("variant_id"), limit=160) or None
        if component_type == "EVENT_OPTION":
            if not event_option_id or item_id or variant_id:
                raise TournamentCommerceValidationError(
                    f"{bundle['name']}: event component target is invalid."
                )
            if event_option_id not in event_by_id:
                raise TournamentCommerceValidationError(
                    f"{bundle['name']}: event component is unavailable."
                )
            target_id = event_option_id
        else:
            variant = variant_by_id.get(str(variant_id or ""))
            if (
                not item_id
                or not variant_id
                or event_option_id
                or not variant
                or variant["item_id"] != item_id
            ):
                raise TournamentCommerceValidationError(
                    f"{bundle['name']}: item option component is invalid."
                )
            target_id = variant_id
        identity = (bundle_id, component_type, str(target_id))
        if identity in component_keys:
            raise TournamentCommerceValidationError(
                f"{bundle['name']}: duplicate component."
            )
        component_keys.add(identity)
        components.append(
            {
                "id": _clean_text(row.get("id"), limit=160)
                or f"{bundle_id}:component:{position + 1}",
                "bundle_id": bundle_id,
                "component_type": component_type,
                "event_option_id": event_option_id,
                "item_id": item_id,
                "variant_id": variant_id,
                "quantity": _safe_int(
                    row.get("quantity", 1),
                    field=f"{bundle['name']} component quantity",
                    minimum=1,
                    maximum=MAX_CART_ITEM_QUANTITY,
                ),
            }
        )

    components_by_bundle: dict[str, list[dict[str, Any]]] = {}
    for row in components:
        components_by_bundle.setdefault(row["bundle_id"], []).append(row)
    for bundle in bundles:
        if bundle["status"] == "ACTIVE" and not components_by_bundle.get(
            bundle["id"]
        ):
            raise TournamentCommerceValidationError(
                f"{bundle['name']}: an active bundle needs at least one component."
            )

    variants_by_item: dict[str, list[dict[str, Any]]] = {}
    for row in variants:
        variants_by_item.setdefault(row["item_id"], []).append(row)
    for item in items:
        if item["status"] == "ACTIVE" and not any(
            row["status"] == "ACTIVE"
            for row in variants_by_item.get(item["id"], [])
        ):
            raise TournamentCommerceValidationError(
                f"{item['name']}: an active item needs at least one active option."
            )

    promotions: list[dict[str, Any]] = []
    for row in raw_promotions:
        promotion_id = _required_id(row.get("id"), field="Offer id")
        name = _clean_text(row.get("name"), limit=160)
        if not name:
            raise TournamentCommerceValidationError("Offer name is required.")
        promotion_type = _upper(row.get("promotion_type"))
        target_type = _upper(row.get("target_type"))
        status = _upper(row.get("status") or "DRAFT")
        if promotion_type not in PROMOTION_TYPES:
            raise TournamentCommerceValidationError(
                f"{name}: unsupported offer type."
            )
        if target_type not in PROMOTION_TARGET_TYPES:
            raise TournamentCommerceValidationError(
                f"{name}: unsupported offer target."
            )
        if status not in CATALOG_STATUSES:
            raise TournamentCommerceValidationError(
                f"{name}: unsupported offer status."
            )
        item_id = _clean_text(row.get("item_id"), limit=160) or None
        variant_id = _clean_text(row.get("variant_id"), limit=160) or None
        bundle_id = _clean_text(row.get("bundle_id"), limit=160) or None
        valid_target = (
            target_type == "ITEM"
            and item_id in item_by_id
            and not variant_id
            and not bundle_id
        ) or (
            target_type == "ITEM_VARIANT"
            and item_id in item_by_id
            and variant_id in variant_by_id
            and variant_by_id[str(variant_id)]["item_id"] == item_id
            and not bundle_id
        ) or (
            target_type == "BUNDLE"
            and bundle_id in bundle_by_id
            and not item_id
            and not variant_id
        )
        if not valid_target:
            raise TournamentCommerceValidationError(
                f"{name}: offer target is invalid."
            )
        start = _datetime(row.get("starts_at"), field=f"{name} start date")
        end = _datetime(row.get("ends_at"), field=f"{name} end date")
        if start and end and end < start:
            raise TournamentCommerceValidationError(
                f"{name}: offer end must follow its start."
            )
        giveaway_limit = _safe_int(
            row.get("giveaway_limit"),
            field=f"{name} giveaway limit",
            minimum=1,
            allow_none=True,
        )
        if promotion_type != "DATE_WINDOW_FREE" and giveaway_limit is None:
            raise TournamentCommerceValidationError(
                f"{name}: a giveaway limit is required."
            )
        promotions.append(
            {
                "id": promotion_id,
                "name": name,
                "promotion_type": promotion_type,
                "target_type": target_type,
                "item_id": item_id,
                "variant_id": variant_id,
                "bundle_id": bundle_id,
                "starts_at": _iso(start),
                "ends_at": _iso(end),
                "_available_from": start,
                "_available_until": end,
                "giveaway_limit": giveaway_limit,
                "per_registration_limit": _safe_int(
                    row.get("per_registration_limit", 1),
                    field=f"{name} free quantity per registration",
                    minimum=1,
                    maximum=MAX_CART_ITEM_QUANTITY,
                ),
                "priority": _safe_int(
                    row.get("priority", 100),
                    field=f"{name} priority",
                    minimum=0,
                ),
                "status": status,
            }
        )
    _unique_ids(promotions, label="offer")

    normalized = {
        "currency": "USD",
        "catalog_revision": int(catalog.get("catalog_revision") or 0),
        "items": sorted(items, key=lambda row: (row["sort_order"], row["id"])),
        "variants": sorted(
            variants,
            key=lambda row: (row["item_id"], row["sort_order"], row["id"]),
        ),
        "bundles": sorted(
            bundles, key=lambda row: (row["sort_order"], row["id"])
        ),
        "bundle_components": sorted(
            components,
            key=lambda row: (
                row["bundle_id"],
                row["component_type"],
                str(row.get("event_option_id") or row.get("variant_id") or ""),
            ),
        ),
        "promotions": sorted(
            promotions, key=lambda row: (row["priority"], row["id"])
        ),
        "event_options": sorted(event_options, key=lambda row: row["id"]),
    }
    supplied_fingerprint = _clean_text(
        catalog.get("catalog_fingerprint"), limit=128
    )
    normalized["catalog_fingerprint"] = supplied_fingerprint or stable_fingerprint(
        _catalog_fingerprint_payload(normalized)
    )
    return normalized


normalize_commerce_catalog = normalize_tournament_commerce_catalog


def tournament_commerce_catalog_payload(
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the flat JSON payload expected by the catalog replacement RPC."""

    normalized = normalize_tournament_commerce_catalog(catalog)

    def public_row(
        row: Mapping[str, Any], *, exclude: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
        return {
            key: value
            for key, value in row.items()
            if not key.startswith("_")
            and key != "remaining"
            and key not in exclude
        }

    payload = {
        "currency": "USD",
        "items": [public_row(row) for row in normalized["items"]],
        "variants": [
            public_row(row, exclude=frozenset({"price_minor"}))
            for row in normalized["variants"]
        ],
        "bundles": [public_row(row) for row in normalized["bundles"]],
        "bundle_components": [
            public_row(row) for row in normalized["bundle_components"]
        ],
        "promotions": [public_row(row) for row in normalized["promotions"]],
    }
    payload["catalog_fingerprint"] = stable_fingerprint(payload)
    return payload


def _promotion_eligible(
    promotion: Mapping[str, Any],
    *,
    quote_at: datetime,
    registration_submitted_at: datetime | None,
    registrant_position: int | None,
    used_claims: int,
) -> tuple[bool, int | None, str]:
    if promotion.get("status") != "ACTIVE":
        return False, 0, "inactive"
    promotion_type = str(promotion["promotion_type"])
    limit = promotion.get("giveaway_limit")
    if promotion_type == "DATE_WINDOW_FREE":
        eligibility_time = registration_submitted_at or quote_at
        if not _available_at(promotion, at=eligibility_time):
            return False, 0, "outside_date_window"
        return True, None, "eligible"
    if promotion_type == "FIRST_N_REGISTRANTS":
        if registrant_position is None:
            return False, 0, "registrant_position_unavailable"
        if int(registrant_position) > int(limit or 0):
            return False, 0, "registrant_giveaway_full"
        return True, int(promotion["per_registration_limit"]), "eligible"
    remaining = max(0, int(limit or 0) - max(0, int(used_claims or 0)))
    if remaining <= 0:
        return False, 0, "claim_giveaway_full"
    return True, remaining, "eligible"


def _promotion_matches_line(
    promotion: Mapping[str, Any], line: Mapping[str, Any]
) -> bool:
    target_type = str(promotion.get("target_type"))
    return (
        target_type == "ITEM"
        and line.get("line_type") == "ITEM"
        and str(line.get("item_id")) == str(promotion.get("item_id"))
    ) or (
        target_type == "ITEM_VARIANT"
        and line.get("line_type") == "ITEM"
        and str(line.get("variant_id")) == str(promotion.get("variant_id"))
    ) or (
        target_type == "BUNDLE"
        and line.get("line_type") == "BUNDLE"
        and str(line.get("bundle_id")) == str(promotion.get("bundle_id"))
    )


def _allocate_promotions(
    *,
    lines: list[Mapping[str, Any]],
    promotions: list[Mapping[str, Any]],
    promo_remaining: Mapping[str, int | None],
) -> tuple[dict[int, list[tuple[dict[str, Any], int]]], int]:
    """Exact, deterministic maximum-savings free-unit allocation."""

    ranked_promotions = [
        dict(promotion)
        for promotion in sorted(
            promotions,
            key=lambda row: (int(row["priority"]), str(row["id"])),
        )
        if int(promo_remaining.get(str(promotion["id"])) or 0) > 0
    ]
    ranked_lines = sorted(
        [
            (index, dict(line))
            for index, line in enumerate(lines)
            if int(line.get("quantity") or 0) > 0
            and int(line.get("final_unit_minor") or 0) > 0
        ],
        key=lambda entry: (
            -int(entry[1].get("final_unit_minor") or 0),
            str(entry[1].get("line_key") or ""),
            entry[0],
        ),
    )
    if not ranked_promotions or not ranked_lines:
        return {}, 0

    node_count = 2 + len(ranked_promotions) + len(ranked_lines)
    graph: list[list[list[int]]] = [[] for _ in range(node_count)]
    source = 0
    promotion_offset = 1
    line_offset = promotion_offset + len(ranked_promotions)
    sink = node_count - 1

    def add_edge(start: int, end: int, capacity: int, cost: int) -> int:
        forward_index = len(graph[start])
        reverse_index = len(graph[end])
        graph[start].append([end, reverse_index, capacity, cost])
        graph[end].append([start, forward_index, 0, -cost])
        return forward_index

    total_capacity = 0
    for promotion_index, promotion in enumerate(ranked_promotions):
        capacity = int(promo_remaining.get(str(promotion["id"])) or 0)
        total_capacity += capacity
        add_edge(source, promotion_offset + promotion_index, capacity, 0)
    for line_index, (_original_index, line) in enumerate(ranked_lines):
        add_edge(
            line_offset + line_index,
            sink,
            int(line.get("quantity") or 0),
            0,
        )

    allocation_edges: list[
        tuple[int, dict[str, Any], int, int, int]
    ] = []
    money_scale = 1_000_000_000
    for promotion_index, promotion in enumerate(ranked_promotions):
        promotion_node = promotion_offset + promotion_index
        for line_index, (original_index, line) in enumerate(ranked_lines):
            if not _promotion_matches_line(promotion, line):
                continue
            capacity = min(
                int(promo_remaining.get(str(promotion["id"])) or 0),
                int(line.get("quantity") or 0),
            )
            if capacity <= 0:
                continue
            tie_cost = promotion_index * 10_000 + line_index
            edge_index = add_edge(
                promotion_node,
                line_offset + line_index,
                capacity,
                -int(line.get("final_unit_minor") or 0) * money_scale
                + tie_cost,
            )
            allocation_edges.append(
                (
                    original_index,
                    promotion,
                    promotion_node,
                    edge_index,
                    capacity,
                )
            )

    while total_capacity > 0:
        distances: list[int | None] = [None] * node_count
        previous: list[tuple[int, int] | None] = [None] * node_count
        in_queue = [False] * node_count
        distances[source] = 0
        queue: deque[int] = deque([source])
        in_queue[source] = True
        while queue:
            node = queue.popleft()
            in_queue[node] = False
            for edge_index, edge in enumerate(graph[node]):
                destination, _reverse, capacity, cost = edge
                if capacity <= 0 or distances[node] is None:
                    continue
                candidate_distance = int(distances[node]) + cost
                if (
                    distances[destination] is None
                    or candidate_distance < int(distances[destination])
                ):
                    distances[destination] = candidate_distance
                    previous[destination] = (node, edge_index)
                    if not in_queue[destination]:
                        queue.append(destination)
                        in_queue[destination] = True
        if distances[sink] is None or int(distances[sink]) >= 0:
            break
        path_capacity = total_capacity
        node = sink
        while node != source:
            prior = previous[node]
            if prior is None:
                path_capacity = 0
                break
            start, edge_index = prior
            path_capacity = min(path_capacity, graph[start][edge_index][2])
            node = start
        if path_capacity <= 0:
            break
        node = sink
        while node != source:
            prior = previous[node]
            if prior is None:
                break
            start, edge_index = prior
            edge = graph[start][edge_index]
            edge[2] -= path_capacity
            graph[node][edge[1]][2] += path_capacity
            node = start
        total_capacity -= path_capacity

    allocations: dict[int, list[tuple[dict[str, Any], int]]] = {}
    savings = 0
    for original_index, promotion, node, edge_index, initial_capacity in (
        allocation_edges
    ):
        remaining_capacity = graph[node][edge_index][2]
        quantity = initial_capacity - remaining_capacity
        if quantity <= 0:
            continue
        allocations.setdefault(original_index, []).append(
            (promotion, quantity)
        )
        savings += (
            int(lines[original_index].get("final_unit_minor") or 0)
            * quantity
        )
    for line_allocations in allocations.values():
        line_allocations.sort(
            key=lambda entry: (
                int(entry[0]["priority"]),
                str(entry[0]["id"]),
            )
        )
    return allocations, savings


def _bundle_candidates(
    *,
    normalized: Mapping[str, Any],
    available_counts: Mapping[str, int],
    unit_prices: Mapping[str, int],
    at: datetime,
) -> list[_BundleCandidate]:
    components_by_bundle: dict[str, list[Mapping[str, Any]]] = {}
    for component in normalized["bundle_components"]:
        components_by_bundle.setdefault(
            str(component["bundle_id"]), []
        ).append(component)
    candidates: list[_BundleCandidate] = []
    for bundle in normalized["bundles"]:
        if bundle["status"] != "ACTIVE" or not _available_at(bundle, at=at):
            continue
        component_counts: dict[str, int] = {}
        for component in components_by_bundle.get(str(bundle["id"]), []):
            if component["component_type"] == "EVENT_OPTION":
                key = f"event:{component['event_option_id']}"
            else:
                key = f"variant:{component['variant_id']}"
            component_counts[key] = component_counts.get(key, 0) + int(
                component["quantity"]
            )
        if not component_counts or any(
            key not in available_counts or key not in unit_prices
            for key in component_counts
        ):
            continue
        if any(
            available_counts.get(key, 0) < quantity
            for key, quantity in component_counts.items()
        ):
            continue
        regular = sum(
            int(unit_prices[key]) * quantity
            for key, quantity in component_counts.items()
        )
        price = int(bundle["price_minor"])
        candidates.append(
            _BundleCandidate(
                bundle=dict(bundle),
                components=tuple(sorted(component_counts.items())),
                regular_minor=regular,
                price_minor=price,
                savings_minor=regular - price,
            )
        )
    if len(candidates) > MAX_BUNDLE_CANDIDATES:
        raise TournamentCommerceValidationError(
            "This cart matches too many bundles to price safely."
        )
    return sorted(
        candidates,
        key=lambda candidate: (
            int(candidate.bundle["sort_order"]),
            str(candidate.bundle["id"]),
        ),
    )


def _optimize_bundles(
    *,
    candidates: list[_BundleCandidate],
    available_counts: Mapping[str, int],
    source_lines: Mapping[str, Mapping[str, Any]],
    promotions: list[Mapping[str, Any]],
    promo_remaining: Mapping[str, int | None],
) -> tuple[int, ...]:
    resource_keys = tuple(sorted(available_counts))
    initial = tuple(int(available_counts[key]) for key in resource_keys)
    positions = {key: index for index, key in enumerate(resource_keys)}
    choice_savings: list[tuple[int, ...]] = []
    for candidate in candidates:
        max_by_stock = min(
            initial[positions[key]] // quantity
            for key, quantity in candidate.components
        )
        max_quantity = min(
            max_by_stock,
            int(candidate.bundle.get("max_per_registration") or 1),
        )
        savings_by_quantity = []
        for quantity in range(max_quantity + 1):
            promotion_savings = _allocate_promotions(
                lines=[
                    {
                        "line_key": f"bundle:{candidate.bundle['id']}",
                        "line_type": "BUNDLE",
                        "bundle_id": candidate.bundle["id"],
                        "quantity": quantity,
                        "final_unit_minor": candidate.price_minor,
                    }
                ],
                promotions=promotions,
                promo_remaining=promo_remaining,
            )[1]
            savings_by_quantity.append(
                candidate.savings_minor * quantity + promotion_savings
            )
        choice_savings.append(tuple(savings_by_quantity))

    def better(
        score: int,
        counts: tuple[int, ...],
        best_score: int,
        best_counts: tuple[int, ...],
    ) -> bool:
        if score != best_score:
            return score > best_score
        if sum(counts) != sum(best_counts):
            return sum(counts) < sum(best_counts)
        return counts > best_counts

    @lru_cache(maxsize=50_000)
    def solve(
        index: int, remaining: tuple[int, ...]
    ) -> tuple[int, tuple[int, ...]]:
        if index >= len(candidates):
            remaining_lines = [
                {
                    **dict(source_lines[key]),
                    "quantity": int(remaining[position]),
                }
                for position, key in enumerate(resource_keys)
                if int(remaining[position]) > 0
            ]
            promotion_savings = _allocate_promotions(
                lines=remaining_lines,
                promotions=promotions,
                promo_remaining=promo_remaining,
            )[1]
            return promotion_savings, ()
        candidate = candidates[index]
        component_positions = [
            (positions[key], needed) for key, needed in candidate.components
        ]
        max_quantity = min(
            min(
                remaining[position] // needed
                for position, needed in component_positions
            ),
            int(candidate.bundle.get("max_per_registration") or 1),
        )
        best_score = -10**18
        best_counts: tuple[int, ...] = ()
        for quantity in range(max_quantity + 1):
            next_remaining = list(remaining)
            for position, needed in component_positions:
                next_remaining[position] -= needed * quantity
            tail_savings, tail_counts = solve(
                index + 1, tuple(next_remaining)
            )
            score = choice_savings[index][quantity] + tail_savings
            counts = (quantity,) + tail_counts
            if not best_counts or better(
                score, counts, best_score, best_counts
            ):
                best_score = score
                best_counts = counts
        return best_score, best_counts

    return solve(0, initial)[1] if candidates else ()


def _scaled_components(
    components: Iterable[Mapping[str, Any]], quantity: int
) -> list[dict[str, Any]]:
    return [
        {
            **dict(component),
            "total_quantity": int(component["quantity_per_bundle"]) * quantity,
        }
        for component in components
    ]


def quote_tournament_commerce(
    catalog: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    quote_at: datetime | None = None,
    registration_submitted_at: datetime | None = None,
    registrant_position: int | None = None,
    promotion_usage: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build a server-authoritative quote for events and optional extras."""

    normalized = normalize_tournament_commerce_catalog(catalog)
    if not isinstance(request, Mapping):
        raise TournamentCommerceValidationError("Commerce selection is invalid.")
    now = quote_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    if registration_submitted_at and registration_submitted_at.tzinfo is None:
        registration_submitted_at = registration_submitted_at.replace(
            tzinfo=timezone.utc
        )
    if registration_submitted_at:
        registration_submitted_at = registration_submitted_at.astimezone(
            timezone.utc
        )

    event_ids = sorted(
        {
            _required_id(value, field="Event option id")
            for value in (request.get("event_option_ids") or [])
        }
    )
    if len(event_ids) > MAX_EVENT_SELECTIONS:
        raise TournamentCommerceValidationError(
            "Too many tournament events were selected."
        )
    event_by_id = {
        str(row["id"]): row for row in normalized["event_options"]
    }
    event_lines: dict[str, dict[str, Any]] = {}
    for event_id in event_ids:
        event = event_by_id.get(event_id)
        if not event or not event["enabled"] or event.get("status") in {
            "closed",
            "cancelled",
            "archived",
        }:
            raise TournamentCommerceValidationError(
                "A selected tournament event is no longer available."
            )
        key = f"event:{event_id}"
        event_lines[key] = {
            "line_key": key,
            "line_type": "EVENT",
            "event_option_id": event_id,
            "item_id": None,
            "variant_id": None,
            "bundle_id": None,
            "promotion_id": None,
            "label": event["label"],
            "option_label": event.get("option_label"),
            "quantity": 1,
            "list_unit_minor": int(event["price_minor"]),
            "final_unit_minor": int(event["price_minor"]),
            "requires_fulfillment": False,
        }

    selections_by_variant: dict[str, int] = {}
    for selection in request.get("item_selections") or []:
        if not isinstance(selection, Mapping):
            raise TournamentCommerceValidationError(
                "An extras selection is invalid."
            )
        variant_id = _required_id(
            selection.get("variant_id"), field="Item option id"
        )
        quantity = int(
            _safe_int(
                selection.get("quantity", 1),
                field="Item quantity",
                minimum=1,
                maximum=MAX_CART_ITEM_QUANTITY,
            )
            or 0
        )
        selections_by_variant[variant_id] = (
            selections_by_variant.get(variant_id, 0) + quantity
        )
        if selections_by_variant[variant_id] > MAX_CART_ITEM_QUANTITY:
            raise TournamentCommerceValidationError(
                "An extras quantity exceeds the per-registration limit."
            )
    if len(selections_by_variant) > MAX_CART_DISTINCT_ITEMS:
        raise TournamentCommerceValidationError(
            "Too many different extras were selected."
        )

    item_by_id = {str(row["id"]): row for row in normalized["items"]}
    variant_by_id = {
        str(row["id"]): row for row in normalized["variants"]
    }
    item_totals: dict[str, int] = {}
    item_lines: dict[str, dict[str, Any]] = {}
    inventory_requirements: dict[str, dict[str, Any]] = {}
    for variant_id in sorted(selections_by_variant):
        quantity = selections_by_variant[variant_id]
        variant = variant_by_id.get(variant_id)
        item = item_by_id.get(str((variant or {}).get("item_id") or ""))
        if (
            not variant
            or not item
            or variant["status"] != "ACTIVE"
            or item["status"] != "ACTIVE"
            or not _available_at(item, at=now)
        ):
            raise TournamentCommerceValidationError(
                "A selected extra is no longer available."
            )
        item_id = str(item["id"])
        item_totals[item_id] = item_totals.get(item_id, 0) + quantity
        if item_totals[item_id] > int(item["max_per_registration"]):
            raise TournamentCommerceValidationError(
                f"{item['name']}: select no more than "
                f"{item['max_per_registration']} per registration."
            )
        if variant.get("remaining") is not None and quantity > int(
            variant["remaining"]
        ):
            raise TournamentCommerceValidationError(
                f"{item['name']} — {variant['name']} is no longer available "
                "in that quantity."
            )
        key = f"variant:{variant_id}"
        item_lines[key] = {
            "line_key": key,
            "line_type": "ITEM",
            "event_option_id": None,
            "item_id": item_id,
            "variant_id": variant_id,
            "bundle_id": None,
            "promotion_id": None,
            "label": item["name"],
            "option_label": variant["name"],
            "sku": variant.get("sku") or "",
            "quantity": quantity,
            "list_unit_minor": int(variant["price_minor"]),
            "final_unit_minor": int(variant["price_minor"]),
            "requires_fulfillment": bool(item["requires_fulfillment"]),
            "fulfillment_instructions": item.get(
                "fulfillment_instructions"
            )
            or "",
        }
        inventory_requirements[variant_id] = {
            "item_id": item_id,
            "variant_id": variant_id,
            "quantity": quantity,
            "item_inventory_limit": item.get("inventory_limit"),
            "variant_inventory_limit": variant.get("inventory_limit"),
            "item_remaining": item.get("remaining"),
            "variant_remaining": variant.get("remaining"),
        }
    for item_id, quantity in item_totals.items():
        item = item_by_id[item_id]
        if item.get("remaining") is not None and quantity > int(
            item["remaining"]
        ):
            raise TournamentCommerceValidationError(
                f"{item['name']} is no longer available in that quantity."
            )

    quote_request = {
        "event_option_ids": event_ids,
        "item_selections": [
            {"variant_id": variant_id, "quantity": selections_by_variant[variant_id]}
            for variant_id in sorted(selections_by_variant)
        ],
    }
    source_lines = {**event_lines, **item_lines}
    available_counts = {
        key: int(line["quantity"]) for key, line in source_lines.items()
    }
    unit_prices = {
        key: int(line["list_unit_minor"]) for key, line in source_lines.items()
    }

    usage = {str(key): int(value) for key, value in (promotion_usage or {}).items()}
    promotion_diagnostics: list[dict[str, Any]] = []
    promo_remaining: dict[str, int | None] = {}
    eligible_promotions: list[dict[str, Any]] = []
    for promotion in normalized["promotions"]:
        eligible, available, reason = _promotion_eligible(
            promotion,
            quote_at=now,
            registration_submitted_at=registration_submitted_at,
            registrant_position=registrant_position,
            used_claims=usage.get(str(promotion["id"]), 0),
        )
        per_registration = int(promotion["per_registration_limit"])
        promo_remaining[str(promotion["id"])] = (
            min(per_registration, available)
            if available is not None
            else per_registration
        )
        promotion_diagnostics.append(
            {
                "promotion_id": promotion["id"],
                "name": promotion["name"],
                "eligible": eligible,
                "available_quantity": promo_remaining[str(promotion["id"])],
                "reason": reason,
            }
        )
        if eligible and int(promo_remaining[str(promotion["id"])] or 0) > 0:
            eligible_promotions.append(dict(promotion))

    candidates = _bundle_candidates(
        normalized=normalized,
        available_counts=available_counts,
        unit_prices=unit_prices,
        at=now,
    )
    bundle_counts = _optimize_bundles(
        candidates=candidates,
        available_counts=available_counts,
        source_lines=source_lines,
        promotions=eligible_promotions,
        promo_remaining=promo_remaining,
    )

    remaining = dict(available_counts)
    base_lines: list[dict[str, Any]] = []
    for candidate, quantity in zip(candidates, bundle_counts):
        if quantity <= 0:
            continue
        component_snapshot: list[dict[str, Any]] = []
        for key, component_quantity in candidate.components:
            total_component_quantity = component_quantity * quantity
            remaining[key] -= total_component_quantity
            source_line = source_lines.get(key) or {}
            component_snapshot.append(
                {
                    "resource_key": key,
                    "component_type": (
                        "EVENT_OPTION"
                        if key.startswith("event:")
                        else "ITEM_VARIANT"
                    ),
                    "event_option_id": source_line.get("event_option_id"),
                    "item_id": source_line.get("item_id"),
                    "variant_id": source_line.get("variant_id"),
                    "label": source_line.get("label"),
                    "option_label": source_line.get("option_label"),
                    "sku": source_line.get("sku"),
                    "quantity_per_bundle": component_quantity,
                    "total_quantity": total_component_quantity,
                    "unit_price_minor": int(unit_prices[key]),
                    "requires_fulfillment": bool(
                        source_line.get("requires_fulfillment")
                    ),
                    "fulfillment_instructions": source_line.get(
                        "fulfillment_instructions"
                    )
                    or "",
                }
            )
        base_lines.append(
            {
                "line_key": f"bundle:{candidate.bundle['id']}",
                "line_type": "BUNDLE",
                "event_option_id": None,
                "item_id": None,
                "variant_id": None,
                "bundle_id": candidate.bundle["id"],
                "promotion_id": None,
                "label": candidate.bundle["name"],
                "option_label": None,
                "quantity": quantity,
                "list_unit_minor": candidate.regular_minor,
                "final_unit_minor": candidate.price_minor,
                "requires_fulfillment": any(
                    component["requires_fulfillment"]
                    for component in component_snapshot
                ),
                "component_snapshot": component_snapshot,
            }
        )
    for key in sorted(remaining):
        quantity = int(remaining[key])
        if quantity <= 0:
            continue
        base_lines.append({**source_lines[key], "quantity": quantity})

    allocations, _promotion_savings = _allocate_promotions(
        lines=base_lines,
        promotions=eligible_promotions,
        promo_remaining=promo_remaining,
    )
    final_lines: list[dict[str, Any]] = []
    promotion_summary: dict[str, dict[str, Any]] = {}
    for index, line in enumerate(base_lines):
        original_quantity = int(line["quantity"])
        allocated = 0
        line_allocations = allocations.get(index, [])
        for promotion, quantity in line_allocations:
            allocated += quantity
            promoted = {
                **line,
                "line_key": f"{line['line_key']}:promotion:{promotion['id']}",
                "promotion_id": promotion["id"],
                "promotion_name": promotion["name"],
                "quantity": quantity,
                "final_unit_minor": 0,
            }
            if promoted.get("component_snapshot"):
                promoted["component_snapshot"] = _scaled_components(
                    promoted["component_snapshot"], quantity
                )
            promoted["list_total_minor"] = int(
                promoted["list_unit_minor"]
            ) * quantity
            promoted["final_total_minor"] = 0
            promoted["savings_minor"] = promoted["list_total_minor"]
            final_lines.append(promoted)
            summary = promotion_summary.setdefault(
                str(promotion["id"]),
                {
                    "promotion_id": promotion["id"],
                    "name": promotion["name"],
                    "promotion_type": promotion["promotion_type"],
                    "quantity": 0,
                    "savings_minor": 0,
                },
            )
            summary["quantity"] += quantity
            summary["savings_minor"] += int(line["final_unit_minor"]) * quantity
        remaining_quantity = original_quantity - allocated
        if remaining_quantity > 0:
            standard = {
                **line,
                "line_key": (
                    f"{line['line_key']}:standard"
                    if line_allocations
                    else line["line_key"]
                ),
                "quantity": remaining_quantity,
            }
            if standard.get("component_snapshot"):
                standard["component_snapshot"] = _scaled_components(
                    standard["component_snapshot"], remaining_quantity
                )
            standard["list_total_minor"] = (
                int(standard["list_unit_minor"]) * remaining_quantity
            )
            standard["final_total_minor"] = (
                int(standard["final_unit_minor"]) * remaining_quantity
            )
            standard["savings_minor"] = (
                standard["list_total_minor"] - standard["final_total_minor"]
            )
            final_lines.append(standard)

    final_lines.sort(key=lambda row: str(row["line_key"]))
    list_subtotal = sum(int(row["list_total_minor"]) for row in final_lines)
    total = sum(int(row["final_total_minor"]) for row in final_lines)

    applied_bundles: list[dict[str, Any]] = []
    for candidate in candidates:
        matching = [
            row
            for row in final_lines
            if str(row.get("bundle_id")) == str(candidate.bundle["id"])
        ]
        quantity = sum(int(row["quantity"]) for row in matching)
        if quantity <= 0:
            continue
        regular = candidate.regular_minor * quantity
        bundle_price = candidate.price_minor * quantity
        final = sum(int(row["final_total_minor"]) for row in matching)
        applied_bundles.append(
            {
                "bundle_id": candidate.bundle["id"],
                "name": candidate.bundle["name"],
                "quantity": quantity,
                "regular_minor": regular,
                "bundle_price_minor": bundle_price,
                "final_minor": final,
                "promotion_savings_minor": bundle_price - final,
                "savings_minor": regular - final,
            }
        )

    fulfillment: list[dict[str, Any]] = []
    for line in final_lines:
        if line.get("line_type") == "ITEM" and line.get(
            "requires_fulfillment"
        ):
            fulfillment.append(
                {
                    "line_key": line["line_key"],
                    "item_id": line.get("item_id"),
                    "variant_id": line.get("variant_id"),
                    "bundle_id": None,
                    "label": line["label"],
                    "option_label": line.get("option_label"),
                    "sku": line.get("sku"),
                    "quantity": line["quantity"],
                    "instructions": line.get("fulfillment_instructions") or "",
                }
            )
        elif line.get("line_type") == "BUNDLE":
            for component in line.get("component_snapshot") or []:
                if not component.get("requires_fulfillment"):
                    continue
                fulfillment.append(
                    {
                        "line_key": line["line_key"],
                        "item_id": component.get("item_id"),
                        "variant_id": component.get("variant_id"),
                        "bundle_id": line.get("bundle_id"),
                        "label": component.get("label"),
                        "option_label": component.get("option_label"),
                        "sku": component.get("sku"),
                        "quantity": component.get("total_quantity"),
                        "instructions": component.get(
                            "fulfillment_instructions"
                        )
                        or "",
                    }
                )

    quote = {
        "quote_version": 1,
        "currency": normalized["currency"],
        "quoted_at": now.isoformat(),
        "registration_submitted_at": (
            registration_submitted_at.isoformat()
            if registration_submitted_at
            else None
        ),
        "registrant_position": registrant_position,
        "catalog_fingerprint": normalized["catalog_fingerprint"],
        "request": quote_request,
        "request_fingerprint": stable_fingerprint(quote_request),
        "lines": final_lines,
        "applied_bundles": applied_bundles,
        "applied_promotions": [
            promotion_summary[key] for key in sorted(promotion_summary)
        ],
        "promotion_diagnostics": promotion_diagnostics,
        "inventory": [
            inventory_requirements[key]
            for key in sorted(inventory_requirements)
        ],
        "fulfillment": fulfillment,
        "list_subtotal_minor": list_subtotal,
        "discount_minor": list_subtotal - total,
        "total_minor": total,
        "offline_payment": True,
        "payment_status": "UNPAID",
    }
    review_binding = {
        key: quote[key]
        for key in (
            "quote_version",
            "currency",
            "registrant_position",
            "catalog_fingerprint",
            "request",
            "request_fingerprint",
            "lines",
            "applied_bundles",
            "applied_promotions",
            "promotion_diagnostics",
            "inventory",
            "fulfillment",
            "list_subtotal_minor",
            "discount_minor",
            "total_minor",
            "offline_payment",
            "payment_status",
        )
    }
    quote["review_binding_version"] = 1
    quote["quote_fingerprint"] = stable_fingerprint(review_binding)
    return quote


__all__ = [
    "CATALOG_STATUSES",
    "COMPONENT_TYPES",
    "ITEM_KINDS",
    "PROMOTION_TARGET_TYPES",
    "PROMOTION_TYPES",
    "TournamentCommerceValidationError",
    "normalize_commerce_catalog",
    "normalize_tournament_commerce_catalog",
    "quote_tournament_commerce",
    "stable_fingerprint",
    "tournament_commerce_catalog_payload",
]
