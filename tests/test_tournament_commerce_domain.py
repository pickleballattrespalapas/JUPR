from datetime import datetime, timezone

import pytest

from jupr_app.domain.tournament_commerce import (
    TournamentCommerceValidationError,
    normalize_tournament_commerce_catalog,
    quote_tournament_commerce,
    tournament_commerce_catalog_payload,
)


EVENT = "00000000-0000-4000-8000-000000000001"
ITEM = "00000000-0000-4000-8000-000000000002"
VARIANT = "00000000-0000-4000-8000-000000000003"
BUNDLE = "00000000-0000-4000-8000-000000000004"
PROMOTION = "00000000-0000-4000-8000-000000000005"
NOW = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)


def _catalog(
    *,
    item_price=2000,
    item_remaining=10,
    quantity_limit=4,
    bundles=None,
    bundle_components=None,
    promotions=None,
):
    return {
        "currency": "USD",
        "items": [
            {
                "id": ITEM,
                "name": "Tournament shirt",
                "kind": "MERCHANDISE",
                "status": "ACTIVE",
                "base_price_minor": item_price,
                "inventory_limit": 10,
                "remaining": item_remaining,
                "max_per_registration": quantity_limit,
                "requires_fulfillment": True,
            }
        ],
        "variants": [
            {
                "id": VARIANT,
                "item_id": ITEM,
                "name": "Medium",
                "status": "ACTIVE",
                "price_delta_minor": 0,
                "inventory_limit": 10,
                "remaining": item_remaining,
            }
        ],
        "event_options": [
            {
                "id": EVENT,
                "label": "Doubles",
                "price_minor": 4000,
                "enabled": True,
                "status": "open",
            }
        ],
        "bundles": bundles or [],
        "bundle_components": bundle_components or [],
        "promotions": promotions or [],
    }


def _request(quantity=1, *, include_event=False):
    return {
        "event_option_ids": [EVENT] if include_event else [],
        "item_selections": [{"variant_id": VARIANT, "quantity": quantity}],
    }


def test_active_item_requires_an_active_option():
    catalog = _catalog()
    catalog["variants"][0]["status"] = "ARCHIVED"

    with pytest.raises(
        TournamentCommerceValidationError,
        match="active item needs at least one active option",
    ):
        normalize_tournament_commerce_catalog(catalog)


def test_bundle_is_applied_automatically_and_snapshots_fulfillment_parts():
    catalog = _catalog(
        bundles=[
            {
                "id": BUNDLE,
                "name": "Player pack",
                "status": "ACTIVE",
                "price_minor": 5000,
                "max_per_registration": 1,
            }
        ],
        bundle_components=[
            {
                "bundle_id": BUNDLE,
                "component_type": "EVENT_OPTION",
                "event_option_id": EVENT,
                "quantity": 1,
            },
            {
                "bundle_id": BUNDLE,
                "component_type": "ITEM_VARIANT",
                "item_id": ITEM,
                "variant_id": VARIANT,
                "quantity": 1,
            },
        ],
    )

    quote = quote_tournament_commerce(
        catalog,
        _request(include_event=True),
        quote_at=NOW,
    )

    assert quote["list_subtotal_minor"] == 6000
    assert quote["discount_minor"] == 1000
    assert quote["total_minor"] == 5000
    assert quote["lines"][0]["line_type"] == "BUNDLE"
    assert len(quote["lines"][0]["component_snapshot"]) == 2
    assert quote["fulfillment"] == [
        {
            "line_key": f"bundle:{BUNDLE}",
            "bundle_id": BUNDLE,
            "item_id": ITEM,
            "variant_id": VARIANT,
            "label": "Tournament shirt",
            "option_label": "Medium",
            "sku": "",
            "quantity": 1,
            "instructions": "",
        }
    ]


def test_catalog_save_payload_keeps_bundle_price_but_not_derived_variant_price():
    catalog = _catalog(
        bundles=[
            {
                "id": BUNDLE,
                "name": "Player pack",
                "status": "DRAFT",
                "price_minor": 5000,
                "max_per_registration": 1,
            }
        ]
    )

    payload = tournament_commerce_catalog_payload(catalog)

    assert payload["bundles"][0]["price_minor"] == 5000
    assert "price_minor" not in payload["variants"][0]


def test_optimizer_compares_bundle_savings_with_item_giveaway_globally():
    catalog = _catalog(
        item_price=1000,
        bundles=[
            {
                "id": BUNDLE,
                "name": "Two-shirt pack",
                "status": "ACTIVE",
                "price_minor": 1500,
                "max_per_registration": 1,
            }
        ],
        bundle_components=[
            {
                "bundle_id": BUNDLE,
                "component_type": "ITEM_VARIANT",
                "item_id": ITEM,
                "variant_id": VARIANT,
                "quantity": 2,
            }
        ],
        promotions=[
            {
                "id": PROMOTION,
                "name": "One free shirt",
                "promotion_type": "DATE_WINDOW_FREE",
                "target_type": "ITEM_VARIANT",
                "item_id": ITEM,
                "variant_id": VARIANT,
                "starts_at": "2026-06-01T00:00:00+00:00",
                "ends_at": "2026-08-01T00:00:00+00:00",
                "per_registration_limit": 1,
                "status": "ACTIVE",
            }
        ],
    )

    quote = quote_tournament_commerce(
        catalog,
        _request(quantity=2),
        quote_at=NOW,
    )

    assert quote["list_subtotal_minor"] == 2000
    assert quote["discount_minor"] == 1000
    assert quote["total_minor"] == 1000
    assert not quote["applied_bundles"]
    assert quote["applied_promotions"][0]["promotion_id"] == PROMOTION


def test_first_n_registrants_is_free_only_within_authoritative_rank():
    catalog = _catalog(
        promotions=[
            {
                "id": PROMOTION,
                "name": "First ten free",
                "promotion_type": "FIRST_N_REGISTRANTS",
                "target_type": "ITEM",
                "item_id": ITEM,
                "giveaway_limit": 10,
                "per_registration_limit": 1,
                "status": "ACTIVE",
            }
        ]
    )

    tenth = quote_tournament_commerce(
        catalog,
        _request(),
        quote_at=NOW,
        registrant_position=10,
    )
    eleventh = quote_tournament_commerce(
        catalog,
        _request(),
        quote_at=NOW,
        registrant_position=11,
    )

    assert tenth["total_minor"] == 0
    assert tenth["discount_minor"] == 2000
    assert eleventh["total_minor"] == 2000
    assert eleventh["discount_minor"] == 0


def test_first_claim_limit_and_inventory_are_enforced():
    catalog = _catalog(
        item_remaining=2,
        promotions=[
            {
                "id": PROMOTION,
                "name": "First ten claims",
                "promotion_type": "FIRST_N_CLAIMS",
                "target_type": "ITEM_VARIANT",
                "item_id": ITEM,
                "variant_id": VARIANT,
                "giveaway_limit": 10,
                "per_registration_limit": 2,
                "status": "ACTIVE",
            }
        ],
    )

    quote = quote_tournament_commerce(
        catalog,
        _request(quantity=2),
        quote_at=NOW,
        promotion_usage={PROMOTION: 9},
    )
    assert quote["discount_minor"] == 2000
    assert quote["total_minor"] == 2000

    with pytest.raises(
        TournamentCommerceValidationError,
        match="no longer available in that quantity",
    ):
        quote_tournament_commerce(
            catalog,
            _request(quantity=3),
            quote_at=NOW,
        )


def test_existing_registration_keeps_original_date_window_eligibility():
    catalog = _catalog(
        promotions=[
            {
                "id": PROMOTION,
                "name": "Register by June 30",
                "promotion_type": "DATE_WINDOW_FREE",
                "target_type": "ITEM",
                "item_id": ITEM,
                "starts_at": "2026-06-01T00:00:00+00:00",
                "ends_at": "2026-06-30T23:59:59+00:00",
                "per_registration_limit": 1,
                "status": "ACTIVE",
            }
        ]
    )

    quote = quote_tournament_commerce(
        catalog,
        _request(),
        quote_at=NOW,
        registration_submitted_at=datetime(
            2026, 6, 30, 10, 0, tzinfo=timezone.utc
        ),
    )

    assert quote["total_minor"] == 0
    assert quote["promotion_diagnostics"][0]["reason"] == "eligible"
