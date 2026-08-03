from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException

from jupr_app.services import public_tournament_commerce_service as service
from services.api import public_tournament_commerce_routes as routes


UUID = "00000000-0000-4000-8000-000000000001"


class _Query:
    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self


class _Supabase:
    def table(self, _name):
        return _Query()


def _route(app: FastAPI, *, path: str, method: str):
    return next(
        row.endpoint
        for row in app.routes
        if getattr(row, "path", "") == path
        and method in getattr(row, "methods", set())
    )


def test_disabled_public_routes_do_not_resolve_club_or_database(monkeypatch):
    calls = []
    app = FastAPI()
    monkeypatch.setattr(routes, "is_tournament_commerce_enabled", lambda: False)
    routes.install_public_tournament_commerce_routes(
        app,
        get_club=lambda *_args: calls.append("club"),
        get_supabase_client=lambda: calls.append("database"),
        public_club_payload=lambda *_args: {},
    )

    get_endpoint = _route(
        app, path="/clubs/{club_slug}/tournament-commerce", method="GET"
    )
    post_endpoint = _route(
        app,
        path="/clubs/{club_slug}/tournament-commerce/quote",
        method="POST",
    )

    with pytest.raises(HTTPException) as get_error:
        get_endpoint("tres-palapas", UUID)
    with pytest.raises(HTTPException) as post_error:
        post_endpoint(
            "tres-palapas",
            routes.PublicTournamentCommerceQuoteRequest(
                tournament_id=UUID,
                event_option_ids=[],
                item_selections=[],
            ),
        )

    assert get_error.value.status_code == 404
    assert post_error.value.status_code == 404
    assert calls == []


def test_public_quote_route_does_not_expose_existing_order_state(monkeypatch):
    app = FastAPI()
    monkeypatch.setattr(routes, "is_tournament_commerce_enabled", lambda: True)
    monkeypatch.setattr(
        routes,
        "quote_public_tournament_commerce",
        lambda *_args, **_kwargs: {
            "ok": True,
            "quote": {"quote_fingerprint": "safe-public-quote"},
            "current_order": {
                "payment_status": "PAID",
                "updated_at": "2026-07-01T00:00:00Z",
            },
        },
    )
    routes.install_public_tournament_commerce_routes(
        app,
        get_club=lambda *_args: {"id": "club"},
        get_supabase_client=lambda: object(),
        public_club_payload=lambda *_args: {"slug": "tres-palapas"},
    )
    endpoint = _route(
        app,
        path="/clubs/{club_slug}/tournament-commerce/quote",
        method="POST",
    )

    response = endpoint(
        "tres-palapas",
        routes.PublicTournamentCommerceQuoteRequest(
            tournament_id=UUID,
            registration_id="reg_untrusted",
            event_option_ids=[],
            item_selections=[],
        ),
    )

    assert response["quote"]["quote_fingerprint"] == "safe-public-quote"
    assert "current_order" not in response


def test_named_commerce_waves_remain_distinct_and_open_allows_both(
    monkeypatch,
):
    monkeypatch.setenv("JUPR_ENABLE_TOURNAMENT_COMMERCE", "1")
    monkeypatch.setenv(
        "JUPR_ENABLE_STAGING_TOURNAMENT_COMMERCE_WRITES", "1"
    )
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-service-role")

    monkeypatch.setenv("JUPR_ENV", "production")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "public-intake-auth")
    with pytest.raises(PermissionError, match="production write policy"):
        service.require_tournament_commerce_mutation_runtime(
            actor_type="PUBLIC_REGISTRANT"
        )

    monkeypatch.setenv("JUPR_ENV", "staging")
    service.require_tournament_commerce_mutation_runtime(
        actor_type="PUBLIC_REGISTRANT"
    )
    with pytest.raises(PermissionError, match="Tournament commerce writes are disabled"):
        service.require_tournament_commerce_mutation_runtime(actor_type="ADMIN")

    monkeypatch.setenv(
        "JUPR_STAGING_WRITE_WAVE", "tournament-commerce-admin"
    )
    service.require_tournament_commerce_mutation_runtime(actor_type="ADMIN")
    with pytest.raises(PermissionError, match="Tournament commerce writes are disabled"):
        service.require_tournament_commerce_mutation_runtime(
            actor_type="PUBLIC_REGISTRANT"
        )

    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "open")
    service.require_tournament_commerce_mutation_runtime(actor_type="ADMIN")
    service.require_tournament_commerce_mutation_runtime(
        actor_type="PUBLIC_REGISTRANT"
    )
    status = service.tournament_commerce_runtime_status()
    assert status["admin_write_ready"] is True
    assert status["public_registration_write_ready"] is True


def test_public_bundle_components_include_concrete_event_and_option_labels(
    monkeypatch,
):
    monkeypatch.setenv("JUPR_ENABLE_TOURNAMENT_COMMERCE", "1")
    monkeypatch.setattr(
        service,
        "_require_public_tournament_registration_visibility",
        lambda *_args, tournament_id, **_kwargs: tournament_id,
    )
    monkeypatch.setattr(service, "_current_order", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        service,
        "_load_catalog",
        lambda *_args, **_kwargs: (
            {
                "catalog_revision": 1,
                "catalog_fingerprint": "stored-fingerprint",
            },
            {
                "currency": "USD",
                "items": [
                    {
                        "id": "shirt",
                        "name": "Tournament shirt",
                        "kind": "MERCHANDISE",
                        "status": "ACTIVE",
                        "base_price_minor": 2500,
                        "max_per_registration": 2,
                        "requires_fulfillment": True,
                        "fulfillment_instructions": "Pick up at check-in.",
                    }
                ],
                "variants": [
                    {
                        "id": "shirt-large",
                        "item_id": "shirt",
                        "name": "Large",
                        "sku": "SHIRT-L",
                        "status": "ACTIVE",
                        "price_delta_minor": 0,
                    }
                ],
                "bundles": [
                    {
                        "id": "entry-shirt",
                        "name": "Entry and shirt",
                        "status": "ACTIVE",
                        "price_minor": 5000,
                    }
                ],
                "bundle_components": [
                    {
                        "id": "bundle-event",
                        "bundle_id": "entry-shirt",
                        "component_type": "EVENT_OPTION",
                        "event_option_id": "event-open",
                        "quantity": 1,
                    },
                    {
                        "id": "bundle-shirt",
                        "bundle_id": "entry-shirt",
                        "component_type": "ITEM_VARIANT",
                        "item_id": "shirt",
                        "variant_id": "shirt-large",
                        "quantity": 1,
                    },
                ],
                "promotions": [],
                "event_options": [
                    {
                        "id": "event-open",
                        "division_name": "Open Doubles",
                        "day_label": "Saturday",
                        "price_minor": 3500,
                        "enabled": True,
                        "status": "open",
                    }
                ],
            },
        ),
    )

    catalog = service.build_public_tournament_commerce_catalog(
        object(),
        club_id="club",
        tournament_id=UUID,
    )
    components = {
        row["component_type"]: row for row in catalog["bundle_components"]
    }

    assert components["EVENT_OPTION"]["label"] == "Open Doubles"
    assert components["EVENT_OPTION"]["option_label"] == "Saturday"
    assert components["ITEM_VARIANT"]["label"] == "Tournament shirt"
    assert components["ITEM_VARIANT"]["option_label"] == "Large"
    assert components["ITEM_VARIANT"]["sku"] == "SHIRT-L"


@pytest.mark.parametrize(
    ("tournament_rows", "settings_rows"),
    [
        ([], []),
        (
            [{"id": UUID, "club_id": "club", "status": "ARCHIVED"}],
            [{"tournament_id": UUID, "registration_status": "open"}],
        ),
        (
            [{"id": UUID, "club_id": "club", "status": "REGISTRATION_OPEN"}],
            [{"tournament_id": UUID, "registration_status": "draft"}],
        ),
        (
            [{"id": UUID, "club_id": "club", "status": "REGISTRATION_OPEN"}],
            [
                {
                    "tournament_id": UUID,
                    "registration_status": "open",
                    "registration_open_at": "2100-01-01T00:00:00Z",
                }
            ],
        ),
    ],
)
def test_public_catalog_refuses_nonpublic_registration_before_commerce_reads(
    monkeypatch,
    tournament_rows,
    settings_rows,
):
    monkeypatch.setenv("JUPR_ENABLE_TOURNAMENT_COMMERCE", "1")
    calls = []

    class Query:
        def __init__(self, table):
            self.table = table

        def select(self, *_args, **_kwargs):
            return self

        def eq(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def execute(self):
            calls.append(self.table)
            rows = (
                tournament_rows
                if self.table == "tournaments"
                else settings_rows
            )
            return type("Response", (), {"data": rows})()

    class Supabase:
        def table(self, table):
            return Query(table)

    monkeypatch.setattr(
        service,
        "_current_order",
        lambda *_args, **_kwargs: pytest.fail(
            "order data must not be read before public authorization"
        ),
    )
    monkeypatch.setattr(
        service,
        "_load_catalog",
        lambda *_args, **_kwargs: pytest.fail(
            "catalog data must not be read before public authorization"
        ),
    )

    with pytest.raises(PermissionError, match="unavailable"):
        service.build_public_tournament_commerce_catalog(
            Supabase(),
            club_id="club",
            tournament_id=UUID,
        )

    assert calls[0] == "tournaments"
    assert calls == ["tournaments"] or calls == [
        "tournaments",
        "tournament_registration_settings",
    ]


def test_public_catalog_allows_current_open_registration(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_TOURNAMENT_COMMERCE", "1")
    calls = []

    class Query:
        def __init__(self, table):
            self.table = table

        def select(self, *_args, **_kwargs):
            return self

        def eq(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def execute(self):
            calls.append(self.table)
            rows = {
                "tournaments": [
                    {
                        "id": UUID,
                        "club_id": "club",
                        "status": "REGISTRATION_OPEN",
                    }
                ],
                "tournament_registration_settings": [
                    {
                        "tournament_id": UUID,
                        "registration_status": "open",
                        "registration_open_at": "2020-01-01T00:00:00Z",
                        "registration_close_at": "2100-01-01T00:00:00Z",
                    }
                ],
            }[self.table]
            return type("Response", (), {"data": rows})()

    class Supabase:
        def table(self, table):
            return Query(table)

    monkeypatch.setattr(service, "_current_order", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        service,
        "_load_catalog",
        lambda *_args, **_kwargs: (
            None,
            {
                "currency": "USD",
                "items": [],
                "variants": [],
                "bundles": [],
                "bundle_components": [],
                "promotions": [],
                "event_options": [],
            },
        ),
    )

    catalog = service.build_public_tournament_commerce_catalog(
        Supabase(),
        club_id="club",
        tournament_id=UUID,
    )

    assert catalog["available"] is False
    assert calls == ["tournaments", "tournament_registration_settings"]


def test_token_bound_edit_can_read_its_scoped_catalog_after_public_close(
    monkeypatch,
):
    monkeypatch.setenv("JUPR_ENABLE_TOURNAMENT_COMMERCE", "1")
    monkeypatch.setattr(
        service,
        "_require_public_tournament_registration_visibility",
        lambda *_args, **_kwargs: pytest.fail(
            "token-bound edit must not reuse anonymous visibility"
        ),
    )
    monkeypatch.setattr(service, "_current_order", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        service,
        "_load_catalog",
        lambda *_args, **_kwargs: (
            None,
            {
                "currency": "USD",
                "items": [],
                "variants": [],
                "bundles": [],
                "bundle_components": [],
                "promotions": [],
                "event_options": [],
            },
        ),
    )

    catalog = service.build_public_tournament_commerce_catalog(
        object(),
        club_id="club",
        tournament_id=UUID,
        registration_id="token-bound-registration",
        token_bound_edit=True,
    )

    assert catalog["available"] is False


def test_public_quote_checks_registration_visibility_before_order_data(
    monkeypatch,
):
    monkeypatch.setenv("JUPR_ENABLE_TOURNAMENT_COMMERCE", "1")
    monkeypatch.setattr(
        service,
        "_require_public_tournament_registration_visibility",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PermissionError(
                "Tournament extras are unavailable for this registration."
            )
        ),
    )
    monkeypatch.setattr(
        service,
        "_current_order",
        lambda *_args, **_kwargs: pytest.fail(
            "order data must not be read before public authorization"
        ),
    )

    with pytest.raises(PermissionError, match="unavailable"):
        service.quote_public_tournament_commerce(
            object(),
            club_id="club",
            tournament_id=UUID,
            request={"event_option_ids": [], "item_selections": []},
        )


def test_prepare_transaction_returns_server_quote_on_review_conflict(
    monkeypatch,
):
    monkeypatch.setattr(
        service, "is_tournament_commerce_enabled", lambda: True
    )
    monkeypatch.setattr(service, "_execute", lambda *_args, **_kwargs: [])
    current_quote = {
        "quote_fingerprint": "new-server-fingerprint",
        "request_fingerprint": "request-fingerprint",
        "request": {"event_option_ids": [], "item_selections": []},
    }
    monkeypatch.setattr(
        service,
        "quote_public_tournament_commerce",
        lambda *_args, **_kwargs: {
            "quote": current_quote,
            "current_order": None,
        },
    )

    with pytest.raises(
        service.TournamentCommerceQuoteChangedError
    ) as conflict:
        service.prepare_public_registration_commerce_transaction(
            _Supabase(),
            club_id="club",
            tournament_id=UUID,
            registration_id="reg_test",
            registration_email="person@example.com",
            event_option_ids=[],
            commerce={
                "item_selections": [],
                "expected_quote_fingerprint": "old-browser-review",
                "idempotency_key": UUID,
            },
        )

    assert conflict.value.current_quote == current_quote


def test_public_routes_never_accept_browser_price_snapshots():
    source = Path(
        "services/api/public_tournament_commerce_routes.py"
    ).read_text(encoding="utf-8")
    registration_routes = Path(
        "services/api/public_tournament_registration_routes.py"
    ).read_text(encoding="utf-8")

    assert "p_quote_snapshot" not in source
    assert "p_quote_snapshot" not in registration_routes
    assert "expected_quote_fingerprint" in registration_routes
    assert "item_selections" in registration_routes


def test_large_reads_use_keyset_pagination_and_batched_in_filters():
    source = Path(
        "jupr_app/services/public_tournament_commerce_service.py"
    ).read_text(encoding="utf-8")

    assert "PAGE_SIZE = 500" in source
    assert "IN_FILTER_BATCH_SIZE = 100" in source
    assert "query = query.gt(order_column, cursor)" in source
    assert "values[start : start + IN_FILTER_BATCH_SIZE]" in source
