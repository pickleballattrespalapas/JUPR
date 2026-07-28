from pathlib import Path


MIGRATION = Path("supabase/migrations/20260728010000_tournament_commerce.sql")


def _source() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


COMMERCE_TABLES = (
    "tournament_commerce_catalog_state",
    "tournament_commerce_items",
    "tournament_commerce_item_variants",
    "tournament_commerce_bundles",
    "tournament_commerce_bundle_components",
    "tournament_commerce_promotions",
    "tournament_commerce_orders",
    "tournament_commerce_order_revisions",
    "tournament_commerce_order_lines",
    "tournament_commerce_inventory_reservations",
    "tournament_commerce_promotion_claims",
    "tournament_commerce_fulfillment",
    "tournament_commerce_operations",
    "tournament_commerce_audit_log",
)


def test_forward_only_commerce_tables_are_forced_rls_and_browser_revoked():
    source = _source()

    for table in COMMERCE_TABLES:
        assert f"create table if not exists public.{table}" in source
        assert f"alter table public.{table} enable row level security" in source
        assert f"alter table public.{table} force row level security" in source
        assert (
            f"revoke all on table public.{table} from public, anon, authenticated"
            in source
        )


def test_only_service_role_can_execute_commerce_mutation_rpcs():
    source = _source()
    functions = (
        "server_replace_tournament_commerce_catalog",
        "server_apply_tournament_commerce_order",
        "server_cancel_tournament_commerce_order",
        "server_update_tournament_commerce_fulfillment",
        "server_update_tournament_commerce_payment",
        "server_create_public_tournament_registration_with_commerce",
        "server_update_public_tournament_registration_with_commerce",
    )

    for function in functions:
        assert f"create or replace function public.{function}(" in source
        assert f"revoke all on function public.{function}(" in source
        assert f"grant execute on function public.{function}(" in source

    mutation_grants = source[source.index("grant execute on function public.server_replace") :]
    assert "to service_role" in mutation_grants
    assert "to anon" not in mutation_grants
    assert "to authenticated" not in mutation_grants


def test_order_revisions_and_audit_rows_are_immutable():
    source = _source()

    assert "private.tournament_commerce_reject_immutable_change()" in source
    assert "create trigger tournament_commerce_revision_immutable" in source
    assert "create trigger tournament_commerce_audit_immutable" in source
    assert (
        "before update or delete on public.tournament_commerce_order_revisions"
        in source
    )
    assert (
        "before update or delete on public.tournament_commerce_audit_log"
        in source
    )


def test_registration_create_and_edit_are_atomic_with_commerce():
    source = _source()

    assert (
        "server_create_public_tournament_registration_with_commerce(" in source
    )
    assert (
        "server_update_public_tournament_registration_with_commerce(" in source
    )
    assert "insert into public.tournament_registrations" in source
    assert "insert into public.tournament_registration_selections" in source
    assert "server_apply_tournament_commerce_order" in source
    assert "from public.tournaments tournament" in source
    assert "for update;" in source
    assert "private.tournament_commerce_operation_begin(" in source


def test_idempotency_operation_begin_serializes_concurrent_first_attempts():
    source = _source()
    start = source.index(
        "create or replace function private.tournament_commerce_operation_begin("
    )
    end = source.index("revoke all on function private.tournament_commerce_operation_begin(", start)
    operation_begin = source[start:end]

    assert (
        "on conflict (club_id, tournament_id, idempotency_key) do nothing"
        in operation_begin
    )
    assert operation_begin.index("on conflict") < operation_begin.index(
        "for update"
    )
    assert "jupr_tournament_commerce_idempotency_conflict" in operation_begin


def test_registration_cancellation_releases_commerce_with_audited_revision():
    source = _source()

    assert "create trigger tournament_commerce_registration_cancel_sync" in source
    assert "before update of status on public.tournament_registrations" in source
    assert "registration_status_cancel_sync" in source
    assert "cancellation_reason" in source
    assert "active = false" in source


def test_inventory_claims_and_large_catalog_reads_have_supporting_indexes():
    source = _source()

    for index in (
        "tournament_commerce_inventory_variant_fk_idx",
        "tournament_commerce_claims_promotion_fk_idx",
        "tournament_registrations_tournament_commerce_page_idx",
        "tournament_commerce_orders_registration_fk_idx",
        "tournament_commerce_fulfillment_tournament_fk_idx",
    ):
        assert f"create index if not exists {index}" in source
