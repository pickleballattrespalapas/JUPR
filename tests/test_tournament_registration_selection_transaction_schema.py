import re
from pathlib import Path


MIGRATION = Path(
    "supabase/migrations/20260717141224_selection_update_transaction_guards.sql"
)
RELATIONSHIP_LOCK_SCOPE_FOLLOW_UP = Path(
    "supabase/migrations/20260717142402_selection_relationship_update_lock_scope.sql"
)


def _sql() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


def _normalized_sql() -> str:
    return re.sub(r"\s+", " ", _sql()).strip()


def _normalized_follow_up_sql() -> str:
    sql = RELATIONSHIP_LOCK_SCOPE_FOLLOW_UP.read_text(encoding="utf-8").lower()
    return re.sub(r"\s+", " ", sql).strip()


def _trigger_statement(trigger_name: str) -> str:
    match = re.search(
        rf"create trigger {re.escape(trigger_name)}\b.*?;",
        _normalized_sql(),
    )
    assert match is not None, f"missing trigger {trigger_name}"
    return match.group(0)


def _migration_preflight() -> str:
    match = re.search(
        r"do \$migration_preflight\$.*?\$migration_preflight\$;",
        _normalized_sql(),
    )
    assert match is not None, "missing migration preflight"
    return match.group(0)


def test_selection_transaction_guard_uses_canonical_supabase_migration():
    assert MIGRATION.is_file()
    assert MIGRATION.parent.as_posix() == "supabase/migrations"
    assert not Path(
        "migrations/20260717141224_selection_update_transaction_guards.sql"
    ).exists()


def test_selection_update_rpc_has_stable_signature_json_contract_and_security():
    sql = _normalized_sql()
    signature = (
        "public.admin_update_tournament_registration_selection( "
        "text, text, timestamptz, jsonb )"
    )

    assert (
        "create or replace function "
        "public.admin_update_tournament_registration_selection( "
        "p_tournament_id text, p_selection_id text, "
        "p_expected_updated_at timestamptz, p_patch jsonb ) returns jsonb"
    ) in sql
    assert "language plpgsql security invoker set search_path = ''" in sql
    assert "security definer" not in sql
    assert f"revoke all on function {signature} from public, anon, authenticated" in sql
    assert f"grant execute on function {signature} to service_role" in sql

    assert "return jsonb_build_object( 'ok', true, 'selection', to_jsonb(v_after) )" in sql
    assert "return jsonb_build_object( 'ok', false, 'code', 'selection_not_found' )" in sql
    assert (
        "return jsonb_build_object( 'ok', false, "
        "'code', 'selection_write_conflict', 'reason', 'stale_version' )"
    ) in sql
    assert (
        "return jsonb_build_object( 'ok', false, "
        "'code', 'selection_write_conflict', "
        "'reason', 'duplicate_event_family' )"
    ) in sql
    assert (
        "return jsonb_build_object( 'ok', false, "
        "'code', 'selection_write_conflict', "
        "'reason', 'partner_relationship_changed' )"
    ) in sql


def test_selection_insert_and_event_update_share_the_transaction_guard():
    sql = _normalized_sql()

    assert (
        "create or replace function "
        "private.guard_tournament_registration_selection_identity() "
        "returns trigger language plpgsql security invoker set search_path = ''"
    ) in sql

    trigger = _trigger_statement("guard_tournament_selection_identity")
    assert "before insert or update of" in trigger
    assert "on public.tournament_registration_selections" in trigger
    assert " for each row " in trigger
    assert (
        "execute function "
        "private.guard_tournament_registration_selection_identity()"
    ) in trigger
    assert " delete " not in f" {trigger} "

    assert "tg_op = 'insert'" in sql
    assert "pg_catalog.current_setting('jupr.selection_edit_rpc', true)" in sql
    assert "pg_catalog.set_config('jupr.selection_edit_rpc', 'on', true)" in sql


def test_every_selection_update_advances_a_monotonic_write_version():
    sql = _normalized_sql()

    assert (
        "create or replace function "
        "private.advance_tournament_registration_selection_updated_at() "
        "returns trigger language plpgsql security invoker set search_path = ''"
    ) in sql
    assert (
        "new.updated_at := greatest( pg_catalog.clock_timestamp(), "
        "old.updated_at + interval '1 microsecond' )"
    ) in sql

    trigger = _trigger_statement("advance_tournament_selection_updated_at")
    assert "before update" in trigger
    assert " update of " not in f" {trigger} "
    assert "on public.tournament_registration_selections" in trigger
    assert " for each row " in trigger
    assert (
        "execute function "
        "private.advance_tournament_registration_selection_updated_at()"
    ) in trigger


def test_relationship_insert_and_update_writes_are_serialized_but_deletes_are_not():
    sql = _normalized_sql()
    trigger_contracts = {
        "guard_tournament_partner_request_change": (
            "public.tournament_registration_partner_requests"
        ),
        "guard_tournament_team_link_change": (
            "public.tournament_registration_team_links"
        ),
        "guard_tournament_team_member_change": (
            "public.tournament_registration_team_members"
        ),
    }

    assert (
        "create or replace function "
        "private.guard_tournament_registration_relationship_change() "
        "returns trigger language plpgsql security invoker set search_path = ''"
    ) in sql
    for trigger_name, table_name in trigger_contracts.items():
        trigger = _trigger_statement(trigger_name)
        assert "before insert or update" in trigger
        assert f"on {table_name}" in trigger
        assert " for each row " in trigger
        assert (
            "execute function "
            "private.guard_tournament_registration_relationship_change()"
        ) in trigger
        assert " delete " not in f" {trigger} "


def test_relationship_lock_scope_follow_up_uses_only_new_selection_references():
    assert RELATIONSHIP_LOCK_SCOPE_FOLLOW_UP.is_file()
    sql = _normalized_follow_up_sql()
    function_match = re.search(
        r"create or replace function "
        r"private\.guard_tournament_registration_relationship_change\(\) "
        r"returns trigger.*?end \$function\$;",
        sql,
    )
    assert function_match is not None
    guard_function = function_match.group(0)

    assert (
        "v_selection_ids := array[new.requester_selection_id, "
        "new.target_selection_id]"
    ) in guard_function
    assert (
        "v_selection_ids := array[new.selection1_id, new.selection2_id]"
    ) in guard_function
    assert "v_selection_ids := array[new.selection_id]" in guard_function
    assert "old." not in guard_function
    assert (
        "perform private.lock_tournament_registration_selection_scope("
        "v_selection_ids)"
    ) in guard_function

    signature = "private.guard_tournament_registration_relationship_change()"
    assert (
        f"revoke all on function {signature} "
        "from public, anon, authenticated"
    ) in sql
    assert f"grant execute on function {signature} to service_role" in sql
    assert "notify pgrst, 'reload schema';" in sql


def test_selection_transaction_guards_only_lock_active_relationship_states():
    sql = _normalized_sql()

    assert "request.status = 'pending'" in sql
    assert "team_link.status in ('confirmed', 'admin_confirmed')" in sql
    assert "team_member.status = 'active'" in sql
    assert "new.status = 'pending'" in sql
    assert "new.status in ('confirmed', 'admin_confirmed')" in sql
    assert "new.status = 'active'" in sql

    assert "idx_tournament_team_links_confirmed_selection1" in sql
    assert "idx_tournament_team_links_confirmed_selection2" in sql
    assert "idx_tournament_team_members_active_selection" in sql


def test_preflight_rejects_inconsistent_legacy_selection_references():
    preflight = _migration_preflight()

    assert "to_regclass('public.tournament_registration_days') is null" in preflight
    assert (
        "left join public.tournament_registrations as registration "
        "on registration.id = selection.registration_id"
    ) in preflight
    assert (
        "left join public.tournament_event_options as event_option "
        "on event_option.id = selection.event_option_id"
    ) in preflight
    assert (
        "left join public.tournament_registration_days as registration_day "
        "on registration_day.id = selection.registration_day_id"
    ) in preflight
    assert "registration.tournament_id <> selection.tournament_id" in preflight
    assert "event_option.tournament_id <> selection.tournament_id" in preflight
    assert (
        "event_option.registration_day_id <> selection.registration_day_id"
    ) in preflight
    assert "registration_day.tournament_id <> selection.tournament_id" in preflight
    assert (
        "jupr_selection_invalid_target: clean inconsistent legacy selection "
        "registration, event, or day references"
    ) in preflight


def test_preflight_rejects_active_members_without_confirmed_links():
    preflight = _migration_preflight()

    assert (
        "team_member.status = 'active' and team_link.status not in "
        "('confirmed', 'admin_confirmed')"
    ) in preflight
    assert (
        "jupr_relation_invalid: clean inconsistent tournament team members"
    ) in preflight


def test_event_moves_recheck_enabled_day_and_open_event_under_lock():
    sql = _normalized_sql()

    assert (
        "v_event_changed := v_candidate.event_option_id is distinct from "
        "v_before.event_option_id"
    ) in sql
    assert "if v_event_changed then" in sql
    assert (
        "from public.tournament_registration_days as registration_day "
        "where registration_day.id = v_target_event.registration_day_id "
        "and registration_day.tournament_id = btrim(p_tournament_id) for share"
    ) in sql
    assert "or not v_target_day.enabled" in sql
    assert "or not v_target_event.enabled" in sql
    assert (
        "lower(coalesce(nullif(btrim(v_target_event.status), ''), 'draft')) "
        "not in ('open', 'tentative', 'confirmed')"
    ) in sql
    assert (
        "jupr_selection_invalid_target: target event is not open on an "
        "enabled registration day"
    ) in sql


def test_selection_transaction_contract_keeps_stable_error_markers():
    sql = _sql()
    stable_markers = {
        "jupr_selection_duplicate_family",
        "jupr_relation_invalid",
        "jupr_relation_selection_not_found",
        "jupr_selection_identity_immutable",
        "jupr_selection_event_update_requires_rpc",
        "jupr_selection_invalid_target",
        "jupr_selection_invalid_patch",
        "selection_not_found",
        "selection_write_conflict",
        "stale_version",
        "duplicate_event_family",
        "partner_relationship_changed",
    }

    for marker in stable_markers:
        assert marker in sql


def test_selection_transaction_migration_reloads_postgrest_schema_cache():
    assert "notify pgrst, 'reload schema';" in _sql()
