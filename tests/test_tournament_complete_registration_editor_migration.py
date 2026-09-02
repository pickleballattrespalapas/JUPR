import re
from pathlib import Path


MIGRATION = Path(
    "supabase/migrations/20260807150000_tournament_complete_registration_editor.sql"
)
LEGACY_SELECTION_RPC_MIGRATION = Path(
    "supabase/migrations/20260717141224_selection_update_transaction_guards.sql"
)
RPC_DECLARATION = (
    "create or replace function "
    "public.admin_update_tournament_registration_selection("
)


def _source(path: Path = MIGRATION) -> str:
    return path.read_text(encoding="utf-8")


def _normalized_sql() -> str:
    source = re.sub(r"--.*$", "", _source().lower(), flags=re.MULTILINE)
    return re.sub(r"\s+", " ", source).strip()


def _rpc_block(source: str) -> str:
    start = source.index(RPC_DECLARATION)
    end = source.index("$function$;", start) + len("$function$;")
    return source[start:end]


def _normalized(source: str) -> str:
    return re.sub(r"\s+", " ", source).strip()


def test_complete_registration_editor_uses_the_canonical_migration_path() -> None:
    assert MIGRATION.is_file()
    assert MIGRATION.parent.as_posix() == "supabase/migrations"
    assert not Path(
        "migrations/20260807150000_tournament_complete_registration_editor.sql"
    ).exists()


def test_explicit_skill_boundaries_and_partner_gender_are_forward_only() -> None:
    sql = _normalized_sql()

    assert "add column if not exists skill_min_rating numeric(4,2) null" in sql
    assert "add column if not exists skill_max_rating numeric(4,2) null" in sql
    assert "add column if not exists partner_gender text null" in sql
    assert "drop column" not in sql
    assert "drop table" not in sql


def test_five_eligibility_modes_have_coherent_cross_field_constraints() -> None:
    sql = _normalized_sql()

    assert sql.index(
        "drop constraint if exists tournament_event_options_eligibility_mode_chk"
    ) < sql.index("set eligibility_mode = 'minimum'")

    assert (
        "eligibility_mode in ( 'standard', 'minimum', 'open', "
        "'combined_rating_cap', 'custom' )"
    ) in sql
    assert (
        "eligibility_mode = 'combined_rating_cap' "
        "and combined_rating_cap is not null and combined_rating_cap > 0 "
        "and combined_rating_cap <= 14"
    ) in sql
    assert (
        "eligibility_mode <> 'combined_rating_cap' "
        "and combined_rating_cap is null"
    ) in sql
    assert (
        "eligibility_mode = 'minimum' and skill_min_rating is not null "
        "and skill_min_rating between 1 and 7 and skill_max_rating is null"
    ) in sql
    assert (
        "eligibility_mode = 'custom' and combined_rating_cap is null "
        "and (skill_min_rating is not null or skill_max_rating is not null)"
    ) in sql
    assert (
        "skill_max_rating is null or "
        "(skill_max_rating > 1 and skill_max_rating <= 7.5)"
    ) in sql
    assert "skill_min_rating < skill_max_rating" in sql

    for constraint in (
        "tournament_event_options_eligibility_mode_chk",
        "tournament_event_options_combined_cap_chk",
        "tournament_event_options_skill_bounds_chk",
        "tournament_event_options_team_contract_chk",
    ):
        assert f"validate constraint {constraint}" in sql


def test_four_player_team_contract_preserves_its_existing_eligibility_engine() -> None:
    sql = _normalized_sql()

    assert (
        "competition_format = 'four_player_team' "
        "and eligibility_mode = 'standard' and team_roster_size = 4"
    ) in sql
    assert "team_gender_rule = 'two_men_two_women'" in sql
    assert "team_tiebreak_mode in ('singles', 'skinny_relay')" in sql
    assert "eligibility_mode in ('standard', 'minimum', 'open', 'custom')" not in sql


def test_legacy_open_and_minimum_metadata_are_backfilled_before_validation() -> None:
    sql = _normalized_sql()

    assert "message = 'jupr_tournament_minimum_skill_backfill_invalid'" in sql
    assert "set eligibility_mode = 'minimum', skill_min_rating =" in sql
    assert "set eligibility_mode = 'open', skill_min_rating = null" in sql
    assert "btrim(coalesce(event.skill_label, '')) ~ '\\+\\s*$'" in sql


def test_selection_rpc_only_adds_partner_gender_to_the_guarded_contract() -> None:
    legacy = _rpc_block(_source(LEGACY_SELECTION_RPC_MIGRATION))
    replacement = _rpc_block(_source())

    assert "    'partner_gender',\n" in replacement
    assert "    partner_gender = v_candidate.partner_gender,\n" in replacement

    # The original transaction/relationship/CAS structure remains present;
    # the editor additionally allows a complete manual partner when no
    # canonical relationship exists.
    for marker in (
        "private.lock_tournament_registration_selection_scope",
        "DUPLICATE_EVENT_FAMILY",
        "PARTNER_RELATIONSHIP_CHANGED",
        "STALE_VERSION",
        "pg_catalog.set_config('jupr.selection_edit_rpc', 'on', true)",
    ):
        assert marker.lower() in replacement.lower()
        assert marker.lower() in legacy.lower()
    assert "manual partner name, email, age, and gender are required" in replacement
    assert "v_candidate.partner_gender is distinct from v_before.partner_gender" in replacement
    assert "v_candidate.partner_mode in ('NONE', 'NEEDS_PARTNER')" in replacement
    assert "v_candidate.show_on_partner_board := false" in replacement


def test_selection_rpc_retains_invoker_security_and_service_role_only_grant() -> None:
    sql = _normalized_sql()
    signature = (
        "public.admin_update_tournament_registration_selection( "
        "text, text, timestamptz, jsonb )"
    )

    assert "language plpgsql security invoker set search_path = ''" in sql
    assert "security definer" not in sql
    assert f"revoke all on function {signature} from public, anon, authenticated" in sql
    assert f"grant execute on function {signature} to service_role" in sql
    assert "perform private.lock_tournament_registration_selection_scope(" in sql
    assert "pg_catalog.set_config('jupr.selection_edit_rpc', 'on', true)" in sql
    assert "'reason', 'partner_relationship_changed'" in sql
    assert "'reason', 'duplicate_event_family'" in sql
    assert "'reason', 'stale_version'" in sql
    assert "notify pgrst, 'reload schema';" in sql


def test_selection_delete_rpc_is_atomic_cas_and_service_role_only() -> None:
    sql = _normalized_sql()
    signature = (
        "public.admin_delete_tournament_registration_selection( "
        "text, text, timestamptz )"
    )

    assert (
        "create or replace function "
        "public.admin_delete_tournament_registration_selection( "
        "p_tournament_id text, p_selection_id text, "
        "p_expected_updated_at timestamptz )"
    ) in sql
    assert "perform private.lock_tournament_registration_selection_scope(" in sql
    assert "'code', 'selection_relationship_locked'" in sql
    assert "'code', 'selection_imported_to_draw'" in sql
    assert "and selection.updated_at = p_expected_updated_at" in sql
    assert f"revoke all on function {signature} from public, anon, authenticated" in sql
    assert f"grant execute on function {signature} to service_role" in sql
