from pathlib import Path


BASE_MIGRATION = Path(
    'supabase/migrations/20260728020000_combined_rating_team_tournaments.sql'
)
TYPE_FIX_MIGRATION = Path(
    'supabase/migrations/'
    '20261024000000_fix_tournament_team_eligibility_trigger_type.sql'
)


def test_playoff_threshold_case_is_parenthesized() -> None:
    sql = BASE_MIGRATION.read_text(encoding='utf-8')
    assert 'if cardinality(v_seed_ids) < case' not in sql
    assert "if cardinality(v_seed_ids) < (\n       case" in sql


def test_team_eligibility_trigger_compares_legacy_text_id_to_uuid_as_text() -> None:
    sql = TYPE_FIX_MIGRATION.read_text(encoding='utf-8')

    assert 'event.tournament_id = new.tournament_id::text' in sql
    assert 'event.tournament_id = new.tournament_id\n' not in sql
    assert 'review.tournament_id = new.tournament_id' in sql


def test_team_eligibility_trigger_fix_changes_only_the_mixed_type_comparison() -> None:
    marker = (
        'create or replace function '
        'public.enforce_combined_rating_draw_eligibility()'
    )
    base_function = BASE_MIGRATION.read_text(encoding='utf-8').split(
        marker,
        1,
    )[1].split('$$;', 1)[0]
    fixed_function = TYPE_FIX_MIGRATION.read_text(encoding='utf-8').split(
        marker,
        1,
    )[1].split('$$;', 1)[0]

    assert fixed_function == base_function.replace(
        'event.tournament_id = new.tournament_id',
        'event.tournament_id = new.tournament_id::text',
    )


def test_team_eligibility_trigger_fix_preserves_postgres_security_contract() -> None:
    sql = TYPE_FIX_MIGRATION.read_text(encoding='utf-8').lower()
    function = sql.split(
        'create or replace function '
        'public.enforce_combined_rating_draw_eligibility()',
        1,
    )[1].split('revoke all on function', 1)[0]

    assert 'returns trigger' in function
    assert 'language plpgsql' in function
    assert 'security invoker' in function
    assert "set search_path = ''" in function
    assert (
        'revoke all on function '
        'public.enforce_combined_rating_draw_eligibility()\n'
        '  from public, anon, authenticated;'
    ) in sql
    assert (
        'grant execute on function '
        'public.enforce_combined_rating_draw_eligibility()\n'
        '  to service_role;'
    ) in sql
