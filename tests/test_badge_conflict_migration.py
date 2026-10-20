from pathlib import Path
import re


MIGRATION = Path('supabase/migrations/20261109004103_badge_conflicts_are_not_serialization_failures.sql')


def test_badge_conflict_migration_targets_only_reviewed_business_raises():
    source = MIGRATION.read_text()
    targets = re.findall(r"\('public\.([a-z_0-9]+)\([^']*\)', ([0-9]+)\)", source)
    assert targets == [
        ('apply_program_badges_v1', '1'),
        ('admin_resolve_round_robin_v1', '3'),
        ('admin_manage_badges_v1', '1'),
        ('validate_badge_season_award_v1', '1'),
    ]
    assert 'pg_get_functiondef(routine_oid)' in source
    assert 'old_conflicts + fixed_conflicts <> routine.conflict_count' in source
    assert "'errcode=''PT409'''" in source
    assert 'pg_terminate_backend' not in source
    assert 'drop function' not in source.lower()
    assert 'security definer' not in source.lower()
    assert "notify pgrst, 'reload schema'" in source
    # Existing future-dated badge prerequisites must not overwrite this repair
    # during a from-scratch ordered migration replay.
    for prerequisite in (
        '20261109001100_badge_reactivation_and_admin_seasons.sql',
        '20261109001300_program_badge_reconciliation_actor.sql',
        '20261109001400_program_badge_round_robin_state.sql',
    ):
        assert prerequisite < MIGRATION.name


def test_badge_sql_regressions_expect_business_conflicts_and_rollback_safety():
    program = Path('tests/sql/program_badge_transactions.sql').read_text()
    admin = Path('tests/sql/badge_admin_transactions.sql').read_text()
    assert 'when serialization_failure' not in program + admin
    assert program.count("when sqlstate 'PT409'") == 3
    assert admin.count("when sqlstate 'PT409'") == 2
    assert 'stale_before is distinct from stale_after' in program


def test_staging_harness_scopes_both_regressions_and_rolls_back_before_residual_checks():
    import runpy
    from uuid import UUID

    build = runpy.run_path('tests/sql/build_badge_staging_verification.py')['build']
    query = build(UUID('00000000-0000-0000-0000-000000000001'))
    assert 'begin isolation level repeatable read;' in query
    assert "set local statement_timeout='15s';" in query
    assert "set local lock_timeout='2s';" in query
    assert 'from public.players order by id limit 1;' not in query
    assert query.count("where club_id=current_setting('jupr.audit_badge_fixture_club') order by id limit 1;") == 2
    assert "club := 'badge-transaction-'" not in query
    assert query.count('insert into public.clubs(') == 1
    assert query.count('insert into public.players(') == 1
    assert 'program badge transaction tests passed' in query
    assert 'badge admin transactional tests passed' in query
    assert query.count('rollback;') == 1
    assert 'commit;' not in query
    assert query.index('rollback;') < query.index('as remaining_clubs')
    assert query.count('as remaining_') == 10
    assert 'values(-1000000001,' in query
    assert 'values(-1000000002,' in query
    assert 'values(-1000000003,' in query
