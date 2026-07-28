from pathlib import Path


def test_team_league_hardening_uses_valid_postgres_forms() -> None:
    paths = [
        Path('supabase/migrations/20260728040000_team_league_awards_hardening.sql'),
        Path('supabase/migrations/20260728041000_team_league_registration_identity_recovery.sql'),
    ]
    sql = '\n'.join(path.read_text(encoding='utf-8') for path in paths)
    assert 'pg_catalog.coalesce(' not in sql
    assert 'pg_catalog.least(' not in sql
    assert 'pg_catalog.greatest(' not in sql
    assert 'public.digest(' not in sql
    assert 'extensions.digest(' in sql
