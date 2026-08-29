from pathlib import Path


MIGRATION = Path(
    "supabase/migrations/20261108023000_tournament_podium_row_versions.sql"
)


def test_podium_row_version_migration_installs_complete_contract() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    assert "alter table public.tournament_podium" in sql
    assert "add column if not exists updated_at timestamptz not null" in sql
    assert "alter column updated_at set not null" in sql
    assert "create or replace function public.advance_tournament_podium_updated_at()" in sql
    assert "old.updated_at + interval '1 microsecond'" in sql
    assert "create trigger trg_tournament_podium_advance_updated_at" in sql
    assert "before update on public.tournament_podium" in sql
    assert "revoke all on function public.advance_tournament_podium_updated_at()" in sql
    assert "grant execute on function public.advance_tournament_podium_updated_at()" in sql
