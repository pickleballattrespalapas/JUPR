from pathlib import Path


MIGRATION = Path(
    "supabase/migrations/"
    "20261108016000_standard_tournament_substitute_policy.sql"
)


def _sql() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


def test_standard_event_substitute_flags_are_repaired_without_touching_team_events() -> None:
    sql = _sql()

    repair = sql.split("alter table public.tournament_event_options", 1)[0]
    assert "set team_allow_substitutes = false" in repair
    assert "event.team_allow_substitutes is true" in repair
    assert "<> 'four_player_team'" in repair


def test_database_constraint_preserves_only_four_player_team_roster_policy() -> None:
    sql = _sql()

    assert "tournament_event_options_substitutes_four_player_only_v1" in sql
    assert "= 'four_player_team'\n    or team_allow_substitutes is false" in sql
    assert "not valid" in sql
    assert "validate constraint" in sql
    assert "check-in never assigns substitutes" in sql


def test_migration_uses_portable_postgres_conditional_forms() -> None:
    sql = _sql()

    for special_form in ("coalesce", "nullif", "least", "greatest"):
        assert f"pg_catalog.{special_form}" not in sql
