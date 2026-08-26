from __future__ import annotations

import re
from pathlib import Path


MIGRATION = Path(
    "supabase/migrations/20260826001529_repair_tournament_game_draw_scoped_uniqueness.sql"
)


def _normalized_sql() -> str:
    return re.sub(r"\s+", " ", MIGRATION.read_text(encoding="utf-8").lower()).strip()


def test_migration_removes_both_obsolete_tournament_wide_objects() -> None:
    sql = _normalized_sql()

    assert "drop constraint if exists tournament_games_rr_unique" in sql
    assert "drop constraint if exists tournament_games_playoff_unique" in sql
    assert "drop index if exists public.tournament_games_rr_unique" in sql
    assert "drop index if exists public.tournament_games_playoff_unique" in sql


def test_migration_preserves_draw_scoped_and_legacy_uniqueness() -> None:
    sql = _normalized_sql()

    assert (
        "create unique index if not exists uq_tournament_games_draw_rr "
        "on public.tournament_games ( tournament_id, draw_id, rr_round_number, "
        "rr_slot_number ) where draw_id is not null and stage = 'round_robin'"
    ) in sql
    assert (
        "create unique index if not exists uq_tournament_games_draw_playoff "
        "on public.tournament_games ( tournament_id, draw_id, playoff_game_code ) "
        "where draw_id is not null and stage = 'playoff' and playoff_game_code is not null"
    ) in sql
    assert (
        "create unique index if not exists uq_tournament_games_legacy_rr "
        "on public.tournament_games ( tournament_id, rr_round_number, rr_slot_number ) "
        "where draw_id is null and stage = 'round_robin'"
    ) in sql
    assert (
        "create unique index if not exists uq_tournament_games_legacy_playoff "
        "on public.tournament_games ( tournament_id, playoff_game_code ) where draw_id is null "
        "and stage = 'playoff' and playoff_game_code is not null"
    ) in sql
