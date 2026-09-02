import re
from pathlib import Path


BASE_MIGRATION = Path(
    "supabase/migrations/"
    "20260719204700_tournament_operations_guard_surface.sql"
)
ROWTYPE_FIX_MIGRATION = Path(
    "supabase/migrations/"
    "20261025000000_fix_tournament_draw_child_trigger_rowtype.sql"
)
FUNCTION_DECLARATION = (
    "create or replace function "
    "public.touch_tournament_draw_version_from_child()"
)
GAME_ONLY_OLD_NEW_FIELDS = {
    "stage",
    "team_a_id",
    "team_b_id",
    "team_a_source",
    "team_b_source",
    "score_a",
    "score_b",
    "winner_team_id",
    "loser_team_id",
    "finalized_at",
}


def _function_definition(path: Path) -> str:
    sql = path.read_text(encoding="utf-8")
    start = sql.index(FUNCTION_DECLARATION)
    end = sql.index("$$;", start) + len("$$;")
    return sql[start:end]


def _if_block(source: str, header: str) -> tuple[int, int]:
    start = source.index(header)
    depth = 0
    token_pattern = re.compile(r"end\s+if\s*;|\bif\b", re.IGNORECASE)
    for token in token_pattern.finditer(source, start):
        if token.group(0).lower().startswith("end"):
            depth -= 1
            if depth == 0:
                return start, token.end()
        else:
            depth += 1
    raise AssertionError(f"Unclosed PL/pgSQL IF block: {header}")


def _game_rowtype_violations(source: str) -> set[str]:
    game_start, game_end = _if_block(
        source,
        "if tg_table_name = 'tournament_games' then",
    )
    violations: set[str] = set()
    for match in re.finditer(
        r"\b(?:old|new)\.([a-z_][a-z0-9_]*)",
        source,
        re.IGNORECASE,
    ):
        field = match.group(1).lower()
        if (
            field in GAME_ONLY_OLD_NEW_FIELDS
            and not game_start <= match.start() < game_end
        ):
            violations.add(field)
    return violations


def test_forward_fix_only_moves_the_game_branch_boundary() -> None:
    base = _function_definition(BASE_MIGRATION)
    fixed = _function_definition(ROWTYPE_FIX_MIGRATION)

    close_before_guards = """    end if;
  end if;
  if v_game_derivation_change and exists ("""
    keep_branch_open = """    end if;
  if v_game_derivation_change and exists ("""
    close_after_guards = """  end if;
  if tg_table_name = 'tournament_podium' and exists ("""
    close_game_then_continue = """  end if;
  end if;
  if tg_table_name = 'tournament_podium' and exists ("""

    assert base.count(close_before_guards) == 1
    expected = base.replace(close_before_guards, keep_branch_open, 1)
    assert expected.count(close_after_guards) == 1
    expected = expected.replace(
        close_after_guards,
        close_game_then_continue,
        1,
    )

    assert fixed == expected


def test_game_only_old_new_fields_are_unreachable_for_other_trigger_rowtypes() -> None:
    base = _function_definition(BASE_MIGRATION)
    fixed = _function_definition(ROWTYPE_FIX_MIGRATION)

    # This reproduces the old compatibility defect: stage was resolved outside
    # the tournament_games table guard when a tournament_teams row fired the
    # same trigger function.
    assert _game_rowtype_violations(base) == {"stage"}
    assert _game_rowtype_violations(fixed) == set()

    game_start, game_end = _if_block(
        fixed,
        "if tg_table_name = 'tournament_games' then",
    )
    game_block = fixed[game_start:game_end]
    assert "JUPR_TOURNAMENT_SCORE_PODIUM_LOCK" in game_block
    assert "JUPR_TOURNAMENT_SCORE_PUBLISHED_LOCK" in game_block
    assert "JUPR_TOURNAMENT_DOWNSTREAM_SCORE_LOCK" in game_block


def test_forward_fix_preserves_trigger_security_and_attachment_contract() -> None:
    sql = ROWTYPE_FIX_MIGRATION.read_text(encoding="utf-8")
    lower_sql = sql.lower()
    function = _function_definition(ROWTYPE_FIX_MIGRATION).lower()

    assert "returns trigger" in function
    assert "language plpgsql" in function
    assert "security invoker" in function
    assert "set search_path = public, pg_temp" in function
    assert (
        "revoke all on function "
        "public.touch_tournament_draw_version_from_child()\n"
        "  from public, anon, authenticated;"
    ) in lower_sql
    assert (
        "grant execute on function "
        "public.touch_tournament_draw_version_from_child()\n"
        "  to service_role;"
    ) in lower_sql

    # CREATE OR REPLACE retains the three existing trigger attachments. The
    # repair neither changes their timing nor introduces any table lock DDL.
    assert "create trigger" not in lower_sql
    assert "drop trigger" not in lower_sql
    assert "alter table" not in lower_sql
