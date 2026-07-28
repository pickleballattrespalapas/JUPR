from pathlib import Path


def test_playoff_threshold_case_is_parenthesized() -> None:
    sql = Path(
        'supabase/migrations/20260728020000_combined_rating_team_tournaments.sql'
    ).read_text(encoding='utf-8')
    assert 'if cardinality(v_seed_ids) < case' not in sql
    assert "if cardinality(v_seed_ids) < (\n       case" in sql
