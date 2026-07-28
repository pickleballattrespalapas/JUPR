from pathlib import Path


def test_tournament_commerce_migration_avoids_composite_multi_target_into() -> None:
    sql = Path(
        'supabase/migrations/20260728010000_tournament_commerce.sql'
    ).read_text(encoding='utf-8')
    assert 'into v_variant, v_item' not in sql
    assert 'into v_fulfillment, v_order' not in sql
    assert sql.count('into v_pair') == 3
    assert sql.count('v_pair record;') == 2
