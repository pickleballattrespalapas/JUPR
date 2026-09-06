from __future__ import annotations

from pathlib import Path
from dataclasses import replace

from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS
from jupr_app.domain.tournament_podium import PODIUM_BADGE_MAP


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT
    / "supabase/migrations/20261108024000_tournament_podium_badge_catalog.sql"
)

COMPATIBILITY_COLUMNS = {
    "name_v2": "name",
    "prestige_v2": "prestige",
    "category_v2": "category",
    "is_active_v2": "is_active",
    "is_stackable_v2": "is_stackable",
    "lore_v2": "lore",
    "hint_v2": "hint",
    "scope_v2": "scope",
}


def _normalized(value: str) -> str:
    return " ".join(value.lower().split())


def test_podium_badge_seed_matches_the_python_canonical_definitions() -> None:
    sql = _normalized(MIGRATION.read_text(encoding="utf-8"))
    expected_ids = set(PODIUM_BADGE_MAP.values())
    canonical = {
        badge.badge_id: badge
        for badge in BADGE_DEFINITIONS
        if badge.badge_id in expected_ids
    }

    assert set(canonical) == expected_ids
    for badge_id in sorted(expected_ids):
        badge = replace(canonical[badge_id], **historical_presentation(MIGRATION.read_text(), badge_id))
        expected_row = _normalized(
            "( " + ", ".join(
                (
                    repr(badge.badge_id),
                    repr(badge.name),
                    str(badge.prestige),
                    repr(badge.category),
                    str(badge.is_stackable).lower(),
                    str(badge.is_active).lower(),
                    repr(badge.rarity),
                    "null::integer" if badge.tier is None else str(badge.tier),
                    repr(badge.icon_key),
                    repr(badge.lore),
                    repr(badge.hint),
                    repr(badge.scope),
                    "'live'::public.badge_state",
                )
            ) + " )"
        )
        assert expected_row in sql
        assert sql.count(f"'{badge_id}'") == 2


def test_podium_badge_seed_is_insert_missing_only_with_hard_postcondition() -> None:
    sql = _normalized(MIGRATION.read_text(encoding="utf-8"))

    assert "insert into public.badges" in sql
    assert "on conflict (badge_id) do nothing" in sql
    assert "do update" not in sql
    assert "update public.badges" not in sql
    assert "delete from public.badges" not in sql
    assert "missing_badge_ids" in sql
    assert "cardinality(missing_badge_ids) > 0" in sql
    assert "raise exception" in sql
    assert "to_regclass('public.badges')" in sql
    assert "to_regtype('public.badge_state')" in sql


def test_podium_badge_seed_aligns_only_present_compatibility_columns() -> None:
    sql = _normalized(MIGRATION.read_text(encoding="utf-8"))

    assert "from pg_catalog.pg_attribute" in sql
    assert "attribute.attrelid = 'public.badges'::regclass" in sql
    assert "attribute.attnum > 0" in sql
    assert "not attribute.attisdropped" in sql
    assert "compatibility_insert_columns text := ''" in sql
    assert "compatibility_select_columns text := ''" in sql
    assert "format(', %i', compatibility_column)" in sql
    assert "format(', canonical.%i', canonical_column)" in sql
    for compatibility_column, canonical_column in COMPATIBILITY_COLUMNS.items():
        assert f"('{compatibility_column}', '{canonical_column}')" in sql


def test_podium_badge_seed_version_is_ordered_and_unique() -> None:
    migration_files = sorted((ROOT / "supabase/migrations").glob("*.sql"))
    versions = [path.name.split("_", 1)[0] for path in migration_files]

    assert len(versions) == len(set(versions))
    assert len("20261108024000") == 14
    assert "20261108024000".isdigit()
    assert "20261108023000" < "20261108024000"


def historical_presentation(sql: str, badge_id: str) -> dict[str, str]:
    import re
    pattern = r"\(\s*'" + re.escape(badge_id) + r"'\s*,([\s\S]*?)\)"
    row = re.search(pattern, sql).group(0)
    strings = [value.replace("''", "'") for value in re.findall(r"'((?:''|[^'])*)'", row)]
    # id, name, category, rarity, icon, lore, hint, scope, state
    return dict(zip(("category", "lore", "hint"), (strings[2], strings[5], strings[6])))
