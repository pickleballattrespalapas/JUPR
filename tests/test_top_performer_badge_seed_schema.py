from __future__ import annotations

from pathlib import Path

from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "supabase/migrations/20260720014744_seed_top_performer_badges.sql"

EXPECTED = {
    "top_performer_highest_rating": {
        "name": "Top Performer: Highest Rating",
        "prestige": 130,
        "lore": "The league closes with your rating on the peak.",
        "hint": "Finish the season with the highest mark.",
    },
    "top_performer_most_improved": {
        "name": "Top Performer: Most Improved",
        "prestige": 125,
        "lore": "The biggest climb shows up in the final tape.",
        "hint": "Make the largest rating leap in the league.",
    },
    "top_performer_best_win_pct": {
        "name": "Top Performer: Best Win %",
        "prestige": 120,
        "lore": "The league’s cleanest record shines at the top.",
        "hint": "Finish with the best win percentage.",
    },
    "top_performer_most_wins": {
        "name": "Top Performer: Most Wins",
        "prestige": 115,
        "lore": "No one stacks wins faster when the season closes.",
        "hint": "Lead the league in total wins.",
    },
}

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


def test_top_performer_seed_matches_the_python_canonical_definitions() -> None:
    canonical = {badge.badge_id: badge for badge in BADGE_DEFINITIONS if badge.badge_id in EXPECTED}

    assert set(canonical) == set(EXPECTED)
    for badge_id, expected in EXPECTED.items():
        badge = canonical[badge_id]
        assert badge.name == expected["name"]
        assert badge.prestige == expected["prestige"]
        assert badge.lore == badge.hint
        assert badge.category == "Trophies"
        assert badge.is_stackable is True
        assert badge.is_active is True
        assert badge.rarity == "legendary"
        assert badge.tier is None
        assert badge.icon_key == "trophy"
        assert badge.scope == "league"
        assert badge.state == "live"


def test_top_performer_seed_is_insert_missing_only_and_has_a_hard_postcondition() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    normalized = " ".join(sql.lower().split())

    assert "insert into public.badges" in normalized
    assert "on conflict (badge_id) do nothing" in normalized
    assert "do update" not in normalized
    assert "missing_badge_ids" in normalized
    assert "cardinality(missing_badge_ids) > 0" in normalized
    assert "raise exception" in normalized
    for badge_id, expected in EXPECTED.items():
        expected_row = " ".join(
            (
                f"( '{badge_id}',",
                f"'{expected['name']}',",
                f"{expected['prestige']},",
                "'Top Performer Awards', true, true, 'legendary', null, 'trophy',",
                f"'{expected['lore']}',",
                f"'{expected['hint']}',",
                "'league', 'live'::public.badge_state )",
            )
        )
        expected_row = expected_row.replace("null,", "null::integer,", 1)
        assert sql.count(f"'{badge_id}'") == 2
        assert expected_row.lower() in normalized


def test_top_performer_seed_aligns_only_present_v2_compatibility_columns() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    normalized = " ".join(sql.lower().split())

    assert "from pg_catalog.pg_attribute" in normalized
    assert "attribute.attrelid = 'public.badges'::regclass" in normalized
    assert "attribute.attnum > 0" in normalized
    assert "not attribute.attisdropped" in normalized
    assert "compatibility_insert_columns text := ''" in normalized
    assert "compatibility_select_columns text := ''" in normalized
    assert "if exists" in normalized
    assert "format(', %i', compatibility_column)" in normalized
    assert "format(', canonical.%i', canonical_column)" in normalized
    assert "update public.badges" not in normalized
    for compatibility_column, canonical_column in COMPATIBILITY_COLUMNS.items():
        assert f"('{compatibility_column}', '{canonical_column}')" in normalized


def test_top_performer_seed_version_is_ordered_and_unique() -> None:
    migration_files = sorted((ROOT / "supabase/migrations").glob("*.sql"))
    versions = [path.name.split("_", 1)[0] for path in migration_files]

    assert len(versions) == len(set(versions))
    assert len("20260720014744") == 14
    assert "20260720014744".isdigit()
    assert "20260719220000" < "20260720014744" < "20261020000000"
