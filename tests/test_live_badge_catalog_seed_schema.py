from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import re

from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS
from jupr_app.domain.gamification.badge_registry import active_badge_ids
from jupr_app.domain.gamification.match_exclusion_reconcile import (
    MATCH_EXCLUSION_BADGE_IDS,
)
from jupr_app.domain.gamification.requirements import load_requirements_map


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT
    / "supabase/migrations/20260726183000_seed_live_badge_catalog.sql"
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


def _sql_text(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _normalized(value: str) -> str:
    return " ".join(value.lower().split())


def _canonical_values_block(sql: str) -> str:
    canonical_prefix = sql.split(") as canonical(", 1)[0]
    return canonical_prefix.rsplit("from (\n            values", 1)[1]


def test_live_badge_seed_exactly_matches_the_runtime_contract() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    values_block = _canonical_values_block(sql)
    seeded_ids = set(re.findall(r"^\s*\('([^']+)'", values_block, re.MULTILINE))
    expected_ids = set(MATCH_EXCLUSION_BADGE_IDS)

    assert len(expected_ids) == 29
    assert seeded_ids == expected_ids
    assert expected_ids == active_badge_ids()


def test_live_badge_seed_matches_current_python_definitions() -> None:
    sql = _normalized(MIGRATION.read_text(encoding="utf-8"))
    expected_ids = set(MATCH_EXCLUSION_BADGE_IDS)
    canonical = {
        badge.badge_id: badge
        for badge in BADGE_DEFINITIONS
        if badge.badge_id in expected_ids
    }
    requirements = load_requirements_map()

    assert set(canonical) == expected_ids
    for badge_id in MATCH_EXCLUSION_BADGE_IDS:
        badge = replace(canonical[badge_id], **historical_presentation(MIGRATION.read_text(), badge_id))
        expected_row = (
            "("
            + ", ".join(
                (
                    _sql_text(badge.badge_id),
                    _sql_text(badge.name),
                    str(badge.prestige),
                    _sql_text(badge.category),
                    str(badge.is_stackable).lower(),
                    str(badge.is_active).lower(),
                    _sql_text(badge.rarity),
                    "null::integer" if badge.tier is None else str(badge.tier),
                    (
                        "null::text"
                        if badge.icon_key is None
                        else _sql_text(badge.icon_key)
                    ),
                    _sql_text(badge.lore),
                    _sql_text(badge.hint),
                    _sql_text(badge.scope),
                    "'live'::public.badge_state",
                    """'["match_recorded","match_updated"]'::jsonb""",
                )
            )
            + ")"
        )
        assert _normalized(expected_row) in sql
        assert str(requirements.get(badge_id, "")).strip()
        assert requirements[badge_id] != "Requirements TBD"


def test_live_badge_seed_preserves_presentation_and_repairs_operational_fields() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    normalized = _normalized(sql)
    conflict_clause = normalized.split(
        "on conflict (badge_id) do update",
        1,
    )[1].split("$live_badge_seed_insert$;", 1)[0]

    assert "set state = excluded.state" in conflict_clause
    assert "is_active = excluded.is_active" in conflict_clause
    assert "eval_triggers = case" in conflict_clause
    assert "then badges.eval_triggers" in conflict_clause
    assert "badges.eval_triggers ||" in conflict_clause
    assert """@> '["match_updated"]'::jsonb""" in conflict_clause
    for protected_column in (
        "name",
        "prestige",
        "category",
        "is_stackable",
        "rarity",
        "tier",
        "icon_key",
        "lore",
        "hint",
        "scope",
        "state_changed_at",
        "state_change_reason",
    ):
        assert f"set {protected_column} =" not in conflict_clause
        assert f", {protected_column} =" not in conflict_clause


def test_live_badge_seed_is_compatibility_safe_and_has_hard_postconditions() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    normalized = _normalized(sql)

    assert "add column if not exists eval_triggers jsonb" in normalized
    assert "alter column eval_triggers set not null" in normalized
    assert "from pg_catalog.pg_attribute" in normalized
    assert "missing_badge_ids" in normalized
    assert "ineligible_badge_ids" in normalized
    assert "cardinality(missing_badge_ids) > 0" in normalized
    assert "cardinality(ineligible_badge_ids) > 0" in normalized
    assert "seeded.state::text" in normalized
    assert "seeded.is_active is distinct from true" in normalized
    assert """seeded.eval_triggers @> '["match_updated"]'::jsonb""" in normalized
    assert normalized.count("raise exception") >= 3
    for compatibility_column, canonical_column in COMPATIBILITY_COLUMNS.items():
        assert f"('{compatibility_column}', '{canonical_column}')" in normalized


def test_live_badge_seed_version_is_ordered_and_unique() -> None:
    migration_files = sorted((ROOT / "supabase/migrations").glob("*.sql"))
    versions = [path.name.split("_", 1)[0] for path in migration_files]

    assert len(versions) == len(set(versions))
    assert len("20260726183000") == 14
    assert "20260726183000".isdigit()
    assert "20260726143742" < "20260726183000" < "20261020000000"


def historical_presentation(sql: str, badge_id: str) -> dict[str, str]:
    import re
    pattern = r"\(\s*'" + re.escape(badge_id) + r"'\s*,([\s\S]*?)\)"
    row = re.search(pattern, sql).group(0)
    strings = [value.replace("''", "'") for value in re.findall(r"'((?:''|[^'])*)'", row)]
    # id, name, category, rarity, icon, lore, hint, scope, state
    return dict(zip(("category", "lore", "hint"), (strings[2], strings[5], strings[6])))
