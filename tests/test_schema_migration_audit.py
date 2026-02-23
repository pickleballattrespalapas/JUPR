from __future__ import annotations

import runpy
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "schema_migration_audit.py"
MODULE = runpy.run_path(str(SCRIPT_PATH))
AUDIT_SQL_MIGRATIONS = MODULE["audit_sql_migrations"]


def _write_sql(tmp_path: Path, name: str, sql: str) -> Path:
    path = tmp_path / name
    path.write_text(sql, encoding="utf-8")
    return path


def test_alter_table_only_is_parsed_with_real_table_name(tmp_path: Path) -> None:
    migration = _write_sql(
        tmp_path,
        "202601010001_alter_only.sql",
        "ALTER TABLE ONLY public.foo ADD COLUMN bar integer;",
    )

    violations = AUDIT_SQL_MIGRATIONS([migration], {"public.foo"})

    assert violations == []


def test_alter_table_rename_to_updates_known_table_name(tmp_path: Path) -> None:
    migration = _write_sql(
        tmp_path,
        "202601010002_rename.sql",
        "ALTER TABLE public.foo RENAME TO bar;\nALTER TABLE public.bar ADD COLUMN baz integer;",
    )

    violations = AUDIT_SQL_MIGRATIONS([migration], {"public.foo"})

    assert violations == []


def test_alter_table_set_schema_moves_table_for_later_statements(tmp_path: Path) -> None:
    migration = _write_sql(
        tmp_path,
        "202601010003_set_schema.sql",
        "ALTER TABLE public.foo SET SCHEMA other;\nALTER TABLE other.foo ADD COLUMN baz integer;",
    )

    violations = AUDIT_SQL_MIGRATIONS([migration], {"public.foo"})

    assert violations == []
