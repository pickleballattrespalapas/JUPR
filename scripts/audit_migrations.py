#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]

CREATE_SCHEMA_RE = re.compile(r"\bCREATE\s+SCHEMA\s+(?:IF\s+NOT\s+EXISTS\s+)?(?P<schema>[\w\"]+)", re.IGNORECASE)
CREATE_TABLE_RE = re.compile(
    r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?P<table>(?:[\w\"]+\.)?[\w\"]+)",
    re.IGNORECASE,
)
DROP_TABLE_RE = re.compile(
    r"\bDROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?P<table>(?:[\w\"]+\.)?[\w\"]+)",
    re.IGNORECASE,
)
ALTER_TABLE_RE = re.compile(
    r"\bALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?P<table>(?:[\w\"]+\.)?[\w\"]+)",
    re.IGNORECASE,
)
RENAME_RE = re.compile(r"\bRENAME\s+TO\s+(?P<newname>[\w\"]+)", re.IGNORECASE)


@dataclass
class MigrationSource:
    framework: str
    directory: Path
    files: list[Path]


@dataclass
class Violation:
    kind: str
    migration: str
    detail: str


@dataclass
class AuditState:
    schemas: set[str] = field(default_factory=lambda: {"public"})
    existing_tables: set[str] = field(default_factory=set)
    dropped_tables: set[str] = field(default_factory=set)
    renamed_from: dict[str, str] = field(default_factory=dict)


def normalize_identifier(raw: str) -> str:
    token = raw.strip().strip('"')
    if "." in token:
        schema, table = token.split(".", 1)
    else:
        schema, table = "public", token
    schema = schema.replace('"', "")
    table = table.replace('"', "")
    return f"{schema.lower()}.{table.lower()}"


def discover_migration_sources(repo_root: Path) -> list[MigrationSource]:
    candidates = [
        ("supabase", repo_root / "supabase" / "migrations", "*.sql"),
        ("prisma", repo_root / "prisma" / "migrations", "migration.sql"),
        ("alembic", repo_root / "alembic" / "versions", "*.py"),
        ("sql-root", repo_root / "migrations", "*.sql"),
    ]
    discovered: list[MigrationSource] = []
    for framework, directory, pattern in candidates:
        if directory.exists() and directory.is_dir():
            files = sorted(directory.glob(pattern))
            if files:
                discovered.append(MigrationSource(framework, directory, files))
    return discovered


def strip_sql_comments(sql: str) -> str:
    sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql


def parse_create_tables(sql_text: str) -> set[str]:
    cleaned = strip_sql_comments(sql_text)
    return {normalize_identifier(m.group("table")) for m in CREATE_TABLE_RE.finditer(cleaned)}


def detect_authoritative_snapshot(repo_root: Path) -> Path | None:
    candidates = [
        repo_root / "supabase" / "schema.sql",
        repo_root / "docs" / "schema" / "live_schema_snapshot_20260219.sql",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def extract_timestamp_prefix(filename: str) -> str:
    m = re.match(r"(?P<ts>\d+)", filename)
    return m.group("ts") if m else ""


def inventory_anomalies(files: Iterable[Path]) -> tuple[dict[str, list[str]], list[str]]:
    duplicates: dict[str, list[str]] = {}
    missing_ts: list[str] = []
    for path in files:
        ts = extract_timestamp_prefix(path.name)
        if not ts:
            missing_ts.append(path.name)
            continue
        duplicates.setdefault(ts, []).append(path.name)
    duplicates = {k: v for k, v in duplicates.items() if len(v) > 1}
    return duplicates, missing_ts


def audit_sql_migrations(files: list[Path], initial_tables: set[str]) -> list[Violation]:
    state = AuditState(existing_tables=set(initial_tables))
    violations: list[Violation] = []

    for path in files:
        sql_text = path.read_text(encoding="utf-8")
        cleaned = strip_sql_comments(sql_text)

        for m in CREATE_SCHEMA_RE.finditer(cleaned):
            state.schemas.add(m.group("schema").strip('"').lower())

        for m in CREATE_TABLE_RE.finditer(cleaned):
            table = normalize_identifier(m.group("table"))
            state.existing_tables.add(table)
            state.dropped_tables.discard(table)

        for m in DROP_TABLE_RE.finditer(cleaned):
            table = normalize_identifier(m.group("table"))
            if table not in state.existing_tables:
                violations.append(Violation("drop_missing", path.name, f"DROP TABLE references unknown table {table}"))
            state.existing_tables.discard(table)
            state.dropped_tables.add(table)

        for m in ALTER_TABLE_RE.finditer(cleaned):
            table = normalize_identifier(m.group("table"))
            if table in state.renamed_from:
                violations.append(
                    Violation(
                        "alter_old_name",
                        path.name,
                        f"ALTER TABLE uses old name {table}; renamed to {state.renamed_from[table]}",
                    )
                )
            if table in state.dropped_tables:
                violations.append(Violation("alter_dropped", path.name, f"ALTER TABLE references dropped table {table}"))
            elif table not in state.existing_tables:
                violations.append(Violation("alter_missing", path.name, f"ALTER TABLE references unknown table {table}"))

            trailing = cleaned[m.end() : m.end() + 250]
            rename_match = RENAME_RE.search(trailing)
            if rename_match:
                target = rename_match.group("newname")
                if "." in target:
                    new_table = normalize_identifier(target)
                else:
                    old_schema = table.split(".", 1)[0]
                    new_table = normalize_identifier(f"{old_schema}.{target}")
                state.existing_tables.discard(table)
                state.existing_tables.add(new_table)
                state.renamed_from[table] = new_table

    return violations


def print_violations(header: str, violations: list[Violation]) -> None:
    print(f"\n== Violations ({header}) ==")
    if not violations:
        print("No violations found.")
        return
    for violation in violations:
        print(f"- [{violation.kind}] {violation.migration}: {violation.detail}")
    alter_missing = [v for v in violations if v.kind == "alter_missing"]
    print(f"Summary: {len(violations)} total, {len(alter_missing)} alter-missing.")


def main() -> int:
    sources = discover_migration_sources(REPO_ROOT)
    if not sources:
        print("No migration directories found.")
        return 1

    print("== Migration source discovery ==")
    for source in sources:
        print(f"- {source.framework}: {source.directory.relative_to(REPO_ROOT)} ({len(source.files)} files)")

    snapshot = detect_authoritative_snapshot(REPO_ROOT)
    snapshot_tables: set[str] = set()
    if snapshot:
        snapshot_tables = parse_create_tables(snapshot.read_text(encoding="utf-8"))
        print(f"\nAuthoritative snapshot: {snapshot.relative_to(REPO_ROOT)} ({len(snapshot_tables)} CREATE TABLE entries)")
    else:
        print("\nAuthoritative snapshot: none detected")

    exit_code = 0
    for source in sources:
        print(f"\n== {source.framework} inventory ({source.directory.relative_to(REPO_ROOT)}) ==")
        for file in source.files:
            print(f"  {file.name}")
        duplicates, missing_ts = inventory_anomalies(source.files)
        if duplicates:
            print("  Duplicate timestamp prefixes:")
            for ts, names in sorted(duplicates.items()):
                print(f"    {ts}: {', '.join(names)}")
        if missing_ts:
            print(f"  Files missing numeric timestamp prefix: {', '.join(sorted(missing_ts))}")

        if source.framework in {"supabase", "sql-root"}:
            strict_violations = audit_sql_migrations(source.files, set())
            print_violations(f"{source.framework}: strict-empty-db", strict_violations)
            if strict_violations:
                exit_code = 2
            if source.framework == "supabase" and snapshot_tables:
                snapshot_violations = audit_sql_migrations(source.files, snapshot_tables)
                print_violations(f"{source.framework}: snapshot-seeded", snapshot_violations)
                if snapshot_violations:
                    exit_code = 2

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
