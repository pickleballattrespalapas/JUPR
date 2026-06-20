#!/usr/bin/env python3
"""Guardrail: ensure unique version prefixes in supabase migrations."""

from __future__ import annotations

import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
SUPABASE_MIGRATIONS_DIR = ROOT / "supabase" / "migrations"


def migration_files() -> list[pathlib.Path]:
    return sorted(SUPABASE_MIGRATIONS_DIR.glob("*.sql"))


def version_prefix(filename: str) -> str:
    return filename.split("_", 1)[0]


def main() -> int:
    files = migration_files()
    if not files:
        print("[supabase-migration-version-guard] No supabase migration files found.")
        return 0

    groups: dict[str, list[str]] = defaultdict(list)
    for path in files:
        groups[version_prefix(path.name)].append(path.name)

    duplicates = {prefix: names for prefix, names in groups.items() if len(names) > 1}
    if not duplicates:
        print("[supabase-migration-version-guard] All migration version prefixes are unique.")
        return 0

    print(
        "[supabase-migration-version-guard] Duplicate migration version prefixes detected:",
        file=sys.stderr,
    )
    for prefix in sorted(duplicates):
        print(f"  - {prefix}", file=sys.stderr)
        for name in sorted(duplicates[prefix]):
            print(f"      * {name}", file=sys.stderr)

    print(
        "Use unique YYYYMMDDHHMMSS_name.sql prefixes for files in supabase/migrations/.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
