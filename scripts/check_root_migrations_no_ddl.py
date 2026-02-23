#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = ROOT / "migrations"

DDL_PATTERN = re.compile(
    r"\b(CREATE\s+TABLE|ALTER\s+TABLE|CREATE\s+POLICY|CREATE\s+TRIGGER|CREATE\s+FUNCTION|ENABLE\s+ROW\s+LEVEL\s+SECURITY|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def main() -> int:
    offenders: list[tuple[Path, int, str]] = []
    for sql_file in sorted(MIGRATIONS_DIR.glob("*.sql")):
        for i, line in enumerate(sql_file.read_text().splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("--"):
                continue
            if DDL_PATTERN.search(line):
                offenders.append((sql_file.relative_to(ROOT), i, stripped))

    if offenders:
        print("Root migrations/ must not contain operational DDL. Move SQL into supabase/migrations/.")
        for path, line, text in offenders:
            print(f" - {path}:{line}: {text}")
        return 1

    print("OK: no root migrations DDL detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
