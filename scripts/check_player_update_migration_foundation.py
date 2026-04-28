#!/usr/bin/env python3
"""Guardrail: player update migrations must create subscriptions before updates.

Ensures any Supabase migration that runs an UPDATE against
public.player_profile_update_subscriptions has an earlier Supabase migration that
creates that table.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SUPABASE_MIGRATIONS = ROOT / "supabase" / "migrations"

UPDATE_PATTERN = re.compile(r"\bupdate\s+public\.player_profile_update_subscriptions\b", re.IGNORECASE)
CREATE_PATTERN = re.compile(
    r"\bcreate\s+table\s+(if\s+not\s+exists\s+)?public\.player_profile_update_subscriptions\b",
    re.IGNORECASE,
)


def main() -> int:
    sql_files = sorted(SUPABASE_MIGRATIONS.glob("*.sql"), key=lambda p: p.name)
    if not sql_files:
        print("[player-update-foundation] No supabase migrations found.", file=sys.stderr)
        return 1

    creators: list[str] = []
    failures: list[str] = []

    for path in sql_files:
        text = path.read_text(encoding="utf-8")
        if CREATE_PATTERN.search(text):
            creators.append(path.name)

        if UPDATE_PATTERN.search(text) and not creators:
            failures.append(path.name)

    if failures:
        print(
            "[player-update-foundation] Found UPDATE migration(s) that run before table creation:",
            file=sys.stderr,
        )
        for name in failures:
            print(f"  - {name}", file=sys.stderr)
        print(
            "Add an earlier supabase migration that creates public.player_profile_update_subscriptions.",
            file=sys.stderr,
        )
        return 1

    print("[player-update-foundation] OK: creation migration exists before any UPDATE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
