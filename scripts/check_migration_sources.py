#!/usr/bin/env python3
"""Guardrail: prevent undocumented new root migrations.

Fails when new files are added under migrations/*.sql unless explicitly documented
in docs/migrations_root_explanations.md.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXPLANATIONS_FILE = ROOT / "docs" / "migrations_root_explanations.md"


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def added_root_migrations(base_ref: str) -> list[str]:
    diff = run_git("diff", "--name-status", "--diff-filter=A", f"{base_ref}...HEAD")
    added: list[str] = []
    for line in diff.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        status, path = parts
        if status != "A":
            continue
        if re.fullmatch(r"migrations/.+\.sql", path):
            added.append(path)
    return sorted(set(added))


def documented_entries() -> set[str]:
    if not EXPLANATIONS_FILE.exists():
        return set()
    text = EXPLANATIONS_FILE.read_text(encoding="utf-8")
    # Matches bullets like:
    # - `migrations/example.sql`: reason
    matches = re.findall(r"-\s+`(migrations/[^`]+\.sql)`\s*:", text)
    return set(matches)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-ref",
        default="origin/staging",
        help="Git base ref to compare with current HEAD (default: origin/staging)",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Warn instead of exiting non-zero.",
    )
    args = parser.parse_args()

    try:
        added = added_root_migrations(args.base_ref)
    except subprocess.CalledProcessError as exc:
        print("[migration-guard] Could not evaluate git diff.", file=sys.stderr)
        print(exc.stderr, file=sys.stderr)
        return 2

    if not added:
        print("[migration-guard] No new root migrations detected.")
        return 0

    documented = documented_entries()
    undocumented = [path for path in added if path not in documented]

    if not undocumented:
        print("[migration-guard] New root migrations are documented.")
        return 0

    print("[migration-guard] Undocumented root migrations detected:", file=sys.stderr)
    for path in undocumented:
        print(f"  - {path}", file=sys.stderr)

    print(
        "Add entries to docs/migrations_root_explanations.md "
        "or move schema migrations into supabase/migrations/.",
        file=sys.stderr,
    )

    if args.warn_only:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
