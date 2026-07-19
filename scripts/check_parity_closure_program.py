#!/usr/bin/env python3
"""Ensure every Partial parity row has exactly one closure contract."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "docs" / "next_streamlit_parity_matrix.md"
PROGRAM_PATH = ROOT / "docs" / "next_parity_closure_program.md"

MATRIX_ROW_RE = re.compile(
    r"^\|\s*`(?P<key>[^`]+)`\s*\|.*?\|\s*`(?P<status>Done|Partial|API needed|Auth needed|Not started|Deferred)`",
    re.MULTILINE,
)
PROGRAM_ROW_RE = re.compile(
    r"^\|\s*`(?P<key>[^`]+)`\s*\|\s*`(?P<track>[^`]+)`\s*\|",
    re.MULTILINE,
)
ALLOWED_TRACKS = {
    "public-data",
    "static-support",
    "public-tournaments",
    "auth",
    "match-player",
    "league-comms",
    "badges-tools",
    "live-ladder",
    "tournament-admin",
}


def partial_keys(text: str) -> set[str]:
    return {
        match.group("key")
        for match in MATRIX_ROW_RE.finditer(text)
        if match.group("status") == "Partial"
    }


def check_program(
    matrix_path: Path = MATRIX_PATH,
    program_path: Path = PROGRAM_PATH,
) -> list[str]:
    errors: list[str] = []
    if not matrix_path.exists():
        return [f"Missing parity matrix: {matrix_path}"]
    if not program_path.exists():
        return [f"Missing parity closure program: {program_path}"]

    expected = partial_keys(matrix_path.read_text(encoding="utf-8"))
    rows = list(PROGRAM_ROW_RE.finditer(program_path.read_text(encoding="utf-8")))
    counts = Counter(match.group("key") for match in rows)
    documented = set(counts)

    missing = sorted(expected - documented)
    if missing:
        errors.append("Partial pages missing closure contracts: " + ", ".join(missing))

    stale = sorted(documented - expected)
    if stale:
        errors.append(
            "Closure contracts no longer marked Partial in the matrix: " + ", ".join(stale)
        )

    duplicates = sorted(key for key, count in counts.items() if count != 1)
    if duplicates:
        errors.append("Closure contracts must appear exactly once: " + ", ".join(duplicates))

    invalid_tracks = sorted(
        {match.group("track") for match in rows} - ALLOWED_TRACKS
    )
    if invalid_tracks:
        errors.append("Unknown closure tracks: " + ", ".join(invalid_tracks))

    if len(expected) != 45:
        errors.append(
            f"Expected the current closure wave to contain 45 Partial pages, found {len(expected)}. "
            "Update the program deliberately when matrix statuses change."
        )

    return errors


def main() -> int:
    errors = check_program()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    count = len(partial_keys(MATRIX_PATH.read_text(encoding="utf-8")))
    print(f"Parity closure program covers all {count} Partial page definitions exactly once.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
