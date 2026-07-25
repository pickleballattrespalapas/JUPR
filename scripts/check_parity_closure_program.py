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
REQUIRED_HIGH_RISK_ROW_TOKENS = {
    "match_uploader": (
        "JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_SINGLES",
        "`Blocked`",
        "atomic writer",
    ),
    "match_log": (
        "JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_DESTRUCTIVE",
        "`Blocked`",
        "atomic idempotent recovery",
    ),
    "tournament_ops": (
        "automated-ready",
        "atomic CAS RPC",
    ),
}


def _row_text(text: str, key: str) -> str:
    match = re.search(rf"^\|\s*`{re.escape(key)}`\s*\|.*$", text, re.MULTILINE)
    return match.group(0) if match else ""


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

    matrix_text = matrix_path.read_text(encoding="utf-8")
    program_text = program_path.read_text(encoding="utf-8")
    expected = partial_keys(matrix_text)
    rows = list(PROGRAM_ROW_RE.finditer(program_text))
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

    for key, tokens in REQUIRED_HIGH_RISK_ROW_TOKENS.items():
        if key not in expected:
            continue
        for label, text in (("matrix", matrix_text), ("closure program", program_text)):
            row = _row_text(text, key)
            missing_tokens = [token for token in tokens if token not in row]
            if missing_tokens:
                errors.append(
                    f"{label.title()} row {key} is missing high-risk contract token(s): "
                    + ", ".join(missing_tokens)
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
