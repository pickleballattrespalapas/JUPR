#!/usr/bin/env python3
"""Validate that every Streamlit page is represented in the Next parity matrix."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs" / "next_streamlit_parity_matrix.md"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jupr_app.ui.page_registry import PAGE_DEFINITIONS  # noqa: E402

MATRIX_KEY_RE = re.compile(r"^\|\s*`(?P<key>[^`]+)`\s*\|", re.MULTILINE)
REQUIRED_SECTION_HEADINGS = (
    "## Purpose",
    "## Cutover gates",
    "## Parity matrix",
    "## Recommended implementation sequence",
    "## Maintenance rule",
)


def expected_page_keys() -> set[str]:
    return {str(page.key) for page in PAGE_DEFINITIONS}


def documented_page_keys(text: str) -> set[str]:
    return {match.group("key") for match in MATRIX_KEY_RE.finditer(text)}


def check_matrix(doc_path: Path = DOC_PATH) -> list[str]:
    errors: list[str] = []
    if not doc_path.exists():
        return [f"Missing parity matrix document: {doc_path}"]

    text = doc_path.read_text(encoding="utf-8")
    expected = expected_page_keys()
    documented = documented_page_keys(text)

    missing = sorted(expected - documented)
    if missing:
        errors.append(
            "docs/next_streamlit_parity_matrix.md is missing Streamlit page keys: "
            + ", ".join(missing)
        )

    unknown = sorted(documented - expected)
    if unknown:
        errors.append(
            "docs/next_streamlit_parity_matrix.md contains unknown page keys not in page_registry.py: "
            + ", ".join(unknown)
        )

    for heading in REQUIRED_SECTION_HEADINGS:
        if heading not in text:
            errors.append(f"Parity matrix is missing required section heading: {heading}")

    if "| Streamlit key | Streamlit label | Access |" not in text:
        errors.append("Parity matrix table header is missing or malformed.")

    return errors


def main() -> int:
    errors = check_matrix()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Parity matrix covers {len(expected_page_keys())} Streamlit page definitions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
