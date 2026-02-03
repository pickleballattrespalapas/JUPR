#!/usr/bin/env python3
"""Audit badge requirements and report which badges still use placeholder text."""

from __future__ import annotations

import sys
from pathlib import Path

FALLBACK_REQUIREMENTS = "Requirements TBD"


def _bootstrap_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def _is_missing_requirement(text: str | None) -> bool:
    if text is None:
        return True
    cleaned = str(text).strip()
    if not cleaned:
        return True
    if cleaned == FALLBACK_REQUIREMENTS:
        return True
    return "requirements tbd" in cleaned.lower()


def main() -> int:
    _bootstrap_repo_root()
    try:
        from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS
        from jupr_app.domain.gamification.requirements import requirement_for
    except Exception as exc:  # pragma: no cover - import guard for CLI usage
        print(
            "Unable to import JUPR badge modules. Run from the repo root with "
            "`python scripts/audit_badge_requirements.py`.",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    missing = []
    for badge in BADGE_DEFINITIONS:
        requirements_text = requirement_for(badge.badge_id)
        if _is_missing_requirement(requirements_text):
            missing.append(
                {
                    "badge_id": badge.badge_id,
                    "name": badge.name,
                    "category": badge.category,
                }
            )

    total = len(BADGE_DEFINITIONS)
    print(f"Missing badge requirements: {len(missing)} of {total}")
    for entry in missing:
        print(f"{entry['badge_id']}\t{entry['name']}\t{entry['category']}")

    if missing:
        print("\nMarkdown stub pack:")
        for entry in missing:
            print(f"## {entry['badge_id']} — {entry['name']}")
            print("Unlock: Requirements TBD.")
            print()

    return 2 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
