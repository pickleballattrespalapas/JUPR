#!/usr/bin/env python3
"""Audit badge requirement text coverage for the Badge Codex badge catalog."""
from __future__ import annotations

import sys
from pathlib import Path


FALLBACK_TEXT = "Requirements TBD"


def _bootstrap_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _load_badges():
    from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS

    return BADGE_DEFINITIONS


def _missing_requirement(raw: str | None) -> bool:
    cleaned = str(raw or "").strip()
    if not cleaned:
        return True
    if cleaned == FALLBACK_TEXT:
        return True
    return FALLBACK_TEXT.lower() in cleaned.lower()


def main() -> int:
    _bootstrap_repo_root()
    try:
        badges = _load_badges()
        from jupr_app.domain.gamification.requirements import requirement_for
    except Exception as exc:  # pragma: no cover - defensive CLI
        print(f"[audit_badge_requirements] import error: {exc}")
        print("Run this script from the repo root with dependencies installed.")
        return 1

    missing = []
    for badge in badges:
        badge_id = str(getattr(badge, "badge_id", "") or "")
        if not badge_id:
            continue
        name = str(getattr(badge, "name", "Badge") or "Badge")
        category = getattr(badge, "category", None)
        requirement_text = requirement_for(badge_id)
        if _missing_requirement(requirement_text):
            missing.append(
                {
                    "badge_id": badge_id,
                    "name": name,
                    "category": str(category or ""),
                }
            )

    missing.sort(key=lambda row: row["badge_id"])
    print(f"Missing badge requirements: {len(missing)}")
    for entry in missing:
        print(f"{entry['badge_id']}\t{entry['name']}\t{entry['category']}")

    if missing:
        print("\nMarkdown stub pack:")
        for entry in missing:
            print(f"## {entry['badge_id']} — {entry['name']}")
            print(f"Unlock: {FALLBACK_TEXT}.")
            print("")

    return 2 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
