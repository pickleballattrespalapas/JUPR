#!/usr/bin/env python3
# How to run it:
#   python scripts/audit_badge_requirements.py
# What success output looks like:
#   [audit_badge_requirements] Badge requirement audit
#   Badges checked: 42
#   Badges missing requirements: 0
#   Result: PASS
#   What to do next: No missing requirements detected. ✅
# What failure output looks like:
#   [audit_badge_requirements] Badge requirement audit
#   Badges checked: 42
#   Badges missing requirements: 3
#   Result: FAIL
#   What to do next: Fill in the missing requirement text and re-run the audit.
# How to check exit code (echo $?):
#   echo $?
"""Audit badge requirement text coverage for the Badge Codex badge catalog."""
from __future__ import annotations

import os
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
        print("[audit_badge_requirements] Badge requirement audit")
        print(f"[audit_badge_requirements] import error: {exc}")
        print("Run this script from the repo root with dependencies installed.")
        return 1

    try:
        verbose = os.getenv("JUPR_AUDIT_VERBOSE") == "1"
        missing = []
        badges_checked = 0
        for badge in badges:
            badge_id = str(getattr(badge, "badge_id", "") or "")
            if not badge_id:
                continue
            badges_checked += 1
            name = str(getattr(badge, "name", "Badge") or "Badge")
            category = getattr(badge, "category", None)
            requirement_text = requirement_for(badge_id)
            is_missing = _missing_requirement(requirement_text)
            if verbose:
                preview = " ".join(str(requirement_text or "").split())
                preview = preview[:80]
                missing_flag = "Y" if is_missing else "N"
                print(
                    "[audit_badge_requirements][verbose] "
                    f"{badge_id} | {name} | {preview} | missing={missing_flag}"
                )
            if is_missing:
                missing.append(
                    {
                        "badge_id": badge_id,
                        "name": name,
                        "category": str(category or ""),
                    }
                )

        missing.sort(key=lambda row: row["badge_id"])
        print("[audit_badge_requirements] Badge requirement audit")
        print(f"Badges checked: {badges_checked}")
        print(f"Badges missing requirements: {len(missing)}")
        print(f"Result: {'FAIL' if missing else 'PASS'}")

        if missing:
            print("What to do next: Fill in the missing requirement text and re-run the audit.")
            print("\nMissing badge requirements:")
            for entry in missing:
                print(f"{entry['badge_id']}\t{entry['name']}\t{entry['category']}")

            print("\nMarkdown stub pack:")
            for entry in missing:
                print(f"## {entry['badge_id']} — {entry['name']}")
                print(f"Unlock: {FALLBACK_TEXT}.")
                print("")
        else:
            print("What to do next: No missing requirements detected. ✅")
            print(
                "Troubleshooting: If you still see TBD in the UI, the audit may not "
                "be using the same resolver. Re-run with JUPR_AUDIT_VERBOSE=1 and paste "
                "the output in your report."
            )

        return 2 if missing else 0
    except Exception as exc:  # pragma: no cover - defensive CLI
        print("[audit_badge_requirements] Badge requirement audit")
        print(f"[audit_badge_requirements] unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
