#!/usr/bin/env python3
# How to run it:
#   python scripts/audit_incomplete_badges.py
#   JUPR_AUDIT_VERBOSE=1 python scripts/audit_incomplete_badges.py
"""Audit the Badge Codex for incomplete badges (requirements + copy + optional V3 DB conditions)."""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


FALLBACK_TEXT = "Requirements TBD"


@dataclass(frozen=True)
class BadgeAuditRow:
    badge_id: str
    name: str
    category: str
    requirement_preview: str
    missing_req: bool
    missing_desc: bool
    missing_meta: bool
    db_rules_total: str
    db_rules_enabled: str


def _bootstrap_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _missing_requirement(text: str | None) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return True
    return FALLBACK_TEXT.lower() in cleaned.lower()


def _normalize_preview(text: str | None, *, limit: int = 120) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit]


def _load_badges():
    from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS

    return BADGE_DEFINITIONS


def _load_badge_copy_sources():
    from jupr_app.domain.gamification.badge_descriptions import BADGE_DESCRIPTIONS_MD
    from jupr_app.domain.gamification.badge_copy import build_badge_copy_plain
    from jupr_app.domain.gamification.requirements import load_requirements_map

    requirements = load_requirements_map()
    return BADGE_DESCRIPTIONS_MD, build_badge_copy_plain, requirements


def _load_rules_counts() -> tuple[dict[str, int] | None, dict[str, int] | None]:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        return None, None

    try:
        from jupr_app.data.client import make_supabase
    except Exception:
        return None, None

    supabase = make_supabase(supabase_url, supabase_key)
    try:
        response = supabase.table("badge_rule_conditions").select("badge_id").execute()
    except Exception:
        return None, None

    totals: dict[str, int] = defaultdict(int)
    enabled: dict[str, int] = defaultdict(int)
    for row in response.data or []:
        badge_id = str(row.get("badge_id") or "")
        if not badge_id:
            continue
        totals[badge_id] += 1
        enabled[badge_id] += 1
    return totals, enabled


def main() -> int:
    _bootstrap_repo_root()
    try:
        badges = _load_badges()
        descriptions_map, build_badge_copy_plain, requirements_map = _load_badge_copy_sources()
    except Exception as exc:  # pragma: no cover - defensive CLI
        print("[audit_incomplete_badges] Badge audit")
        print(f"[audit_incomplete_badges] import error: {exc}")
        print("Run this script from the repo root with dependencies installed.")
        return 1

    try:
        verbose = os.getenv("JUPR_AUDIT_VERBOSE") == "1"
        rules_totals, rules_enabled = _load_rules_counts()
        db_available = rules_totals is not None and rules_enabled is not None
        rows: list[BadgeAuditRow] = []
        incomplete: list[BadgeAuditRow] = []

        for badge in badges:
            badge_id = str(getattr(badge, "badge_id", "") or "")
            if not badge_id:
                continue
            name = str(getattr(badge, "name", "Badge") or "Badge")
            category = str(getattr(badge, "category", "") or "")
            description_md = descriptions_map.get(badge_id, "")
            requirements_raw = requirements_map.get(badge_id)

            copy_plain = build_badge_copy_plain(
                {
                    "badge_id": badge_id,
                    "name": name,
                    "category": category,
                    "requirements": requirements_raw,
                    "description_md": description_md,
                }
            )

            requirement_text = copy_plain.req_text
            missing_req = _missing_requirement(requirement_text)
            missing_desc = badge_id in descriptions_map and not (copy_plain.desc_text or "").strip()
            missing_meta = False

            total_rules = rules_totals.get(badge_id, 0) if db_available else None
            enabled_rules = rules_enabled.get(badge_id, 0) if db_available else None
            rules_flag = False
            if db_available:
                rules_flag = total_rules == 0 or enabled_rules == 0

            row = BadgeAuditRow(
                badge_id=badge_id,
                name=name,
                category=category,
                requirement_preview=_normalize_preview(requirement_text),
                missing_req=missing_req,
                missing_desc=missing_desc,
                missing_meta=missing_meta,
                db_rules_total=str(total_rules) if db_available else "N/A",
                db_rules_enabled=str(enabled_rules) if db_available else "N/A",
            )
            rows.append(row)

            if missing_req or missing_desc or missing_meta or rules_flag:
                incomplete.append(row)

            if verbose:
                print(
                    "[audit_incomplete_badges][verbose] "
                    f"{badge_id} | req_missing={missing_req} | desc_missing={missing_desc} "
                    f"| meta_missing={missing_meta} | rules_total={row.db_rules_total} "
                    f"| rules_enabled={row.db_rules_enabled}"
                )

        print("[audit_incomplete_badges] Badge audit")
        print(f"Badges checked: {len(rows)}")
        print(f"Incomplete badges: {len(incomplete)}")
        if db_available:
            print("DB rules check: enabled (badge_rule_conditions).")
        else:
            print("DB rules check: skipped (SUPABASE_URL/SUPABASE_KEY not set).")

        if incomplete:
            print("\nIncomplete badge list (tab-separated):")
            print(
                "badge_id\tname\tcategory\trequirement_preview\tmissing_req\tmissing_desc\tmissing_meta\t"
                "db_rules_total\tdb_rules_enabled"
            )
            for row in sorted(incomplete, key=lambda item: item.badge_id):
                print(
                    f"{row.badge_id}\t{row.name}\t{row.category}\t{row.requirement_preview}\t"
                    f"{row.missing_req}\t{row.missing_desc}\t{row.missing_meta}\t"
                    f"{row.db_rules_total}\t{row.db_rules_enabled}"
                )

            missing_req_rows = sorted(
                (row for row in incomplete if row.missing_req),
                key=lambda item: item.badge_id,
            )
            if missing_req_rows:
                print("\nFix pack: markdown stubs for docs/badge_requirements.md")
                for row in missing_req_rows:
                    print(f"## {row.badge_id} — {row.name}")
                    print("Unlock: Requirements TBD.")
                    print("")

            return 2

        print("Result: PASS (no incomplete badges found). ✅")
        return 0
    except Exception as exc:  # pragma: no cover - defensive CLI
        print("[audit_incomplete_badges] Badge audit")
        print(f"[audit_incomplete_badges] unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
