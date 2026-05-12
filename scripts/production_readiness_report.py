#!/usr/bin/env python3
"""Generate a branch promotion readiness report for JUPR."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

AREAS = [
    "Streamlit UI",
    "domain/services",
    "Supabase migrations",
    "FastAPI",
    "Next.js",
    "workers",
    "docs",
    "tests/CI",
    "other",
]


@dataclass
class CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str


def run_command(*args: str) -> CommandResult:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    return CommandResult(" ".join(args), result.returncode, result.stdout.strip(), result.stderr.strip())


def ahead_behind(base: str, head: str) -> str:
    result = run_command("git", "rev-list", "--left-right", "--count", f"{base}...{head}")
    if result.returncode != 0 or not result.stdout:
        return "unavailable"
    parts = result.stdout.split()
    if len(parts) != 2:
        return "unavailable"
    behind, ahead = parts
    return f"{head} is ahead by {ahead}, behind by {behind} versus {base}"


def changed_files(base: str, head: str) -> list[str]:
    result = run_command("git", "diff", "--name-only", f"{base}...{head}")
    if result.returncode != 0 or not result.stdout:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def added_files(base: str, head: str) -> list[str]:
    result = run_command("git", "diff", "--name-status", "--diff-filter=A", f"{base}...{head}")
    if result.returncode != 0 or not result.stdout:
        return []
    added: list[str] = []
    for line in result.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0] == "A":
            added.append(parts[1].strip())
    return added


def classify_path(path: str) -> str:
    if path.startswith(("app.py", "streamlit_app.py", "jupr_court_board/", "jupr_app/ui/")):
        return "Streamlit UI"
    if path.startswith("services/api/"):
        return "FastAPI"
    if path.startswith(("jupr_app/services/", "services/", "jupr_app/domain/")):
        return "domain/services"
    if path.startswith(("supabase/migrations/", "migrations/")):
        return "Supabase migrations"
    if path.startswith("apps/web/"):
        return "Next.js"
    if "/workers/" in path or path.startswith("workers/"):
        return "workers"
    if path.startswith("docs/") or path == "README.txt":
        return "docs"
    if path.startswith("tests/") or path.startswith((".github/workflows/", "Makefile")):
        return "tests/CI"
    return "other"


def build_risk_flags(files: list[str]) -> list[str]:
    risk_flags: list[str] = []
    joined = "\n".join(files)

    if any(p.startswith(("services/api/", "apps/web/")) and "admin" in p and "score" in p for p in files):
        risk_flags.append("FastAPI/Next.js admin score entry related files touched")
    if any(p.startswith(("apps/web/", "services/api/", "README.txt", "docs/")) for p in files):
        tracked = run_command("git", "diff", "-U0", "rollback-feb8...HEAD", "--", "apps/web", "services/api", "README.txt", "docs")
        if "JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY" in tracked.stdout or "NEXT_PUBLIC_JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY" in tracked.stdout:
            risk_flags.append("Admin score entry feature flag defaults changed")
    if any(p.startswith("apps/web/") for p in files):
        web_diff = run_command("git", "diff", "-U0", "rollback-feb8...HEAD", "--", "apps/web")
        if "SERVICE_ROLE" in web_diff.stdout.upper() or "SUPABASE_SERVICE_ROLE_KEY" in web_diff.stdout:
            risk_flags.append("Potential service-role key reference in apps/web changes")
    if "process_matches" in joined and not any("services" in p for p in files if "process_matches" in p):
        risk_flags.append("process_matches import/use touched outside service layer")
    if any("match_processing.py" in p for p in files):
        risk_flags.append("match_processing.py changed")
    if any("admin_auth" in p.lower() and "streamlit" in p.lower() or "admin_auth" in p.lower() and p.startswith("jupr_app/") for p in files):
        risk_flags.append("Streamlit admin auth related changes detected")
    if any(p.startswith(("supabase/migrations/", "migrations/")) for p in files):
        risk_flags.append("Supabase migration files changed")

    return risk_flags


def build_report(base: str, head: str) -> dict:
    files = changed_files(base, head)
    added = added_files(base, head)

    grouped: dict[str, list[str]] = defaultdict(list)
    for path in files:
        grouped[classify_path(path)].append(path)

    supabase_added = [p for p in added if p.startswith("supabase/migrations/")]
    root_added = [p for p in added if p.startswith("migrations/")]

    checks = {
        "check_supabase_migration_versions": run_command("python", "scripts/check_supabase_migration_versions.py").__dict__,
        "check_migration_sources": run_command("python", "scripts/check_migration_sources.py", "--base-ref", base).__dict__,
    }

    return {
        "base": base,
        "head": head,
        "ahead_behind": ahead_behind(base, head),
        "changed_files_count": len(files),
        "changed_files_grouped": {k: grouped.get(k, []) for k in AREAS},
        "migration_summary": {
            "supabase_migrations_added": supabase_added,
            "root_migrations_added": root_added,
            "root_migration_warning": bool(root_added),
            "checks": checks,
        },
        "risk_flags": build_risk_flags(files),
        "recommended_checklist": [
            "staging env verified",
            "staging smoke passed",
            "production SQL reviewed",
            "Next admin score entry disabled",
            "Streamlit smoke passed",
            "rollback plan ready",
        ],
    }


def format_text(report: dict) -> str:
    lines = [
        "# Production Promotion Readiness Report",
        f"Base: {report['base']}",
        f"Head: {report['head']}",
        "",
        "## 1) Branch diff summary",
        f"- Ahead/behind: {report['ahead_behind']}",
        f"- Changed files: {report['changed_files_count']}",
    ]
    for area, paths in report["changed_files_grouped"].items():
        lines.append(f"- {area}: {len(paths)}")
        for path in paths:
            lines.append(f"  - {path}")

    m = report["migration_summary"]
    lines.extend([
        "",
        "## 2) Migration summary",
        "- Supabase migrations added:",
    ])
    lines.extend([f"  - {p}" for p in m["supabase_migrations_added"]] or ["  - (none)"])
    lines.append("- Root migrations added:")
    lines.extend([f"  - {p}" for p in m["root_migrations_added"]] or ["  - (none)"])
    if m["root_migration_warning"]:
        lines.append("- WARNING: root migrations under migrations/ were added")
    for name, result in m["checks"].items():
        status = "PASS" if result["returncode"] == 0 else "FAIL"
        lines.append(f"- {name}: {status}")
        if result["stdout"]:
            lines.append(f"  stdout: {result['stdout']}")
        if result["stderr"]:
            lines.append(f"  stderr: {result['stderr']}")

    lines.extend(["", "## 3) Risk flags"])
    if report["risk_flags"]:
        lines.extend([f"- {x}" for x in report["risk_flags"]])
    else:
        lines.append("- No explicit risk flags detected.")

    lines.extend(["", "## 4) Recommended checklist"])
    lines.extend([f"- [ ] {x}" for x in report["recommended_checklist"]])

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="rollback-feb8")
    parser.add_argument("--head", default="Test")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build_report(args.base, args.head)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
