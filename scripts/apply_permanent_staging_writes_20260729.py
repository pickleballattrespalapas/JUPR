#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one exact match, found {count}")
    return text.replace(old, new, 1)


def replace_regex(text: str, pattern: str, replacement: str, label: str, *, expected: int = 1) -> str:
    updated, count = re.subn(pattern, replacement, text, flags=re.MULTILINE | re.DOTALL)
    if count != expected:
        raise RuntimeError(f"{label}: expected {expected} match(es), found {count}")
    return updated


def update_write_wave_registry() -> None:
    path = ROOT / "scripts/staging_write_waves.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        'NO_WRITE_WAVE = "none"\n',
        'NO_WRITE_WAVE = "none"\nOPEN_WRITE_WAVE = "open"\n',
        "open wave constant",
    )
    text = replace_once(
        text,
        "# Each deployment may open exactly one reviewed staging wave. `none` blocks\n"
        "# business-data mutation routes; append-only permission-denial auditing remains\n"
        "# active as a security control. Values include\n"
        "# dependency gates that must be open for that wave; every other controlled gate\n"
        "# is forced off before Fly builds the release.\n",
        "# Named waves remain available for diagnosis and emergency isolation. The `open`\n"
        "# wave is the normal staging posture and enables every reviewed staging mutation\n"
        "# gate. Production-only email and public-live override flags remain disabled.\n",
        "write-wave policy comment",
    )
    text = replace_once(
        text,
        "}\n\nDORMANT_STAGING_WRITE_FLAGS = (",
        "}\n\n"
        "STAGING_WRITE_WAVES[OPEN_WRITE_WAVE] = tuple(\n"
        "    sorted(\n"
        "        {\n"
        "            flag\n"
        "            for wave, flags in STAGING_WRITE_WAVES.items()\n"
        "            if wave != NO_WRITE_WAVE\n"
        "            for flag in flags\n"
        "        }\n"
        "    )\n"
        ")\n\n"
        "DORMANT_STAGING_WRITE_FLAGS = (",
        "open wave flag union",
    )
    text = replace_once(
        text,
        "}\n\n\ndef _route_template_pattern",
        "}\n\n"
        "STAGING_WRITE_WAVE_ROUTES[OPEN_WRITE_WAVE] = tuple(\n"
        "    sorted(\n"
        "        {\n"
        "            route\n"
        "            for wave, routes in STAGING_WRITE_WAVE_ROUTES.items()\n"
        "            if wave != NO_WRITE_WAVE\n"
        "            for route in routes\n"
        "        }\n"
        "    )\n"
        ")\n\n\n"
        "def _route_template_pattern",
        "open wave route union",
    )
    text = replace_once(
        text,
        'REGISTRATION_SECRET_WAVES = frozenset(\n    {"public-intake-auth", "tournament-registration"}\n)\n',
        'REGISTRATION_SECRET_WAVES = frozenset(\n'
        '    {"public-intake-auth", "tournament-registration", OPEN_WRITE_WAVE}\n'
        ')\n'
        'PUBLIC_LIVE_SECRET_WAVES = frozenset({"public-live", OPEN_WRITE_WAVE})\n',
        "open wave secret sets",
    )
    text = replace_once(
        text,
        'description="Configure one allowlisted, least-privilege Fly staging write wave."',
        'description="Configure the Fly staging write posture; open is the permanent test default."',
        "write-wave CLI description",
    )
    path.write_text(text, encoding="utf-8")


def update_fly_workflow() -> None:
    path = ROOT / ".github/workflows/fly_api_staging_deploy.yml"
    text = path.read_text(encoding="utf-8")
    text, count = re.subn(
        r"\$\{\{ github\.event_name == 'workflow_dispatch' && github\.event\.inputs\.write_wave \|\| 'none' \}\}",
        "${{ github.event_name == 'workflow_dispatch' && github.event.inputs.write_wave || 'open' }}",
        text,
    )
    if count != 2:
        raise RuntimeError(f"open workflow defaults: expected 2 matches, found {count}")
    text = replace_once(
        text,
        '        default: "none"\n        options:\n          - none\n',
        '        default: "open"\n        options:\n          - open\n          - none\n',
        "workflow-dispatch open option",
    )
    text = replace_once(
        text,
        "      JUPR_STAGING_WRITE_WAVE: none\n",
        "      JUPR_STAGING_WRITE_WAVE: open\n",
        "job open default",
    )
    text = replace_regex(
        text,
        r"      - name: Configure one least-privilege staging write wave\n.*?(?=\n      - name: Set up Python)",
        "      - name: Configure permanently writable staging\n"
        "        shell: bash\n"
        "        env:\n"
        "          SELECTED_WRITE_WAVE: ${{ github.event_name == 'workflow_dispatch' && github.event.inputs.write_wave || 'open' }}\n"
        "        run: |\n"
        "          set -euo pipefail\n"
        "          python scripts/staging_write_waves.py \\\n"
        "            --wave \"$SELECTED_WRITE_WAVE\" \\\n"
        "            --fly-config fly.staging.toml \\\n"
        "            --github-env \"$GITHUB_ENV\"\n",
        "permanent staging configuration step",
    )
    text = replace_once(
        text,
        '          if wave in {"public-intake-auth", "tournament-registration"}:',
        '          if wave in {"public-intake-auth", "tournament-registration", "open"}:',
        "registration secrets for open",
    )
    text = replace_once(
        text,
        '          if wave == "public-live":',
        '          if wave in {"public-live", "open"}:',
        "public-live secrets for open",
    )
    text = replace_once(
        text,
        '          if [ "$JUPR_STAGING_WRITE_WAVE" = "public-intake-auth" ] || [ "$JUPR_STAGING_WRITE_WAVE" = "tournament-registration" ]; then',
        '          if [ "$JUPR_STAGING_WRITE_WAVE" = "public-intake-auth" ] || [ "$JUPR_STAGING_WRITE_WAVE" = "tournament-registration" ] || [ "$JUPR_STAGING_WRITE_WAVE" = "open" ]; then',
        "stage registration secrets for open",
    )
    text = replace_once(
        text,
        '          if [ "$JUPR_STAGING_WRITE_WAVE" = "public-live" ]; then',
        '          if [ "$JUPR_STAGING_WRITE_WAVE" = "public-live" ] || [ "$JUPR_STAGING_WRITE_WAVE" = "open" ]; then',
        "stage public-live secrets for open",
    )
    path.write_text(text, encoding="utf-8")


def update_admin_fetch_freshness() -> int:
    paths = list((ROOT / "apps/web/lib").glob("admin*.ts"))
    paths.extend((ROOT / "apps/web/app/admin").rglob("page.tsx"))
    old = 'fetch(url, { next: { revalidate: 30 } })'
    new = 'fetch(url, { cache: "no-store" })'
    changed = 0
    match_uploader_changed = False
    for path in sorted(set(paths)):
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if not count:
            continue
        path.write_text(text.replace(old, new), encoding="utf-8")
        changed += count
        if path.name == "adminMatchUploaderApi.ts":
            match_uploader_changed = True
    if changed < 10:
        raise RuntimeError(f"admin no-store conversion expected at least 10 fetches, found {changed}")
    if not match_uploader_changed:
        raise RuntimeError("Match Uploader status fetch was not converted to no-store")
    return changed


RETIRED_SESSION_WORKFLOW = '''name: Retired Staging Write Session

on:
  workflow_dispatch:

permissions: {}

jobs:
  retired:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - name: Explain permanent staging test mode
        run: |
          echo "Temporary staging write leases are retired."
          echo "Staging is permanently writable for authenticated testing."
          echo "Use the separate emergency-disable workflow only when writes must be stopped."
'''

EMERGENCY_DISABLE_WORKFLOW = '''name: Staging Emergency Write Disable
run-name: Staging emergency write disable

on:
  workflow_dispatch:
    inputs:
      confirmation:
        description: "Type DISABLE STAGING WRITES"
        required: true
        type: string

permissions: {}

jobs:
  disable:
    if: >-
      github.repository == 'pickleballattrespalapas/JUPR' &&
      github.actor_id == 250933369
    runs-on: ubuntu-latest
    timeout-minutes: 70
    environment: staging
    permissions:
      actions: write
      contents: read
    concurrency:
      group: jupr-staging-fly-deploy
      cancel-in-progress: false
    steps:
      - name: Check out canonical staging
        uses: actions/checkout@v4
        with:
          ref: staging
          fetch-depth: 1
          persist-credentials: false

      - name: Confirm emergency disable
        shell: bash
        env:
          CONFIRMATION: ${{ github.event.inputs.confirmation }}
        run: |
          set -euo pipefail
          test "$CONFIRMATION" = "DISABLE STAGING WRITES"

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Disable staging writes
        shell: bash
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          set -euo pipefail
          CANDIDATE_SHA="$(git rev-parse HEAD)"
          python scripts/staging_evidence_automation.py dispatch \\
            --workflow fly_api_staging_deploy.yml \\
            --input write_wave=none \\
            --input "expected_candidate_sha=$CANDIDATE_SHA" \\
            > "$RUNNER_TEMP/fly-disable-dispatch.json"
          RUN_ID="$(jq -r '.workflow_run_id' "$RUNNER_TEMP/fly-disable-dispatch.json")"
          python scripts/staging_evidence_automation.py wait-run \\
            --workflow fly_api_staging_deploy.yml \\
            --run-id "$RUN_ID" \\
            --candidate-sha "$CANDIDATE_SHA" \\
            --timeout-seconds 3600
          python scripts/staging_evidence_automation.py fly-health \\
            > "$RUNNER_TEMP/fly-disabled.json"
          python scripts/staging_evidence_automation.py verify-final-none \\
            --fly-json "$RUNNER_TEMP/fly-disabled.json"
'''


def update_control_workflows() -> None:
    (ROOT / ".github/workflows/staging-write-session.yml").write_text(
        RETIRED_SESSION_WORKFLOW,
        encoding="utf-8",
    )
    (ROOT / ".github/workflows/staging-write-recovery.yml").write_text(
        EMERGENCY_DISABLE_WORKFLOW,
        encoding="utf-8",
    )


def configure_open_fly_file() -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/staging_write_waves.py"),
            "--wave",
            "open",
            "--fly-config",
            str(ROOT / "fly.staging.toml"),
        ],
        check=True,
        cwd=ROOT,
    )


def write_tests_and_docs(admin_fetch_changes: int) -> None:
    test_path = ROOT / "tests/test_permanent_staging_write_mode.py"
    test_path.write_text(
        '''from pathlib import Path

from scripts.staging_write_waves import (
    ALL_STAGING_WRITE_FLAGS,
    DORMANT_STAGING_WRITE_FLAGS,
    NO_WRITE_WAVE,
    OPEN_WRITE_WAVE,
    STAGING_WRITE_WAVE_ROUTES,
    STAGING_WRITE_WAVES,
    expected_write_flags,
)


def test_open_wave_enables_every_reviewed_staging_gate() -> None:
    expected = {
        flag
        for wave, flags in STAGING_WRITE_WAVES.items()
        if wave not in {NO_WRITE_WAVE, OPEN_WRITE_WAVE}
        for flag in flags
    }
    assert set(STAGING_WRITE_WAVES[OPEN_WRITE_WAVE]) == expected
    projection = expected_write_flags(OPEN_WRITE_WAVE)
    assert all(projection[flag] for flag in expected)
    assert all(not projection[flag] for flag in DORMANT_STAGING_WRITE_FLAGS)
    assert set(projection) == set(ALL_STAGING_WRITE_FLAGS)


def test_open_wave_allows_every_reviewed_staging_route() -> None:
    expected = {
        route
        for wave, routes in STAGING_WRITE_WAVE_ROUTES.items()
        if wave not in {NO_WRITE_WAVE, OPEN_WRITE_WAVE}
        for route in routes
    }
    assert set(STAGING_WRITE_WAVE_ROUTES[OPEN_WRITE_WAVE]) == expected


def test_fly_staging_defaults_to_permanent_open_mode() -> None:
    text = Path("fly.staging.toml").read_text(encoding="utf-8")
    assert 'JUPR_STAGING_WRITE_WAVE = "open"' in text
    assert 'JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL = "0"' in text
    assert 'JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_IMPORT_HANDOFF = "0"' in text
    projection = expected_write_flags(OPEN_WRITE_WAVE)
    for flag, enabled in projection.items():
        assert f'{flag} = "{1 if enabled else 0}"' in text


def test_staging_deploy_defaults_to_open_and_retains_emergency_none() -> None:
    text = Path(".github/workflows/fly_api_staging_deploy.yml").read_text(encoding="utf-8")
    assert "|| 'open'" in text
    assert 'default: "open"' in text
    assert "- open" in text
    assert "- none" in text
    assert "Push-triggered staging deploys must use write_wave=none" not in text
    assert 'JUPR_EMAIL_MODE: dry_run' in text


def test_automatic_lease_and_recovery_guards_are_retired() -> None:
    session = Path(".github/workflows/staging-write-session.yml").read_text(encoding="utf-8")
    recovery = Path(".github/workflows/staging-write-recovery.yml").read_text(encoding="utf-8")
    assert "issues:" not in session
    assert "schedule:" not in recovery
    assert "workflow_run:" not in recovery
    assert "Staging Emergency Write Disable" in recovery


def test_admin_status_fetches_do_not_cache_guard_state() -> None:
    uploader = Path("apps/web/lib/adminMatchUploaderApi.ts").read_text(encoding="utf-8")
    assert 'fetch(url, { cache: "no-store" })' in uploader
    for path in Path("apps/web").rglob("*.ts*"):
        if "/admin" not in path.as_posix() and not path.name.startswith("admin"):
            continue
        assert "revalidate: 30" not in path.read_text(encoding="utf-8"), path
''',
        encoding="utf-8",
    )
    doc_path = ROOT / "docs/permanent_staging_test_mode.md"
    doc_path.write_text(
        f'''# Permanent staging test mode

Staging is intentionally writable at all times for acceptance testing. The former issue-controlled, 60-minute, one-wave-at-a-time lease system is retired.

## Normal staging posture

- `JUPR_STAGING_WRITE_WAVE=open`
- Every reviewed staging mutation gate is enabled.
- Public and authenticated staging test flows remain available without reopening a window.
- Admin pages continue to require admin authentication.
- API audit logs, idempotency, transaction guards, recovery locks, and confirmation dialogs remain in place.
- Email remains `dry_run`; live player-update email is disabled.

## Hard boundaries retained

- Fly target remains `juprleagues-api-staging`.
- Supabase target remains `sijpxjxvdtrehmqvirfi`.
- Production Supabase `dnoockbwfenunhcibwfn` is never a target.
- Production deployment is never triggered by staging workflows.
- `JUPR_ENABLE_PUBLIC_LIVE_WRITES_PRODUCTION` remains disabled.

## Emergency stop

`Staging Emergency Write Disable` is a manual owner-only workflow that can deploy `write_wave=none`. The next normal staging deployment restores permanent `open` mode.

Admin status fetches use no-store responses so pages cannot remain stuck on a previously deployed guarded state. This implementation converted {admin_fetch_changes} cached admin fetch call(s).
''',
        encoding="utf-8",
    )


def main() -> int:
    update_write_wave_registry()
    update_fly_workflow()
    admin_fetch_changes = update_admin_fetch_freshness()
    update_control_workflows()
    configure_open_fly_file()
    write_tests_and_docs(admin_fetch_changes)
    print(f"Configured permanent staging writes; converted {admin_fetch_changes} admin fetches to no-store.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
