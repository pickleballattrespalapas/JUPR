#!/usr/bin/env python3
"""Crash after atomic exclusion commit; recover through the deployed staging API.

The child runs the real application service but exits at its post-commit
completion boundary. No server fault switch or shared-worker kill is used.
Only a newly prepared, owned disposable fixture can be targeted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import Request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prepare_parity_match_exclusion_fixture as fixture

CRASH_EXIT = 86
API_ORIGIN = "https://juprleagues-api-staging.fly.dev"
REPORT_NAME = "interrupted-exclusion-result.json"


def owned_fixture(report_dir, env, *, fresh=False):
    origin, key, sha = fixture._validate_staging_environment(env)
    if env.get("GITHUB_REF") != "refs/heads/staging":
        raise fixture.FixtureError("Interruption rehearsal requires canonical staging.")
    if env.get("JUPR_PARITY_MUTATION_CONFIRMATION") != "RUN DISPOSABLE STAGING WRITES":
        raise fixture.FixtureError("Disposable staging writes were not confirmed.")
    if env.get("STAGING_API_BASE_URL") != API_ORIGIN:
        raise fixture.FixtureError("Refusing another API origin.")
    manifest = fixture._load_manifest(report_dir / fixture.MANIFEST_NAME)
    email, user_id = fixture._decode_jwt_identity(env)
    if manifest.get("candidate_sha") != sha or manifest.get("operator_email_sha256") != hashlib.sha256(email.encode()).hexdigest():
        raise fixture.FixtureError("Fixture candidate or owner does not match.")
    fixture._manifest_contract(manifest)
    client = fixture._RestClient(origin=origin, service_role_key=key, transport=fixture._default_transport)
    rows = fixture._fixture_rows(client, manifest)
    fixture._validate_owned_rows(rows, manifest, operator_email=email,
        operator_user_id=user_id, require_complete=fresh, require_terminal=False)
    return manifest, rows, client, email


def crash_after_commit(_supabase=None, **kwargs):
    operation = kwargs["operation"]
    if operation.get("operation_status") != "pending_replay" or not operation.get("replay_job_id"):
        raise fixture.FixtureError("Atomic commit did not leave pending replay work.")
    # Deliberately bypass finally/atexit: this is an actual process interruption.
    os._exit(CRASH_EXIT)


def commit_child(report_dir, env):
    manifest, rows, client, email = owned_fixture(report_dir, env, fresh=True)
    if rows["operations"] or rows["replay_jobs"]:
        raise fixture.FixtureError("Interruption requires a fresh fixture.")
    from supabase import create_client
    from jupr_app.services import match_exclusion_durability_service as service
    _, match_ids, versions, _ = fixture._optional_manifest_ids(manifest)
    _, club_id, _, _, _, keys = fixture._manifest_contract(manifest)
    os.environ["JUPR_ENABLE_NEXT_ADMIN_REPLAY"] = "1"
    service._complete_operation = crash_after_commit
    service.apply_atomic_match_exclusions(create_client(client.origin, client.service_role_key),
        club_id=club_id,
        targets=[{"match_id": match_ids["distinct"], "expected_row_version": versions["distinct"]}],
        actor_email=email, actor_role="super_admin", source=fixture.FIXTURE_SOURCE,
        note=f"Disposable staging fixture {club_id}",
        idempotency_key=keys["direct_exclusion"], mode="exclude")
    raise fixture.FixtureError("Child unexpectedly completed without interruption.")


def api_recover(club_id, operation_id, env):
    # UUID and club ownership were verified before constructing this fixed-origin URL.
    request = Request(f"{API_ORIGIN}/admin/clubs/{club_id}/match-log/exclusions/{operation_id}/recover",
        data=json.dumps({"confirmation_text": "RECOVER", "source": fixture.FIXTURE_SOURCE}).encode(),
        headers={"Authorization": "Bearer " + env["STAGING_ADMIN_BEARER_TOKEN"],
                 "Content-Type": "application/json"}, method="POST")
    try:
        with fixture._STAGING_OPENER.open(request, timeout=180) as response:
            status = response.status
            raw = response.read(fixture.MAX_RESPONSE_BYTES + 1)
        if status != 200 or len(raw) > fixture.MAX_RESPONSE_BYTES:
            raise ValueError()
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError()
        return result
    except Exception:
        raise fixture.FixtureError("Deployed staging recovery request failed; fixture retained if nonterminal.") from None


def verify_interrupted(rows, manifest):
    _, match_ids, versions, _ = fixture._optional_manifest_ids(manifest)
    operations = rows["operations"]
    if len(operations) != 1 or len(rows["replay_jobs"]) != 1:
        raise fixture.FixtureError("Expected exactly one committed operation and replay job.")
    operation, replay = operations[0], rows["replay_jobs"][0]
    match = next((row for row in rows["matches"] if row["id"] == match_ids["distinct"]), {})
    if (operation.get("status") != "pending_replay" or replay.get("status") != "pending"
        or operation.get("replay_job_id") != replay.get("id")
        or [int(value) for value in operation.get("excluded_match_ids", [])] != [match_ids["distinct"]]
        or not match.get("deleted_at") or int(match.get("row_version", 0)) != versions["distinct"] + 1):
        raise fixture.FixtureError("Durable post-crash state did not match the commit boundary.")
    return operation


def verify_recovery(result, operation, *, repeated):
    expected_mode = "already_recovered" if repeated else "match_exclusion_recovered"
    if (result.get("ok") is not True or result.get("mode") != expected_mode
        or result.get("operation_status") != "succeeded"
        or result.get("operation_id") != operation["id"]
        or result.get("replay_job_id") != operation["replay_job_id"]
        or result.get("replay_status") != "succeeded"
        or result.get("recovery_stage") is not None
        or [int(value) for value in result.get("excluded_ids", [])] != [int(value) for value in operation["excluded_match_ids"]]
        or (repeated and result.get("idempotent") is not True)):
        raise fixture.FixtureError("Recovery did not preserve the committed operation and replay identity.")


def rehearse(report_dir, env):
    # Validate before even preparing any data; preparation repeats its own guards.
    fixture._validate_staging_environment(env)
    if env.get("JUPR_PARITY_MUTATION_CONFIRMATION") != "RUN DISPOSABLE STAGING WRITES" or env.get("GITHUB_REF") != "refs/heads/staging" or env.get("STAGING_API_BASE_URL") != API_ORIGIN:
        raise fixture.FixtureError("Interruption rehearsal requires confirmed canonical staging.")
    try:
        fixture.prepare_fixture(report_dir=report_dir, github_env=report_dir / "fixture.env", env=env)
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", "--report-dir", str(report_dir)],
            env=dict(env), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
        if child.returncode != CRASH_EXIT:
            raise fixture.FixtureError("Child did not exit at the expected commit boundary.")
        fixture._write_json(report_dir / "child-exit.json", {"exit_code": child.returncode})
        finish_interrupted(report_dir, env)
    except Exception as exc:
        fixture._write_json(report_dir / "rehearsal-failure.json", {
            "error_type": type(exc).__name__,
            "failure": str(exc) if isinstance(exc, fixture.FixtureError) else "Unexpected rehearsal error",
        })
        raise
    finally:
        fixture.cleanup_fixture(report_dir=report_dir, env=env)


def finish_interrupted(report_dir, env):
    manifest, rows, client, email = owned_fixture(report_dir, env)
    operation = verify_interrupted(rows, manifest)
    evidence = {"candidate_sha": manifest["candidate_sha"], "status": "interrupted",
        "operation_id": operation["id"], "replay_job_id": operation["replay_job_id"],
        "child_exit_code": CRASH_EXIT, "observed_status": "pending_replay"}
    fixture._write_json(report_dir / REPORT_NAME, evidence)
    for repeated in (False, True):
        result = api_recover(manifest["club_id"], operation["id"], env)
        verify_recovery(result, operation, repeated=repeated)
    final = fixture._fixture_rows(client, manifest)
    _, user_id = fixture._decode_jwt_identity(env)
    fixture._validate_owned_rows(final, manifest, operator_email=email, operator_user_id=user_id,
        require_complete=False, require_terminal=True)
    if len(final["operations"]) != 1 or len(final["replay_jobs"]) != 1:
        raise fixture.FixtureError("Recovery created duplicate durable work.")
    fixture._write_json(report_dir / REPORT_NAME, {**evidence, "status": "passed", "repeat_idempotent": True})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        if args.resume:
            try:
                finish_interrupted(args.report_dir, os.environ)
            finally:
                fixture.cleanup_fixture(report_dir=args.report_dir, env=os.environ)
        elif args.child:
            commit_child(args.report_dir, os.environ)
        else:
            rehearse(args.report_dir, os.environ)
            print("Interrupted exclusion recovered through staging API; retry and cleanup passed.")
    except fixture.FixtureError as exc:
        print(f"Interrupted exclusion rehearsal: {exc}", file=sys.stderr)
        return 2
    except Exception:
        # Never emit SDK exception bodies, credentials, or child output.
        print("Interrupted exclusion rehearsal failed; inspect sanitized fixture evidence.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
