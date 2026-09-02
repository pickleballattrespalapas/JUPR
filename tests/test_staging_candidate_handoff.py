from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.staging_candidate_handoff import (
    EXPECTED_WRITE_WAVE,
    FLY_HEALTH_URL,
    HandoffError,
    PRODUCTION_FLY_APP,
    PRODUCTION_SUPABASE_PROJECT_REF,
    READ_ONLY_CHECKS,
    STAGING_AUTH_ORIGIN,
    STAGING_FLY_APP,
    STAGING_FLY_ORIGIN,
    STAGING_SUPABASE_PROJECT_REF,
    STAGING_VERCEL_ALIAS,
    VERCEL_ENVIRONMENT_URL,
    build_handoff,
    render_markdown,
    run_read_only_checks,
    validate_identity,
    write_handoff,
)
from scripts.staging_write_waves import expected_write_flags


CANDIDATE_SHA = "a" * 40


def _fly_identity() -> dict[str, object]:
    flags = expected_write_flags(EXPECTED_WRITE_WAVE)
    fingerprint = hashlib.sha256(
        "\n".join(
            f"{name}={1 if enabled else 0}"
            for name, enabled in sorted(flags.items())
        ).encode("utf-8")
    ).hexdigest()
    return {
        "ok": True,
        "environment": "staging",
        "git_commit_sha": CANDIDATE_SHA,
        "image_build_git_sha": CANDIDATE_SHA,
        "fly_app_name": STAGING_FLY_APP,
        "fly_image_ref": (
            "registry.fly.io/juprleagues-api-staging:"
            "deployment-01KSTAGINGHANDOFF123456789"
        ),
        "web_origin": STAGING_VERCEL_ALIAS,
        "supabase_project_ref": STAGING_SUPABASE_PROJECT_REF,
        "jwt_verification_configured": True,
        "jwt_verification_mode": "jwks",
        "jwt_verification_project_ref": STAGING_SUPABASE_PROJECT_REF,
        "staging_write_wave": EXPECTED_WRITE_WAVE,
        "write_wave": EXPECTED_WRITE_WAVE,
        "business_data_write_wave_active": True,
        "security_denial_audit_logging_required": True,
        "public_live_writes_enabled": flags["JUPR_ENABLE_PUBLIC_LIVE_WRITES"],
        "public_live_production_override_enabled": False,
        "controlled_write_flags": flags,
        "controlled_write_flag_fingerprint": fingerprint,
        "write_prerequisites": {
            "service_role_configured": True,
            "api_audit_required": True,
            "worker_run_log_required": True,
            "email_mode": "dry_run",
            "live_player_update_email_enabled": False,
        },
    }


def _web_identity() -> dict[str, object]:
    return {
        "environment": "staging",
        "vercel_environment": "preview",
        "git_commit_sha": CANDIDATE_SHA,
        "vercel_deployment_id": "dpl_StagingHandoff123",
        "vercel_deployment_origin": (
            "https://jupr-abcdefgh-pickleballattrespalapas1.vercel.app"
        ),
        "api_origin": STAGING_FLY_ORIGIN,
        "auth_origin": STAGING_AUTH_ORIGIN,
        "preview_isolation_active": True,
        "preview_auth_isolation_active": True,
    }


def _checks() -> list[dict[str, object]]:
    return [
        {"name": name, "method": "GET", "url": url, "status": 200, "ok": True}
        for name, url, _expect_json, _vercel in READ_ONLY_CHECKS
    ]


class StagingCandidateHandoffTests(unittest.TestCase):
    def test_identity_accepts_only_the_exact_persistent_open_candidate(self) -> None:
        identity = validate_identity(
            candidate_sha=CANDIDATE_SHA,
            fly=_fly_identity(),
            web=_web_identity(),
        )

        self.assertEqual(
            identity,
            {
                "candidate_sha": CANDIDATE_SHA,
                "fly_image_ref": (
                    "registry.fly.io/juprleagues-api-staging:"
                    "deployment-01KSTAGINGHANDOFF123456789"
                ),
                "vercel_deployment_id": "dpl_StagingHandoff123",
                "vercel_deployment_origin": (
                    "https://jupr-abcdefgh-pickleballattrespalapas1.vercel.app"
                ),
                "supabase_project_ref": STAGING_SUPABASE_PROJECT_REF,
                "write_wave": "open",
                "email_mode": "dry_run",
            },
        )

    def test_identity_rejects_mismatches_and_production_targets(self) -> None:
        cases = (
            ("fly", "git_commit_sha", "b" * 40),
            ("web", "git_commit_sha", "b" * 40),
            ("fly", "staging_write_wave", "none"),
            ("fly", "write_wave", "none"),
            ("fly", "fly_app_name", PRODUCTION_FLY_APP),
            ("fly", "supabase_project_ref", PRODUCTION_SUPABASE_PROJECT_REF),
            ("web", "api_origin", "https://api.juprleagues.com"),
            (
                "web",
                "auth_origin",
                f"https://{PRODUCTION_SUPABASE_PROJECT_REF}.supabase.co",
            ),
        )
        for surface, key, value in cases:
            with self.subTest(surface=surface, key=key, value=value):
                fly = _fly_identity()
                web = _web_identity()
                (fly if surface == "fly" else web)[key] = value
                with self.assertRaisesRegex(HandoffError, "identity rejected"):
                    validate_identity(
                        candidate_sha=CANDIDATE_SHA, fly=fly, web=web
                    )

    def test_identity_requires_dry_run_email(self) -> None:
        fly = _fly_identity()
        prerequisites = dict(fly["write_prerequisites"])
        prerequisites["email_mode"] = "live"
        fly["write_prerequisites"] = prerequisites

        with self.assertRaisesRegex(HandoffError, "dry_run"):
            validate_identity(
                candidate_sha=CANDIDATE_SHA, fly=fly, web=_web_identity()
            )

    def test_read_only_smoke_uses_only_hard_coded_gets(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        def fake_get(url: str, **kwargs: object) -> tuple[int, object]:
            calls.append((url, kwargs))
            return 200, {}

        results = run_read_only_checks(get=fake_get)

        self.assertEqual(len(results), len(READ_ONLY_CHECKS))
        self.assertTrue(
            all(result["method"] == "GET" and result["ok"] for result in results)
        )
        self.assertEqual(
            {url for url, _kwargs in calls},
            {url for _name, url, _expect_json, _vercel in READ_ONLY_CHECKS},
        )
        self.assertIn(FLY_HEALTH_URL, {url for url, _kwargs in calls})
        self.assertTrue(
            all(
                url.startswith((STAGING_FLY_ORIGIN, STAGING_VERCEL_ALIAS))
                for url, _kwargs in calls
            )
        )
        self.assertNotIn(
            VERCEL_ENVIRONMENT_URL,
            {url for _name, url, _expect_json, _vercel in READ_ONLY_CHECKS},
        )

    def test_handoff_renders_secret_free_json_and_markdown(self) -> None:
        identity = validate_identity(
            candidate_sha=CANDIDATE_SHA,
            fly=_fly_identity(),
            web=_web_identity(),
        )
        handoff = build_handoff(
            candidate_sha=CANDIDATE_SHA,
            identity=identity,
            checks=_checks(),
            workflow_run_url=(
                "https://github.com/pickleballattrespalapas/JUPR/actions/runs/12345"
            ),
            generated_at="2026-08-20T12:00:00Z",
        )
        with tempfile.TemporaryDirectory() as temporary:
            json_path, markdown_path = write_handoff(Path(temporary), handoff)
            json_text = json_path.read_text(encoding="utf-8")
            stored = json.loads(json_text)
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(stored["status"], "ready_for_manual_testing")
        self.assertEqual(stored["candidate_sha"], CANDIDATE_SHA)
        self.assertEqual(stored["safety"]["requests"], "GET only")
        self.assertIs(stored["safety"]["production_targets_contacted"], False)
        self.assertIn("ready_for_manual_testing", markdown)
        self.assertIn(CANDIDATE_SHA, markdown)
        self.assertIn("Production targets were not contacted", markdown)
        self.assertNotIn("secret", json_text.lower())

    def test_handoff_rejects_non_get_or_failed_checks(self) -> None:
        identity = validate_identity(
            candidate_sha=CANDIDATE_SHA,
            fly=_fly_identity(),
            web=_web_identity(),
        )
        checks = _checks()
        checks[0]["method"] = "POST"

        with self.assertRaisesRegex(HandoffError, "successful GET"):
            build_handoff(
                candidate_sha=CANDIDATE_SHA,
                identity=identity,
                checks=checks,
                workflow_run_url=(
                    "https://github.com/pickleballattrespalapas/JUPR/actions/runs/12345"
                ),
            )

    def test_markdown_rejects_incomplete_handoff(self) -> None:
        with self.assertRaisesRegex(HandoffError, "missing deployment"):
            render_markdown({"status": "ready_for_manual_testing"})

    def test_fly_workflow_uploads_exact_open_candidate_handoff(self) -> None:
        workflow = Path(".github/workflows/fly_api_staging_deploy.yml").read_text(
            encoding="utf-8"
        )
        verify_index = workflow.index("- name: Verify full staging feature surface")
        handoff_index = workflow.index("- name: Build exact-candidate staging handoff")

        self.assertLess(verify_index, handoff_index)
        self.assertIn("id: staging-posture", workflow)
        self.assertIn("printf 'write_wave=%s\\n'", workflow)
        self.assertIn("steps.staging-posture.outputs.write_wave == 'open'", workflow)
        self.assertIn("scripts/staging_candidate_handoff.py", workflow)
        self.assertIn("VERCEL_AUTOMATION_BYPASS_SECRET", workflow)
        self.assertIn("staging-handoff-${{ github.sha }}", workflow)
        self.assertIn("retention-days: 90", workflow)
        self.assertIn(
            'cat "$RUNNER_TEMP/staging-handoff/staging-handoff.md" >> "$GITHUB_STEP_SUMMARY"',
            workflow,
        )
        self.assertNotIn("environment: production", workflow)
        self.assertNotIn(PRODUCTION_SUPABASE_PROJECT_REF, workflow)


if __name__ == "__main__":
    unittest.main()
