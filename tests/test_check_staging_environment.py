from __future__ import annotations

import json
import hashlib
from types import SimpleNamespace

from scripts import check_staging_environment as cse


class _FakeQuery:
    def __init__(self, table_name: str, fail_tables: set[str]):
        self.table_name = table_name
        self.fail_tables = fail_tables

    def select(self, _cols: str):
        return self

    def limit(self, _n: int):
        return self

    def execute(self):
        if self.table_name in self.fail_tables:
            raise RuntimeError("boom")
        return SimpleNamespace(data=[])


class _FakeSupabase:
    def __init__(self, fail_tables: set[str] | None = None):
        self.fail_tables = fail_tables or set()

    def table(self, table_name: str):
        return _FakeQuery(table_name, self.fail_tables)


def _args(**kwargs):
    base = {
        "expect_full_next_admin": False,
        "api_base_url": None,
        "require_api": False,
        "require_supabase": False,
        "require_supabase_isolation": False,
        "expected_supabase_project_ref": None,
        "club_slug": "tres-palapas",
        "club_id": "tres_palapas",
        "write_wave": None,
        "expected_git_sha": None,
        "expected_fly_app_name": None,
        "expected_web_origin": None,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_missing_jupr_env_fails(monkeypatch):
    monkeypatch.delenv("JUPR_ENV", raising=False)
    rc, summary = cse.run_checks(_args())
    assert rc == 1
    assert summary["ok"] is False
    assert any("JUPR_ENV" in e for e in summary["errors"])


def test_production_is_rejected(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "production")
    rc, summary = cse.run_checks(_args())
    assert rc == 2
    assert summary["ok"] is False
    assert any("production" in err for err in summary["errors"])


def test_score_entry_is_a_write_wave_gate_not_a_full_read_surface_flag(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY", "true")
    rc, summary = cse.run_checks(_args())
    assert rc == 0
    assert summary["ok"] is True
    assert "JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY" not in summary["next_admin_flags"]["required"]
    assert summary["staging_write_wave"]["flags"]["JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY"] is True
    assert summary["warnings"]


def test_expect_full_next_admin_flags_reports_missing(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")
    rc, summary = cse.run_checks(_args(expect_full_next_admin=True))
    assert rc == 1
    assert summary["ok"] is False
    assert any("Full Next admin staging requested" in err for err in summary["errors"])


def test_expect_full_next_admin_passes_when_flags_enabled(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")
    for name in cse.FULL_NEXT_ADMIN_FLAGS:
        monkeypatch.setenv(name, "1")
    for name in cse.ALL_STAGING_WRITE_FLAGS:
        monkeypatch.setenv(name, "0")
    for name in cse.ALWAYS_DISABLED_FLAGS:
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "none")
    monkeypatch.setenv("JUPR_EMAIL_MODE", "dry_run")
    rc, summary = cse.run_checks(
        _args(expect_full_next_admin=True, write_wave="none")
    )
    assert rc == 0
    assert summary["ok"] is True
    assert summary["next_admin_flags"]["required_enabled_count"] == len(cse.FULL_NEXT_ADMIN_FLAGS)


def test_staging_redirect_requires_redirect_address(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_EMAIL_MODE", "staging_redirect")
    rc, summary = cse.run_checks(_args())
    assert rc == 1
    assert any("JUPR_STAGING_EMAIL_REDIRECT_TO" in err for err in summary["errors"])


def test_secret_values_are_not_printed(monkeypatch, capsys):
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "super-secret-value")
    monkeypatch.setenv("SUPABASE_URL", "https://user:pass@example.supabase.co/path")
    monkeypatch.setattr(cse, "make_supabase", lambda _u, _k: _FakeSupabase())
    rc = cse.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "super-secret-value" not in out
    assert "user:pass" not in out


def test_mocked_supabase_table_checks(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon")
    monkeypatch.setattr(cse, "make_supabase", lambda _u, _k: _FakeSupabase())
    rc, summary = cse.run_checks(_args(require_supabase=True))
    assert rc == 0
    assert summary["checked_tables"]["clubs"]["status"] == "ok"
    monkeypatch.setattr(cse, "make_supabase", lambda _u, _k: _FakeSupabase({"replay_jobs"}))
    rc2, summary2 = cse.run_checks(_args(require_supabase=True))
    assert rc2 == 1
    assert summary2["checked_tables"]["replay_jobs"]["status"] == "error"


def test_supabase_schema_inventory_covers_full_next_workflows():
    required = {
        "league_live_sessions",
        "league_live_rounds",
        "league_live_courts",
        "player_profile_update_subscriptions",
        "player_profile_update_outbox",
        "tournament_registration_settings",
        "tournament_registration_partner_requests",
        "tournament_registration_team_links",
        "tournament_event_draws",
        "tournament_games",
        "tournament_podium",
        "worker_run_log",
    }
    assert required.issubset(set(cse.SUPABASE_OBJECTS))


def test_required_supabase_isolation_needs_expected_project_ref(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("SUPABASE_URL", "https://stageproject.supabase.co")
    rc, summary = cse.run_checks(_args(require_supabase_isolation=True))
    assert rc == 1
    assert summary["supabase_isolation"]["verified"] is False
    assert any("isolation verification requires" in err for err in summary["errors"])


def test_supabase_isolation_rejects_project_mismatch(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("SUPABASE_URL", "https://productionref.supabase.co")
    rc, summary = cse.run_checks(
        _args(require_supabase_isolation=True, expected_supabase_project_ref="stagingref")
    )
    assert rc == 1
    assert any("project mismatch" in err for err in summary["errors"])


def test_supabase_isolation_accepts_matching_project(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("SUPABASE_URL", "https://stagingref.supabase.co")
    rc, summary = cse.run_checks(
        _args(require_supabase_isolation=True, expected_supabase_project_ref="stagingref")
    )
    assert rc == 0
    assert summary["supabase_isolation"]["verified"] is True


def test_supabase_project_ref_parser_requires_a_canonical_exact_origin():
    assert cse._supabase_project_ref("https://stagingref.supabase.co") == "stagingref"
    assert cse._supabase_project_ref("https://stagingref.supabase.co/") == "stagingref"
    for unsafe in (
        "http://stagingref.supabase.co",
        "https://stagingref.extra.supabase.co",
        "https://user@stagingref.supabase.co",
        "https://stagingref.supabase.co:443",
        "https://stagingref.supabase.co/rest/v1",
        "https://stagingref.supabase.co?target=staging",
    ):
        assert cse._supabase_project_ref(unsafe) is None


def test_mocked_api_checks(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")

    def fake_get(url: str):
        if url.endswith("/admin/operations/status"):
            return 401, None, "HTTP 401"
        if url.endswith("/health"):
            return 200, {"ok": True}, None
        if url.endswith("/leaderboards"):
            return 200, {"club": {}, "leaderboard": []}, None
        if "/clubs/tres-palapas" in url:
            return 200, {"id": "1", "slug": "tres-palapas", "name": "Tres"}, None
        return 200, {"ok": True}, None

    monkeypatch.setattr(cse, "_http_get_json", fake_get)
    rc, summary = cse.run_checks(_args(api_base_url="https://api.example.com", require_api=True))
    assert rc == 0
    assert summary["checked_endpoints"]["/health"]["status"] == "ok"
    assert "/admin/operations/status" in summary["checked_endpoints"]
    assert (
        summary["checked_endpoints"]["/admin/operations/status"]["protected"]
        is True
    )
    assert "/admin/clubs/tres_palapas/tools/status" in summary["checked_endpoints"]
    assert "/admin/clubs/tres_palapas/tournaments/setup/status" in summary["checked_endpoints"]
    assert "/admin/clubs/tres_palapas/match-canonical-audit/status" in summary["checked_endpoints"]

    monkeypatch.setattr(cse, "_http_get_json", lambda _url: (500, None, "HTTP 500"))
    rc2, summary2 = cse.run_checks(_args(api_base_url="https://api.example.com", require_api=True))
    assert rc2 == 1
    assert summary2["checked_endpoints"]["/health"]["status"] == "error"


def test_full_next_api_check_requires_enabled_status_payloads(monkeypatch):
    sha = "a" * 40
    project_ref = "stagingref"
    app_name = "juprleagues-api-staging"
    expected_gates = cse.expected_write_flags("none")
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "none")
    monkeypatch.setenv("JUPR_EMAIL_MODE", "dry_run")
    monkeypatch.setenv("SUPABASE_URL", f"https://{project_ref}.supabase.co")
    for name in cse.FULL_NEXT_ADMIN_FLAGS:
        monkeypatch.setenv(name, "1")
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES", raising=False)
    for name, enabled in expected_gates.items():
        monkeypatch.setenv(name, "1" if enabled else "0")
    for name in cse.ALWAYS_DISABLED_FLAGS:
        monkeypatch.setenv(name, "0")

    def expected_surface_flag(flag: str) -> bool:
        return expected_gates.get(flag, flag in cse.FULL_NEXT_ADMIN_FLAGS)

    def status_payload(path: str):
        gated_statuses = {
            "/match-uploader/status": "JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER",
            "/players/editor/status": "JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR",
            "/player-updates/status": "JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES",
            "/verified-updates/status": "JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES",
            "/support-requests/status": "JUPR_ENABLE_NEXT_ADMIN_SUPPORT_REQUESTS",
            "/league-manager/live/status": "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN",
            "/weekly-recap/status": "JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP",
            "/badges/status": "JUPR_ENABLE_NEXT_ADMIN_BADGE_DIAGNOSTICS",
            "/tools/status": "JUPR_ENABLE_NEXT_ADMIN_TOOLS",
        }
        enabled = next(
            (
                expected_surface_flag(flag)
                for suffix, flag in gated_statuses.items()
                if path.endswith(suffix)
            ),
            True,
        )
        payload = {"enabled": enabled}
        if path.endswith(
            (
                "/player-updates/status",
                "/verified-updates/status",
                "/weekly-recap/status",
            )
        ):
            payload["mutations_enabled"] = expected_gates[
                "JUPR_ENABLE_NEXT_ADMIN_COMMUNICATIONS_MUTATIONS"
            ]
        if path.endswith("/player-updates/status"):
            payload["auto_send_enabled"] = False
        elif path.endswith("/league-manager/status"):
            payload.update(awards_write_enabled=False, league_manager_writes_enabled=False)
        elif path.endswith("/league-manager/live/status"):
            payload["submit_enabled"] = False
        elif path.endswith("/tournaments/admin/status"):
            payload.update(
                mutation_runtime={
                    "surface_flags": {
                        surface: {"name": flag, "enabled": expected_gates[flag]}
                        for surface, flag in {
                            "tournament": "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_MUTATIONS",
                            "setup": "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_SETUP_MUTATIONS",
                            "registration": "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_REGISTRATION_MUTATIONS",
                            "import_handoff": "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_IMPORT_HANDOFF",
                            "operations": "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OPERATIONS_MUTATIONS",
                            "tournament_live": "JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_LIVE_WRITES",
                        }.items()
                    },
                    "service_role_ready": True,
                },
                operations_runtime={
                    "operations_mutations_enabled": False,
                    "official_publish_enabled": False,
                    "email_handoff_enabled": False,
                    "auto_player_updates_enabled": False,
                    "email_mode": "dry_run",
                },
            )
        elif path.endswith("/tournament-live/status"):
            payload.update(
                write_flag={
                    "name": "JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_LIVE_WRITES",
                    "enabled": False,
                },
                writes_enabled=False,
            )
        elif path.endswith(("/challenge-ladder/status", "/moneyball/status", "/jupr-live/status")):
            payload["writes_enabled"] = False
        elif path.endswith("/match-canonical-audit/status"):
            payload["normalize_writes_enabled"] = False
        return payload

    def fake_get(url: str):
        if url.endswith("/health"):
            fingerprint = hashlib.sha256(
                "\n".join(
                    f"{name}={1 if enabled else 0}"
                    for name, enabled in sorted(expected_gates.items())
                ).encode("utf-8")
            ).hexdigest()
            return 200, {
                "ok": True,
                "environment": "staging",
                "git_commit_sha": sha,
                "image_build_git_sha": sha,
                "fly_app_name": app_name,
                "fly_image_ref": "registry.fly.io/staging@sha256:123",
                "web_origin": "https://jupr-git-staging-pickleballattrespalapas1.vercel.app",
                "supabase_project_ref": project_ref,
                "staging_write_wave": "none",
                "business_data_write_wave_active": False,
                "security_denial_audit_logging_required": True,
                "jwt_verification_configured": True,
                "jwt_verification_mode": "jwks",
                "jwt_verification_project_ref": project_ref,
                "public_live_writes_enabled": False,
                "public_live_production_override_enabled": False,
                "controlled_write_flags": expected_gates,
                "controlled_write_flag_fingerprint": fingerprint,
                "write_prerequisites": {
                    "service_role_configured": True,
                    "api_audit_required": True,
                    "worker_run_log_required": True,
                    "email_mode": "dry_run",
                    "live_player_update_email_enabled": False,
                },
            }, None
        if url.endswith("/admin/operations/status"):
            return 401, None, "HTTP 401"
        if "/admin/clubs/" in url and url.endswith("/status"):
            return 200, status_payload(url), None
        if url.endswith("/health/live-sessions"):
            return 200, {"ok": True}, None
        if url.endswith("/leaderboards"):
            return 200, {"club": {}, "leaderboard": []}, None
        if url.endswith("/live-sessions"):
            return 200, {"write_enabled": False}, None
        return 200, {"id": "1", "slug": "tres-palapas", "name": "Tres"}, None

    monkeypatch.setattr(cse, "_http_get_json", fake_get)
    rc, summary = cse.run_checks(
        _args(
            api_base_url="https://api.example.com",
            require_api=True,
            expect_full_next_admin=True,
            write_wave="none",
            expected_git_sha=sha,
            expected_fly_app_name=app_name,
            expected_web_origin="https://jupr-git-staging-pickleballattrespalapas1.vercel.app",
            expected_supabase_project_ref=project_ref,
        )
    )
    assert rc == 0
    assert (
        summary["checked_endpoints"]["/admin/operations/status"]["protected"]
        is True
    )
    assert all(
        summary["checked_endpoints"][template.format(club_id="tres_palapas")]["status"] == "ok"
        for template in cse.ADMIN_STATUS_PATHS
    )
    for suffix in (
        "player-updates/status",
        "verified-updates/status",
        "weekly-recap/status",
    ):
        checked = summary["checked_endpoints"][
            f"/admin/clubs/tres_palapas/{suffix}"
        ]
        assert checked["enabled"] is True
        assert checked["mutations_enabled"] is False

    def fake_web_origin_drift_get(url: str):
        status, payload, error = fake_get(url)
        if url.endswith("/health"):
            payload = {**payload, "web_origin": "https://pickleballclubsandwich.com"}
        return status, payload, error

    monkeypatch.setattr(cse, "_http_get_json", fake_web_origin_drift_get)
    drift_rc, drift_summary = cse.run_checks(
        _args(
            api_base_url="https://api.example.com",
            require_api=True,
            expect_full_next_admin=True,
            write_wave="none",
            expected_git_sha=sha,
            expected_fly_app_name=app_name,
            expected_web_origin="https://jupr-git-staging-pickleballattrespalapas1.vercel.app",
            expected_supabase_project_ref=project_ref,
        )
    )
    assert drift_rc == 1
    assert any("web_origin" in error for error in drift_summary["errors"])

    def fake_image_sha_drift_get(url: str):
        status, payload, error = fake_get(url)
        if url.endswith("/health"):
            payload = {**payload, "image_build_git_sha": "b" * 40}
        return status, payload, error

    monkeypatch.setattr(cse, "_http_get_json", fake_image_sha_drift_get)
    image_drift_rc, image_drift_summary = cse.run_checks(
        _args(
            api_base_url="https://api.example.com",
            require_api=True,
            expect_full_next_admin=True,
            write_wave="none",
            expected_git_sha=sha,
            expected_fly_app_name=app_name,
            expected_web_origin="https://jupr-git-staging-pickleballattrespalapas1.vercel.app",
            expected_supabase_project_ref=project_ref,
        )
    )
    assert image_drift_rc == 1
    assert any(
        "image_build_git_sha" in error
        for error in image_drift_summary["errors"]
    )

    disabled_path = "/admin/clubs/tres_palapas/tournaments/setup/status"

    def fake_disabled_get(url: str):
        if url.endswith(disabled_path):
            return 200, {"enabled": False}, None
        return fake_get(url)

    monkeypatch.setattr(cse, "_http_get_json", fake_disabled_get)
    rc2, summary2 = cse.run_checks(
        _args(
            api_base_url="https://api.example.com",
            require_api=True,
            expect_full_next_admin=True,
            write_wave="none",
            expected_git_sha=sha,
            expected_fly_app_name=app_name,
            expected_web_origin="https://jupr-git-staging-pickleballattrespalapas1.vercel.app",
            expected_supabase_project_ref=project_ref,
        )
    )
    assert rc2 == 1
    assert summary2["checked_endpoints"][disabled_path]["status"] == "error"


def test_json_output_is_valid(monkeypatch, capsys):
    monkeypatch.setenv("JUPR_ENV", "staging")
    rc = cse.main(["--json"])
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert rc == 0
    assert isinstance(parsed, dict)
    assert "next_admin_flags" in parsed


def test_open_write_wave_is_a_supported_full_staging_posture(monkeypatch):
    args = cse.build_parser().parse_args(["--write-wave", "open"])
    assert args.write_wave == "open"

    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "open")
    monkeypatch.setenv("JUPR_EMAIL_MODE", "dry_run")
    monkeypatch.setenv("JUPR_REGISTRATION_EDIT_SECRET", "edit-" + "a" * 40)
    monkeypatch.setenv("JUPR_REGISTRATION_CONFIRMATION_SECRET", "confirm-" + "b" * 40)
    monkeypatch.setenv("JUPR_PUBLIC_LIVE_TOKEN_SECRET", "token-" + "c" * 40)
    monkeypatch.setenv("JUPR_PUBLIC_LIVE_RATE_LIMIT_SECRET", "rate-" + "d" * 40)
    for name in cse.FULL_NEXT_ADMIN_FLAGS:
        monkeypatch.setenv(name, "1")
    for name, enabled in cse.expected_write_flags("open").items():
        monkeypatch.setenv(name, "1" if enabled else "0")
    for name in cse.ALWAYS_DISABLED_FLAGS:
        monkeypatch.setenv(name, "0")

    rc, summary = cse.run_checks(
        _args(expect_full_next_admin=True, write_wave="open")
    )

    assert rc == 0, summary
    assert summary["ok"] is True
    assert summary["staging_write_wave"]["known"] is True
    assert summary["staging_write_wave"]["expected"] == "open"
    assert summary["staging_write_wave"]["actual"] == "open"
    assert summary["staging_write_wave"]["public_live_token_secret_configured"] is True
    assert summary["staging_write_wave"]["public_live_rate_limit_secret_configured"] is True


def test_full_next_flag_check_can_defer_to_required_live_api(monkeypatch):
    for name in cse.FULL_NEXT_ADMIN_FLAGS:
        monkeypatch.delenv(name, raising=False)
    summary = {"errors": [], "warnings": []}

    cse._check_full_next_flags(
        summary,
        expect_full_next_admin=True,
        defer_to_api=True,
    )

    assert summary["errors"] == []
    assert len(summary["warnings"]) == 1
    assert "live API status checks are authoritative" in summary["warnings"][0]
