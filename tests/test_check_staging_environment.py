from __future__ import annotations

import json
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


def test_staging_allows_next_admin_score_entry_as_full_surface_flag(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY", "true")
    rc, summary = cse.run_checks(_args())
    assert rc == 0
    assert summary["ok"] is True
    assert summary["next_admin_flags"]["required"]["JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY"] is True
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
    rc, summary = cse.run_checks(_args(expect_full_next_admin=True))
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


def test_mocked_api_checks(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")

    def fake_get(url: str):
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
    assert "/admin/clubs/tres_palapas/tools/status" in summary["checked_endpoints"]
    assert "/admin/clubs/tres_palapas/tournaments/setup/status" in summary["checked_endpoints"]
    assert "/admin/clubs/tres_palapas/match-canonical-audit/status" in summary["checked_endpoints"]

    monkeypatch.setattr(cse, "_http_get_json", lambda _url: (500, None, "HTTP 500"))
    rc2, summary2 = cse.run_checks(_args(api_base_url="https://api.example.com", require_api=True))
    assert rc2 == 1
    assert summary2["checked_endpoints"]["/health"]["status"] == "error"


def test_full_next_api_check_requires_enabled_status_payloads(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")
    for name in cse.FULL_NEXT_ADMIN_FLAGS:
        monkeypatch.setenv(name, "1")

    def fake_get(url: str):
        if url.endswith("/health"):
            return 200, {"ok": True}, None
        if url.endswith("/admin/operations/status"):
            return 200, {"environment": "staging", "write_pilot_enabled": True, "enabled_workflows": []}, None
        if "/admin/clubs/" in url and url.endswith("/status"):
            return 200, {"enabled": True}, None
        if url.endswith("/leaderboards"):
            return 200, {"club": {}, "leaderboard": []}, None
        return 200, {"id": "1", "slug": "tres-palapas", "name": "Tres"}, None

    monkeypatch.setattr(cse, "_http_get_json", fake_get)
    rc, summary = cse.run_checks(
        _args(api_base_url="https://api.example.com", require_api=True, expect_full_next_admin=True)
    )
    assert rc == 0
    assert all(
        summary["checked_endpoints"][template.format(club_id="tres_palapas")]["enabled"] is True
        for template in cse.ADMIN_STATUS_PATHS
    )

    disabled_path = "/admin/clubs/tres_palapas/tournaments/setup/status"

    def fake_disabled_get(url: str):
        if url.endswith(disabled_path):
            return 200, {"enabled": False}, None
        return fake_get(url)

    monkeypatch.setattr(cse, "_http_get_json", fake_disabled_get)
    rc2, summary2 = cse.run_checks(
        _args(api_base_url="https://api.example.com", require_api=True, expect_full_next_admin=True)
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
