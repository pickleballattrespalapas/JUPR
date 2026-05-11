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
        return {"data": []}


class _FakeSupabase:
    def __init__(self, fail_tables: set[str] | None = None):
        self.fail_tables = fail_tables or set()

    def table(self, table_name: str):
        return _FakeQuery(table_name, self.fail_tables)


def _args(**kwargs):
    base = {
        "allow_next_admin_score_entry": False,
        "json": False,
        "api_base_url": None,
        "require_api": False,
        "require_supabase": False,
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


def test_next_admin_entry_guard(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY", "true")
    rc, summary = cse.run_checks(_args())
    assert rc == 1
    assert any("JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY" in e for e in summary["errors"])

    rc2, summary2 = cse.run_checks(_args(allow_next_admin_score_entry=True))
    assert rc2 == 0
    assert summary2["ok"] is True


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


def test_mocked_api_checks(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")

    def fake_get(url: str):
        if url.endswith("/health"):
            return 200, {"ok": True}, None
        if url.endswith("/clubs/tres-palapas"):
            return 200, {"id": "1", "slug": "tres-palapas", "name": "Tres"}, None
        return 200, {"club": {}, "leaderboard": []}, None

    monkeypatch.setattr(cse, "_http_get_json", fake_get)
    rc, summary = cse.run_checks(_args(api_base_url="https://api.example.com", require_api=True))
    assert rc == 0
    assert summary["checked_endpoints"]["/health"]["status"] == "ok"

    monkeypatch.setattr(cse, "_http_get_json", lambda _url: (500, None, "HTTP 500"))
    rc2, summary2 = cse.run_checks(_args(api_base_url="https://api.example.com", require_api=True))
    assert rc2 == 1
    assert summary2["checked_endpoints"]["/health"]["status"] == "error"


def test_json_output_is_valid(monkeypatch, capsys):
    monkeypatch.setenv("JUPR_ENV", "staging")
    rc = cse.main(["--json"])
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert rc == 0
    assert isinstance(parsed, dict)
    assert "ok" in parsed
