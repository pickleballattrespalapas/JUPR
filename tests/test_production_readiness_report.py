from __future__ import annotations

import json
from types import SimpleNamespace

import scripts.production_readiness_report as prr


def test_classify_changed_files_correctly() -> None:
    assert prr.classify_path("streamlit_app.py") == "Streamlit UI"
    assert prr.classify_path("jupr_app/services/match_service.py") == "domain/services"
    assert prr.classify_path("supabase/migrations/20260101010101_x.sql") == "Supabase migrations"
    assert prr.classify_path("services/api/main.py") == "FastAPI"
    assert prr.classify_path("apps/web/app/page.tsx") == "Next.js"
    assert prr.classify_path("jupr_app/workers/badge_worker.py") == "workers"
    assert prr.classify_path("docs/saas_staging_deploy.md") == "docs"
    assert prr.classify_path("tests/test_api_health.py") == "tests/CI"


def test_detect_migration_additions(monkeypatch) -> None:
    def fake_run(*args: str):
        cmd = " ".join(args)
        if "--name-status" in cmd:
            return prr.CommandResult(cmd, 0, "A supabase/migrations/202601.sql\nA migrations/202602.sql", "")
        if "--name-only" in cmd:
            return prr.CommandResult(cmd, 0, "supabase/migrations/202601.sql\nmigrations/202602.sql", "")
        if "rev-list" in cmd:
            return prr.CommandResult(cmd, 0, "0 2", "")
        return prr.CommandResult(cmd, 0, "ok", "")

    monkeypatch.setattr(prr, "run_command", fake_run)
    report = prr.build_report("rollback-feb8", "Test")
    assert report["migration_summary"]["supabase_migrations_added"] == ["supabase/migrations/202601.sql"]
    assert report["migration_summary"]["root_migrations_added"] == ["migrations/202602.sql"]
    assert report["migration_summary"]["root_migration_warning"] is True


def test_detect_high_risk_and_service_role(monkeypatch) -> None:
    changed = ["apps/web/admin/score/page.tsx", "jupr_app/match_processing.py", "supabase/migrations/a.sql"]

    def fake_run(*args: str):
        cmd = " ".join(args)
        if "apps/web" in cmd and "diff" in cmd:
            return prr.CommandResult(cmd, 0, "SUPABASE_SERVICE_ROLE_KEY\nJUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=1", "")
        return prr.CommandResult(cmd, 0, "", "")

    monkeypatch.setattr(prr, "run_command", fake_run)
    flags = prr.build_risk_flags(changed)
    assert any("admin score entry" in x for x in flags)
    assert any("service-role" in x for x in flags)
    assert any("match_processing.py" in x for x in flags)
    assert any("Supabase migration files changed" in x for x in flags)


def test_json_output_valid(monkeypatch, capsys) -> None:
    fake_report = {
        "base": "rollback-feb8",
        "head": "Test",
        "ahead_behind": "ok",
        "changed_files_count": 0,
        "changed_files_grouped": {k: [] for k in prr.AREAS},
        "migration_summary": {"supabase_migrations_added": [], "root_migrations_added": [], "root_migration_warning": False, "checks": {}},
        "risk_flags": [],
        "recommended_checklist": ["a"],
    }

    monkeypatch.setattr(prr, "build_report", lambda base, head: fake_report)
    monkeypatch.setattr("sys.argv", ["production_readiness_report.py", "--json"])
    prr.main()
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["base"] == "rollback-feb8"
