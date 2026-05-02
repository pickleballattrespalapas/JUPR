from pathlib import Path


def test_admin_tools_has_replay_jobs_migration_guidance():
    source = Path("jupr_app/ui/pages/admin_tools.py").read_text()
    assert "20260502120000_replay_jobs.sql" in source
    assert "NOTIFY pgrst, 'reload schema';" in source
