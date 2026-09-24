from __future__ import annotations

from types import SimpleNamespace

from jupr_app.workers.badge_queue_worker import main, run_badge_queue_worker


def test_run_badge_queue_worker_passes_arguments(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")

    captured = {}

    class EmptyQuery:
        def insert(self, _payload):
            return self

        def update(self, _payload):
            return self

        def eq(self, _key, _value):
            return self

        def execute(self):
            return SimpleNamespace(data=[])

    def rpc(name, args):
        captured["season_enqueue"] = (name, args)
        return EmptyQuery()
    fake_client = SimpleNamespace(table=lambda _name: EmptyQuery(), rpc=rpc)

    def fake_make_supabase(url, key):
        captured["url"] = url
        captured["key"] = key
        return fake_client

    def fake_process(supabase, club_id, **kwargs):
        captured["supabase"] = supabase
        captured["club_id"] = club_id
        captured["kwargs"] = kwargs
        return {"total_processed": 3, "total_errored": 0, "loops": 1, "stopped_reason": "empty"}

    monkeypatch.setattr("jupr_app.workers.badge_queue_worker.make_supabase", fake_make_supabase)
    monkeypatch.setattr(
        "jupr_app.workers.badge_queue_worker.process_badge_eval_queue_until_empty", fake_process
    )

    summary = run_badge_queue_worker(
        "tres_palapas",
        max_total_jobs=500,
        batch_max_jobs=11,
        per_batch_time_budget_seconds=1.5,
        max_wall_clock_seconds=90,
        max_errors=7,
    )

    assert captured["url"] == "https://example.supabase.co"
    assert captured["key"] == "service-role"
    assert captured["supabase"] is fake_client
    assert captured["club_id"] == "tres_palapas"
    assert captured["kwargs"] == {
        "max_total_jobs": 500,
        "batch_max_jobs": 11,
        "per_batch_time_budget_seconds": 1.5,
        "max_wall_clock_seconds": 90,
        "max_errors": 7,
    }
    assert summary["ok"] is True
    assert summary["key_source"] == "SUPABASE_SERVICE_ROLE_KEY"


def test_main_missing_config_returns_clean_failure(monkeypatch, capsys):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)

    rc = main(["--club-id", "tres_palapas"])
    out = capsys.readouterr().out

    assert rc == 2
    assert '"ok": false' in out.lower()
    assert "SUPABASE_URL" in out


def test_main_rejects_anon_only_config(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")

    rc = main(["--club-id", "tres_palapas", "--max-total-jobs", "2"])
    out = capsys.readouterr().out

    assert rc == 2
    assert "SUPABASE_SERVICE_ROLE_KEY" in out
    assert "cannot use SUPABASE_ANON_KEY" in out
