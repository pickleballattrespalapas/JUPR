from types import SimpleNamespace

from jupr_app.services.admin_tools_service import (
    build_admin_worker_status,
    run_admin_badge_queue_worker,
    run_admin_badge_recompute_job,
)


class FakeQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.filters = []
        self.insert_payload = None
        self.limit_value = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def limit(self, value):
        self.limit_value = int(value)
        return self

    def insert(self, payload):
        self.insert_payload = dict(payload)
        return self

    def execute(self):
        rows = self.storage.setdefault(self.table_name, [])
        if self.insert_payload is not None:
            row = {"id": f"row-{len(rows) + 1}", **self.insert_payload}
            rows.append(row)
            return SimpleNamespace(data=[row])
        scoped = list(rows)
        for key, expected in self.filters:
            scoped = [row for row in scoped if str(row.get(key)) == str(expected)]
        if self.limit_value is not None:
            scoped = scoped[: self.limit_value]
        return SimpleNamespace(data=scoped)


class FakeSupabase:
    def __init__(self):
        self.storage = {
            "badge_eval_queue": [
                {"id": "q1", "club_id": "club", "status": "pending"},
                {"id": "q2", "club_id": "club", "status": "error"},
            ],
            "badge_recompute_runs": [],
            "admin_activity_log": [],
        }

    def table(self, name):
        return FakeQuery(self.storage, name)


def test_admin_worker_status_counts_queue(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOOLS", "1")
    payload = build_admin_worker_status(FakeSupabase(), club_id="club")
    assert payload["queue_counts"]["pending"]["count"] == 1
    assert payload["queue_counts"]["error"]["count"] == 1


def test_badge_queue_worker_requires_confirmation(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOOLS", "1")
    try:
        run_admin_badge_queue_worker(
            FakeSupabase(),
            club_id="club",
            mode="batch",
            max_jobs=10,
            time_budget_seconds=5,
            actor_email="admin@example.com",
            actor_role="super_admin",
            confirmation_text="PROCESS",
        )
    except ValueError as exc:
        assert "PROCESS BADGE QUEUE" in str(exc)
    else:
        raise AssertionError("expected confirmation error")


def test_badge_queue_worker_writes_audit(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOOLS", "1")
    monkeypatch.setattr("jupr_app.services.admin_tools_service.process_badge_eval_queue", lambda *_args, **_kwargs: {"processed": 2, "errored": 0})
    supabase = FakeSupabase()
    result = run_admin_badge_queue_worker(
        supabase,
        club_id="club",
        mode="batch",
        max_jobs=10,
        time_budget_seconds=5,
        actor_email="admin@example.com",
        actor_role="super_admin",
        confirmation_text="PROCESS BADGE QUEUE",
    )
    assert result["result"]["processed"] == 2
    assert supabase.storage["admin_activity_log"]


def test_badge_recompute_apply_requires_confirmation(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOOLS", "1")
    try:
        run_admin_badge_recompute_job(
            FakeSupabase(),
            club_id="club",
            mode="append-only",
            player_id=1,
            badge_id=None,
            league_id=None,
            context_id=None,
            since=None,
            until=None,
            include_non_live=False,
            allow_strict_global=False,
            match_limit=5000,
            actor_email="admin@example.com",
            actor_role="super_admin",
            confirmation_text="RUN",
        )
    except ValueError as exc:
        assert "RUN BADGE RECOMPUTE" in str(exc)
    else:
        raise AssertionError("expected confirmation error")


def test_badge_recompute_invokes_python_domain(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOOLS", "1")
    monkeypatch.setattr(
        "jupr_app.services.admin_tools_service.run_badge_recompute",
        lambda *_args, **_kwargs: {"mode": _kwargs.get("mode"), "new_awards_count": 3},
    )
    supabase = FakeSupabase()
    result = run_admin_badge_recompute_job(
        supabase,
        club_id="club",
        mode="append-only",
        player_id=1,
        badge_id="high_roller",
        league_id=None,
        context_id=None,
        since=None,
        until=None,
        include_non_live=False,
        allow_strict_global=False,
        match_limit=5000,
        actor_email="admin@example.com",
        actor_role="super_admin",
        confirmation_text="RUN BADGE RECOMPUTE",
    )
    assert result["summary"]["new_awards_count"] == 3
    assert supabase.storage["admin_activity_log"]
