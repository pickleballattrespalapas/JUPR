from copy import deepcopy
from types import SimpleNamespace

from jupr_app.services.admin_tools_service import (
    build_admin_tournament_match_backfill_preview,
    build_admin_worker_status,
    run_admin_badge_queue_worker,
    run_admin_badge_recompute_job,
)


class FakeQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.filters = []
        self.in_filters = []
        self.insert_payload = None
        self.limit_value = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def in_(self, key, values):
        self.in_filters.append((key, {str(value) for value in values}))
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
        for key, expected_values in self.in_filters:
            scoped = [row for row in scoped if str(row.get(key)) in expected_values]
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
            "badge_eval_runs": [
                {"id": "run1", "status": "done", "scope_json": {"club_id": "club"}},
                {"id": "run2", "status": "done", "scope_json": {"club_id": "other"}},
            ],
            "admin_activity_log": [],
            "tournaments": [],
            "tournament_games": [],
            "tournament_teams": [],
            "matches": [],
        }

    def table(self, name):
        return FakeQuery(self.storage, name)


def test_admin_worker_status_counts_queue(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOOLS", "1")
    payload = build_admin_worker_status(FakeSupabase(), club_id="club")
    assert payload["queue_counts"]["pending"]["count"] == 1
    assert payload["queue_counts"]["error"]["count"] == 1
    assert payload["badge_recompute_run_count"]["count"] == 1


def test_tournament_match_backfill_preview_is_read_only_and_classifies_candidates(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOOLS", "1")
    supabase = FakeSupabase()
    supabase.storage["tournaments"] = [
        {"id": "tour-1", "club_id": "club", "name": "Summer Cup"},
        {"id": "tour-other", "club_id": "other", "name": "Other Club Cup"},
    ]
    supabase.storage["tournament_teams"] = [
        {"id": "team-1", "tournament_id": "tour-1", "player1_id": 1, "player2_id": 2},
        {"id": "team-2", "tournament_id": "tour-1", "player1_id": 3, "player2_id": 4},
    ]
    supabase.storage["tournament_games"] = [
        {"id": "ready", "tournament_id": "tour-1", "team_a_id": "team-1", "team_b_id": "team-2", "score_a": 11, "score_b": 7, "finalized_at": "2026-07-01T00:00:00Z"},
        {"id": "empty", "tournament_id": "tour-1", "team_a_id": "team-1", "team_b_id": "team-2", "score_a": 0, "score_b": 0, "finalized_at": "2026-07-01T00:00:00Z"},
        {"id": "tied", "tournament_id": "tour-1", "team_a_id": "team-1", "team_b_id": "team-2", "score_a": 10, "score_b": 10, "finalized_at": "2026-07-01T00:00:00Z"},
        {"id": "incomplete", "tournament_id": "tour-1", "team_a_id": "team-1", "team_b_id": "missing", "score_a": 11, "score_b": 8, "finalized_at": "2026-07-01T00:00:00Z"},
        {"id": "published", "tournament_id": "tour-1", "team_a_id": "team-1", "team_b_id": "team-2", "score_a": 11, "score_b": 9, "finalized_at": "2026-07-01T00:00:00Z"},
        {"id": "not-final", "tournament_id": "tour-1", "team_a_id": "team-1", "team_b_id": "team-2", "score_a": 11, "score_b": 6, "finalized_at": None},
        {"id": "other-club", "tournament_id": "tour-other", "team_a_id": "x", "team_b_id": "y", "score_a": 11, "score_b": 6, "finalized_at": "2026-07-01T00:00:00Z"},
    ]
    supabase.storage["matches"] = [
        {"id": 99, "club_id": "club", "tournament_game_id": "published"},
        {"id": 100, "club_id": "other", "tournament_game_id": "ready"},
    ]
    before = deepcopy(supabase.storage)

    result = build_admin_tournament_match_backfill_preview(supabase, club_id="club")

    assert result["read_only"] is True
    assert result["summary"]["tournament_count"] == 1
    assert result["summary"]["finalized_game_count"] == 5
    assert result["summary"]["already_published_count"] == 1
    assert result["summary"]["missing_match_count"] == 4
    assert result["summary"]["ready_count"] == 1
    assert result["summary"]["blocked_count"] == 3
    statuses = {row["game_id"]: row["status"] for row in result["candidates"]}
    assert statuses == {
        "empty": "empty_score",
        "incomplete": "incomplete_team",
        "ready": "ready",
        "tied": "tied_score",
    }
    ready = next(row for row in result["candidates"] if row["game_id"] == "ready")
    assert ready["match_payload"]["tournament_game_id"] == "ready"
    assert ready["match_payload"]["t1_p2"] == 2
    assert supabase.storage == before


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
