from copy import deepcopy
from types import SimpleNamespace

from jupr_app.services.admin_tools_service import (
    apply_admin_tournament_match_backfill,
    build_admin_rating_report,
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
            "players": [],
            "league_ratings": [],
            "leagues_metadata": [],
        }

    def table(self, name):
        return FakeQuery(self.storage, name)


def test_admin_worker_status_counts_queue(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOOLS", "1")
    payload = build_admin_worker_status(FakeSupabase(), club_id="club")
    assert payload["queue_counts"]["pending"]["count"] == 1
    assert payload["queue_counts"]["error"]["count"] == 1
    assert payload["badge_recompute_run_count"]["count"] == 1


def test_admin_rating_reports_are_club_scoped_read_only_and_match_streamlit_calculations(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOOLS", "1")
    supabase = FakeSupabase()
    supabase.storage["players"] = [
        {"id": 1, "club_id": "club", "name": "Alice", "rating": 1600, "starting_rating": 1400, "wins": 3, "losses": 1, "matches_played": 4, "active": True, "inactive_at": None},
        {"id": 2, "club_id": "club", "name": "Bob", "rating": 1400, "starting_rating": 1400, "wins": 1, "losses": 2, "matches_played": 3, "active": False, "inactive_at": "2026-01-01T00:00:00Z"},
        {"id": 3, "club_id": "club", "name": "Legacy (MERGED into Alice)", "rating": 2000, "starting_rating": 1200, "wins": 9, "losses": 0, "matches_played": 9, "active": True, "inactive_at": None},
        {"id": 4, "club_id": "club", "name": "Zero", "rating": 1200, "starting_rating": 1200, "wins": 0, "losses": 0, "matches_played": 0, "active": True, "inactive_at": None},
        {"id": 5, "club_id": "other", "name": "Outside", "rating": 2400, "starting_rating": 1200, "wins": 10, "losses": 0, "matches_played": 10, "active": True, "inactive_at": None},
    ]
    supabase.storage["leagues_metadata"] = [
        {"club_id": "club", "league_name": "Open"},
        {"club_id": "other", "league_name": "Secret"},
    ]
    supabase.storage["league_ratings"] = [
        {"club_id": "club", "player_id": 1, "league_name": "Open", "rating": 1680, "starting_rating": 1600, "wins": 4, "losses": 1, "matches_played": 5, "is_active": True},
        {"club_id": "club", "player_id": 2, "league_name": "Open", "rating": 1400, "starting_rating": 1440, "wins": 1, "losses": 3, "matches_played": 4, "is_active": False},
        {"club_id": "other", "player_id": 5, "league_name": "Open", "rating": 2400, "starting_rating": 1200, "wins": 10, "losses": 0, "matches_played": 10, "is_active": True},
    ]
    before = deepcopy(supabase.storage)

    overall = build_admin_rating_report(supabase, club_id="club")
    league = build_admin_rating_report(supabase, club_id="club", league_name="Open")

    assert overall["read_only"] is True
    assert overall["available_scopes"] == ["OVERALL", "Open"]
    assert [row["name"] for row in overall["rows"]] == ["Alice", "Zero"]
    assert overall["rows"][0] == {
        "player_id": 1,
        "name": "Alice",
        "jupr": 4.0,
        "wins": 3,
        "losses": 1,
        "matches_played": 4,
        "win_percent": 75.0,
        "gain": 0.5,
    }
    assert overall["rows"][1]["win_percent"] == 0.0
    assert [row["name"] for row in league["rows"]] == ["Alice", "Bob"]
    assert league["rows"][0]["jupr"] == 4.2
    assert league["rows"][0]["gain"] == 0.2
    assert league["rows"][1]["gain"] == -0.1
    try:
        build_admin_rating_report(supabase, club_id="club", league_name="Secret")
    except ValueError as exc:
        assert "available" in str(exc).lower()
    else:
        raise AssertionError("expected cross-club report scope rejection")
    assert supabase.storage == before


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
        {"id": "team-3", "tournament_id": "tour-1", "player1_id": 3, "player2_id": 99},
    ]
    supabase.storage["tournament_games"] = [
        {"id": "ready", "tournament_id": "tour-1", "team_a_id": "team-1", "team_b_id": "team-2", "score_a": 11, "score_b": 7, "finalized_at": "2026-07-01T00:00:00Z"},
        {"id": "empty", "tournament_id": "tour-1", "team_a_id": "team-1", "team_b_id": "team-2", "score_a": 0, "score_b": 0, "finalized_at": "2026-07-01T00:00:00Z"},
        {"id": "tied", "tournament_id": "tour-1", "team_a_id": "team-1", "team_b_id": "team-2", "score_a": 10, "score_b": 10, "finalized_at": "2026-07-01T00:00:00Z"},
        {"id": "incomplete", "tournament_id": "tour-1", "team_a_id": "team-1", "team_b_id": "missing", "score_a": 11, "score_b": 8, "finalized_at": "2026-07-01T00:00:00Z"},
        {"id": "missing-player", "tournament_id": "tour-1", "team_a_id": "team-1", "team_b_id": "team-3", "score_a": 11, "score_b": 8, "finalized_at": "2026-07-01T00:00:00Z"},
        {"id": "published", "tournament_id": "tour-1", "team_a_id": "team-1", "team_b_id": "team-2", "score_a": 11, "score_b": 9, "finalized_at": "2026-07-01T00:00:00Z"},
        {"id": "not-final", "tournament_id": "tour-1", "team_a_id": "team-1", "team_b_id": "team-2", "score_a": 11, "score_b": 6, "finalized_at": None},
        {"id": "other-club", "tournament_id": "tour-other", "team_a_id": "x", "team_b_id": "y", "score_a": 11, "score_b": 6, "finalized_at": "2026-07-01T00:00:00Z"},
    ]
    supabase.storage["matches"] = [
        {"id": 99, "club_id": "club", "tournament_game_id": "published"},
        {"id": 100, "club_id": "other", "tournament_game_id": "ready"},
    ]
    supabase.storage["players"] = [
        {"id": 1, "club_id": "club", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
        {"id": 2, "club_id": "club", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
        {"id": 3, "club_id": "club", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
        {"id": 4, "club_id": "club", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
    ]
    before = deepcopy(supabase.storage)

    result = build_admin_tournament_match_backfill_preview(supabase, club_id="club")

    assert result["read_only"] is True
    assert result["summary"]["tournament_count"] == 1
    assert result["summary"]["finalized_game_count"] == 6
    assert result["summary"]["already_published_count"] == 1
    assert result["summary"]["missing_match_count"] == 5
    assert result["summary"]["ready_count"] == 1
    assert result["summary"]["blocked_count"] == 4
    statuses = {row["game_id"]: row["status"] for row in result["candidates"]}
    assert statuses == {
        "empty": "empty_score",
        "incomplete": "incomplete_team",
        "missing-player": "missing_player",
        "ready": "ready",
        "tied": "tied_score",
    }
    ready = next(row for row in result["candidates"] if row["game_id"] == "ready")
    assert ready["match_payload"]["tournament_game_id"] == "ready"
    assert ready["match_payload"]["t1_p2"] == 2
    assert len(result["preview_fingerprint"]) == 64
    assert result["confirmation_text"] == "BACKFILL TOURNAMENT MATCHES"
    assert supabase.storage == before


def test_tournament_match_backfill_apply_requires_current_selected_preview(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOOLS", "1")
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    supabase = FakeSupabase()
    supabase.storage["tournaments"] = [
        {"id": "tour-1", "club_id": "club", "name": "Summer Cup"},
    ]
    supabase.storage["tournament_teams"] = [
        {"id": "team-1", "tournament_id": "tour-1", "player1_id": 1, "player2_id": 2},
        {"id": "team-2", "tournament_id": "tour-1", "player1_id": 3, "player2_id": 4},
    ]
    supabase.storage["tournament_games"] = [
        {"id": "ready", "tournament_id": "tour-1", "team_a_id": "team-1", "team_b_id": "team-2", "score_a": 11, "score_b": 7, "finalized_at": "2026-07-01T00:00:00Z"},
    ]
    supabase.storage["players"] = [
        {"id": value, "club_id": "club", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0}
        for value in (1, 2, 3, 4)
    ]
    preview = build_admin_tournament_match_backfill_preview(supabase, club_id="club")
    captured: dict[str, object] = {}

    def fake_process(match_payloads, **kwargs):
        captured["payloads"] = deepcopy(match_payloads)
        captured["club_id"] = kwargs.get("club_id")
        for payload in match_payloads:
            supabase.storage["matches"].append(
                {
                    "id": f"match-{len(supabase.storage['matches']) + 1}",
                    "club_id": kwargs.get("club_id"),
                    "tournament_game_id": payload.get("tournament_game_id"),
                }
            )
        return {"inserted": len(match_payloads), "badge_summary": {"mode": "test"}}

    monkeypatch.setattr("jupr_app.services.admin_tools_service.process_matches", fake_process)
    try:
        apply_admin_tournament_match_backfill(
            supabase,
            club_id="club",
            game_ids=["ready"],
            preview_fingerprint=preview["preview_fingerprint"],
            preview_limit=1000,
            confirmation_text="BACKFILL TOURNAMENT MATCHES",
            actor_email="owner@example.com",
            actor_role="super_admin",
        )
    except ValueError as exc:
        assert "stale" in str(exc).lower()
    else:
        raise AssertionError("expected preview-limit fingerprint rejection")
    assert supabase.storage["matches"] == []
    assert supabase.storage["admin_activity_log"] == []

    result = apply_admin_tournament_match_backfill(
        supabase,
        club_id="club",
        game_ids=["ready"],
        preview_fingerprint=preview["preview_fingerprint"],
        confirmation_text="BACKFILL TOURNAMENT MATCHES",
        actor_email="owner@example.com",
        actor_role="super_admin",
    )

    assert result["inserted_count"] == 1
    assert result["selected_game_ids"] == ["ready"]
    assert captured["club_id"] == "club"
    assert captured["payloads"][0]["t1_p2"] == 2
    assert [row["action_type"] for row in supabase.storage["admin_activity_log"]] == [
        "tournament_match_backfill_apply_started",
        "tournament_match_backfill_applied",
    ]


def test_tournament_match_backfill_apply_rejects_stale_preview_without_writes(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOOLS", "1")
    supabase = FakeSupabase()
    before = deepcopy(supabase.storage)
    called = {"process": False}
    monkeypatch.setattr(
        "jupr_app.services.admin_tools_service.process_matches",
        lambda *_args, **_kwargs: called.update(process=True),
    )

    try:
        apply_admin_tournament_match_backfill(
            supabase,
            club_id="club",
            game_ids=["missing"],
            preview_fingerprint="stale",
            confirmation_text="BACKFILL TOURNAMENT MATCHES",
            actor_email="owner@example.com",
            actor_role="super_admin",
        )
    except ValueError as exc:
        assert "stale" in str(exc).lower()
    else:
        raise AssertionError("expected stale preview rejection")
    assert called["process"] is False
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
