from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from jupr_app.services.admin_tournament_guarded_operation import StaleTournamentAdminStateError
from jupr_app.services.admin_tournament_pool_service import save_admin_tournament_draw_pools
from tests.test_admin_match_log_service import FakeSupabase

VERSION = "2026-10-31T10:00:00Z"


def fixture_tables():
    return {
        "tournaments": [{"id": "tournament", "club_id": "club", "status": "PUBLISHED"}],
        "tournament_event_draws": [{"id": "draw", "tournament_id": "tournament", "name": "Mixed 3.0", "updated_at": VERSION, "status": "draft"}],
        "tournament_teams": [{"id": f"team-{n}", "team_number": n, "tournament_id": "tournament", "draw_id": "draw", "updated_at": VERSION} for n in range(1, 7)],
        "tournament_games": [], "tournament_podium": [], "tournament_day_live_draws": [], "tournament_day_live_queue": [], "admin_activity_log": [],
    }


def pool_config():
    return {"version": 1, "pools": [
        {"id": "A", "name": "Pool A", "team_ids": ["team-1", "team-2", "team-3"], "advance_count": 2},
        {"id": "B", "name": "Pool B", "team_ids": ["team-4", "team-5", "team-6"], "advance_count": 2},
    ]}


class PoolSupabase(FakeSupabase):
    def __init__(self, tables):
        super().__init__(tables)
        self.tables = tables
        self.rpc_calls = []
        self.rpc_error = None

    def rpc(self, name, params):
        self.rpc_calls.append((name, params))
        def execute():
            if self.rpc_error:
                raise RuntimeError(self.rpc_error)
            draw = self.tables["tournament_event_draws"][0]
            draw.update(pool_config=deepcopy(params["p_pool_config"]), updated_at="2026-10-31T10:01:00Z")
            return SimpleNamespace(data={"ok": True, "draw": deepcopy(draw)})
        return SimpleNamespace(execute=execute)


def save(db, **overrides):
    kwargs = dict(club_id="club", tournament_id="tournament", draw_id="draw", pool_config=pool_config(),
                  expected_draw_updated_at=VERSION,
                  expected_team_versions=[{"id": f"team-{n}", "updated_at": VERSION} for n in range(1, 7)],
                  actor_email="admin@example.com", actor_role="club_owner", confirmation_text="SAVE POOLS")
    kwargs.update(overrides)
    return save_admin_tournament_draw_pools(db, **kwargs)


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")


def test_pool_save_uses_exact_snapshot_rpc_and_records_audit():
    tables = fixture_tables()
    db = PoolSupabase(tables)
    result = save(db)
    assert result["draw"]["pool_config"] == pool_config()
    assert result["draw"]["updated_at"] != VERSION
    name, params = db.rpc_calls[0]
    assert name == "admin_save_tournament_draw_pools_cas"
    assert params["p_club_id"] == "club"
    assert len(params["p_expected_teams"]) == 6
    assert tables["admin_activity_log"][0]["after_json"]["draw"]["pool_config"] == pool_config()


def test_clear_pool_split_is_explicit_and_audited():
    tables = fixture_tables()
    tables["tournament_event_draws"][0]["pool_config"] = pool_config()
    db = PoolSupabase(tables)
    result = save(db, pool_config=None)
    assert result["pool_config"] is None
    assert tables["admin_activity_log"][0]["before_json"]["draw"]["pool_config"] == pool_config()


def test_preview_never_writes():
    db = PoolSupabase(fixture_tables())
    result = save(db, dry_run=True)
    assert result["write_count"] == 0
    assert result["pool_config"] == pool_config()
    assert not db.rpc_calls
    assert not db.tables["admin_activity_log"]


@pytest.mark.parametrize("overrides", [
    {"club_id": "other-club"},
    {"confirmation_text": ""},
    {"expected_draw_updated_at": "2026-10-30T00:00:00Z"},
    {"expected_team_versions": []},
    {"expected_team_versions": [{"id": "team-1", "updated_at": VERSION}]},
])
def test_scope_confirmation_or_stale_snapshot_rejected_before_rpc(overrides):
    db = PoolSupabase(fixture_tables())
    with pytest.raises((ValueError, StaleTournamentAdminStateError)):
        save(db, **overrides)
    assert not db.rpc_calls


@pytest.mark.parametrize("table", ["tournament_games", "tournament_podium", "tournament_day_live_draws", "tournament_day_live_queue"])
def test_existing_participation_locks_assignments_and_advance_counts(table):
    tables = fixture_tables()
    tables[table] = [{"id": "existing", "tournament_id": "tournament", "draw_id": "draw"}]
    db = PoolSupabase(tables)
    with pytest.raises(ValueError, match="before games or live play"):
        save(db)
    assert not db.rpc_calls


def test_missing_or_duplicate_team_cannot_be_saved():
    db = PoolSupabase(fixture_tables())
    config = pool_config()
    config["pools"][1]["team_ids"][-1] = "team-1"
    with pytest.raises(ValueError):
        save(db, pool_config=config)
    assert not db.rpc_calls


def test_rpc_stale_race_cannot_claim_save():
    db = PoolSupabase(fixture_tables())
    db.rpc_error = "JUPR_TOURNAMENT_TEAM_SNAPSHOT_STALE"
    with pytest.raises(StaleTournamentAdminStateError):
        save(db)
    assert not db.tables["admin_activity_log"]


def test_required_dependency_read_fails_closed():
    db = PoolSupabase(fixture_tables())
    original = db.table
    def table(name):
        if name == "tournament_day_live_draws":
            raise RuntimeError("unavailable")
        return original(name)
    db.table = table
    with pytest.raises(RuntimeError, match="Could not verify"):
        save(db)
    assert not db.rpc_calls


@pytest.mark.parametrize("kind", ["editor", "registration", "bulk"])
def test_existing_pool_split_blocks_roster_replacement_before_writes(kind):
    from jupr_app.services.admin_tournament_team_service import replace_admin_tournament_draw_teams
    from jupr_app.services.admin_tournament_registration_import_service import import_admin_tournament_registrations_to_draw
    from jupr_app.services.admin_tournament_bulk_team_import_service import import_admin_tournament_bulk_teams
    tables = fixture_tables()
    tables["tournament_event_draws"][0]["pool_config"] = pool_config()
    db = PoolSupabase(tables)
    before = deepcopy(tables["tournament_teams"])
    kwargs = dict(club_id="club", tournament_id="tournament", draw_id="draw", actor_email="admin@example.com", actor_role="club_owner")
    with pytest.raises(ValueError, match="Remove the pool split"):
        if kind == "editor":
            replace_admin_tournament_draw_teams(db, **kwargs, teams=[], confirmation_text="SAVE TEAMS")
        elif kind == "registration":
            import_admin_tournament_registrations_to_draw(db, **kwargs, confirmation_text="IMPORT REGISTRATIONS")
        else:
            import_admin_tournament_bulk_teams(db, **kwargs, raw_text="", confirmation_text="IMPORT TEAMS")
    assert tables["tournament_teams"] == before
    assert not db.rpc_calls


def test_ops_state_fingerprint_changes_when_pool_assignment_changes():
    from jupr_app.services.admin_tournament_ops_service import get_admin_tournament_ops_state_fingerprint
    tables = fixture_tables()
    db = PoolSupabase(tables)
    before = get_admin_tournament_ops_state_fingerprint(db, club_id="club", tournament_id="tournament")
    tables["tournament_event_draws"][0]["pool_config"] = pool_config()
    after = get_admin_tournament_ops_state_fingerprint(db, club_id="club", tournament_id="tournament")
    assert before != after


def test_required_audit_failure_is_reported_without_replaying_pool_write(monkeypatch):
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    monkeypatch.setattr("jupr_app.services.admin_tournament_pool_service.write_admin_activity_log", lambda *_args: SimpleNamespace(ok=False, warning="audit unavailable"))
    db = PoolSupabase(fixture_tables())
    with pytest.raises(RuntimeError, match="audit record"):
        save(db)
    assert len(db.rpc_calls) == 1
    assert db.tables["tournament_event_draws"][0]["pool_config"] == pool_config()
