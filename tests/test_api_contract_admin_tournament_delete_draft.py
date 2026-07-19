from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase
from tests.test_api_contract_admin_tournament import _install_auth

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def draft_tournament_tables():
    return {
        "tournaments": [
            {"club_id": "club", "id": "draft_1", "name": "Empty Draft", "status": "DRAFT", "created_at": "2026-03-01T00:00:00Z", "updated_at": "2026-03-02T00:00:00Z"}
        ],
        "tournament_registration_settings": [{"id": "regset_draft", "tournament_id": "draft_1", "registration_slug": "empty-draft"}],
        "tournament_registration_days": [{"id": "day_draft", "tournament_id": "draft_1", "label": "Friday", "sort_order": 1}],
        "tournament_event_options": [{"id": "event_draft", "tournament_id": "draft_1", "registration_day_id": "day_draft", "division_name": "3.5"}],
        "tournament_registrations": [],
        "tournament_registration_selections": [],
        "tournament_event_draws": [],
        "tournament_teams": [],
        "tournament_games": [],
        "tournament_podium": [],
        "admin_activity_log": [],
    }


class AtomicDeleteSupabase(FakeSupabase):
    def __init__(self, tables):
        super().__init__(tables)
        self.atomic_delete_calls = 0

    def rpc(self, name, payload):
        assert name == "admin_delete_empty_tournament_draft_cas"
        supabase = self

        class Query:
            def execute(self):
                supabase.atomic_delete_calls += 1
                tournament = next(
                    (
                        row
                        for row in supabase.tables["tournaments"]
                        if row["id"] == payload["p_tournament_id"]
                        and row["club_id"] == payload["p_club_id"]
                        and row["updated_at"] == payload["p_expected_updated_at"]
                    ),
                    None,
                )
                if tournament is None:
                    raise RuntimeError("JUPR_TOURNAMENT_STALE")
                usage = {
                    "registrations": len(supabase.tables["tournament_registrations"]),
                    "registration_selections": len(supabase.tables["tournament_registration_selections"]),
                    "event_draws": len(supabase.tables["tournament_event_draws"]),
                    "teams": len(supabase.tables["tournament_teams"]),
                    "games": len(supabase.tables["tournament_games"]),
                    "podium": len(supabase.tables["tournament_podium"]),
                }
                if any(usage.values()):
                    raise RuntimeError("JUPR_TOURNAMENT_NOT_EMPTY")
                for table_name in (
                    "tournament_registration_settings",
                    "tournament_event_options",
                    "tournament_registration_days",
                ):
                    supabase.tables[table_name] = [
                        row for row in supabase.tables[table_name] if row.get("tournament_id") != payload["p_tournament_id"]
                    ]
                supabase.tables["tournaments"].remove(tournament)
                return SimpleNamespace(data={"ok": True, "usage_summary": usage})

        return Query()


def test_admin_tournament_delete_empty_draft_contract(monkeypatch):
    tables = draft_tournament_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/draft_1/delete-draft",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "DELETE DRAFT"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_draft_deleted"
    assert payload["tournament_id"] == "draft_1"
    assert tables["tournaments"] == []
    assert tables["tournament_registration_settings"] == []
    assert tables["tournament_event_options"] == []
    assert tables["tournament_registration_days"] == []
    assert tables["admin_activity_log"][0]["action_type"] == "delete_draft_tournament_admin"
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True


def test_staging_delete_uses_one_atomic_versioned_rpc_and_replays(monkeypatch):
    tables = draft_tournament_tables()
    tables["tournament_admin_operations"] = []
    supabase = AtomicDeleteSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_MUTATIONS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)
    request = {
        "expected_updated_at": "2026-03-02T00:00:00Z",
        "confirmation_text": "DELETE DRAFT",
    }
    client = TestClient(app)

    first = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/draft_1/delete-draft",
        headers={"Authorization": "Bearer local"},
        json=request,
    )
    replay = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/draft_1/delete-draft",
        headers={"Authorization": "Bearer local"},
        json=request,
    )

    assert first.status_code == 200, first.text
    assert replay.status_code == 200, replay.text
    assert replay.json()["idempotent_replay"] is True
    assert supabase.atomic_delete_calls == 1
    assert tables["tournaments"] == []
    assert tables["tournament_registration_settings"] == []
    assert tables["tournament_event_options"] == []
    assert tables["tournament_registration_days"] == []
