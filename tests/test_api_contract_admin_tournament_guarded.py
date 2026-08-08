from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_api_contract_admin_tournament import FakeSupabase, tournament_tables

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def _install(monkeypatch, supabase) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "tournament-registration")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_REGISTRATION_MUTATIONS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


def test_registration_write_has_versioned_audit_lifecycle_and_idempotent_replay(monkeypatch) -> None:
    tables = tournament_tables()
    tables["tournament_registrations"][0]["updated_at"] = "2026-03-03T00:00:00Z"
    tables["tournament_admin_operations"] = []
    supabase = FakeSupabase(tables)
    _install(monkeypatch, supabase)
    request = {
        "payment_status": "refunded",
        "notes": "Operator verified refund.",
        "expected_updated_at": "2026-03-03T00:00:00Z",
        "confirmation_text": "SAVE REGISTRATION",
        "source": "test_guarded_registration",
    }
    client = TestClient(app)

    first = client.patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/registration_1",
        headers={"Authorization": "Bearer local"},
        json=request,
    )
    replay = client.patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/registration_1",
        headers={"Authorization": "Bearer local"},
        json=request,
    )

    assert first.status_code == 200, first.text
    assert replay.status_code == 200, replay.text
    assert first.json()["idempotent_replay"] is False
    assert replay.json()["idempotent_replay"] is True
    assert replay.json()["operation_key"] == first.json()["operation_key"]
    assert len(tables["tournament_admin_operations"]) == 1
    assert tables["tournament_admin_operations"][0]["status"] == "completed"
    assert [row["action_type"] for row in tables["admin_activity_log"]] == [
        "tournament_registration_update_intent",
        "update_tournament_registration_admin",
        "tournament_registration_update_completion",
    ]


def test_completed_archive_replays_before_already_archived_preflight(monkeypatch) -> None:
    tables = tournament_tables()
    tables["tournament_admin_operations"] = []
    supabase = FakeSupabase(tables)
    _install(monkeypatch, supabase)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_MUTATIONS", "1")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "tournament-mutations")
    request = {
        "action": "archive",
        "expected_updated_at": "2026-03-02T00:00:00Z",
        "confirmation_text": "ARCHIVE",
    }
    client = TestClient(app)

    first = client.patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/status-action",
        headers={"Authorization": "Bearer local"},
        json=request,
    )
    replay = client.patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/status-action",
        headers={"Authorization": "Bearer local"},
        json=request,
    )

    assert first.status_code == 200, first.text
    assert replay.status_code == 200, replay.text
    assert first.json()["idempotent_replay"] is False
    assert replay.json()["idempotent_replay"] is True
    assert len(tables["tournament_admin_operations"]) == 1
    assert tables["tournaments"][0]["status"] == "ARCHIVED"


def test_stale_registration_write_has_no_operation_audit_or_domain_write(monkeypatch) -> None:
    tables = tournament_tables()
    tables["tournament_registrations"][0]["updated_at"] = "2026-03-04T00:00:00Z"
    tables["tournament_admin_operations"] = []
    before = dict(tables["tournament_registrations"][0])
    supabase = FakeSupabase(tables)
    _install(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/registration_1",
        headers={"Authorization": "Bearer local"},
        json={
            "payment_status": "refunded",
            "expected_updated_at": "2026-03-03T00:00:00Z",
            "confirmation_text": "SAVE REGISTRATION",
        },
    )

    assert response.status_code == 409
    assert "changed after it was loaded" in response.json()["detail"]
    assert tables["tournament_registrations"][0] == before
    assert tables["tournament_admin_operations"] == []
    assert tables["admin_activity_log"] == []


def test_imported_draw_refusal_precedes_registration_mutation(monkeypatch) -> None:
    tables = tournament_tables()
    tables["tournament_teams"][0].update(
        {
            "registration_day_id": "day_1",
            "event_option_id": "event_1",
            "source": "REGISTRATION",
        }
    )
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_ENV", "test")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )

    response = TestClient(app).patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/registration_1",
        headers={"Authorization": "Bearer local"},
        json={
            "registration_status": "cancelled",
            "expected_updated_at": "2026-03-03T00:00:00Z",
            "confirmation_text": "SAVE REGISTRATION",
        },
    )

    assert response.status_code == 400
    assert "Tournament Ops" in response.json()["detail"]
    assert tables["tournament_registrations"][0]["status"] == "confirmed"
    assert tables["admin_activity_log"] == []


def test_tournament_date_preflight_uses_unchanged_start_date_and_writes_nothing(monkeypatch) -> None:
    tables = tournament_tables()
    tables["tournament_admin_operations"] = []
    before = dict(tables["tournaments"][0])
    supabase = FakeSupabase(tables)
    _install(monkeypatch, supabase)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_MUTATIONS", "1")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "tournament-mutations")

    response = TestClient(app).patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1",
        headers={"Authorization": "Bearer local"},
        json={
            "end_date": "2026-04-01",
            "expected_updated_at": "2026-03-02T00:00:00Z",
            "confirmation_text": "SAVE TOURNAMENT",
        },
    )

    assert response.status_code == 400
    assert "end date cannot be before" in response.json()["detail"]
    assert tables["tournaments"][0] == before
    assert tables["tournament_admin_operations"] == []
    assert tables["admin_activity_log"] == []


def test_tournament_patch_preserves_explicit_null_to_clear_a_date(monkeypatch) -> None:
    tables = tournament_tables()
    tables["tournament_admin_operations"] = []
    supabase = FakeSupabase(tables)
    _install(monkeypatch, supabase)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_MUTATIONS", "1")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "tournament-mutations")

    response = TestClient(app).patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1",
        headers={"Authorization": "Bearer local"},
        json={
            "end_date": None,
            "expected_updated_at": "2026-03-02T00:00:00Z",
            "confirmation_text": "SAVE TOURNAMENT",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["tournament"]["end_date"] is None
    assert tables["tournaments"][0]["end_date"] is None
    assert tables["tournament_admin_operations"][0]["status"] == "completed"


def test_bulk_update_requires_every_selected_row_version_before_intent(monkeypatch) -> None:
    tables = tournament_tables()
    tables["tournament_registrations"][0]["updated_at"] = "2026-03-03T00:00:00Z"
    tables["tournament_admin_operations"] = []
    before = dict(tables["tournament_registrations"][0])
    supabase = FakeSupabase(tables)
    _install(monkeypatch, supabase)
    client = TestClient(app)
    detail = client.get(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1",
        headers={"Authorization": "Bearer local"},
    ).json()

    response = client.patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/bulk",
        headers={"Authorization": "Bearer local"},
        json={
            "registration_ids": ["registration_1"],
            "payment_status": "refunded",
            "expected_state_fingerprint": detail["state_fingerprint"],
            "expected_versions": {},
            "confirmation_text": "BULK UPDATE REGISTRATIONS",
        },
    )

    assert response.status_code == 409
    assert "changed after the bulk selection" in response.json()["detail"]
    assert tables["tournament_registrations"][0] == before
    assert tables["tournament_admin_operations"] == []
    assert tables["admin_activity_log"] == []
