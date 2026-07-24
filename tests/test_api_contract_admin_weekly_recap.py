from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from jupr_app.services import admin_weekly_recap_service as recap_service
from jupr_app.services.admin_weekly_recap_service import StaleWeeklyRecapStateError, apply_weekly_recap_edits
from services.api.main import app


def weekly_recap_tables() -> dict[str, list[dict]]:
    return {
        "weekly_recaps": [
            {
                "id": "recap-1",
                "club_id": "club",
                "week_start": "2026-07-06",
                "week_end": "2026-07-12",
                "status": "draft",
                "row_version": 1,
                "generated_json": {"numbers": {"matches": 4}, "spotlight": []},
                "edits_json": {},
                "final_json": {"numbers": {"matches": 4}, "spotlight": []},
            }
        ],
        "admin_activity_log": [],
    }


def _install_env(monkeypatch, supabase) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "local-service")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr(
        "services.api.admin_weekly_recap_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_weekly_recap_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


def test_admin_weekly_recap_status_disabled_contract(monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP", raising=False)
    response = TestClient(app).get("/admin/clubs/club/weekly-recap/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is False
    assert payload["list_endpoint"] is None


def test_weekly_recap_public_status_has_no_counts_or_secret_readiness(monkeypatch) -> None:
    supabase = FakeSupabase(weekly_recap_tables())
    _install_env(monkeypatch, supabase)
    response = TestClient(app).get("/admin/clubs/club/weekly-recap/status")
    assert response.status_code == 200
    payload = response.json()
    assert "recap_count" not in payload
    assert "published_count" not in payload
    assert "service_role_configured" not in payload
    assert payload["mutations_enabled"] is True


def test_admin_weekly_recap_list_contract(monkeypatch):
    supabase = FakeSupabase(weekly_recap_tables())
    _install_env(monkeypatch, supabase)

    response = TestClient(app).get(
        "/admin/clubs/club/weekly-recap/recaps",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["count"] == 1
    assert payload["recaps"][0]["week_start"] == "2026-07-06"


def test_staging_weekly_recap_reads_stay_open_while_mutations_are_double_guarded(
    monkeypatch,
) -> None:
    supabase = FakeSupabase(weekly_recap_tables())
    _install_env(monkeypatch, supabase)
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "none")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_COMMUNICATIONS_MUTATIONS", "0")
    client = TestClient(app)

    status = client.get("/admin/clubs/club/weekly-recap/status")
    recaps = client.get(
        "/admin/clubs/club/weekly-recap/recaps",
        headers={"Authorization": "Bearer local"},
    )
    denied_by_wave = client.post(
        "/admin/clubs/club/weekly-recap/generate",
        headers={"Authorization": "Bearer local"},
        json={
            "week_start": "2026-07-06",
            "week_end": "2026-07-12",
            "confirmation_text": "GENERATE",
        },
    )

    assert status.status_code == 200
    assert status.json()["enabled"] is True
    assert status.json()["mutations_enabled"] is False
    assert recaps.status_code == 200
    assert recaps.json()["count"] == 1
    assert denied_by_wave.status_code == 403

    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "communications")
    denied_by_service = client.post(
        "/admin/clubs/club/weekly-recap/generate",
        headers={"Authorization": "Bearer local"},
        json={
            "week_start": "2026-07-06",
            "week_end": "2026-07-12",
            "confirmation_text": "GENERATE",
        },
    )
    assert denied_by_service.status_code == 403
    assert "Communications mutations are disabled" in denied_by_service.json()["detail"]

    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_COMMUNICATIONS_MUTATIONS", "1")
    confirmation_required = client.post(
        "/admin/clubs/club/weekly-recap/generate",
        headers={"Authorization": "Bearer local"},
        json={
            "week_start": "2026-07-06",
            "week_end": "2026-07-12",
            "confirmation_text": "GENERATE",
        },
    )
    assert confirmation_required.status_code == 400
    assert "GENERATE RECAP" in confirmation_required.json()["detail"]


def test_admin_weekly_recap_generate_requires_confirmation(monkeypatch):
    supabase = FakeSupabase(weekly_recap_tables())
    _install_env(monkeypatch, supabase)

    response = TestClient(app).post(
        "/admin/clubs/club/weekly-recap/generate",
        headers={"Authorization": "Bearer local"},
        json={"week_start": "2026-07-06", "week_end": "2026-07-12", "confirmation_text": "GENERATE"},
    )

    assert response.status_code == 400
    assert "GENERATE RECAP" in response.json()["detail"]


def test_admin_weekly_recap_save_requires_confirmation(monkeypatch):
    supabase = FakeSupabase(weekly_recap_tables())
    _install_env(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/weekly-recap/recaps/2026-07-06",
        headers={"Authorization": "Bearer local"},
        json={"edits_json": {}, "confirmation_text": "SAVE"},
    )

    assert response.status_code == 400
    assert "SAVE RECAP" in response.json()["detail"]


def test_apply_weekly_recap_edits_uses_spotlight_candidates():
    generated = {"spotlight": [{"key": "TOP_PERFORMER_WEEK", "order": 1, "include": True}], "looking_ahead": []}
    edits = {
        "looking_ahead": ["League finals", "Tournament signup"],
        "spotlight_overrides": {
            "TOP_PERFORMER_WEEK": {
                "players": ["cand-1"],
                "description": "Best range performance.",
                "order": 1,
                "include": True,
            }
        },
    }
    candidates = {
        "TOP_PERFORMER_WEEK": [
            {"candidate_id": "cand-1", "label": "Top Performer", "display": "Alex +0.120", "player_ids": [1]}
        ]
    }

    final = apply_weekly_recap_edits(generated, edits, candidates)

    assert final["looking_ahead"] == ["League finals", "Tournament signup"]
    assert final["spotlight"][0]["players"] == ["Alex +0.120"]
    assert final["spotlight"][0]["description"] == "Best range performance."


def test_explicit_blank_spotlight_omits_instead_of_selecting_fallback() -> None:
    generated = {"spotlight": [{"key": "TOP_PERFORMER_WEEK", "candidate_ids": ["old"], "include": True}]}
    candidates = {"TOP_PERFORMER_WEEK": [{"candidate_id": "new", "label": "Top", "display": "New player"}]}
    final = apply_weekly_recap_edits(
        generated,
        {"spotlight_overrides": {"TOP_PERFORMER_WEEK": {"players": [], "include": True}}},
        candidates,
    )
    assert final["spotlight"] == []


def test_explicit_stale_spotlight_candidate_requires_reload() -> None:
    generated = {"spotlight": [{"key": "TOP_PERFORMER_WEEK", "candidate_ids": ["old"], "include": True}]}
    with pytest.raises(StaleWeeklyRecapStateError, match="candidates changed"):
        apply_weekly_recap_edits(
            generated,
            {"spotlight_overrides": {"TOP_PERFORMER_WEEK": {"players": ["missing"], "include": True}}},
            {"TOP_PERFORMER_WEEK": []},
        )


def test_unpublish_is_status_only_and_preserves_published_content(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP", "1")
    before = {
        "id": "recap-1",
        "club_id": "club",
        "week_start": "2026-07-06",
        "week_end": "2026-07-12",
        "status": "published",
        "row_version": 4,
        "generated_json": {"spotlight": [{"key": "saved"}]},
        "edits_json": {"looking_ahead": ["saved"]},
        "final_json": {"numbers": {"matches": 9}},
    }
    captured: dict[str, object] = {}
    monkeypatch.setattr(recap_service, "_fetch_recap_row", lambda *_args, **_kwargs: dict(before))
    monkeypatch.setattr(recap_service, "_candidates_for_row", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(recap_service, "_required_audit_intent", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(recap_service, "_audit", lambda *_args, **_kwargs: [])

    def capture_upsert(_supabase, **kwargs):
        captured.update(kwargs["payload"])
        return {**before, **kwargs["payload"], "row_version": 5}

    monkeypatch.setattr(recap_service, "_upsert_recap_row", capture_upsert)
    result = recap_service.publish_admin_weekly_recap(
        object(),
        club_id="club",
        week_start="2026-07-06",
        action="unpublish",
        edits_json={"looking_ahead": ["browser drift"]},
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="UNPUBLISH RECAP",
        expected_row_version=4,
    )
    assert result["recap"]["status"] == "draft"
    assert captured["generated_json"] == before["generated_json"]
    assert captured["edits_json"] == before["edits_json"]
    assert captured["final_json"] == before["final_json"]


def test_admin_weekly_recap_stale_save_is_conflict(monkeypatch):
    supabase = FakeSupabase(weekly_recap_tables())
    _install_env(monkeypatch, supabase)

    def fake_save(*_args, **_kwargs):
        raise StaleWeeklyRecapStateError("Weekly recap changed. Reload before saving.")

    monkeypatch.setattr("services.api.admin_weekly_recap_routes.save_admin_weekly_recap", fake_save)
    response = TestClient(app).patch(
        "/admin/clubs/club/weekly-recap/recaps/2026-07-06",
        headers={"Authorization": "Bearer local"},
        json={
            "edits_json": {},
            "expected_row_version": 1,
            "confirmation_text": "SAVE RECAP",
        },
    )

    assert response.status_code == 409
    assert "Reload" in response.json()["detail"]
