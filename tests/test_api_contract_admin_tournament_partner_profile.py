from copy import deepcopy
from types import SimpleNamespace

import pytest

from tests.test_api_contract_tournament_registration import integrity_client

ENDPOINT = "/admin/clubs/club-1/tournaments/admin/tournaments/t1/partner-profile-resolution"


@pytest.fixture
def admin_partner_client(integrity_client, monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "tournament-registration")
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.test", user_id="admin-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )
    return integrity_client


def test_admin_lookup_works_after_registration_closes_without_writes(admin_partner_client, monkeypatch):
    api, storage = admin_partner_client
    monkeypatch.setenv("JUPR_ENABLE_STAGING_PUBLIC_INTAKE_WRITES", "0")
    storage["tournament_registration_settings"][0]["registration_status"] = "closed"
    storage["players"] = [{
        "id": 301, "club_id": "club-1", "name": "Kirsten Fixture", "rating": 1649.3827156,
        "email": "private@example.test", "phone": "private", "age": 50, "gender": "Women",
    }]
    before = deepcopy(storage)
    response = api.post(ENDPOINT, json={"name": "Kirsten"})
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store, private"
    assert response.json() == {
        "ok": True, "profile_match_kind": "name_partial",
        "profile_candidates": [{"id": "301", "display_name": "Kirsten Fixture", "dupr_id": "",
                                "doubles_skill": pytest.approx(4.123456789), "singles_skill": None}],
    }
    assert all(storage[key] == value for key, value in before.items())
    assert all(not value for key, value in storage.items() if key not in before)


def test_admin_lookup_is_club_scoped_and_does_not_hide_inactive_players(admin_partner_client):
    api, storage = admin_partner_client
    base = {"club_id": "club-1", "name": "Kirsten Fixture", "rating": 1600}
    storage["players"] = [
        {**base, "id": 1, "club_id": "other-club"},
        {**base, "id": 2, "active": False},
        {**base, "id": 3, "name": "Kirsten Fixture (MERGED into Player #2)", "display_name": "Kirsten Fixture"},
    ]
    response = api.post(ENDPOINT, json={"name": "Kirsten Fixture"})
    assert response.status_code == 200
    assert [row["id"] for row in response.json()["profile_candidates"]] == ["2"]
    assert response.json()["profile_match_kind"] == "name_exact"
    response = api.post(ENDPOINT.replace("club-1", "other-club"), json={"name": "Kirsten"})
    assert response.status_code == 400
    assert "profile_candidates" not in response.json()


@pytest.mark.parametrize("patch", [{"name": "  "}, {"email": "invalid-email"}])
def test_admin_lookup_rejects_invalid_identity(admin_partner_client, patch):
    api, _ = admin_partner_client
    response = api.post(ENDPOINT, json={"name": "Kirsten", **patch})
    assert response.status_code == 400


def test_admin_lookup_requires_tournament_permission(admin_partner_client, monkeypatch):
    api, _ = admin_partner_client
    monkeypatch.setattr("services.api.admin_tournament_routes.resolve_admin_role",
                        lambda **_kwargs: SimpleNamespace(role="viewer"))
    response = api.post(ENDPOINT, json={"name": "Kirsten"})
    assert response.status_code == 403
    assert "profile_candidates" not in response.json()


def test_admin_lookup_requires_authentication(integrity_client, monkeypatch):
    api, _ = integrity_client
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "tournament-registration")
    response = api.post(ENDPOINT, json={"name": "Kirsten"})
    assert response.status_code == 401


def test_admin_lookup_retains_staging_gate(admin_partner_client, monkeypatch):
    api, _ = admin_partner_client
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "none")
    response = api.post(ENDPOINT, json={"name": "Kirsten"})
    assert response.status_code == 403
