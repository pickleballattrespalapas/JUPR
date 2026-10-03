from copy import deepcopy

import pytest

from tests.test_api_contract_tournament_registration import client, integrity_client

ENDPOINT = "/clubs/tres-palapas/tournament-registration/partner-profile-resolution"


def test_partner_lookup_before_demographics_and_with_existing_registration(integrity_client):
    api, storage = integrity_client
    storage["players"][0].update({"name": "Verified Alex", "email": "partner@example.com", "phone": "private", "gender": "Men", "age": 50})
    storage["tournament_registrations"].append({"id": "existing", "tournament_id": "t1", "email": "partner@example.com", "player_id": 10})
    before = deepcopy(storage)
    for extra, kind in [({}, "name_exact"), ({"email": "partner@example.com"}, "email_exact")]:
        response = api.post(ENDPOINT, json={"registration_slug": "tres-open", "name": "Verified Alex", **extra})
        assert response.status_code == 200
        data = response.json()
        assert data["profile_match_kind"] == kind
        assert data["profile_candidates"] == [{"id": "10", "display_name": "Verified Alex", "dupr_id": "", "doubles_skill": 4.0, "singles_skill": None}]
        assert data["profile_policy"]["public_submission_links_player"] is False
        assert "existing_registration" not in str(data)
    assert all(storage[key] == value for key, value in before.items())
    assert all(not value for key, value in storage.items() if key not in before)


def test_partner_lookup_includes_inactive_players_and_stays_bounded_and_club_scoped(integrity_client):
    api, storage = integrity_client
    base = {"club_id": "club-1", "name": "Same Name", "rating": 1600, "active": True}
    storage["players"] = [
        {**base, "id": 30, "club_id": "other-club"},
        {**base, "id": 31, "active": False},
        *[{**base, "id": value} for value in range(40, 45)],
    ]
    response = api.post(ENDPOINT, json={"registration_slug": "tres-open", "name": "Same Name"})
    assert response.status_code == 200
    assert [row["id"] for row in response.json()["profile_candidates"]] == ["31", "40", "41"]
    response = api.post(ENDPOINT, json={"registration_slug": "tres-open", "name": "Same"})
    assert response.status_code == 200
    assert response.json()["profile_match_kind"] == "name_partial"
    assert [row["id"] for row in response.json()["profile_candidates"]] == ["31", "40", "41", "42", "43", "44"]


@pytest.mark.parametrize("active", [True, False])
@pytest.mark.parametrize("name,email,kind", [
    ("Returning Player", None, "name_exact"),
    ("Returning", None, "name_partial"),
    ("Player", "returning@example.test", "email_exact"),
])
def test_inactive_partner_lookup_preserves_rating_privacy_and_activity(integrity_client, active, name, email, kind):
    api, storage = integrity_client
    storage["players"] = [{
        "id": 200, "club_id": "club-1", "name": "Returning Player",
        "active": active, "inactive_at": "2026-01-27T20:35:42+00:00",
        "last_game_at": "2025-12-24T04:18:00+00:00", "rating": 2046.3415,
        "email": "returning@example.test", "phone": "private", "age": 50,
        "gender": "Women", "dupr_id": "DUPR-200",
    }]
    before = deepcopy(storage)
    response = api.post(ENDPOINT, json={"registration_slug": "tres-open", "name": name, "email": email})
    assert response.status_code == 200
    data = response.json()
    assert data["profile_match_kind"] == kind
    assert data["profile_candidates"] == [{
        "id": "200", "display_name": "Returning Player", "dupr_id": "DUPR-200",
        "doubles_skill": pytest.approx(2046.3415 / 400), "singles_skill": None,
    }]
    assert data["profile_policy"]["public_submission_links_player"] is False
    assert all(storage[key] == value for key, value in before.items())
    assert all(not value for key, value in storage.items() if key not in before)


@pytest.mark.parametrize("active", [True, False])
@pytest.mark.parametrize("query", [
    {"name": "Retired Player"},
    {"name": "Retired"},
    {"name": "Retired Player", "email": "retired@example.test"},
])
def test_merged_partner_profiles_are_not_discovered_by_alias_or_email(integrity_client, active, query):
    api, storage = integrity_client
    storage["players"] = [{
        "id": 30, "club_id": "club-1", "rating": 1600, "active": active,
        "inactive_at": "2026-01-27T20:35:42+00:00",
        "name": "Retired Player (MERGED into Canonical Player #31)",
        "display_name": "Retired Player", "first_name": "Retired", "last_name": "Player",
        "email": "retired@example.test",
    }]
    response = api.post(ENDPOINT, json={"registration_slug": "tres-open", **query})
    assert response.status_code == 200
    assert response.json()["profile_candidates"] == []
    assert response.json()["profile_match_kind"] == "none"


@pytest.mark.parametrize("query", ["Va", "verdu", "Vale Ver", "VERDUGO val", "valeria verdugo"])
def test_partner_suggestions_match_first_last_and_accented_names(integrity_client, query):
    api, storage = integrity_client
    storage["players"] = [{"id": 80, "club_id": "club-1", "name": "Valéria Verdugo", "rating": 1600, "active": True, "email": "private@example.com", "phone": "private"}]
    response = api.post(ENDPOINT, json={"registration_slug": "tres-open", "name": query})
    assert response.status_code == 200
    assert response.json()["profile_candidates"] == [{"id": "80", "display_name": "Valéria Verdugo", "dupr_id": "", "doubles_skill": 4.0, "singles_skill": None}]


def test_partner_search_is_bounded_ranked_and_finds_profiles_beyond_first_page(integrity_client):
    api, storage = integrity_client
    base = {"club_id": "club-1", "rating": 1600, "active": True}
    storage["players"] = [{**base, "id": index, "name": f"Unrelated Player {index}"} for index in range(2100)]
    storage["players"].extend([
        {**base, "id": 8000, "name": "Joanna Taylor"},
        *[{**base, "id": 9000 + index, "name": f"Ann Player {index}"} for index in range(10)],
        {**base, "id": 9999, "name": "Canonical Name", "display_name": "Vale Verdugo"},
    ])
    response = api.post(ENDPOINT, json={"registration_slug": "tres-open", "name": "ann"})
    assert response.status_code == 200
    assert len(response.json()["profile_candidates"]) == 8
    assert all(row["display_name"].startswith("Ann ") for row in response.json()["profile_candidates"])
    response = api.post(ENDPOINT, json={"registration_slug": "tres-open", "name": "Vale Verdugo"})
    assert response.json()["profile_match_kind"] == "name_exact"
    assert response.json()["profile_candidates"][0]["id"] == "9999"
    for query in ["v", "%_", "Nobody"]:
        response = api.post(ENDPOINT, json={"registration_slug": "tres-open", "name": query})
        assert response.json()["profile_candidates"] == []


@pytest.mark.parametrize("active", [True, False])
def test_registrant_preflight_also_suggests_inactive_players(integrity_client, active):
    api, storage = integrity_client
    storage["players"][0].update({"name": "Valeria Verdugo", "active": active, "inactive_at": "2026-01-27"})
    response = api.post("/clubs/tres-palapas/tournament-registration/profile-resolution", json={
        "registration_slug": "tres-open", "first_name": "Vale", "last_name": "Verdugo",
        "email": "new@example.test", "age": 40, "gender": "Women",
    })
    assert response.status_code == 200
    assert response.json()["profile_match_kind"] == "name_partial"
    assert response.json()["profile_candidates"][0]["display_name"] == "Valeria Verdugo"
    assert response.json()["profile_candidates"][0]["doubles_skill"] == 4.0


@pytest.mark.parametrize("patch", [{"name": " "}, {"email": "Baumann"}, {"website": "bot.example"}])
def test_partner_lookup_rejects_invalid_input(client, patch):
    response = client.post(ENDPOINT, json={"registration_slug": "tres-open", "name": "Partner Name", **patch})
    assert response.status_code == 400


def test_partner_lookup_obeys_intake_guard(client, monkeypatch):
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "none")
    response = client.post(ENDPOINT, json={"registration_slug": "tres-open", "name": "Partner Name"})
    assert response.status_code == 403


def test_partner_lookup_requires_open_public_tournament(integrity_client):
    api, storage = integrity_client
    storage["tournament_registration_settings"][0]["registration_status"] = "closed"
    response = api.post(ENDPOINT, json={"registration_slug": "tres-open", "name": "Verified Alex"})
    assert response.status_code == 400
    assert "profile_candidates" not in response.json()
