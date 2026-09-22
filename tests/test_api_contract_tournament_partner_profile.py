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


def test_partner_lookup_supports_partial_names_and_stays_active_and_club_scoped(integrity_client):
    api, storage = integrity_client
    base = {"club_id": "club-1", "name": "Same Name", "rating": 1600, "active": True}
    storage["players"] = [
        {**base, "id": 30, "club_id": "other-club"},
        {**base, "id": 31, "active": False},
        *[{**base, "id": value} for value in range(40, 45)],
    ]
    response = api.post(ENDPOINT, json={"registration_slug": "tres-open", "name": "Same Name"})
    assert response.status_code == 200
    assert [row["id"] for row in response.json()["profile_candidates"]] == ["40", "41", "42"]
    response = api.post(ENDPOINT, json={"registration_slug": "tres-open", "name": "Same"})
    assert response.status_code == 200
    assert response.json()["profile_match_kind"] == "name_partial"
    assert [row["id"] for row in response.json()["profile_candidates"]] == ["40", "41", "42", "43", "44"]


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


def test_registrant_preflight_also_suggests_partial_names(integrity_client):
    api, storage = integrity_client
    storage["players"][0]["name"] = "Valeria Verdugo"
    response = api.post("/clubs/tres-palapas/tournament-registration/profile-resolution", json={
        "registration_slug": "tres-open", "first_name": "Vale", "last_name": "Verdugo",
        "email": "new@example.test", "age": 40, "gender": "Women",
    })
    assert response.status_code == 200
    assert response.json()["profile_match_kind"] == "name_partial"
    assert response.json()["profile_candidates"][0]["display_name"] == "Valeria Verdugo"


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
