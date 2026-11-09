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
        candidate = data["profile_candidates"][0]
        saved = candidate.pop("registered_partner")
        assert saved["key"].startswith("tr_")
        assert saved == {"key": saved["key"], "has_email": True, "has_age": True, "has_gender": True}
        assert candidate == {"id": "10", "display_name": "Verified Alex", "dupr_id": "", "doubles_skill": 4.0, "singles_skill": None}
        assert data["profile_policy"]["public_submission_links_player"] is False
        assert "existing_registration" not in str(data)
    assert all(storage[key] == value for key, value in before.items())
    assert all(not value for key, value in storage.items() if key not in before)


@pytest.mark.parametrize("linked", [True, False])
def test_registered_partner_lookup_and_submission_reuse_saved_details(integrity_client, linked):
    api, storage = integrity_client
    storage["players"][0].update({"name": "Paola Masera", "rating": 1860.6572})
    partner = {"id": "paola-registration", "tournament_id": "t1", "display_name": "Paola Masera",
               "email": "private-partner@example.test", "age": 48, "gender": "Women", "doubles_skill": 4.65,
               "player_id": 10 if linked else None, "status": "confirmed"}
    storage["tournament_registrations"].append(partner)
    storage["tournament_event_options"][0].update({"partner_required": True, "event_type": "MIXED_DOUBLES"})
    before = deepcopy(partner)
    response = api.post(ENDPOINT, json={"registration_slug": "tres-open", "name": "Paola Masera"})
    assert response.status_code == 200
    candidates = response.json()["profile_candidates"]
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["registered_partner"]["has_age"] is True
    assert "private-partner" not in response.text and "paola-registration" not in response.text
    assert "age" not in candidate and "gender" not in candidate and "email" not in candidate
    result = api.post("/clubs/tres-palapas/tournament-registration", json={
        "registration_slug": "tres-open", "first_name": "Joe", "last_name": "Fixture",
        "email": "requester@example.test", "age": 38, "gender": "Men", "doubles_skill": 4.0,
        "terms_accepted": True, "selections": [{"event_option_id": "event1", "partner_mode": "HAS_PARTNER",
            "partner_name": "Paola Masera", "partner_registration_key": candidate["registered_partner"]["key"]}],
    })
    assert result.status_code == 200, result.text
    saved = storage["tournament_registration_selections"][-1]
    assert saved["partner_email"] == partner["email"]
    assert saved["partner_age"] == 48 and saved["partner_gender"] == "Women"
    assert saved["partner_skill"] == pytest.approx(1860.6572 / 400 if linked else 4.65)
    assert storage["tournament_registrations"][0] == before
    assert len(storage["tournament_registrations"]) == 2


def test_registration_only_partner_is_discovered_without_club_profile(integrity_client):
    api, storage = integrity_client
    storage["tournament_registrations"].append({"id": "unlinked", "tournament_id": "t1", "display_name": "New Partner",
        "email": "private@example.test", "age": 40, "gender": "Women", "doubles_skill": 3.5})
    response = api.post(ENDPOINT, json={"registration_slug": "tres-open", "name": "New Partner"})
    assert response.status_code == 200
    candidate = response.json()["profile_candidates"][0]
    assert candidate["id"].startswith("tr_")
    assert candidate["registered_partner"]["has_email"] is True
    assert response.json()["profile_match_kind"] == "name_exact"


@pytest.mark.parametrize("invalid", ["tampered", "other-tournament", "cancelled", "self"])
def test_registered_partner_reference_cannot_bypass_scope_or_status(integrity_client, invalid):
    from jupr_app.domain.tournament_public_references import build_public_tournament_reference

    api, storage = integrity_client
    partner = {"id": "target", "tournament_id": "t1", "display_name": "Saved Partner",
               "email": "partner@example.test", "age": 40, "gender": "Women", "doubles_skill": 4.0}
    if invalid == "other-tournament":
        partner["tournament_id"] = "t2"
    if invalid == "cancelled":
        partner["status"] = "cancelled"
    if invalid == "self":
        partner["email"] = "requester@example.test"
    storage["tournament_registrations"].append(partner)
    storage["tournament_event_options"][0].update({"partner_required": True, "event_type": "MIXED_DOUBLES"})
    key = build_public_tournament_reference(tournament_id=partner["tournament_id"], namespace="partner-registration", source_id="target")
    if invalid == "tampered":
        key += "bad"
    # Use the shared validator directly for the self case: the normal new-form
    # endpoint correctly sends an already-registered requester to email recovery.
    from jupr_app.services.public_tournament_registration_service import validate_and_clean_tournament_selection
    from tests.test_public_tournament_registration_service import FakeSupabase
    before = deepcopy(storage)
    with pytest.raises(ValueError, match="own partner|no longer available"):
        validate_and_clean_tournament_selection(FakeSupabase(storage), club_id="club-1", tournament_id="t1",
            event=storage["tournament_event_options"][0], settings=storage["tournament_registration_settings"][0],
            player_profile={"email": "requester@example.test", "age": 40, "gender": "Men", "doubles_skill": 4.0},
            raw_selection={"event_option_id": "event1", "partner_mode": "HAS_PARTNER", "partner_name": "Saved Partner",
                           "partner_registration_key": key})
    assert storage == before


def test_registered_partner_missing_details_and_server_eligibility(integrity_client):
    from jupr_app.services.public_tournament_registration_service import validate_and_clean_tournament_selection
    from tests.test_public_tournament_registration_service import FakeSupabase

    api, storage = integrity_client
    partner = {"id": "incomplete", "tournament_id": "t1", "display_name": "Saved Partner",
               "email": "partner@example.test", "gender": "Women", "doubles_skill": 4.5}
    storage["tournament_registrations"].append(partner)
    event = storage["tournament_event_options"][0]
    event.update({"partner_required": True, "event_type": "MIXED_DOUBLES", "eligibility_mode": "COMBINED_RATING_CAP", "combined_rating_cap": 8})
    response = api.post(ENDPOINT, json={"registration_slug": "tres-open", "name": "Saved Partner"})
    saved = response.json()["profile_candidates"][0]["registered_partner"]
    assert saved["has_age"] is False and saved["has_gender"] is True
    args = dict(club_id="club-1", tournament_id="t1", event=event, settings=storage["tournament_registration_settings"][0],
        player_profile={"email": "requester@example.test", "age": 40, "gender": "Men", "doubles_skill": 4.0})
    selection = {"event_option_id": "event1", "partner_mode": "HAS_PARTNER", "partner_name": "Saved Partner",
                 "partner_registration_key": saved["key"], "partner_skill": 1.0}
    with pytest.raises(ValueError, match="partner age is required"):
        validate_and_clean_tournament_selection(FakeSupabase(storage), raw_selection=selection, **args)
    with pytest.raises(ValueError, match="Combined|combined"):
        validate_and_clean_tournament_selection(FakeSupabase(storage), raw_selection={**selection, "partner_age": 40}, **args)


def test_duplicate_registered_names_require_an_explicit_choice(integrity_client):
    api, storage = integrity_client
    storage["tournament_registrations"].extend([
        {"id": f"duplicate-{index}", "tournament_id": "t1", "display_name": "Same Partner", "email": f"private{index}@example.test"}
        for index in range(2)
    ])
    response = api.post(ENDPOINT, json={"registration_slug": "tres-open", "name": "Same Partner"})
    candidates = response.json()["profile_candidates"]
    assert len(candidates) == 2
    assert len({row["registered_partner"]["key"] for row in candidates}) == 2
    assert "private0" not in response.text and "private1" not in response.text


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
