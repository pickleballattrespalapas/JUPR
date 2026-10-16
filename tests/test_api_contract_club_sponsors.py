from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from jupr_app.services.club_sponsor_service import SponsorConflictError, list_club_sponsors, save_club_sponsor, use_club_sponsor
from jupr_app.services.tournament_sponsor_service import normalize_sponsors, public_sponsors, valid_asset_path
from services.api.main import app
from tests.test_api_contract_admin_tournament_setup import FakeSupabase, install_env

SOURCE, TARGET, SPONSOR = (str(uuid4()) for _ in range(3))
LOGO = f"club/{SOURCE}/{uuid4().hex}.webp"


class Storage(dict):
    def __init__(self, tables):
        super().__init__(tables)
        self.objects = {LOGO: b"validated-webp"}

    def from_(self, _bucket):
        return self

    def download(self, path):
        return self.objects[path]

    def upload(self, path, data, _options):
        self.objects[path] = data

    def create_signed_url(self, path, _ttl):
        return {"signedURL": f"https://example.com/{path}"}


def fixture():
    sb = FakeSupabase()
    sb.storage = Storage(sb.storage)
    sb.storage["club_sponsors"] = [{
        "id": SPONSOR, "club_id": "club", "name": "Homes and Land of Baja", "name_key": "homes and land of baja",
        "website": "https://example.com", "public_description": "Local homes.",
        "contact_name": "Private contact", "contact_email": "private@example.com", "contact_phone": "+52 123",
        "logo_path": LOGO, "logo_tournament_id": SOURCE, "updated_at": "2026-10-01T00:00:00Z",
    }]
    sb.storage["tournaments"] += [{"id": SOURCE, "club_id": "club"}, {"id": TARGET, "club_id": "club"}]
    return sb


def test_reuse_copies_private_logo_and_keeps_tournament_snapshot_independent():
    sb = fixture()
    original = dict(sb.storage["club_sponsors"][0])
    selected = use_club_sponsor(sb, club_id="club", tournament_id=TARGET, sponsor_id=SPONSOR)
    assert valid_asset_path(selected["logo_path"], club_id="club", tournament_id=TARGET)
    assert sb.storage.objects[selected["logo_path"]] == sb.storage.objects[LOGO]
    assert selected["tier"] == "supporting" and selected["notes"] == selected["level"] == ""
    assert selected["club_sponsor_id"] == SPONSOR
    assert not any(key.startswith("contact_") for key in selected)
    assert use_club_sponsor(sb, club_id="club", tournament_id=TARGET, sponsor_id=SPONSOR) == selected
    saved = normalize_sponsors([selected], club_id="club", tournament_id=TARGET)
    assert saved[0]["club_sponsor_id"] == SPONSOR
    update = {**original, "name": "New business name", "logo_path": selected["logo_path"]}
    save_club_sponsor(sb, club_id="club", sponsor_id=SPONSOR, tournament_id=TARGET,
                      value=update, expected_updated_at=original["updated_at"])
    published = public_sponsors(sb, saved, club_id="club", tournament_id=TARGET)[0]
    assert published["name"] == original["name"]
    assert "private@example.com" not in str(published)
    assert "club_sponsor_id" not in published


def test_club_isolation_for_read_use_update_and_logo_signing():
    sb = fixture()
    assert list_club_sponsors(sb, club_id="other") == []
    with pytest.raises(LookupError):
        use_club_sponsor(sb, club_id="other", tournament_id=TARGET, sponsor_id=SPONSOR)
    with pytest.raises(ValueError, match="does not belong"):
        save_club_sponsor(sb, club_id="other", tournament_id=TARGET, sponsor_id=str(uuid4()),
                          value={"name": "Bad", "logo_path": LOGO}, expected_updated_at=None)
    sb.storage["club_sponsors"][0]["logo_path"] = f"other/{SOURCE}/{uuid4().hex}.webp"
    assert list_club_sponsors(sb, club_id="club")[0]["logo_url"] == ""
    with pytest.raises(ValueError, match="unavailable"):
        use_club_sponsor(sb, club_id="club", tournament_id=TARGET, sponsor_id=SPONSOR)


def test_updates_conflict_duplicates_and_retries():
    sb = fixture()
    original = dict(sb.storage["club_sponsors"][0])
    with pytest.raises(SponsorConflictError, match="changed"):
        save_club_sponsor(sb, club_id="club", sponsor_id=SPONSOR, tournament_id=SOURCE,
                          value={**original, "website": "new.example.com"}, expected_updated_at="old")
    with pytest.raises(SponsorConflictError, match="already saved"):
        save_club_sponsor(sb, club_id="club", sponsor_id=str(uuid4()), tournament_id=TARGET,
                          value={"name": "  HOMES  AND LAND OF BAJA "}, expected_updated_at=None)
    values = {**original, "contact_email": "new@example.com"}
    updated = save_club_sponsor(sb, club_id="club", sponsor_id=SPONSOR, tournament_id=SOURCE,
                                value=values, expected_updated_at=original["updated_at"])
    assert updated["contact_email"] == "new@example.com"
    retried = save_club_sponsor(sb, club_id="club", sponsor_id=SPONSOR, tournament_id=SOURCE,
                                value=values, expected_updated_at=original["updated_at"])
    assert retried["updated_at"] == updated["updated_at"]
    assert len(sb.storage["club_sponsors"]) == 1


def test_new_profile_and_contact_validation():
    sb = fixture()
    profile_id = str(uuid4())
    values = {"name": "Los Barriles Life", "website": "example.com", "contact_name": "Contact"}
    created = save_club_sponsor(sb, club_id="club", sponsor_id=profile_id, tournament_id=TARGET,
                                value=values, expected_updated_at=None)
    assert created["website"] == "https://example.com" and created["logo_path"] == ""
    assert created["contact_name"] == "Contact"
    with pytest.raises(ValueError, match="email"):
        save_club_sponsor(sb, club_id="club", sponsor_id=profile_id, tournament_id=TARGET,
                          value={**values, "contact_email": "not an email"}, expected_updated_at=created["updated_at"])


def test_routes_permission_club_ownership_and_runtime(monkeypatch):
    sb = fixture()
    install_env(monkeypatch, sb)
    monkeypatch.setenv("JUPR_ENV", "test")
    client = TestClient(app)
    headers = {"Authorization": "Bearer local"}
    assert client.get("/admin/clubs/club/sponsors", headers=headers).json()["sponsors"][0]["contact_email"] == "private@example.com"
    assert client.get("/admin/clubs/other/sponsors", headers=headers).json() == {"sponsors": []}
    endpoint = f"/admin/clubs/club/tournaments/setup/tournaments/{TARGET}/saved-sponsors/{SPONSOR}/use"
    assert client.post(endpoint, headers=headers).status_code == 200
    assert client.post(endpoint.replace("/club/", "/other/"), headers=headers).status_code == 404
    profile = f"/admin/clubs/club/sponsors/{uuid4()}"
    body = {"tournament_id": TARGET, "name": "New sponsor", "contact_email": "contact@example.com"}
    assert client.put(profile, json=body, headers=headers).status_code == 200
    assert client.put(profile.replace("/club/", "/other/"), json=body, headers=headers).status_code == 404
    body["name"] = "Homes and Land of Baja"
    assert client.put(profile, json=body, headers=headers).status_code == 409
    monkeypatch.setenv("JUPR_ENV", "production")
    assert client.post(endpoint, headers=headers).status_code == 403
    assert client.put(profile, json=body, headers=headers).status_code == 403
    monkeypatch.setenv("JUPR_ENV", "test")
    monkeypatch.setattr("services.api.admin_tournament_setup_routes.resolve_admin_role", lambda **_kw: SimpleNamespace(role="operator"))
    assert client.get("/admin/clubs/club/sponsors", headers=headers).status_code == 403
    assert client.post(endpoint, headers=headers).status_code == 403
    assert client.put(profile, json=body, headers=headers).status_code == 403
