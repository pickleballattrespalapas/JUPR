from __future__ import annotations

import base64
from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4

import pytest
from PIL import Image

from jupr_app.services.tournament_sponsor_service import (
    encode_logo, normalize_sponsors, public_sponsors, valid_asset_path, website_url,
)

CLUB = "tres_palapas"
TOURNAMENT = str(uuid4())
PATH = f"{CLUB}/{TOURNAMENT}/{uuid4().hex}.webp"


def test_uploaded_image_is_static_reencoded_bounded_and_has_no_metadata():
    data = BytesIO()
    Image.new("RGBA", (1600, 800), (0, 100, 150, 0)).save(data, format="PNG")
    output = encode_logo(base64.b64encode(data.getvalue()).decode())
    with Image.open(BytesIO(output)) as logo:
        assert logo.format == "WEBP"
        assert logo.size == (1024, 512)
        assert "exif" not in logo.info
        assert logo.getpixel((0, 0))[3] == 0


@pytest.mark.parametrize("data", [b"<svg onload='alert(1)'/>", b"not an image", b""])
def test_invalid_images_rejected(data):
    with pytest.raises(ValueError):
        encode_logo(base64.b64encode(data).decode())


def test_oversized_and_animated_images_rejected():
    data = BytesIO()
    Image.new("RGB", (4097, 1)).save(data, format="PNG")
    with pytest.raises(ValueError, match="4096"):
        encode_logo(base64.b64encode(data.getvalue()).decode())
    data = BytesIO()
    Image.new("RGB", (2, 2), "red").save(data, format="PNG", save_all=True, append_images=[Image.new("RGB", (2, 2), "blue")], duration=100, loop=0)
    with pytest.raises(ValueError, match="static"):
        encode_logo(base64.b64encode(data.getvalue()).decode())


@pytest.mark.parametrize("url", ["javascript:alert(1)", "data:text/html,x", "https://a.com\\@b.com", "https://user:pass@example.com", "https://example.com\n"])
def test_unsafe_links_rejected(url):
    # Surrounding whitespace is harmless and intentionally normalized.
    if url.endswith("\n"):
        assert website_url(url) == "https://example.com"
    else:
        with pytest.raises(ValueError):
            website_url(url)


def test_bare_website_and_legacy_labels():
    rows = normalize_sponsors([{"name": "Legacy sponsor", "level": "Title sponsor", "website": "example.com"}], club_id=CLUB, tournament_id=TOURNAMENT)
    assert rows[0]["website"] == "https://example.com"
    assert rows[0]["level"] == "Title sponsor"
    assert rows[0]["tier"] == "supporting"


def test_cross_tournament_logo_rejected():
    assert valid_asset_path(PATH, club_id=CLUB, tournament_id=TOURNAMENT)
    with pytest.raises(ValueError, match="does not belong"):
        normalize_sponsors([{"name": "Sponsor", "logo_path": PATH}], club_id=CLUB, tournament_id=str(uuid4()))


def test_public_allowlist_hidden_sponsors_and_storage_failure():
    def unavailable(*_args):
        raise RuntimeError("storage offline")
    supabase = SimpleNamespace(storage=SimpleNamespace(from_=unavailable))
    rows = public_sponsors(supabase, [
        {"id": "visible", "name": "Coastal Homes", "tier": "presenting", "logo_path": PATH, "notes": "private contract", "website": "javascript:alert(1)"},
        {"id": "hidden", "name": "Secret draft", "is_visible": False},
    ], club_id=CLUB, tournament_id=TOURNAMENT)
    assert len(rows) == 1
    assert rows[0]["name"] == "Coastal Homes"
    assert rows[0]["logo_url"] == rows[0]["website"] == ""
    assert "notes" not in rows[0] and "logo_path" not in rows[0]


def test_upload_route_denies_other_club_and_operator(monkeypatch):
    from fastapi.testclient import TestClient
    from tests.test_api_contract_admin_tournament_setup import FakeSupabase, install_env
    from services.api.main import app
    sb = FakeSupabase()
    install_env(monkeypatch, sb)
    monkeypatch.setenv("JUPR_ENV", "test")
    uploads = []
    monkeypatch.setattr("services.api.admin_tournament_setup_routes.upload_logo", lambda *_a, **_kw: uploads.append(1))
    client = TestClient(app)
    response = client.post("/admin/clubs/other-club/tournaments/setup/tournaments/t1/sponsor-logos", json={"image_base64": "eA=="}, headers={"Authorization": "Bearer local"})
    assert response.status_code == 404
    monkeypatch.setattr("services.api.admin_tournament_setup_routes.resolve_admin_role", lambda **_kw: SimpleNamespace(role="operator"))
    response = client.post("/admin/clubs/club/tournaments/setup/tournaments/t1/sponsor-logos", json={"image_base64": "eA=="}, headers={"Authorization": "Bearer local"})
    assert response.status_code == 403
    assert not uploads


def test_public_route_reads_published_settings_not_private_draft(monkeypatch):
    from fastapi.testclient import TestClient
    from tests.test_api_contract_admin_tournament_setup import FakeSupabase, install_env
    from services.api.main import app
    sb = FakeSupabase()
    install_env(monkeypatch, sb)
    from fastapi import FastAPI
    from services.api.public_tournament_registration_routes import install_public_tournament_registration_routes
    app = FastAPI()
    install_public_tournament_registration_routes(app, get_club=lambda _slug: {"id": "club", "name": "Test"}, get_supabase_client=lambda: sb, public_club_payload=lambda club, slug: club)
    sb.storage["tournaments"][0]["status"] = "ACTIVE"
    sb.storage["tournament_registration_settings"][0].update({"registration_status": "open", "builder_draft_json": {"published_at": "2026-09-06T00:00:00Z", "basics": {"sponsors_json": [{"name": "Private sponsor"}]}}, "sponsors_json": [{"id": "s1", "name": "Published sponsor", "tier": "presenting", "public_description": "Public introduction", "notes": "Internal only"}]})
    sb.storage["tournament_builder_drafts"] = [{"tournament_id": "t1", "basics": {"sponsors_json": [{"name": "Private sponsor"}]}}]
    client = TestClient(app)
    response = client.get("/clubs/test/tournaments/t1/sponsors")
    assert response.status_code == 200
    assert response.json()["sponsors"][0]["name"] == "Published sponsor"
    assert response.json()["sponsors"][0]["public_description"] == "Public introduction"
    assert "Internal only" not in response.text
    assert "Private sponsor" not in response.text
    sb.storage["tournaments"][0]["club_id"] = "other"
    assert client.get("/clubs/test/tournaments/t1/sponsors").status_code == 404


def test_public_description_round_trip_and_legacy_default():
    original = [{"id": "one", "name": "Sponsor", "public_description": "  Local homes.\nLocal knowledge.  ", "notes": "Private pricing"}]
    saved = normalize_sponsors(original, club_id=CLUB, tournament_id=TOURNAMENT)
    reloaded = normalize_sponsors(saved, club_id=CLUB, tournament_id=TOURNAMENT)
    assert reloaded[0]["public_description"] == "Local homes.\nLocal knowledge."
    public = public_sponsors(None, reloaded, club_id=CLUB, tournament_id=TOURNAMENT)
    assert public[0]["public_description"] == reloaded[0]["public_description"]
    assert "notes" not in public[0]
    assert normalize_sponsors([{"name": "Legacy"}], club_id=CLUB, tournament_id=TOURNAMENT)[0]["public_description"] == ""
    saved[0]["public_description"] = ""
    assert public_sponsors(None, saved, club_id=CLUB, tournament_id=TOURNAMENT)[0]["public_description"] == ""


def test_public_description_length_validation():
    row = {"name": "Sponsor", "public_description": "a" * 500}
    assert len(normalize_sponsors([row], club_id=CLUB, tournament_id=TOURNAMENT)[0]["public_description"]) == 500
    row["public_description"] += "a"
    with pytest.raises(ValueError, match="500 characters"):
        normalize_sponsors([row], club_id=CLUB, tournament_id=TOURNAMENT)
