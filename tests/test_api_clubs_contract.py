import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    def __init__(self, db, table_name):
        self._db = db
        self._table_name = table_name
        self._eq = {}

    def select(self, _fields):
        return self

    def eq(self, field, value):
        self._eq[field] = value
        return self

    def limit(self, _n):
        return self

    def execute(self):
        execute_error = self._db.get(f"{self._table_name}__execute_error")
        if execute_error is not None:
            raise execute_error
        rows = list(self._db.get(self._table_name, []))
        for key, expected in self._eq.items():
            rows = [r for r in rows if r.get(key) == expected]
        return _FakeResponse(rows[:1])


class _FakeSupabase:
    def __init__(self, db):
        self._db = db

    def table(self, table_name):
        return _FakeQuery(self._db, table_name)


@pytest.fixture
def client(monkeypatch):
    db = {
        "clubs": [
            {
                "id": "club-1",
                "slug": "tres-palapas",
                "name": "Tres Palapas",
                "tagline": "Welcome",
                "support_email": "support@trespalapas.com",
                "public_base_url": "https://app.example.com",
                "logo_url": "https://cdn.example.com/logo.png",
                "primary_color": "#00AA88",
                "is_active": True,
            },
            {
                "id": "club-2",
                "slug": "other-club",
                "name": "Other Club",
                "tagline": None,
                "support_email": None,
                "public_base_url": None,
                "logo_url": None,
                "primary_color": None,
                "is_active": True,
            },
        ],
        "players": [{"club_id": "legacy-club"}],
    }
    monkeypatch.setattr("services.api.main.get_supabase_client", lambda: _FakeSupabase(db))
    return TestClient(app)


def test_club_slug_lookup_returns_normalized_fields(client):
    response = client.get("/clubs/tres-palapas")

    assert response.status_code == 200
    assert response.json() == {
        "id": "club-1",
        "slug": "tres-palapas",
        "name": "Tres Palapas",
        "tagline": "Welcome",
        "support_email": "support@trespalapas.com",
        "public_base_url": "https://app.example.com",
        "logo_url": "https://cdn.example.com/logo.png",
        "primary_color": "#00AA88",
        "is_active": True,
    }


def test_club_id_fallback_lookup_works(client):
    response = client.get("/clubs/club-2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "club-2"
    assert payload["slug"] == "other-club"


def test_club_legacy_players_fallback_returns_minimal_public_shape(client):
    response = client.get("/clubs/legacy-club")

    assert response.status_code == 200
    assert response.json() == {
        "id": "legacy-club",
        "slug": "legacy-club",
        "name": "legacy-club",
        "tagline": None,
        "support_email": None,
        "public_base_url": None,
        "logo_url": None,
        "primary_color": None,
        "is_active": True,
    }


def test_missing_club_returns_404_when_no_fallback_data_exists(client):
    response = client.get("/clubs/missing-club")

    assert response.status_code == 404


def test_club_players_fallback_works_when_clubs_table_is_missing(monkeypatch):
    db = {
        "clubs__execute_error": RuntimeError('relation "clubs" does not exist'),
        "players": [{"club_id": "legacy-club"}],
    }
    monkeypatch.setattr("services.api.main.get_supabase_client", lambda: _FakeSupabase(db))
    client = TestClient(app)

    legacy_response = client.get("/clubs/legacy-club")
    assert legacy_response.status_code == 200
    assert legacy_response.json() == {
        "id": "legacy-club",
        "slug": "legacy-club",
        "name": "legacy-club",
        "tagline": None,
        "support_email": None,
        "public_base_url": None,
        "logo_url": None,
        "primary_color": None,
        "is_active": True,
    }

    missing_response = client.get("/clubs/missing-club")
    assert missing_response.status_code == 404
