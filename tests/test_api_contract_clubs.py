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
        rows = list(self._db.get(self._table_name, []))
        for key, expected in self._eq.items():
            rows = [row for row in rows if row.get(key) == expected]
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
                "admin_notes": "private",
                "subscription_token": "secret-token",
            }
        ]
    }
    monkeypatch.setattr("services.api.main.get_supabase_client", lambda: _FakeSupabase(db))
    return TestClient(app)


def test_health_contract(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "service": "jupr-api"}


def test_public_club_contract_filters_private_fields(client):
    response = client.get("/clubs/tres-palapas")

    assert response.status_code == 200
    payload = response.json()

    assert payload["id"] == "club-1"
    assert payload["slug"] == "tres-palapas"
    assert payload["name"] == "Tres Palapas"
    assert payload["is_active"] is True

    assert "admin_notes" not in payload
    assert "subscription_token" not in payload
