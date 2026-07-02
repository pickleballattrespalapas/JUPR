import pytest

from tests.conftest import require_api_dependency

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        "services.api.main.get_club",
        lambda club_slug: {
            "id": "club-1",
            "slug": club_slug,
            "name": "Tres Palapas",
        },
    )
    monkeypatch.setattr("services.api.main.get_supabase_client", lambda: object())

    def fake_get_public_leaderboard(*, supabase, club_id, league_name=None):
        assert club_id == "club-1"
        return [
            {
                "rank_position": 1,
                "club_id": "club-1",
                "league_name": league_name or "Open",
                "player_id": "p1",
                "player_name": "Alex",
                "rating": 1600,
                "rating_jupr": 1600,
                "wins": 10,
                "losses": 2,
                "matches_played": 12,
                "is_active": True,
                "updated_at": "2026-05-04T00:00:00Z",
                "email": "private@example.com",
                "admin_flag": "secret",
            },
            {
                "club_id": "club-1",
                "league_name": league_name or "Open",
                "player_id": "p2",
                "player_name": "Blair",
                "rating_jupr": 1500,
                "wins": 8,
                "losses": 4,
                "matches_played": 12,
                "is_active": True,
                "subscription_token": "nope",
            },
        ]

    monkeypatch.setattr("services.api.main.get_public_leaderboard", fake_get_public_leaderboard)
    return TestClient(app)


def test_public_leaderboards_contract_shape(client):
    response = client.get("/clubs/tres-palapas/leaderboards?league_name=Pro")

    assert response.status_code == 200
    payload = response.json()

    assert payload["club"] == {"id": "club-1", "slug": "tres-palapas", "name": "Tres Palapas"}
    assert isinstance(payload["leaderboard"], list)

    first_row = payload["leaderboard"][0]
    assert first_row["rank"] == 1
    assert first_row["player_name"] == "Alex"
    assert "rating" in first_row or "rating_jupr" in first_row
    assert first_row["matches_played"] == 12
    assert "email" not in first_row
    assert "admin_flag" not in first_row

    second_row = payload["leaderboard"][1]
    assert second_row["rank"] == 2
    assert second_row["player_name"] == "Blair"
    assert "rating" in second_row or "rating_jupr" in second_row
    assert second_row["matches_played"] == 12
    assert "subscription_token" not in second_row


def test_public_leaderboards_compat_alias_matches_primary_contract(client):
    # Temporary compatibility alias only; this should match /leaderboards exactly.
    primary = client.get("/clubs/tres-palapas/leaderboards?league_name=Pro")
    compat = client.get("/clubs/tres-palapas/leaderboards/public?league_name=Pro")

    assert primary.status_code == 200
    assert compat.status_code == 200
    assert compat.json() == primary.json()


def test_get_club_falls_back_to_existing_underscore_club_id(monkeypatch):
    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, table_name):
            self.table_name = table_name
            self.filters = {}

        def select(self, *_args, **_kwargs):
            return self

        def eq(self, key, value):
            self.filters[key] = value
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def execute(self):
            if self.table_name == "clubs":
                raise Exception("Could not find the table 'public.clubs' in the schema cache")
            if self.table_name == "players" and self.filters.get("club_id") == "tres_palapas":
                return FakeResponse([{"club_id": "tres_palapas"}])
            return FakeResponse([])

    class FakeSupabase:
        def table(self, table_name):
            return FakeQuery(table_name)

    monkeypatch.setattr("services.api.main.get_supabase_client", lambda: FakeSupabase())
    response = TestClient(app).get("/clubs/tres-palapas")

    assert response.status_code == 200
    assert response.json()["id"] == "tres_palapas"
    assert response.json()["slug"] == "tres-palapas"
