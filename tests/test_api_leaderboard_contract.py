import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        "services.api.main.get_club",
        lambda club_slug: {
            "club_id": "club-1",
            "club_slug": club_slug,
            "club_name": "Test Club",
        },
    )
    monkeypatch.setattr("services.api.main.get_supabase_client", lambda: object())

    def fake_get_public_leaderboard(*, supabase, club_id, league_name=None):
        assert club_id == "club-1"
        return [
            {
                "rank_position": 7,
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
            },
            {
                "club_id": "club-1",
                "league_name": league_name or "Open",
                "player_id": "p2",
                "player_name": "Blair",
                "rating": 1500,
                "wins": 8,
                "losses": 4,
                "matches_played": 12,
                "is_active": True,
                "internal_notes": "hidden",
            },
        ]

    monkeypatch.setattr("services.api.main.get_public_leaderboard", fake_get_public_leaderboard)
    return TestClient(app)


def test_primary_and_compat_routes_return_same_normalized_contract(client):
    primary = client.get("/clubs/test-club/leaderboards?league_name=Pro")
    compat = client.get("/clubs/test-club/leaderboards/public?league_name=Pro")

    assert primary.status_code == 200
    assert compat.status_code == 200
    assert primary.json() == compat.json()

    payload = primary.json()
    assert payload["club"] == {"id": "club-1", "slug": "test-club", "name": "Test Club"}
    assert isinstance(payload["leaderboard"], list)

    first, second = payload["leaderboard"]
    assert first["rank"] == 7
    assert first["rank_position"] == 7
    assert "email" not in first

    assert second["rank"] == 2
    assert "rank_position" not in second or second["rank_position"] is None
    assert "internal_notes" not in second
