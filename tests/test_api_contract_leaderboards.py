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

    def fake_build_public_leaderboard(
        supabase,
        *,
        club_id,
        league_name=None,
        status="active",
        search=None,
        sort="rank",
        player_id=None,
        limit=50,
        offset=0,
    ):
        assert club_id == "club-1"
        assert status == "all"
        assert search == "Alex"
        assert sort == "gain"
        assert player_id == "p1"
        assert limit == 10
        assert offset == 10
        rows = [
            {
                "rank": 1,
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
                "win_pct": 83.3,
                "is_active": True,
                "qualified": True,
                "rating_gain_jupr": 0.1,
                "gap_jupr": None,
                "badges": [{"badge_id": "champ", "name": "Champion", "prestige": 100}],
                "badge_count": 1,
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
        return {
            "scopes": [{"name": "OVERALL", "label": "Overall", "min_games": 0}, {"name": league_name or "Open", "label": league_name or "Open", "min_games": 6}],
            "selected_scope": league_name or "Open",
            "scope": {"name": league_name or "Open", "label": league_name or "Open", "min_games": 6},
            "filters": {"status": status, "search": search or "", "sort": sort},
            "summary": {"ranked_players": 2, "active_players": 2, "inactive_players": 0, "leaderboard_scopes": 2, "filtered_players": 2},
            "leaderboard": rows,
            "snapshot": rows[0],
            "highlights": {"highest_rating": rows[:1], "most_improved": rows[:1], "best_win_pct": rows[:1], "most_wins": rows[:1]},
            "pagination": {"total": 2, "offset": offset, "limit": limit, "has_more": False},
        }

    monkeypatch.setattr("services.api.main.build_public_leaderboard", fake_build_public_leaderboard)
    return TestClient(app)


def test_public_leaderboards_contract_shape(client):
    response = client.get("/clubs/tres-palapas/leaderboards?league_name=Pro&status=all&q=Alex&sort=gain&player_id=p1&limit=10&offset=10")

    assert response.status_code == 200
    payload = response.json()

    assert payload["club"] == {"id": "club-1", "slug": "tres-palapas", "name": "Tres Palapas"}
    assert isinstance(payload["leaderboard"], list)

    first_row = payload["leaderboard"][0]
    assert first_row["rank"] == 1
    assert first_row["player_name"] == "Alex"
    assert "rating" in first_row or "rating_jupr" in first_row
    assert first_row["matches_played"] == 12
    assert first_row["rating_gain_jupr"] == 0.1
    assert first_row["qualified"] is True
    assert first_row["badges"] == [{"badge_id": "champ", "name": "Champion", "prestige": 100}]
    assert "email" not in first_row
    assert "admin_flag" not in first_row

    second_row = payload["leaderboard"][1]
    assert second_row["rank"] == 2
    assert second_row["player_name"] == "Blair"
    assert "rating" in second_row or "rating_jupr" in second_row
    assert second_row["matches_played"] == 12
    assert "subscription_token" not in second_row
    serialized = response.text
    for private_value in ("private@example.com", "secret", "nope"):
        assert private_value not in serialized

    assert payload["selected_scope"] == "Pro"
    assert payload["filters"] == {"status": "all", "search": "Alex", "sort": "gain"}
    assert payload["pagination"] == {"total": 2, "offset": 10, "limit": 10, "has_more": False}


def test_public_leaderboards_validate_pagination_bounds(client):
    assert client.get("/clubs/tres-palapas/leaderboards?limit=0").status_code == 422
    assert client.get("/clubs/tres-palapas/leaderboards?limit=101").status_code == 422
    assert client.get("/clubs/tres-palapas/leaderboards?offset=-1").status_code == 422


def test_public_leaderboards_report_server_projection_failure_without_backend_detail(client, monkeypatch):
    from services.api import main as api_main

    def broken_projection(*_args, **_kwargs):
        raise api_main.LeaderboardDataUnavailable("permission denied for private table")

    monkeypatch.setattr(api_main, "build_public_leaderboard", broken_projection)
    response = client.get("/clubs/tres-palapas/leaderboards")

    assert response.status_code == 503
    assert response.json() == {"detail": "Leaderboard data is temporarily unavailable."}
    assert "permission denied" not in response.text


def test_public_leaderboards_compat_alias_matches_primary_contract(client):
    # Temporary compatibility alias only; this should match /leaderboards exactly.
    query = "league_name=Pro&status=all&q=Alex&sort=gain&player_id=p1&limit=10&offset=10"
    primary = client.get(f"/clubs/tres-palapas/leaderboards?{query}")
    compat = client.get(f"/clubs/tres-palapas/leaderboards/public?{query}")

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


def test_get_known_public_club_survives_supabase_lookup_failure(monkeypatch):
    class BrokenSupabase:
        def table(self, _table_name):
            raise RuntimeError("Supabase unavailable")

    monkeypatch.setattr("services.api.main.get_supabase_client", lambda: BrokenSupabase())

    response = TestClient(app).get("/clubs/tres-palapas")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "tres_palapas"
    assert payload["slug"] == "tres-palapas"
    assert payload["name"] == "Tres Palapas"
