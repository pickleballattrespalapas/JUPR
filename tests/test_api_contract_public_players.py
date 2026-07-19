from __future__ import annotations

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
        lambda club_slug: {"id": "club-1", "slug": club_slug, "name": "Tres Palapas"},
    )
    monkeypatch.setattr("services.api.main.get_supabase_client", lambda: object())
    return TestClient(app)


def test_player_directory_contract_forwards_active_search_sort_and_paging(client, monkeypatch):
    captured = {}

    def fake_directory(_supabase, **kwargs):
        captured.update(kwargs)
        return {
            "players": [{"id": 1, "name": "Public Alias", "display_name": "Public Alias", "rating_jupr": 4.0, "is_active": True}],
            "filters": {"search": "alias", "status": "active", "sort": "name"},
            "summary": {"public_players": 1, "active_players": 1, "inactive_players": 0, "filtered_players": 1},
            "pagination": {"total": 1, "limit": 25, "offset": 25, "has_more": False},
        }

    monkeypatch.setattr("services.api.main.build_public_player_directory", fake_directory)
    response = client.get("/clubs/tres-palapas/players?q=alias&status=active&sort=name&limit=25&offset=25")

    assert response.status_code == 200
    assert captured == {"club_id": "club-1", "search": "alias", "status": "active", "sort": "name", "limit": 25, "offset": 25}
    payload = response.json()
    assert payload["club"] == {"id": "club-1", "slug": "tres-palapas", "name": "Tres Palapas"}
    assert payload["players"][0]["name"] == "Public Alias"
    assert payload["filters"]["status"] == "active"


def test_player_directory_defaults_to_active(client, monkeypatch):
    captured = {}

    def fake_directory(_supabase, **kwargs):
        captured.update(kwargs)
        return {"players": [], "filters": {"search": "", "status": "active", "sort": "rating"}, "summary": {}, "pagination": {}}

    monkeypatch.setattr("services.api.main.build_public_player_directory", fake_directory)
    response = client.get("/clubs/tres-palapas/players")

    assert response.status_code == 200
    assert captured["status"] == "active"
    assert captured["sort"] == "rating"
    assert captured["limit"] == 100
    assert captured["offset"] == 0


def test_player_profile_contract_exposes_parity_sections_and_forwards_limits(client, monkeypatch):
    captured = {}

    def fake_profile(_supabase, **kwargs):
        captured.update(kwargs)
        return {
            "player": {"id": 1, "name": "Public Alias", "display_name": "Public Alias", "rating_jupr": 4.0, "is_active": True},
            "identity": {"display_name": "Public Alias", "public_name_policy": "public_display_name", "verification_status": "available"},
            "verified_updates": {"status": "available", "can_request": True},
            "rating_summary": {"current_rating_jupr": 4.0, "last_10_record": "1-0"},
            "rating_breakdowns": [{"format": "singles", "label": "Singles", "matches": 1, "wins": 1, "losses": 0}],
            "rating_history": [],
            "league_ratings": [],
            "awards": {"badge_count": 0, "badge_award_count": 0, "prestige_total": 0, "badges": [], "trophies": []},
            "relationships": {"best_partner": None, "rival": None, "partners": [], "rivals": []},
            "social": {"available": True, "identity": {"linked": False, "label": "No linked Club Social identity"}, "summary": {"events": 0, "matches": 0, "wins": 0, "losses": 0, "score_diff": 0, "last_appearance": None}, "skill_breakdown": [], "recent_events": []},
            "recent_matches": [],
            "match_history": [],
            "history": {"total_matches": 0, "recent_limit": 7, "history_limit": 80, "has_more": False},
        }

    monkeypatch.setattr("services.api.main.get_public_player_profile", fake_profile)
    response = client.get("/clubs/tres-palapas/players/1?recent_limit=7&history_limit=80")

    assert response.status_code == 200
    assert captured == {"club_id": "club-1", "player_id": "1", "recent_match_limit": 7, "history_limit": 80}
    payload = response.json()
    for field in ("identity", "verified_updates", "rating_summary", "rating_breakdowns", "rating_history", "awards", "relationships", "social", "match_history"):
        assert field in payload
    assert payload["identity"]["display_name"] == "Public Alias"


def test_unknown_player_is_a_stable_404(client, monkeypatch):
    monkeypatch.setattr("services.api.main.get_public_player_profile", lambda *_args, **_kwargs: None)

    response = client.get("/clubs/tres-palapas/players/not-found")

    assert response.status_code == 404
    assert response.json() == {"detail": "player not found"}
