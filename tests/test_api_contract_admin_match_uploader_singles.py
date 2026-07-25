from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def singles_tables():
    return {
        "players": [
            {
                "club_id": "club",
                "id": 1,
                "name": "Alex",
                "rating": 1200,
                "singles_rating": 1200,
                "singles_wins": 0,
                "singles_losses": 0,
                "singles_matches_played": 0,
                "singles_last_game_at": None,
                "singles_replay_baseline": {
                    "rating": 1200,
                    "wins": 0,
                    "losses": 0,
                    "matches_played": 0,
                    "last_game_at": None,
                },
            },
            {
                "club_id": "club",
                "id": 2,
                "name": "Blair",
                "rating": 1200,
                "singles_rating": 1200,
                "singles_wins": 0,
                "singles_losses": 0,
                "singles_matches_played": 0,
                "singles_last_game_at": None,
                "singles_replay_baseline": {
                    "rating": 1200,
                    "wins": 0,
                    "losses": 0,
                    "matches_played": 0,
                    "last_game_at": None,
                },
            },
        ],
        "matches": [],
        "admin_activity_log": [],
    }


def _install_auth(monkeypatch):
    monkeypatch.setattr(
        "services.api.admin_match_uploader_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_match_uploader_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


def _install_env(monkeypatch, supabase):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_SINGLES", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)


def test_admin_match_uploader_singles_dormant_gate_blocks_before_auth(
    monkeypatch,
):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    monkeypatch.delenv(
        "JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_SINGLES",
        raising=False,
    )
    called = {"auth": False}

    def fail_auth(*_args, **_kwargs):
        called["auth"] = True
        raise AssertionError("auth must not run while direct singles is dormant")

    monkeypatch.setattr(
        "services.api.admin_match_uploader_routes.authenticate_bearer",
        fail_auth,
    )

    response = TestClient(app).post(
        "/admin/clubs/club/match-uploader/singles",
        json={
            "t1_p1": 1,
            "t2_p1": 2,
            "score_t1": 11,
            "score_t2": 8,
        },
    )

    assert response.status_code == 403
    assert "transactional write/replay hardening" in response.json()["detail"]
    assert called == {"auth": False}


def _fake_load_data(supabase, club_id):
    import pandas as pd

    players = pd.DataFrame(supabase.tables["players"])
    empty = pd.DataFrame()
    name_to_id = {"Alex": 1, "Blair": 2}
    id_to_name = {1: "Alex", 2: "Blair"}
    return (players, players, empty, empty, empty, empty, empty, name_to_id, id_to_name, False, "")


def test_admin_match_uploader_singles_contract(monkeypatch):
    tables = singles_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)
    captured: dict[str, object] = {}

    def fake_process_singles(match_list, **_kwargs):
        captured["match_list"] = match_list
        tables["matches"].append({"id": "m1", "club_id": "club", "match_format": "singles"})
        return {"inserted": len(match_list), "match_format": "singles", "skipped_incomplete": 0, "skipped_empty": 0, "skipped_unrated": 0}

    monkeypatch.setattr("jupr_app.services.admin_singles_match_service.load_data", _fake_load_data)
    monkeypatch.setattr("jupr_app.services.admin_singles_match_service.process_singles_matches", fake_process_singles)

    response = TestClient(app).post(
        "/admin/clubs/club/match-uploader/singles",
        headers={"Authorization": "Bearer local"},
        json={"date": "2026-05-01", "league": "Singles", "week_tag": "Challenge", "t1_p1": 1, "t2_p1": 2, "score_t1": 11, "score_t2": 8},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "singles_match_uploader"
    assert payload["match_write_committed"] is True
    assert payload["recovery"]["match_log_route"] == "/admin/match-log"
    assert payload["result"]["match_format"] == "singles"
    assert captured["match_list"][0]["match_format"] == "singles"
    assert captured["match_list"][0]["t1_p1"] == 1
    assert captured["match_list"][0]["t2_p1"] == 2
    assert tables["admin_activity_log"][0]["action_type"] == "submit_singles_match_uploader"


def test_admin_match_uploader_singles_blocks_ties(monkeypatch):
    supabase = FakeSupabase(singles_tables())
    _install_env(monkeypatch, supabase)
    monkeypatch.setattr("jupr_app.services.admin_singles_match_service.load_data", _fake_load_data)

    response = TestClient(app).post(
        "/admin/clubs/club/match-uploader/singles",
        headers={"Authorization": "Bearer local"},
        json={"t1_p1": 1, "t2_p1": 2, "score_t1": 9, "score_t2": 9},
    )

    assert response.status_code == 400
    assert "tied" in response.json()["detail"]


def test_admin_match_uploader_unrated_singles_records_without_rating_change(monkeypatch):
    tables = singles_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)
    monkeypatch.setattr(
        "jupr_app.services.admin_singles_match_service.load_data",
        _fake_load_data,
    )
    before = [dict(player) for player in tables["players"]]

    response = TestClient(app).post(
        "/admin/clubs/club/match-uploader/singles",
        headers={"Authorization": "Bearer local"},
        json={
            "date": "2026-05-01",
            "league": "Singles",
            "week_tag": "smoke",
            "t1_p1": 1,
            "t2_p1": 2,
            "score_t1": 11,
            "score_t2": 8,
            "rating_scope": "unrated",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["inserted"] == 1
    assert payload["result"]["skipped_unrated"] == 1
    assert len(tables["matches"]) == 1
    assert tables["matches"][0]["rating_scope"] == "unrated"
    assert tables["matches"][0]["match_format"] == "singles"
    assert tables["matches"][0]["singles_replay_managed"] is True
    assert tables["players"] == before


def test_admin_match_uploader_first_singles_uses_preserved_seed_and_keeps_latest_date(
    monkeypatch,
):
    tables = singles_tables()
    tables["players"][0].update(
        {
            "rating": 1800,
            "singles_rating": None,
            "singles_last_game_at": "2026-06-01T12:00:00+00:00",
            "singles_replay_baseline": {
                "rating": 1400,
                "wins": 0,
                "losses": 0,
                "matches_played": 0,
                "last_game_at": "2026-06-01T12:00:00+00:00",
            },
        }
    )
    tables["players"][1].update(
        {
            "rating": 1700,
            "singles_rating": None,
            "singles_last_game_at": "2026-06-01T12:00:00+00:00",
            "singles_replay_baseline": {
                "rating": 1300,
                "wins": 0,
                "losses": 0,
                "matches_played": 0,
                "last_game_at": "2026-06-01T12:00:00+00:00",
            },
        }
    )
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)
    monkeypatch.setattr(
        "jupr_app.services.admin_singles_match_service.load_data",
        _fake_load_data,
    )

    response = TestClient(app).post(
        "/admin/clubs/club/match-uploader/singles",
        headers={"Authorization": "Bearer local"},
        json={
            "date": "2026-05-01",
            "t1_p1": 1,
            "t2_p1": 2,
            "score_t1": 11,
            "score_t2": 8,
        },
    )

    assert response.status_code == 200
    assert tables["matches"][0]["t1_p1_r"] == 1400
    assert tables["matches"][0]["t2_p1_r"] == 1300
    assert (
        tables["players"][0]["singles_last_game_at"]
        == "2026-06-01T12:00:00+00:00"
    )
    assert (
        tables["players"][1]["singles_last_game_at"]
        == "2026-06-01T12:00:00+00:00"
    )
