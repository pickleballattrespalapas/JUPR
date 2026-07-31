from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


class AtomicSinglesSupabase(FakeSupabase):
    def __init__(self, tables):
        super().__init__(tables)
        self.receipts: dict[tuple[str, str], dict] = {}

    def rpc(self, name, payload):
        if name != "admin_apply_direct_match_entry_atomic_v1":
            raise AssertionError(f"Unexpected RPC: {name}")

        def execute():
            lookup = (
                str(payload["p_club_id"]),
                str(payload["p_idempotency_key"]),
            )
            if lookup in self.receipts:
                return SimpleNamespace(
                    data={
                        **self.receipts[lookup],
                        "idempotent": True,
                        "duplicate_request": False,
                    }
                )
            match_ids = []
            for match in payload["p_match_rows"]:
                match_id = str(len(self.tables["matches"]) + 1)
                self.tables["matches"].append({**match, "id": match_id})
                match_ids.append(match_id)
            for update in payload["p_player_updates"]:
                for player in self.tables["players"]:
                    if (
                        str(player.get("club_id"))
                        == str(payload["p_club_id"])
                        and int(player.get("id")) == int(update["player_id"])
                    ):
                        player.update(update["after"])
            receipt = {
                "ok": True,
                "committed": True,
                "idempotent": False,
                "duplicate_request": False,
                "operation_id": payload["p_operation_id"],
                "idempotency_key": payload["p_idempotency_key"],
                "request_fingerprint": payload["p_request_fingerprint"],
                "match_format": payload["p_match_format"],
                "inserted": len(match_ids),
                "match_ids": match_ids,
                "result_summary": payload["p_result_summary"],
                "player_updates": payload["p_player_updates"],
            }
            self.receipts[lookup] = receipt
            self.tables["admin_activity_log"].append(
                {
                    "action_type": "submit_singles_match_uploader_atomic",
                    "entity_id": payload["p_operation_id"],
                }
            )
            return SimpleNamespace(data=receipt)

        return SimpleNamespace(execute=execute)


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
            "idempotency_key": "test:singles-disabled",
        },
    )

    assert response.status_code == 403
    assert "current write wave" in response.json()["detail"]
    assert called == {"auth": False}


def _fake_load_data(supabase, club_id):
    import pandas as pd

    players = pd.DataFrame(supabase.tables["players"])
    empty = pd.DataFrame()
    metadata = pd.DataFrame(
        [
            {
                "id": "league-summer-social",
                "club_id": club_id,
                "league_name": "Summer Social",
                "match_format": "singles",
                "k_factor": 32,
                "status": "active",
                "is_active": True,
                "ended_at": None,
            }
        ]
    )
    name_to_id = {"Alex": 1, "Blair": 2}
    id_to_name = {1: "Alex", 2: "Blair"}
    return (
        players,
        players,
        empty,
        empty,
        metadata,
        empty,
        empty,
        name_to_id,
        id_to_name,
        False,
        "",
    )


def test_admin_match_uploader_singles_contract(monkeypatch):
    tables = singles_tables()
    supabase = AtomicSinglesSupabase(tables)
    _install_env(monkeypatch, supabase)

    monkeypatch.setattr("jupr_app.services.admin_singles_match_service.load_data", _fake_load_data)

    response = TestClient(app).post(
        "/admin/clubs/club/match-uploader/singles",
        headers={"Authorization": "Bearer local"},
        json={
            "date": "2026-05-01",
            "league": "Summer Social",
            "week_tag": "Week 1",
            "t1_p1": 1,
            "t2_p1": 2,
            "score_t1": 11,
            "score_t2": 8,
            "idempotency_key": "test:singles-contract",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "singles_match_uploader"
    assert payload["match_write_committed"] is True
    assert payload["recovery"]["match_log_route"] == "/admin/match-log"
    assert payload["result"]["match_format"] == "singles"
    assert tables["matches"][0]["match_format"] == "singles"
    assert tables["matches"][0]["league"] == "Summer Social"
    assert tables["matches"][0]["week_tag"] == "Week 1"
    assert tables["matches"][0]["t1_p1"] == 1
    assert tables["matches"][0]["t2_p1"] == 2
    assert (
        tables["admin_activity_log"][0]["action_type"]
        == "submit_singles_match_uploader_atomic"
    )


def test_admin_match_uploader_singles_blocks_ties(monkeypatch):
    supabase = AtomicSinglesSupabase(singles_tables())
    _install_env(monkeypatch, supabase)
    monkeypatch.setattr("jupr_app.services.admin_singles_match_service.load_data", _fake_load_data)

    response = TestClient(app).post(
        "/admin/clubs/club/match-uploader/singles",
        headers={"Authorization": "Bearer local"},
        json={
            "t1_p1": 1,
            "t2_p1": 2,
            "score_t1": 9,
            "score_t2": 9,
            "idempotency_key": "test:singles-tie",
        },
    )

    assert response.status_code == 400
    assert "tied" in response.json()["detail"]


def test_admin_match_uploader_unrated_singles_records_without_rating_change(monkeypatch):
    tables = singles_tables()
    supabase = AtomicSinglesSupabase(tables)
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
            "idempotency_key": "test:singles-unrated",
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
    supabase = AtomicSinglesSupabase(tables)
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
            "idempotency_key": "test:singles-baseline",
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


def test_admin_match_uploader_singles_popup_context_omits_week_tag(monkeypatch):
    tables = singles_tables()
    supabase = AtomicSinglesSupabase(tables)
    _install_env(monkeypatch, supabase)
    monkeypatch.setattr(
        "jupr_app.services.admin_singles_match_service.load_data",
        _fake_load_data,
    )
    monkeypatch.setattr(
        "jupr_app.services.admin_singles_match_service.upsert_or_get_active_event",
        lambda _supabase, *, club_id, name: f"event:{club_id}:{name}",
    )

    response = TestClient(app).post(
        "/admin/clubs/club/match-uploader/singles",
        headers={"Authorization": "Bearer local"},
        json={
            "date": "2026-05-02",
            "league": "POPUP",
            "week_tag": "Week 9",
            "match_type": "PopUp",
            "is_popup": True,
            "context_type": "event",
            "context_name": "Saturday Singles Social",
            "t1_p1": 1,
            "t2_p1": 2,
            "score_t1": 11,
            "score_t2": 7,
            "idempotency_key": "test:singles-popup-context",
        },
    )

    assert response.status_code == 200
    stored = tables["matches"][0]
    assert stored["match_format"] == "singles"
    assert stored["league"] == "POPUP"
    assert stored["match_type"] == "PopUp"
    assert stored["week_tag"] == ""
    assert stored["context_type"] == "event"
    assert stored["context_id"] == "event:club:Saturday Singles Social"
