from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def roster_tables() -> dict[str, list[dict]]:
    return {
        "leagues_metadata": [
            {
                "club_id": "club",
                "league_name": "Tuesday Ladder",
                "is_active": True,
                "status": "active",
                "match_format": "doubles",
                "k_factor": 32,
                "min_games": 3,
                "schedule_config": {},
                "court_board_defaults": {},
                "rules_config": {},
                "awards_config": {},
                "event_tags": {},
            }
        ],
        "players": [
            {"club_id": "club", "id": 1, "name": "Alex", "rating": 1400, "active": True},
            {"club_id": "club", "id": 2, "name": "Blair", "rating": 1200, "active": True},
        ],
        "league_ratings": [
            {"club_id": "club", "id": 10, "player_id": 1, "league_name": "Tuesday Ladder", "rating": 1500, "starting_rating": 1400, "wins": 2, "losses": 1, "matches_played": 3, "is_active": True},
        ],
        "admin_activity_log": [],
    }


def _install_env(monkeypatch, supabase):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only-local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr(
        "services.api.admin_league_manager_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_league_manager_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )
    calls: list[dict] = []
    receipts: dict[str, tuple[tuple, dict]] = {}

    def _atomic_batch(client, **kwargs):
        assert client is supabase
        calls.append(dict(kwargs))
        key = str(kwargs["idempotency_key"])
        request = (
            str(kwargs["club_id"]),
            str(kwargs["league_name"]),
            str(kwargs["action"]),
            tuple(int(value) for value in kwargs["player_ids"]),
            kwargs.get("starting_rating"),
        )
        if key in receipts:
            existing_request, existing_receipt = receipts[key]
            if existing_request != request:
                raise RuntimeError("Roster data changed. Reload before another bulk update.")
            return {**existing_receipt, "idempotent": True}

        tables = client.tables
        league = next(
            (
                row
                for row in tables.get("leagues_metadata", [])
                if str(row.get("club_id")) == str(kwargs["club_id"])
                and str(row.get("league_name")) == str(kwargs["league_name"])
            ),
            None,
        )
        if league is None:
            raise ValueError("league not found")
        raw_status = str(league.get("status") or "").strip().lower()
        if raw_status in {"active", "running", "live"}:
            status = "active"
        elif raw_status in {"paused"}:
            status = "paused"
        elif raw_status in {"ended", "completed", "complete", "done"}:
            status = "ended"
        elif raw_status == "archived":
            status = "archived"
        else:
            status = "draft"
        if bool(league.get("is_active", False)) != (status == "active"):
            raise ValueError("League lifecycle state is inconsistent")
        if status not in {"draft", "active"}:
            raise ValueError(
                "League roster is read-only unless the league is draft or active."
            )

        pid = int(kwargs["player_ids"][0])
        player = next(
            (
                row
                for row in tables.get("players", [])
                if str(row.get("club_id")) == str(kwargs["club_id"])
                and int(row.get("id") or 0) == pid
            ),
            None,
        )
        if player is None:
            raise ValueError("Every selected player must belong to this club.")
        rating = next(
            (
                row
                for row in tables.get("league_ratings", [])
                if str(row.get("club_id")) == str(kwargs["club_id"])
                and str(row.get("league_name")) == str(kwargs["league_name"])
                and int(row.get("player_id") or 0) == pid
            ),
            None,
        )
        if kwargs["action"] == "activate":
            if rating is not None:
                if bool(rating.get("is_active", True)) and not rating.get(
                    "inactive_at"
                ):
                    raise ValueError("player is already active in this league")
                rating.update({"is_active": True, "inactive_at": None})
            else:
                start = kwargs.get("starting_rating")
                if start is None:
                    start = player.get("rating") or 1200
                start = float(start)
                if start <= 20:
                    start *= 400
                rating = {
                    "club_id": str(kwargs["club_id"]),
                    "id": max(
                        [
                            int(row.get("id") or 0)
                            for row in tables.get("league_ratings", [])
                        ],
                        default=0,
                    )
                    + 1,
                    "player_id": pid,
                    "league_name": str(kwargs["league_name"]),
                    "rating": start,
                    "starting_rating": start,
                    "wins": 0,
                    "losses": 0,
                    "matches_played": 0,
                    "is_active": True,
                    "inactive_at": None,
                }
                tables.setdefault("league_ratings", []).append(rating)
        else:
            if rating is None or rating.get("is_active") is False:
                raise ValueError("player is not currently active in this league")
            rating.update(
                {
                    "is_active": False,
                    "inactive_at": "2026-08-17T12:00:00+00:00",
                }
            )
        tables.setdefault("admin_activity_log", []).append(
            {
                "action_type": "update_league_manager_roster_membership_batch_admin",
                "flagged_for_review": True,
            }
        )
        receipt = {
            "ok": True,
            "committed": True,
            "updated_count": 1,
            "operation_id": f"fake-operation-{len(receipts) + 1}",
            "idempotent": False,
            "league_ratings": [dict(rating)],
        }
        receipts[key] = (request, dict(receipt))
        return receipt

    monkeypatch.setattr(
        "jupr_app.services.admin_league_manager_roster_service."
        "update_admin_league_manager_roster_batch",
        _atomic_batch,
    )
    return calls


def test_admin_league_manager_roster_activate_contract(monkeypatch):
    tables = roster_tables()
    supabase = FakeSupabase(tables)
    calls = _install_env(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder/roster/2",
        headers={"Authorization": "Bearer local"},
        json={
            "action": "activate",
            "starting_rating": 4.0,
            "idempotency_key": "legacy:test-activate",
            "confirmation_text": "SAVE ROSTER",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "league_manager_roster_membership_update"
    inserted = [row for row in tables["league_ratings"] if row.get("player_id") == 2][0]
    assert inserted["rating"] == 1600
    assert inserted["starting_rating"] == 1600
    assert inserted["is_active"] is True
    assert payload["detail"]["league_roster_count"] == 2
    assert calls[0]["player_ids"] == [2]
    assert calls[0]["idempotency_key"] == "legacy:test-activate"
    assert calls[0]["confirmation_text"] == "SAVE LEAGUE ROSTER BATCH"
    assert (
        tables["admin_activity_log"][0]["action_type"]
        == "update_league_manager_roster_membership_batch_admin"
    )
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True


def test_admin_league_manager_roster_deactivate_contract(monkeypatch):
    tables = roster_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder/roster/1",
        headers={"Authorization": "Bearer local"},
        json={"action": "deactivate", "confirmation_text": "SAVE ROSTER"},
    )

    assert response.status_code == 200
    assert tables["league_ratings"][0]["is_active"] is False
    assert tables["league_ratings"][0]["inactive_at"]
    assert response.json()["detail"]["league_roster_count"] == 0


def test_admin_league_manager_roster_requires_confirmation(monkeypatch):
    supabase = FakeSupabase(roster_tables())
    _install_env(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder/roster/2",
        headers={"Authorization": "Bearer local"},
        json={"action": "activate", "confirmation_text": "SAVE"},
    )

    assert response.status_code == 400
    assert "SAVE ROSTER" in response.json()["detail"]


def test_admin_league_manager_roster_rejects_missing_or_closed_league_without_write(monkeypatch):
    for mode in ("missing", "paused", "closed"):
        tables = roster_tables()
        if mode == "missing":
            tables["leagues_metadata"] = []
        elif mode == "paused":
            tables["leagues_metadata"][0].update({"status": "paused", "is_active": False})
        else:
            tables["leagues_metadata"][0].update({"status": "ended", "is_active": False})
        _install_env(monkeypatch, FakeSupabase(tables))

        response = TestClient(app).patch(
            "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder/roster/2",
            headers={"Authorization": "Bearer local"},
            json={"action": "activate", "starting_rating": 4.0, "confirmation_text": "SAVE ROSTER"},
        )

        assert response.status_code == 400
        assert not [row for row in tables["league_ratings"] if row.get("player_id") == 2]


def test_admin_league_manager_roster_rejects_inconsistent_lifecycle_state(monkeypatch):
    tables = roster_tables()
    tables["leagues_metadata"][0].update({"status": "active", "is_active": False})
    _install_env(monkeypatch, FakeSupabase(tables))

    response = TestClient(app).patch(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder/roster/2",
        headers={"Authorization": "Bearer local"},
        json={"action": "activate", "starting_rating": 4.0, "confirmation_text": "SAVE ROSTER"},
    )

    assert response.status_code == 400
    assert "lifecycle state is inconsistent" in response.json()["detail"]
    assert not [row for row in tables["league_ratings"] if row.get("player_id") == 2]


def test_admin_league_manager_roster_reactivation_preserves_history(monkeypatch):
    tables = roster_tables()
    tables["league_ratings"].append(
        {
            "club_id": "club",
            "id": 2,
            "player_id": 2,
            "league_name": "Tuesday Ladder",
            "rating": 1450,
            "starting_rating": 1400,
            "wins": 5,
            "losses": 4,
            "matches_played": 9,
            "is_active": False,
            "inactive_at": "2026-01-01T00:00:00Z",
        }
    )
    _install_env(monkeypatch, FakeSupabase(tables))

    response = TestClient(app).patch(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder/roster/2",
        headers={"Authorization": "Bearer local"},
        json={"action": "activate", "starting_rating": 7.0, "confirmation_text": "SAVE ROSTER"},
    )

    assert response.status_code == 200
    row = [item for item in tables["league_ratings"] if item.get("player_id") == 2][0]
    assert row["rating"] == 1450
    assert row["starting_rating"] == 1400
    assert row["matches_played"] == 9
    assert row["is_active"] is True
    assert row["inactive_at"] is None


def test_admin_league_manager_roster_atomic_failure_has_no_partial_write(monkeypatch):
    tables = roster_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)
    monkeypatch.setattr(
        "jupr_app.services.admin_league_manager_roster_service."
        "update_admin_league_manager_roster_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("atomic roster batch failed")
        ),
    )

    response = TestClient(app).patch(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder/roster/2",
        headers={"Authorization": "Bearer local"},
        json={"action": "activate", "starting_rating": 4.0, "confirmation_text": "SAVE ROSTER"},
    )

    assert response.status_code == 500
    assert not [row for row in tables["league_ratings"] if row.get("player_id") == 2]


def test_admin_league_manager_roster_same_key_retry_survives_pause(monkeypatch):
    tables = roster_tables()
    supabase = FakeSupabase(tables)
    calls = _install_env(monkeypatch, supabase)
    request = {
        "action": "activate",
        "starting_rating": 4.0,
        "idempotency_key": "legacy:retry-after-pause",
        "confirmation_text": "SAVE ROSTER",
    }
    client = TestClient(app)

    first = client.patch(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder/roster/2",
        headers={"Authorization": "Bearer local"},
        json=request,
    )
    tables["leagues_metadata"][0].update(
        {"status": "paused", "is_active": False}
    )
    second = client.patch(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder/roster/2",
        headers={"Authorization": "Bearer local"},
        json=request,
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["idempotent"] is True
    assert len(calls) == 2


def test_admin_league_manager_roster_api_requires_service_role(monkeypatch):
    tables = roster_tables()
    _install_env(monkeypatch, FakeSupabase(tables))
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    response = TestClient(app).patch(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder/roster/2",
        headers={"Authorization": "Bearer local"},
        json={"action": "activate", "starting_rating": 4.0, "confirmation_text": "SAVE ROSTER"},
    )

    assert response.status_code == 503
    assert "SUPABASE_SERVICE_ROLE_KEY" in response.json()["detail"]
    assert not [row for row in tables["league_ratings"] if row.get("player_id") == 2]
