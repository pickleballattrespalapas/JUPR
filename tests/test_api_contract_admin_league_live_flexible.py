from __future__ import annotations

from copy import deepcopy

from fastapi.testclient import TestClient

from jupr_app.services.admin_league_live_service import canonical_league_live_game_preview
from services.api.main import app
from tests.test_admin_match_log_service import FakeSupabase
from tests.test_api_contract_admin_league_live import _create
from tests.test_api_contract_admin_league_live_publish import _install_submit_env, _publisher, _tables


def _setup(monkeypatch):
    tables = _tables()
    tables["players"].append({"club_id": "club", "id": 5, "name": "Staff", "rating": 1350})
    supabase = FakeSupabase(tables)
    _install_submit_env(monkeypatch, supabase)
    monkeypatch.setattr(
        "jupr_app.services.admin_league_live_submit_service.submit_admin_match_uploader_batch",
        _publisher(tables),
    )
    client = TestClient(app)
    session = _create(client, total_rounds=3).json()["session"]
    preview = canonical_league_live_game_preview(session)
    matches = []
    for slot in preview["courts"][0]["matches"]:
        matches.append({
            **slot,
            "series_key": slot["row_id"],
            "series_kind": "fixed_games",
            "series_games": 1,
            "game_number": 1,
            "score_t1": 11,
            "score_t2": 8,
            "scheduled_player_ids": [slot[key] for key in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")],
        })
    return tables, client, session, matches, preview


def _publish(client, session, matches, *, unplayed=None, preview=None):
    return client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/submit",
        headers={"Authorization": "Bearer local"},
        json={
            "match_date": "2026-10-18",
            "matches": matches,
            "expected_match_count": 3,
            "expected_updated_at": session["updated_at"],
            "idempotency_key": "flexible-day-publish",
            "confirmation_text": "SUBMIT LEAGUE ROUND",
            "unplayed_games": unplayed or [],
            "preview": preview or {},
        },
    )


def test_actual_staff_substitute_is_published_and_retained_without_rewriting_schedule(monkeypatch):
    tables, client, session, matches, _ = _setup(monkeypatch)
    original = matches[1]["t1_p1"]
    matches[1]["t1_p1"] = 5
    matches[1]["substitution_reason"] = "Player rested for the second game"
    response = _publish(client, session, matches)
    assert response.status_code == 200, response.text
    published = tables["matches"]
    assert sum(row["t1_p1"] == 5 for row in published) == 1
    assert published[1]["scheduled_player_ids"][0] == original
    retained = next(row for row in tables["league_live_sessions"][0]["roster_json"] if row["player_id"] == 5)
    assert retained["player_name"] == "Staff"
    assert retained["status"] == "bench"
    assert retained["bench_reason"] == "not_playing"
    assert all(5 not in [player["player_id"] for player in row["players_json"]] for row in tables["league_live_courts"])
    replay = _publish(client, session, matches)
    assert replay.status_code == 200, replay.text
    assert replay.json()["idempotent_replay"] is True
    assert len(tables["matches"]) == 3


def test_forged_substitution_preview_cannot_change_saved_generated_slot(monkeypatch):
    tables, client, session, matches, preview = _setup(monkeypatch)
    matches[0]["scheduled_player_ids"][0:2] = reversed(matches[0]["scheduled_player_ids"][0:2])
    forged = deepcopy(preview)
    forged["courts"][0]["matches"][0]["t1_p1"] = matches[0]["scheduled_player_ids"][0]
    forged["courts"][0]["matches"][0]["t1_p2"] = matches[0]["scheduled_player_ids"][1]
    response = _publish(client, session, matches, preview=forged)
    assert response.status_code == 400, response.text
    assert not tables["matches"]
    assert not tables["league_live_publish_operations"]


def test_unplayed_slot_is_durable_and_next_round_movement_keeps_partial_results(monkeypatch):
    tables, client, session, matches, _ = _setup(monkeypatch)
    unplayed = [{"series_key": matches[2]["series_key"], "game_number": 1, "court": 1, "reason": "Court stopped due to heat"}]
    response = _publish(client, session, matches[:2], unplayed=unplayed)
    assert response.status_code == 200, response.text
    assert len(tables["matches"]) == 2
    assert tables["league_live_rounds"][0]["preview_json"]["unplayed_games"] == unplayed
    operation = tables["league_live_publish_operations"][0]
    assert operation["request_json"]["unplayed_games"] == unplayed
    current = tables["league_live_sessions"][0]
    plan = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/plan",
        json={"expected_updated_at": current["updated_at"], "matches": tables["league_live_rounds"][0]["matches_json"]},
    )
    assert plan.status_code == 200, plan.text
    assert plan.json()["scored_match_count"] == 2
    moved = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/movement",
        json={"expected_updated_at": current["updated_at"], "expected_operation_key": plan.json()["operation_key"], "confirmation_text": "APPLY LEAGUE MOVEMENT"},
    )
    assert moved.status_code == 200, moved.text
    assert moved.json()["session"]["current_round"] == 2
    replay = _publish(client, session, matches[:2], unplayed=unplayed)
    assert replay.status_code == 200, replay.text
    assert len(tables["matches"]) == 2


def test_attendance_resolves_incoming_players_from_club_and_rejects_cross_club_ids(monkeypatch):
    tables, client, session, matches, _ = _setup(monkeypatch)
    path = f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/plan"
    body = {
        "expected_updated_at": session["updated_at"],
        "matches": matches,
        "roster_change": {
            "action": "attendance", "active_player_ids": [1, 2, 3, 5], "court_sizes": [4],
            "players": [{"player_id": 5, "player_name": "Forged name", "rating": 9999}],
        },
    }
    response = client.post(path, json=body)
    assert response.status_code == 200, response.text
    staff = next(row for row in response.json()["next_roster"] if row["player_id"] == 5)
    assert staff["player_name"] == "Staff"
    assert staff["rating"] == 1350
    withdrawn = next(row for row in response.json()["next_roster"] if row["player_id"] == 4)
    assert withdrawn["status"] == "bench"
    assert withdrawn["bench_reason"] == "not_playing"
    body["roster_change"]["players"][0]["player_id"] = 999
    body["roster_change"]["active_player_ids"] = [1, 2, 3, 999]
    rejected = client.post(path, json=body)
    assert rejected.status_code == 400, rejected.text
    assert "outside this club" in rejected.json()["detail"]


def test_partial_round_retry_retains_unplayed_slots_and_substitute_identity(monkeypatch):
    tables, client, session, matches, _ = _setup(monkeypatch)
    matches[1]["t1_p1"] = 5
    matches[1]["substitution_reason"] = "Staff filled in for one game"
    unplayed = [{"series_key": matches[2]["series_key"], "game_number": 1, "court": 1, "reason": "Heat stopped play"}]

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("Temporary connection failure before any match write")

    monkeypatch.setattr("jupr_app.services.admin_league_live_submit_service.submit_admin_match_uploader_batch", unavailable)
    first = _publish(client, session, matches[:2], unplayed=unplayed)
    assert first.status_code == 503, first.text
    assert tables["league_live_publish_operations"][0]["status"] == "retryable"
    assert not tables["matches"]
    # A non-court session update must rebase the exact retained request safely.
    tables["league_live_sessions"][0]["updated_at"] = "2026-10-18T15:00:00+00:00"
    monkeypatch.setattr("jupr_app.services.admin_league_live_submit_service.submit_admin_match_uploader_batch", _publisher(tables))
    retried = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/retry",
        json={"confirmation_text": "RETRY LEAGUE ROUND"},
    )
    assert retried.status_code == 200, retried.text
    assert len(tables["matches"]) == 2
    assert tables["matches"][1]["t1_p1"] == 5
    assert tables["league_live_rounds"][0]["preview_json"]["unplayed_games"] == unplayed
    assert tables["league_live_publish_operations"][0]["attempt_count"] == 2


def test_empty_unplayed_list_cannot_hide_missing_generated_games(monkeypatch):
    tables, client, session, matches, _ = _setup(monkeypatch)
    response = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/submit",
        json={
            "match_date": "2026-10-18", "matches": matches[:1], "expected_match_count": 1,
            "expected_updated_at": session["updated_at"], "idempotency_key": "missing-slot-test",
            "confirmation_text": "SUBMIT LEAGUE ROUND", "unplayed_games": [],
        },
    )
    assert response.status_code == 400, response.text
    assert not tables["matches"]
