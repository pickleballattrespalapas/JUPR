"""Exercise the complete Live workflow with production flags and no network writes.

HTTP requests stay inside TestClient; Supabase tables are in memory. Authentication
uses the existing local fixture. Only the data loader and atomic match persistence
boundary are substituted, so the real shared publisher's capability checks run.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi.testclient import TestClient

from services.api.main import app
from tests.test_admin_match_log_service import FakeSupabase
from tests.test_api_contract_admin_league_live import _create, _install_env, league_live_tables


AUTH = {"Authorization": "Bearer local"}
LIVE = "/admin/clubs/club/league-manager/live"


def test_five_rounds_complete_with_production_flags_and_standalone_uploader_off(monkeypatch):
    tables = league_live_tables()
    tables.update({
        "matches": [],
        "league_live_publish_operations": [],
        "league_live_guest_players": [],
        "leagues_metadata": [{
            "club_id": "club", "league_name": "Tuesday Ladder", "status": "active",
            "is_active": True, "match_format": "doubles", "rules_config": {},
        }],
    })
    for player in tables["players"]:
        player.update({"active": True, "wins": 0, "losses": 0, "matches_played": 0})
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)
    for name, value in {
        "JUPR_ENV": "production",
        "JUPR_PRODUCTION_WRITE_POLICY": "enabled",
        "JUPR_STAGING_WRITE_WAVE": "none",
        "JUPR_REQUIRE_API_AUDIT_LOG": "1",
        "JUPR_ENABLE_STAGING_NEXT_ADMIN_LEAGUE_MANAGER_WRITES": "0",
        "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT": "1",
        "JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER": "0",
        "JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_PREVIEW": "0",
        "JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_SINGLES": "0",
        "JUPR_EMAIL_MODE": "dry_run",
    }.items():
        monkeypatch.setenv(name, value)

    # Keep the real Live service -> shared uploader -> atomic writer call chain.
    # The test owns only the final database boundary and supplies loader inputs.
    atomic_calls: list[dict[str, Any]] = []

    def load_fixture_data(_supabase, club_id):
        assert _supabase is supabase and club_id == "club"
        names = {player["name"]: player["id"] for player in tables["players"]}
        return (None, None, None, None, None, None, None, names, {}, False, "")

    def persist_atomic_matches(_supabase, *, club_id, matches, **kwargs):
        assert _supabase is supabase and club_id == "club"
        assert kwargs["source"] == "next_league_live_all_match_publish"
        assert kwargs["match_format"] == "doubles"
        assert all(row["context_type"] == "league_live_session" for row in matches)
        assert all(UUID(row["context_id"]).version == 5 for row in matches)
        assert kwargs["idempotency_key"] not in {call["idempotency_key"] for call in atomic_calls}
        atomic_calls.append({"idempotency_key": kwargs["idempotency_key"], "count": len(matches)})
        existing_contexts = {row["context_id"] for row in tables["matches"]}
        assert not existing_contexts.intersection(row["context_id"] for row in matches)
        ids = []
        for row in matches:
            match_id = 9001 + len(tables["matches"])
            ids.append(match_id)
            tables["matches"].append({**row, "id": match_id, "club_id": club_id, "deleted_at": None})
            first_team_wins = row["score_t1"] > row["score_t2"]
            winners = {row["t1_p1"], row["t1_p2"]} if first_team_wins else {row["t2_p1"], row["t2_p2"]}
            for player in tables["players"]:
                player["matches_played"] += 1
                player["wins" if player["id"] in winners else "losses"] += 1
                player["rating"] += 4 if player["id"] in winners else -4
        return {"ok": True, "submitted_count": len(matches), "match_ids": ids, "idempotent_replay": False}

    monkeypatch.setattr("jupr_app.services.admin_match_uploader_service.load_data", load_fixture_data)
    monkeypatch.setattr("jupr_app.services.admin_match_uploader_service.submit_atomic_direct_matches", persist_atomic_matches)

    def request(client, method, path, payload=None, **kwargs):
        response = client.request(method, path, headers=AUTH, json=payload, **kwargs)
        assert response.status_code == 200, response.text
        return response.json()

    with TestClient(app) as client:
        status = request(client, "GET", f"{LIVE}/status")
        assert status["enabled"] is True
        assert status["submit_enabled"] is True
        assert request(client, "GET", "/admin/clubs/club/match-uploader/status")["enabled"] is False
        created = _create(client, total_rounds=5)
        assert created.status_code == 200, created.text
        session = created.json()["session"]
        session_path = f"/admin/clubs/club/league-manager/live-sessions/{session['id']}"

        for number in range(1, 6):
            # Reopen from authoritative state between rounds, just as a reload does.
            detail = request(client, "GET", session_path)
            session = detail["session"]
            assert session["current_round"] == number
            assert session["status"] == "active"
            assert len(detail["rounds"]) == number - 1

            if number == 3:
                for target in ("paused", "active"):
                    session = request(client, "PATCH", f"{session_path}/snapshot", {
                        "status": target,
                        "expected_updated_at": session["updated_at"],
                        "confirmation_text": "SAVE SESSION",
                    })["session"]
                    assert session["status"] == target
                    assert session["current_round"] == number
                resumed = request(client, "GET", "/admin/clubs/club/league-manager/live-sessions", params={
                    "league_name": "Tuesday Ladder", "resumable_only": "true",
                })
                assert [row["id"] for row in resumed["sessions"]] == [session["id"]]

            courts = session["current_court_state_json"]
            preview = request(client, "POST", f"{LIVE}/round-robin/preview", {"courts": courts})
            assert preview["match_count"] == 3
            assert not preview["missing_players"]
            matches = [
                {**row, "score_t1": 11, "score_t2": 6 + index}
                for court in preview["courts"]
                for index, row in enumerate(court["matches"])
            ]
            round_path = f"{session_path}/rounds/{number}"
            publish_request = {
                "round_label": f"Round {number}", "match_date": "2026-10-16",
                "preview": preview, "matches": matches, "expected_match_count": 3,
                "courts": courts, "expected_updated_at": session["updated_at"],
                "idempotency_key": f"production-profile-round-{number}",
                "confirmation_text": "SUBMIT LEAGUE ROUND",
            }
            published = request(client, "POST", f"{round_path}/submit", publish_request)
            assert published["publish_operation"]["status"] == "completed"
            assert published["rating_review"]["status"] == "verified_readback"
            assert published["session"]["current_round"] == number
            assert not published["round"]["operation_key"]
            assert published["round"]["movement_json"] == {}
            assert len(tables["matches"]) == number * 3
            assert len(atomic_calls) == number

            # Response-loss retry must return the stored publication without rating it twice.
            replayed = request(client, "POST", f"{round_path}/submit", publish_request)
            assert replayed["idempotent_replay"] is True
            assert len(atomic_calls) == number
            after_scores = request(client, "GET", session_path)
            session = after_scores["session"]
            scored_round = next(row for row in after_scores["rounds"] if row["round_number"] == number)
            assert len(scored_round["submitted_match_ids"]) == 3
            assert not scored_round["operation_key"]
            plan = request(client, "POST", f"{round_path}/plan", {
                "expected_updated_at": session["updated_at"],
                "matches": scored_round["matches_json"], "courts": courts,
            })
            movement_request = {
                "expected_updated_at": session["updated_at"],
                "expected_operation_key": plan["operation_key"], "courts": courts,
                "confirmation_text": "APPLY LEAGUE MOVEMENT",
            }
            moved = request(client, "POST", f"{round_path}/movement", movement_request)
            session = moved["session"]
            assert session["current_round"] == min(number + 1, 5)
            assert moved["round"]["operation_key"] == plan["operation_key"]
            assert request(client, "POST", f"{round_path}/movement", movement_request)["idempotent_replay"] is True
            assert len(atomic_calls) == number

        completed = request(client, "PATCH", f"{session_path}/snapshot", {
            "status": "complete", "expected_updated_at": session["updated_at"],
            "confirmation_text": "SAVE SESSION",
        })
        assert completed["session"]["status"] == "complete"
        final = request(client, "GET", session_path)
        assert final["session"]["current_round"] == 5
        assert [(row["round_number"], row["status"]) for row in final["rounds"]] == [(n, "submitted") for n in range(1, 6)]
        assert len(final["publish_operations"]) == 5
        assert all(row["status"] == "completed" for row in final["publish_operations"])
        assert all(row["operation_key"] for row in final["rounds"])
        assert len(tables["matches"]) == len({row["context_id"] for row in tables["matches"]}) == 15
        assert len({row["id"] for row in tables["matches"]}) == 15
        assert [call["count"] for call in atomic_calls] == [3] * 5
        assert all(player["matches_played"] == 15 for player in tables["players"])
        assert request(client, "GET", f"{session_path}/export", params={"kind": "matches"})["row_count"] == 15
        assert request(client, "GET", "/admin/clubs/club/league-manager/live-sessions", params={
            "league_name": "Tuesday Ladder", "resumable_only": "true",
        })["sessions"] == []

        # Enabling the Live workflow must not expose the standalone uploader.
        for path, payload in (
            ("round-robin/preview", {"courts": courts}),
            ("batch", {"matches": matches, "idempotency_key": "unrelated-uploader-test", "source": "next_league_live_all_match_publish", "allow_league_live_context": True}),
        ):
            denied = client.post(f"/admin/clubs/club/match-uploader/{path}", headers=AUTH, json=payload)
            assert denied.status_code == 403, denied.text
        assert len(tables["matches"]) == 15
        assert len(atomic_calls) == 5
