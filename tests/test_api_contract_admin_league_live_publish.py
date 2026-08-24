from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from jupr_app.domain.admin_activity_log import ActivityLogWriteResult
from services.api.main import app
from tests.test_admin_match_log_service import FakeSupabase
from tests.test_api_contract_admin_league_live import (
    COURTS,
    MATCHES,
    _create,
    _install_env,
    _plan,
    league_live_tables,
)


def _tables() -> dict[str, Any]:
    tables: dict[str, Any] = league_live_tables()
    for player in tables["players"]:
        player.setdefault("wins", 0)
        player.setdefault("losses", 0)
        player.setdefault("matches_played", 0)
    tables.update(
        {
            "matches": [],
            "league_live_publish_operations": [],
            "league_live_guest_players": [],
        }
    )
    return tables


def _install_submit_env(monkeypatch, supabase: FakeSupabase) -> None:
    _install_env(monkeypatch, supabase)
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "league-live-submit")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT", "1")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")


def _request(session: dict[str, Any], operation_key: str, *, matches=None, expected_count: int | None = None):
    rows = list(matches or MATCHES)
    return {
        "round_label": "Round 1",
        "match_date": "2026-07-19",
        "matches": rows,
        "expected_match_count": int(expected_count if expected_count is not None else len(rows)),
        "courts": COURTS,
        "expected_updated_at": session["updated_at"],
        "expected_operation_key": operation_key,
        "idempotency_key": operation_key,
        "confirmation_text": "SUBMIT LEAGUE ROUND",
    }


def _publisher(
    tables: dict[str, Any],
    *,
    fail_after: int | None = None,
    calls: list[int] | None = None,
    batch_calls: list[dict[str, Any]] | None = None,
):
    def publish(_supabase, *, club_id: str, matches: list[dict[str, Any]], **_kwargs):
        if calls is not None:
            calls.append(len(matches))
        if batch_calls is not None:
            batch_calls.append(dict(_kwargs))
        for index, row in enumerate(matches, start=1):
            tables["matches"].append(
                {
                    "id": 9000 + len(tables["matches"]),
                    "club_id": str(club_id),
                    "deleted_at": None,
                    **dict(row),
                }
            )
            if fail_after is not None and index >= fail_after:
                raise RuntimeError("simulated interrupted match batch")
        for player in tables["players"]:
            if int(player["id"]) in {int(value) for row in matches for value in (row["t1_p1"], row["t1_p2"], row["t2_p1"], row["t2_p2"])}:
                player["matches_played"] = int(player.get("matches_played") or 0) + len(matches)
                player["rating"] = float(player.get("rating") or 1200) + 4.0
        return {"ok": True, "submitted_count": len(matches)}

    return publish


def test_submit_gate_cannot_enable_in_production(monkeypatch) -> None:
    tables = _tables()
    supabase = FakeSupabase(tables)
    _install_submit_env(monkeypatch, supabase)
    monkeypatch.setenv("JUPR_ENV", "production")

    response = TestClient(app).get("/admin/clubs/club/league-manager/live/status")

    assert response.status_code == 200
    assert response.json()["enabled"] is True
    assert response.json()["submit_enabled"] is False
    assert response.json()["round_submit_endpoint"] is None


def test_all_match_publish_is_one_idempotent_python_operation(monkeypatch) -> None:
    tables = _tables()
    supabase = FakeSupabase(tables)
    _install_submit_env(monkeypatch, supabase)
    calls: list[int] = []
    batch_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "jupr_app.services.admin_league_live_submit_service.submit_admin_match_uploader_batch",
        _publisher(tables, calls=calls, batch_calls=batch_calls),
    )
    client = TestClient(app)
    session = _create(client).json()["session"]
    plan = _plan(client, session).json()
    request = _request(session, plan["operation_key"])

    first = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/submit",
        headers={"Authorization": "Bearer local"},
        json=request,
    )

    assert first.status_code == 200, first.text
    payload = first.json()
    assert payload["publish_operation"]["status"] == "completed"
    assert payload["rating_review"]["status"] == "verified_readback"
    assert len(payload["published_match_ids"]) == 1
    assert len(tables["matches"]) == 1
    assert calls == [1]
    assert batch_calls == [{
        "actor_email": "admin@example.com",
        "actor_role": "club_owner",
        "idempotency_key": f"league-live:{plan['operation_key']}",
        "match_format": "doubles",
        "source": "next_league_live_all_match_publish",
        "allow_league_live_context": True,
    }]
    actions = [row["action_type"] for row in tables["admin_activity_log"]]
    assert "submit_league_live_round_intent_admin" in actions
    assert "complete_league_live_round_publish_admin" in actions
    assert actions.index("submit_league_live_round_intent_admin") < actions.index("complete_league_live_round_publish_admin")
    publish_audits = [
        row
        for row in tables["admin_activity_log"]
        if row["action_type"] in {"submit_league_live_round_intent_admin", "complete_league_live_round_publish_admin"}
    ]
    assert {row["entity_id"] for row in publish_audits} == {payload["publish_operation"]["id"]}
    assert all(payload["publish_operation"]["request_fingerprint"] in row["after_json"]["audit_marker"] for row in publish_audits)

    replay = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/submit",
        headers={"Authorization": "Bearer local"},
        json=request,
    )
    assert replay.status_code == 200, replay.text
    assert replay.json()["idempotent_replay"] is True
    assert len(tables["matches"]) == 1
    assert calls == [1]

    exported = client.get(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/export?kind=matches",
        headers={"Authorization": "Bearer local"},
    )
    assert exported.status_code == 200, exported.text
    assert exported.json()["row_count"] == 1
    assert "league_live_session" in exported.json()["csv_text"]


def test_fixed_series_publishes_each_played_game_as_an_official_match(monkeypatch) -> None:
    tables = _tables()
    tables["leagues_metadata"] = [
        {
            "club_id": "club",
            "league_name": "Tuesday Ladder",
            "rules_config": {
                "competition": {
                    "match_structure": {
                        "kind": "fixed_games",
                        "games": 2,
                        "result_counting": "each_game",
                        "completion": "all_games",
                    }
                }
            },
        }
    ]
    supabase = FakeSupabase(tables)
    _install_submit_env(monkeypatch, supabase)
    calls: list[int] = []
    monkeypatch.setattr(
        "jupr_app.services.admin_league_live_submit_service.submit_admin_match_uploader_batch",
        _publisher(tables, calls=calls),
    )
    client = TestClient(app)
    session = _create(client).json()["session"]
    games = [
        {
            **MATCHES[0],
            "series_key": "court-1-match-1",
            "series_kind": "fixed_games",
            "series_games": 2,
            "game_number": game_number,
        }
        for game_number in (1, 2)
    ]
    plan = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/plan",
        headers={"Authorization": "Bearer local"},
        json={"expected_updated_at": session["updated_at"], "matches": games, "courts": COURTS},
    )
    assert plan.status_code == 200, plan.text

    response = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/submit",
        headers={"Authorization": "Bearer local"},
        json=_request(session, plan.json()["operation_key"], matches=games),
    )

    assert response.status_code == 200, response.text
    assert response.json()["published_match_ids"] == [9000, 9001]
    assert calls == [2]
    assert [row["game_number"] for row in tables["matches"]] == [1, 2]
    assert all(row["series_key"] == "court-1-match-1" for row in tables["matches"])


def test_stale_and_incomplete_requests_fail_before_audit_or_publish(monkeypatch) -> None:
    tables = _tables()
    supabase = FakeSupabase(tables)
    _install_submit_env(monkeypatch, supabase)
    calls: list[int] = []
    monkeypatch.setattr(
        "jupr_app.services.admin_league_live_submit_service.submit_admin_match_uploader_batch",
        _publisher(tables, calls=calls),
    )
    client = TestClient(app)
    session = _create(client).json()["session"]
    plan = _plan(client, session).json()
    baseline_audits = len(tables["admin_activity_log"])

    incomplete = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/submit",
        headers={"Authorization": "Bearer local"},
        json=_request(session, plan["operation_key"], expected_count=2),
    )
    assert incomplete.status_code == 400
    assert "1 of 2" in incomplete.json()["detail"]
    assert len(tables["admin_activity_log"]) == baseline_audits
    assert not tables["matches"]

    tables["league_live_sessions"][0]["updated_at"] = "2026-07-19T23:59:00+00:00"
    stale = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/submit",
        headers={"Authorization": "Bearer local"},
        json=_request(session, plan["operation_key"]),
    )
    assert stale.status_code == 409
    assert "session changed" in stale.json()["detail"].lower()
    assert len(tables["admin_activity_log"]) == baseline_audits
    assert not tables["matches"]
    assert calls == []


def test_missing_order20_schema_fails_closed_before_audit_or_publish(monkeypatch) -> None:
    tables = _tables()
    supabase = FakeSupabase(tables, strict_select_tables={"league_live_rounds"})
    _install_submit_env(monkeypatch, supabase)
    calls: list[int] = []
    monkeypatch.setattr(
        "jupr_app.services.admin_league_live_submit_service.submit_admin_match_uploader_batch",
        _publisher(tables, calls=calls),
    )
    client = TestClient(app)
    session = _create(client).json()["session"]
    plan = _plan(client, session).json()
    tables["league_live_rounds"].append({"id": "legacy-round", "session_id": "other"})
    baseline_audits = len(tables["admin_activity_log"])

    response = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/submit",
        headers={"Authorization": "Bearer local"},
        json=_request(session, plan["operation_key"]),
    )

    assert response.status_code == 503
    assert "schema is not ready" in response.json()["detail"]
    assert len(tables["admin_activity_log"]) == baseline_audits
    assert not tables["matches"]
    assert calls == []


def test_missing_canonical_match_context_schema_fails_closed(monkeypatch) -> None:
    tables = _tables()
    supabase = FakeSupabase(tables, strict_select_tables={"matches"})
    _install_submit_env(monkeypatch, supabase)
    client = TestClient(app)
    session = _create(client).json()["session"]
    plan = _plan(client, session).json()
    tables["matches"].append({"id": 5, "club_id": "club"})
    baseline_audits = len(tables["admin_activity_log"])

    response = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/submit",
        headers={"Authorization": "Bearer local"},
        json=_request(session, plan["operation_key"]),
    )

    assert response.status_code == 503
    assert "schema is not ready" in response.json()["detail"]
    assert len(tables["admin_activity_log"]) == baseline_audits
    assert not tables["league_live_publish_operations"]


def test_attempt_two_prewrite_rejection_recovers_once_with_the_same_operation(
    monkeypatch,
) -> None:
    tables = _tables()
    supabase = FakeSupabase(tables)
    _install_submit_env(monkeypatch, supabase)
    calls: list[str] = []

    def reject_before_write(
        _supabase,
        *,
        matches: list[dict[str, Any]],
        idempotency_key: str,
        **_kwargs,
    ) -> None:
        calls.append(idempotency_key)
        assert all(row["context_type"] == "league_live_session" for row in matches)
        raise ValueError("The match plan was rejected before any data was written.")

    monkeypatch.setattr(
        "jupr_app.services.admin_league_live_submit_service.submit_admin_match_uploader_batch",
        reject_before_write,
    )
    client = TestClient(app)
    session = _create(client).json()["session"]
    plan = _plan(client, session).json()
    request = _request(session, plan["operation_key"])
    endpoint = (
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}"
        "/rounds/1/submit"
    )

    first = client.post(
        endpoint,
        headers={"Authorization": "Bearer local"},
        json=request,
    )

    assert first.status_code == 400, first.text
    assert first.json()["detail"] == (
        "The match plan was rejected before any data was written."
    )
    operation = tables["league_live_publish_operations"][0]
    operation_id = operation["id"]
    assert operation["status"] == "retryable"
    assert operation["attempt_count"] == 1
    assert operation["published_match_ids"] == []
    assert not tables["matches"]

    second = client.post(
        endpoint,
        headers={"Authorization": "Bearer local"},
        json=request,
    )

    assert second.status_code == 400, second.text
    assert tables["league_live_publish_operations"][0]["id"] == operation_id
    assert tables["league_live_publish_operations"][0]["attempt_count"] == 2
    assert tables["league_live_publish_operations"][0]["status"] == "retryable"
    assert not tables["matches"]
    assert calls == [
        f"league-live:{plan['operation_key']}",
        f"league-live:{plan['operation_key']}",
    ]

    publish_calls: list[int] = []
    monkeypatch.setattr(
        "jupr_app.services.admin_league_live_submit_service.submit_admin_match_uploader_batch",
        _publisher(tables, calls=publish_calls),
    )
    recovered = client.post(
        endpoint,
        headers={"Authorization": "Bearer local"},
        json=request,
    )

    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["publish_operation"]["id"] == operation_id
    assert recovered.json()["publish_operation"]["status"] == "completed"
    assert tables["league_live_publish_operations"][0]["attempt_count"] == 3
    assert len(tables["matches"]) == 1
    assert publish_calls == [1]

    replay = client.post(
        endpoint,
        headers={"Authorization": "Bearer local"},
        json=request,
    )
    assert replay.status_code == 200, replay.text
    assert replay.json()["idempotent_replay"] is True
    assert len(tables["matches"]) == 1
    assert publish_calls == [1]


def test_response_loss_after_all_match_inserts_reconciles_before_any_republish(monkeypatch) -> None:
    class SimulatedWorkerLoss(BaseException):
        pass

    tables = _tables()
    supabase = FakeSupabase(tables)
    _install_submit_env(monkeypatch, supabase)
    calls: list[int] = []

    def publish_then_lose_response(_supabase, *, club_id: str, matches: list[dict[str, Any]], **_kwargs):
        calls.append(len(matches))
        for row in matches:
            tables["matches"].append(
                {"id": 9100 + len(tables["matches"]), "club_id": club_id, "deleted_at": None, **dict(row)}
            )
        raise SimulatedWorkerLoss()

    monkeypatch.setattr(
        "jupr_app.services.admin_league_live_submit_service.submit_admin_match_uploader_batch",
        publish_then_lose_response,
    )
    client = TestClient(app)
    session = _create(client).json()["session"]
    plan = _plan(client, session).json()
    request = _request(session, plan["operation_key"])

    with pytest.raises(BaseExceptionGroup, match="unhandled errors"):
        client.post(
            f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/submit",
            headers={"Authorization": "Bearer local"},
            json=request,
        )
    assert tables["league_live_publish_operations"][0]["status"] == "publishing"
    assert len(tables["matches"]) == 1

    def must_not_publish(*_args, **_kwargs):
        raise AssertionError("retry must verify deterministic contexts before calling the uploader")

    monkeypatch.setattr(
        "jupr_app.services.admin_league_live_submit_service.submit_admin_match_uploader_batch",
        must_not_publish,
    )
    recovered = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/submit",
        headers={"Authorization": "Bearer local"},
        json=request,
    )

    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["publish_operation"]["status"] == "completed"
    assert recovered.json()["idempotent_replay"] is True
    assert len(tables["matches"]) == 1
    assert calls == [1]
    assert any(
        row["action_type"] == "recover_league_live_round_publish_response_loss_admin"
        for row in tables["admin_activity_log"]
    )


def test_intent_and_completion_audit_failures_report_correct_mutation_uncertainty(monkeypatch) -> None:
    import jupr_app.services.admin_league_live_submit_service as submit_service

    tables = _tables()
    supabase = FakeSupabase(tables)
    _install_submit_env(monkeypatch, supabase)
    calls: list[int] = []
    monkeypatch.setattr(submit_service, "submit_admin_match_uploader_batch", _publisher(tables, calls=calls))
    client = TestClient(app)
    session = _create(client).json()["session"]
    plan = _plan(client, session).json()
    request = _request(session, plan["operation_key"])
    original_write = submit_service.write_admin_activity_log

    monkeypatch.setattr(
        submit_service,
        "write_admin_activity_log",
        lambda *_args, **_kwargs: ActivityLogWriteResult(ok=False, warning="offline"),
    )
    intent_failure = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/submit",
        headers={"Authorization": "Bearer local"},
        json=request,
    )
    assert intent_failure.status_code == 503
    assert "no new match or session mutation" in intent_failure.json()["detail"].lower()
    assert not tables["league_live_publish_operations"]
    assert not tables["matches"]

    audit_attempts = 0

    def fail_completion(supabase_arg, payload):
        nonlocal audit_attempts
        audit_attempts += 1
        if audit_attempts == 2:
            return ActivityLogWriteResult(ok=False, warning="response lost")
        return original_write(supabase_arg, payload)

    monkeypatch.setattr(submit_service, "write_admin_activity_log", fail_completion)
    completion_failure = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/submit",
        headers={"Authorization": "Bearer local"},
        json=request,
    )
    assert completion_failure.status_code == 503
    assert "may have completed" in completion_failure.json()["detail"].lower()
    assert "do not blindly republish" in completion_failure.json()["detail"].lower()
    assert tables["league_live_publish_operations"][0]["status"] == "completed"
    assert tables["league_live_publish_operations"][0].get("completion_audited_at") is None
    assert len(tables["matches"]) == 1

    monkeypatch.setattr(submit_service, "write_admin_activity_log", original_write)
    recovered = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/submit",
        headers={"Authorization": "Bearer local"},
        json=request,
    )
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["idempotent_replay"] is True
    assert tables["league_live_publish_operations"][0]["completion_audited_at"]
    assert len(tables["matches"]) == 1
    assert calls == [1]


def test_completion_audit_marker_response_loss_is_retryable_and_duplicates_are_recognizable(monkeypatch) -> None:
    import jupr_app.services.admin_league_live_submit_service as submit_service

    tables = _tables()
    supabase = FakeSupabase(tables)
    _install_submit_env(monkeypatch, supabase)
    calls: list[int] = []
    monkeypatch.setattr(submit_service, "submit_admin_match_uploader_batch", _publisher(tables, calls=calls))
    original_update = submit_service._update_publish_operation

    def lose_marker_response(supabase_arg, *, club_id, operation_id, patch):
        if "completion_audited_at" in patch:
            raise submit_service.LeagueLivePersistenceError("simulated marker response loss")
        return original_update(
            supabase_arg,
            club_id=club_id,
            operation_id=operation_id,
            patch=patch,
        )

    monkeypatch.setattr(submit_service, "_update_publish_operation", lose_marker_response)
    client = TestClient(app)
    session = _create(client).json()["session"]
    plan = _plan(client, session).json()
    request = _request(session, plan["operation_key"])

    uncertain = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/submit",
        headers={"Authorization": "Bearer local"},
        json=request,
    )
    assert uncertain.status_code == 503
    assert "completion audit may already exist" in uncertain.json()["detail"].lower()
    assert "do not republish" in uncertain.json()["detail"].lower()
    assert tables["league_live_publish_operations"][0]["status"] == "completed"

    monkeypatch.setattr(submit_service, "_update_publish_operation", original_update)
    recovered = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/submit",
        headers={"Authorization": "Bearer local"},
        json=request,
    )
    assert recovered.status_code == 200, recovered.text
    completion_audits = [
        row for row in tables["admin_activity_log"] if row["action_type"] == "complete_league_live_round_publish_admin"
    ]
    assert len(completion_audits) == 2
    assert len({row["entity_id"] for row in completion_audits}) == 1
    assert len({row["after_json"]["audit_marker"] for row in completion_audits}) == 1
    assert len(tables["matches"]) == 1
    assert calls == [1]


def test_published_matches_reconcile_without_republishing(monkeypatch) -> None:
    tables = _tables()
    supabase = FakeSupabase(tables)
    _install_submit_env(monkeypatch, supabase)
    calls: list[int] = []
    monkeypatch.setattr(
        "jupr_app.services.admin_league_live_submit_service.submit_admin_match_uploader_batch",
        _publisher(tables, calls=calls),
    )
    import jupr_app.services.admin_league_live_submit_service as submit_service

    original_save = submit_service.save_admin_league_live_round
    monkeypatch.setattr(
        submit_service,
        "save_admin_league_live_round",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("snapshot unavailable")),
    )
    client = TestClient(app)
    session = _create(client).json()["session"]
    plan = _plan(client, session).json()

    interrupted = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/submit",
        headers={"Authorization": "Bearer local"},
        json=_request(session, plan["operation_key"]),
    )
    assert interrupted.status_code == 503
    assert tables["league_live_publish_operations"][0]["status"] == "published"
    assert len(tables["matches"]) == 1

    monkeypatch.setattr(submit_service, "save_admin_league_live_round", original_save)
    reconciled = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/reconcile",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "RECONCILE LEAGUE ROUND"},
    )
    assert reconciled.status_code == 200, reconciled.text
    assert reconciled.json()["publish_operation"]["status"] == "completed"
    assert len(tables["matches"]) == 1
    assert calls == [1]


def test_partial_publish_can_only_be_compensated_after_active_contexts_are_gone(monkeypatch) -> None:
    tables = _tables()
    supabase = FakeSupabase(tables)
    _install_submit_env(monkeypatch, supabase)
    second_match = {
        "court": 1,
        "t1_p1": 3,
        "t1_p2": 4,
        "t2_p1": 1,
        "t2_p2": 2,
        "score_t1": 9,
        "score_t2": 11,
    }
    matches = [MATCHES[0], second_match]
    monkeypatch.setattr(
        "jupr_app.services.admin_league_live_submit_service.submit_admin_match_uploader_batch",
        _publisher(tables, fail_after=1),
    )
    client = TestClient(app)
    session = _create(client).json()["session"]
    plan_response = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/plan",
        headers={"Authorization": "Bearer local"},
        json={"expected_updated_at": session["updated_at"], "matches": matches, "courts": COURTS},
    )
    plan = plan_response.json()

    interrupted = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/submit",
        headers={"Authorization": "Bearer local"},
        json=_request(session, plan["operation_key"], matches=matches),
    )
    assert interrupted.status_code == 503
    assert tables["league_live_publish_operations"][0]["status"] == "recovery_required"

    blocked = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/compensate",
        headers={"Authorization": "Bearer local"},
        json={
            "recovery_reference": "match-log-op-123",
            "reason": "Removed partial publish and replayed ratings",
            "confirmation_text": "VERIFY LEAGUE COMPENSATION",
        },
    )
    assert blocked.status_code == 409
    assert "active league live match" in blocked.json()["detail"].lower()

    tables["matches"][0]["deleted_at"] = "2026-07-19T20:00:00+00:00"
    compensated = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/compensate",
        headers={"Authorization": "Bearer local"},
        json={
            "recovery_reference": "match-log-op-123",
            "reason": "Removed partial publish and replayed ratings",
            "confirmation_text": "VERIFY LEAGUE COMPENSATION",
        },
    )
    assert compensated.status_code == 200, compensated.text
    assert compensated.json()["publish_operation"]["status"] == "compensated"


def test_guest_creation_is_audited_idempotent_and_rejects_existing_player_name(monkeypatch) -> None:
    tables = _tables()
    supabase = FakeSupabase(tables)
    _install_submit_env(monkeypatch, supabase)

    def add_guest(**kwargs):
        tables["players"].append(
            {
                "id": 99,
                "club_id": kwargs["club_id"],
                "name": kwargs["name"],
                "rating": float(kwargs["rating_jupr"]) * 400.0,
                "wins": 0,
                "losses": 0,
                "matches_played": 0,
            }
        )
        return True, ""

    monkeypatch.setattr("jupr_app.services.admin_league_live_submit_service.safe_add_player", add_guest)
    client = TestClient(app)
    session = _create(client).json()["session"]
    request = {
        "guest_name": "Guest Taylor",
        "starting_jupr": 3.5,
        "reason": "One-night substitute for illness",
        "expected_updated_at": session["updated_at"],
        "idempotency_key": f"guest:{session['id']}:taylor",
        "confirmation_text": "CREATE LIVE GUEST",
    }
    created = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/guests",
        headers={"Authorization": "Bearer local"},
        json=request,
    )
    assert created.status_code == 200, created.text
    assert created.json()["player"]["rating_jupr"] == 3.5
    assert tables["league_live_guest_players"][0]["status"] == "completed"

    replay = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/guests",
        headers={"Authorization": "Bearer local"},
        json=request,
    )
    assert replay.status_code == 200
    assert replay.json()["idempotent_replay"] is True
    assert len([row for row in tables["players"] if row["name"] == "Guest Taylor"]) == 1

    duplicate = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/guests",
        headers={"Authorization": "Bearer local"},
        json={**request, "guest_name": "Alex", "idempotency_key": f"guest:{session['id']}:alex"},
    )
    assert duplicate.status_code == 409
    assert "already uses this name" in duplicate.json()["detail"]


def test_guest_completion_audit_failure_reports_existing_player_uncertainty_and_recovers(monkeypatch) -> None:
    import jupr_app.services.admin_league_live_submit_service as submit_service

    tables = _tables()
    supabase = FakeSupabase(tables)
    _install_submit_env(monkeypatch, supabase)
    add_calls: list[str] = []

    def add_guest(**kwargs):
        add_calls.append(kwargs["name"])
        tables["players"].append(
            {"id": 100, "club_id": kwargs["club_id"], "name": kwargs["name"], "rating": 1400.0}
        )
        return True, ""

    monkeypatch.setattr(submit_service, "safe_add_player", add_guest)
    original_write = submit_service.write_admin_activity_log
    attempts = 0

    def fail_completion(supabase_arg, payload):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            return ActivityLogWriteResult(ok=False, warning="response lost")
        return original_write(supabase_arg, payload)

    monkeypatch.setattr(submit_service, "write_admin_activity_log", fail_completion)
    client = TestClient(app)
    session = _create(client).json()["session"]
    request = {
        "guest_name": "Guest Morgan",
        "starting_jupr": 3.5,
        "reason": "Emergency substitute for one night",
        "expected_updated_at": session["updated_at"],
        "idempotency_key": f"guest:{session['id']}:morgan",
        "confirmation_text": "CREATE LIVE GUEST",
    }

    failed = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/guests",
        headers={"Authorization": "Bearer local"},
        json=request,
    )
    assert failed.status_code == 503
    assert "may have completed" in failed.json()["detail"].lower()
    assert add_calls == ["Guest Morgan"]
    assert tables["league_live_guest_players"][0]["status"] == "intent"

    monkeypatch.setattr(submit_service, "write_admin_activity_log", original_write)
    recovered = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/guests",
        headers={"Authorization": "Bearer local"},
        json=request,
    )
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["player"]["id"] == 100
    assert tables["league_live_guest_players"][0]["status"] == "completed"
    assert add_calls == ["Guest Morgan"]
    guest_audits = [row for row in tables["admin_activity_log"] if "league_live_guest" in row["action_type"]]
    assert {row["entity_id"] for row in guest_audits} == {tables["league_live_guest_players"][0]["id"]}
