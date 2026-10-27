"""Pending review corrections must be durable and precede immutable approval."""
from copy import deepcopy

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from test_generator_submissions import db, completed_session, submit, review_args, Query, real_approval_data
from test_generator_session_details import completed_moneyball
from jupr_app.services import generator_submission_service as service
from jupr_app.services.public_live_operation_service import PublicLiveConflictError
from jupr_app.services.public_play_generator_service import get_public_play_generator_session


def args(session, **overrides):
    match = service._saved_matches(session["event"])[0]
    return dict(club_id="club", session_key=session["session_key"], expected_version=session["version"],
        actor_email="admin@example.invalid", actor_role="administrator", scores=[{
            "round_number": match[0], "match_id": match[1]["id"], "score_a": 5, "score_b": 11}], **overrides)


@pytest.mark.parametrize("mode", ["rated", "unrated"])
@pytest.mark.parametrize("origin", ["public_play_generator", "admin_play_generator"])
@pytest.mark.parametrize("kind,play_format", [("round_robin", "doubles"), ("round_robin", "singles"), ("ladder", "doubles")])
def test_corrections_survive_reload_and_approval_without_changing_other_games(db, real_approval_data, mode, origin, kind, play_format):
    session, token = completed_session(db, rating_mode=mode, kind=kind, play_format=play_format, total_rounds=2)
    session = submit(db, session, token)
    row = db.db["live_sessions"][0]
    row["state"]["mode"] = origin
    before = deepcopy(row)
    players = deepcopy(db.db["players"])
    request = args(session)
    result = service.correct_generator_submission_scores(db, **request)
    item = result["submission"]
    assert item["status"] == "pending"
    assert item["version"] == session["version"] + 1
    assert item["match_count"] == len(service._saved_matches(session["event"]))
    assert db.db["matches"] == [] and db.db["players"] == players
    stored = db.db["live_sessions"][0]
    expected = deepcopy(before["state"])
    target = expected["page_state"]["event"]["rounds"][0]
    match = (target.get("matches") or target["courts"][0]["matches"])[0]
    match.update(scoreA=5, scoreB=11)
    assert stored["state"] == expected
    listed = service.list_generator_submissions(db, club_id="club", actor_role="administrator")["submissions"][0]
    assert listed == item
    audit = db.db["admin_activity_log"][-1]
    assert audit["action_type"] == "correct_generator_submission_scores"
    assert audit["after_json"]["corrections"][0]["before"] == {"scoreA": 11, "scoreB": 7}
    with pytest.raises(PublicLiveConflictError):
        service.review_generator_submission(db, **review_args(session))
    approved = service.review_generator_submission(db, **{**review_args(session), "expected_version": item["version"]})
    assert approved["status"] == "approved"
    assert len(db.db["matches"]) == item["match_count"]
    assert db.db["matches"][0]["score_t1"] == 5 and db.db["matches"][0]["score_t2"] == 11
    assert all(m["score_t1"] == 11 and m["score_t2"] == 7 for m in db.db["matches"][1:])
    assert service.review_generator_submission(db, **{**review_args(session), "expected_version": item["version"]})["idempotent_replay"]
    assert len(db.db["matches"]) == item["match_count"]


@pytest.mark.parametrize("lock", ["active", "processing", "approved", "rejected", "review", "results", "published_at", "published_match_ids", "event_published", "pending_operation"])
def test_locked_results_cannot_be_corrected(db, lock):
    session, token = completed_session(db)
    session = submit(db, session, token)
    row = db.db["live_sessions"][0]
    state = row["state"]
    if lock == "active":
        row["status"] = "active"
    elif lock in {"processing", "approved", "rejected"}:
        state["generator_submission"]["status"] = lock
    elif lock in {"review", "results"}:
        state["generator_submission"][lock] = {"sealed": True}
    elif lock in {"published_at", "published_match_ids"}:
        state.setdefault("official_publish", {})[lock] = ["published"]
    elif lock == "event_published":
        state["page_state"]["event"]["publishedMatchIds"] = ["published"]
    else:
        row["pending_operation_key"] = "pending"
    before = deepcopy(db.db)
    with pytest.raises((ValueError, PublicLiveConflictError)):
        service.correct_generator_submission_scores(db, **args(session))
    assert db.db == before


@pytest.mark.parametrize("bad", [None, "11", 1.5, True, -1, 100, 11])
def test_invalid_scores_are_rejected_atomically(db, bad):
    session, token = completed_session(db)
    session = submit(db, session, token)
    request = args(session)
    second = service._saved_matches(session["event"])[1]
    request["scores"].append({"round_number": second[0], "match_id": second[1]["id"], "score_a": bad, "score_b": 11})
    before = deepcopy(db.db)
    with pytest.raises(ValueError):
        service.correct_generator_submission_scores(db, **request)
    assert db.db == before


@pytest.mark.parametrize("change", ["role", "club", "version", "match", "round", "duplicate", "empty"])
def test_review_identity_and_version_guards(db, change):
    session, token = completed_session(db)
    session = submit(db, session, token)
    request = args(session)
    if change == "role": request["actor_role"] = "operator"
    if change == "club": request["club_id"] = "other-club"
    if change == "version": request["expected_version"] -= 1
    if change == "match": request["scores"][0]["match_id"] = "unknown"
    if change == "round": request["scores"][0]["round_number"] = 100
    if change == "duplicate": request["scores"] *= 2
    if change == "empty": request["scores"] = []
    before = deepcopy(db.db)
    with pytest.raises((ValueError, PermissionError, PublicLiveConflictError)):
        service.correct_generator_submission_scores(db, **request)
    assert db.db == before


@pytest.mark.parametrize("concurrent_field", ["version", "updated_at"])
def test_correction_cannot_overwrite_a_concurrent_approval_or_organizer_edit(db, monkeypatch, concurrent_field):
    session, token = completed_session(db)
    session = submit(db, session, token)
    event = deepcopy(session["event"])
    execute = Query.execute
    def concurrent(query):
        if query.name == "live_sessions" and query.update_payload:
            row = query.db["live_sessions"][0]
            row[concurrent_field] = row["version"] + 1 if concurrent_field == "version" else "changed"
            row["state"]["generator_submission"]["status"] = "processing"
            row["pending_operation_key"] = "concurrent-approval"
        return execute(query)
    monkeypatch.setattr(Query, "execute", concurrent)
    with pytest.raises(PublicLiveConflictError):
        service.correct_generator_submission_scores(db, **args(session))
    row = db.db["live_sessions"][0]
    assert row["state"]["generator_submission"]["status"] == "processing"
    assert row["state"]["page_state"]["event"] == event


def test_lost_response_retry_requires_reload_and_does_not_repeat_correction(db):
    session, token = completed_session(db)
    session = submit(db, session, token)
    request = args(session)
    result = service.correct_generator_submission_scores(db, **request)
    saved = deepcopy(db.db)
    with pytest.raises(PublicLiveConflictError):
        service.correct_generator_submission_scores(db, **request)
    assert db.db == saved
    noop = service.correct_generator_submission_scores(db, **{**request, "expected_version": result["submission"]["version"]})
    assert noop["corrected_games"] == 0 and db.db == saved


def test_required_audit_failure_preserves_scores(db, monkeypatch):
    session, token = completed_session(db)
    session = submit(db, session, token)
    before = deepcopy(db.db["live_sessions"])
    execute = Query.execute
    def fail_audit(query):
        if query.name == "admin_activity_log": raise RuntimeError("Audit unavailable")
        return execute(query)
    monkeypatch.setattr(Query, "execute", fail_audit)
    with pytest.raises(RuntimeError, match="Required audit intent"):
        service.correct_generator_submission_scores(db, **args(session))
    assert db.db["live_sessions"] == before


def test_moneyball_correction_recomputes_results_and_preserves_frozen_setup(db, completed_moneyball):
    session, token = completed_session(db)
    row = db.db["live_sessions"][0]
    row["state"]["page_state"]["event"] = deepcopy(completed_moneyball)
    session = get_public_play_generator_session(db, club_id="club", session_key=session["session_key"])["session"]
    session = submit(db, session, token)
    request = args(session)
    invalid = {**request, "scores": [{**request["scores"][0], "score_a": 10}]}
    with pytest.raises(ValueError, match="win by two"):
        service.correct_generator_submission_scores(db, **invalid)
    service.correct_generator_submission_scores(db, **request)
    result = get_public_play_generator_session(db, club_id="club", session_key=session["session_key"])["session"]
    assert result["standings"] != session["standings"]
    from jupr_app.domain.moneyball_expectations import settlement
    from jupr_app.domain.adaptive_play_engine import generator_event_standings
    corrected = db.db["live_sessions"][0]["state"]["page_state"]["event"]
    assert result["moneyball"] == settlement(corrected, generator_event_standings(corrected))
    assert row["state"]["page_state"]["event"]["moneyballPrivate"] == completed_moneyball["moneyballPrivate"]


def test_http_auth_club_validation_and_write_gates(db, monkeypatch):
    from services.api import admin_play_generator_routes as routes
    session, token = completed_session(db)
    session = submit(db, session, token)
    role = "operator"
    monkeypatch.setattr(routes, "_resolve_role_or_403", lambda **kw: ("admin@example.invalid", role))
    monkeypatch.setattr(routes, "_require_write_gate", lambda *args: None)
    app = FastAPI()
    routes.install_admin_play_generator_routes(app, get_supabase_client=lambda: db)
    client = TestClient(app)
    path = f"/admin/clubs/club/play-generators/sessions/{session['session_key']}/submission/scores"
    request = args(session)
    body = {k: request[k] for k in ("expected_version", "scores")}
    assert client.patch(path, json=body).status_code == 403
    role = "administrator"
    for value in [None, "7", 7.2, True, -1, 100]:
        assert client.patch(path, json={**body, "scores": [{**body["scores"][0], "score_a": value}]}).status_code == 422
    assert client.patch(path, json={**body, "rating_mode": "rated"}).status_code == 422
    assert client.patch(path.replace("/clubs/club/", "/clubs/other-club/"), json=body).status_code == 400
    result = client.patch(path, json=body)
    assert result.status_code == 200, result.text
    assert result.json()["submission"]["matches"][0]["scoreA"] == 5
    assert client.patch(path, json=body).status_code == 409
    def closed(*args): raise HTTPException(403, "Writes closed")
    monkeypatch.setattr(routes, "_require_write_gate", closed)
    assert client.patch(path, json=body).status_code == 403


def test_score_correction_obeys_existing_staging_boundaries():
    from scripts.staging_write_waves import wave_allows_request
    path = "/admin/clubs/club/play-generators/sessions/session/submission/scores"
    assert wave_allows_request("open", "PATCH", path)
    assert wave_allows_request("jupr-live", "PATCH", path)
    assert not wave_allows_request("none", "PATCH", path)
