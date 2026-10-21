from __future__ import annotations

import copy
import pytest
from types import SimpleNamespace
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from test_generator_submissions import db, completed_session, submit, review_args
from jupr_app.services import generator_submission_service as submissions
from jupr_app.services import generator_session_details_service as details
from jupr_app.services.public_live_operation_service import PublicLiveConflictError


def public_args(session, token):
    return dict(club_id="club", session_key=session["session_key"],
        expected_version=session["version"], edit_token=token,
        idempotency_key="edit-session-details", requester_hash="a" * 64,
        title=" Moneyball   October 20 ", rating_mode="rated")


def admin_args(row):
    return dict(club_id="club", session_key=row["session_key"],
        expected_version=row["updated_at"], title="Moneyball October 20", rating_mode="rated",
        actor_email="admin@example.invalid", actor_role="administrator")


def test_organizer_can_edit_completed_session_without_rebuilding_scores(db):
    session, token = completed_session(db)
    event = db.db["live_sessions"][0]["state"]["page_state"]["event"]
    event["moneyballPrivate"] = {"config": {"name": "Old title", "players": [1, 2]}, "frozen": [3.5, 4.0]}
    before = copy.deepcopy(event)
    args = public_args(session, token)
    saved = details.update_public_generator_details(db, **args)["session"]
    row = db.db["live_sessions"][0]
    after = row["state"]["page_state"]["event"]
    assert saved["title"] == row["state"]["event_name"] == "Moneyball October 20"
    assert saved["rating_mode"] == "rated"
    assert after["rounds"] == before["rounds"]
    assert after["participants"] == before["participants"]
    assert after["moneyballPrivate"]["frozen"] == before["moneyballPrivate"]["frozen"]
    assert after["moneyballPrivate"]["config"]["name"] == saved["title"]
    assert details.update_public_generator_details(db, **args)["idempotent_replay"]
    submitted = submit(db, saved, token)
    assert submitted["submission"]["rating_mode"] == "rated"
    with pytest.raises(ValueError, match="locked"):
        details.update_public_generator_details(db, **{**args, "expected_version": submitted["version"], "idempotency_key": "edit-after-submit"})


def test_details_require_owner_and_fresh_version(db):
    session, token = completed_session(db)
    with pytest.raises(PermissionError):
        details.update_public_generator_details(db, **public_args(session, "wrong-token"))
    args = {**public_args(session, token), "idempotency_key": "stale-session-details", "expected_version": session["version"] - 1}
    with pytest.raises(PublicLiveConflictError):
        details.update_public_generator_details(db, **args)


@pytest.mark.parametrize("title,mode", [(" ", "rated"), ("x" * 161, "rated"), ("Title", "unknown")])
def test_details_validate_values(db, title, mode):
    session, token = completed_session(db)
    with pytest.raises(ValueError):
        details.update_public_generator_details(db, **{**public_args(session, token), "title": title, "rating_mode": mode})


def test_unscored_cannot_be_rated(db):
    session, token = completed_session(db)
    db.db["live_sessions"][0]["state"]["page_state"]["event"]["scoringMode"] = "unscored"
    with pytest.raises(ValueError, match="unscored"):
        details.update_public_generator_details(db, **public_args(session, token))


@pytest.mark.parametrize("mode", ["public_play_generator", "admin_play_generator"])
def test_admin_edit_uses_staff_timestamp_and_keeps_admin_response_contract(db, mode):
    session, _ = completed_session(db)
    row = db.db["live_sessions"][0]
    row["state"]["mode"] = mode
    saved = details.update_admin_generator_details(db, **admin_args(row))
    assert saved["session"]["rating_mode"] == "rated"
    assert saved["session"]["version"] == db.db["live_sessions"][0]["updated_at"]
    assert saved["session"]["event"]["rounds"] == session["event"]["rounds"]


def test_admin_corrects_pending_mode_before_rating_publication(db):
    session, token = completed_session(db)
    session = submit(db, session, token)
    row = db.db["live_sessions"][0]
    saved = details.update_admin_generator_details(db, **{**admin_args(row), "expected_version": row["version"]})
    assert saved["session"]["submission"]["rating_mode"] == "rated"
    approved = submissions.review_generator_submission(db, **{**review_args(session), "expected_version": saved["version"]})
    assert approved["session"]["submission"]["approved_mode"] == "rated"
    assert all(m["rating_scope"] == "overall_only" for m in db.db["matches"])
    with pytest.raises(ValueError, match="locked"):
        details.update_admin_generator_details(db, **admin_args(db.db["live_sessions"][0]))


def test_admin_can_submit_public_hub_session_without_organizer_token(db):
    session, _ = completed_session(db, rating_mode="rated")
    row = db.db["live_sessions"][0]
    row["state"]["round_robin_hub_id"] = "completed-hub"
    args = dict(club_id="club", session_key=session["session_key"], expected_version=row["updated_at"],
        organizer_name="Joe", match_date="2026-10-20", actor_email="admin@example.invalid", actor_role="administrator")
    result = submissions.submit_generator_session(db, **args)
    assert result["session"]["submission"]["rating_mode"] == "rated"
    assert result["session"]["version"] == db.db["live_sessions"][0]["updated_at"]
    assert submissions.submit_generator_session(db, **args)["idempotent_replay"]
    assert not db.db["matches"]


def test_http_admin_submission_uses_verified_role_and_staff_version(db, monkeypatch):
    from services.api import admin_play_generator_routes as routes
    session, _ = completed_session(db)
    row = db.db["live_sessions"][0]
    row["state"]["round_robin_hub_id"] = "completed-hub"
    app = FastAPI()
    role = ["operator"]
    monkeypatch.setattr(routes, "_require_write_gate", lambda: None)
    def authenticate(header):
        if header != "Bearer test-admin":
            raise HTTPException(401, "Sign in")
        return SimpleNamespace(email="admin@example.invalid", user_id="test-admin")
    monkeypatch.setattr(routes, "authenticate_bearer", authenticate)
    monkeypatch.setattr(routes, "resolve_admin_role", lambda **kw: SimpleNamespace(role=role[0]))
    routes.install_admin_play_generator_routes(app, get_supabase_client=lambda: db)
    client = TestClient(app)
    path = f"/admin/clubs/club/play-generators/sessions/{session['session_key']}"
    body = dict(expected_version=row["updated_at"], organizer_name="Joe", match_date="2026-10-20")
    assert client.post(path + "/submit", json=body).status_code == 401
    headers = {"Authorization": "Bearer test-admin"}
    assert client.post(path + "/submit", headers=headers, json={**body, "actor_role": "administrator"}).status_code == 403
    role[0] = "administrator"
    response = client.post(path + "/submit", headers=headers, json=body)
    assert response.status_code == 200, response.text
    assert response.json()["session"]["submission"]["status"] == "pending"
    assert response.json()["session"]["version"] == db.db["live_sessions"][0]["updated_at"]
    assert client.post(path + "/submit", headers=headers, json=body).json()["idempotent_replay"]


def test_http_admin_pending_details_and_stale_conflict(db, monkeypatch):
    from services.api import admin_play_generator_routes as routes
    session, token = completed_session(db)
    session = submit(db, session, token)
    app = FastAPI()
    monkeypatch.setattr(routes, "_require_write_gate", lambda: None)
    monkeypatch.setattr(routes, "_resolve_role_or_403", lambda **kw: ("admin@example.invalid", "administrator"))
    routes.install_admin_play_generator_routes(app, get_supabase_client=lambda: db)
    client = TestClient(app)
    path = f"/admin/clubs/club/play-generators/sessions/{session['session_key']}/details"
    body = dict(expected_version=session["version"], title="Moneyball October 20", rating_mode="rated")
    response = client.patch(path, json=body)
    assert response.status_code == 200, response.text
    assert response.json()["session"]["submission"]["rating_mode"] == "rated"
    assert client.patch(path, json=body).status_code == 409


@pytest.mark.parametrize("role", ["", "operator", "viewer"])
def test_other_staff_cannot_claim_public_organizer_privileges(db, role):
    session, _ = completed_session(db)
    row = db.db["live_sessions"][0]
    with pytest.raises(PermissionError):
        submissions.submit_generator_session(db, club_id="club", session_key=session["session_key"],
            expected_version=row["updated_at"], organizer_name="Other", match_date="2026-10-20",
            actor_email="other@example.invalid", actor_role=role)
    with pytest.raises(PermissionError):
        details.update_admin_generator_details(db, **{**admin_args(row), "actor_role": role})
