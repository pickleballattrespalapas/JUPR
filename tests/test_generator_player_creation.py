from __future__ import annotations

import copy
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from test_generator_submissions import db, completed_session, submit, review_args, Query
from jupr_app.services import generator_player_creation_service as creation
from jupr_app.services import generator_submission_service as submissions
from jupr_app.services.admin_player_editor_service import reconcile_admin_player_editor_operation
from jupr_app.services.admin_guarded_write_service import GuardedWriteRecoveryRequired
from jupr_app.services.public_live_operation_service import PublicLiveConflictError

class CreationQuery(Query):
    def matches(self, row):
        expanded = dict(row)
        expanded["before_json->generator_estimate->>session_key"] = (
            (row.get("before_json") or {}).get("generator_estimate") or {}).get("session_key")
        return super().matches(expanded)

    def execute(self):
        if self.payload is not None and self.name in {"players", "admin_guarded_operations"}:
            items = self.payload if isinstance(self.payload, list) else [self.payload]
            maximum = max((int(row.get("id") or 0) for row in self.db.get(self.name, [])), default=0)
            for index, item in enumerate(items):
                item.setdefault("id", maximum + index + 1)
        return super().execute()

@pytest.fixture
def ready(db, monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", "1")
    monkeypatch.setattr(db, "table", lambda name: CreationQuery(db.db, name))
    return db

def new_request(db, *, rating_mode="unrated", play_format="doubles", total_rounds=3):
    session, token = completed_session(db, rating_mode=rating_mode, play_format=play_format, total_rounds=total_rounds)
    session = submit(db, session, token)
    mapping = review_args(session)["player_ids"]
    participant_id = session["event"]["participants"][0]["id"]
    mapping.pop(participant_id)
    return session, {
        "club_id": "club", "session_key": session["session_key"], "expected_version": session["version"],
        "participant_id": participant_id, "name": "New Participant", "player_ids": mapping,
        "idempotency_key": "generator-player:new-participant", "actor_email": "admin@example.invalid", "actor_role": "administrator",
    }

def test_creates_estimated_profile_without_rating_or_existing_player_changes(ready):
    session, request = new_request(ready)
    before = copy.deepcopy(ready.db["players"])
    result = creation.create_generator_submission_player(ready, **request)
    player = ready.db["players"][-1]
    assert result["rating_estimate"]["method"] == "score_estimate"
    assert result["rating_estimate"]["provisional"]
    assert player["rating"] == player["starting_rating"]
    assert player["wins"] == player["losses"] == player["matches_played"] == 0
    assert ready.db["players"][:-1] == before
    assert ready.db["matches"] == []
    queued = submissions.list_generator_submissions(ready, club_id="club", actor_role="administrator")["submissions"][0]
    assert queued["estimated_players"][request["participant_id"]]["player_id"] == player["id"]
    assert queued["participants"][0]["player_id"] == player["id"]

def test_creation_retry_keeps_frozen_estimate_after_rating_and_version_changes(ready):
    _, request = new_request(ready)
    first = creation.create_generator_submission_player(ready, **request)
    ready.db["players"][1]["rating"] = 2400
    ready.db["live_sessions"][0]["version"] += 1
    second = creation.create_generator_submission_player(ready, **request)
    assert second["player"] == first["player"]
    assert second["rating_estimate"] == first["rating_estimate"]
    assert second["idempotent_replay"]
    assert len(ready.db["players"]) == 9
    with pytest.raises(PublicLiveConflictError, match="different request"):
        creation.create_generator_submission_player(ready, **{**request, "name": "Different Name"})

@pytest.mark.parametrize("change", ["stale", "wrong_club", "operator", "duplicate_name", "already_mapped", "unknown_participant"])
def test_preflight_rejections_do_not_create_profile_or_operation(ready, change):
    _, request = new_request(ready)
    if change == "stale": request["expected_version"] -= 1
    if change == "wrong_club": request["club_id"] = "other"
    if change == "operator": request["actor_role"] = "operator"
    if change == "duplicate_name": request["name"] = "player 2"
    if change == "already_mapped": request["player_ids"][request["participant_id"]] = 1
    if change == "unknown_participant": request["participant_id"] = "not-in-session"
    with pytest.raises((ValueError, PermissionError)):
        creation.create_generator_submission_player(ready, **request)
    assert len(ready.db["players"]) == 8
    assert not ready.db.get("admin_guarded_operations")

def test_incomplete_mapping_falls_back_then_final_mapping_refines_jointly(ready):
    session, request = new_request(ready)
    request["player_ids"] = {}
    created = creation.create_generator_submission_player(ready, **request)
    assert created["rating_estimate"]["method"] == "default_estimate"
    mapping = review_args(session)["player_ids"]
    mapping[request["participant_id"]] = created["player"]["id"]
    before = copy.deepcopy(ready.db["players"])
    estimates, plans = creation.plan_generator_estimates(ready, club_id="club", session_key=session["session_key"],
        event=session["event"], player_ids=mapping)
    assert ready.db["players"] == before
    assert estimates[request["participant_id"]]["ratings"]["doubles"]["method"] == "score_estimate"
    assert plans["doubles"][0]["player_id"] == created["player"]["id"]

@pytest.mark.parametrize("rating_mode", ["unrated", "rated"])
def test_approval_estimates_after_reservation_and_retry_does_not_reset_history(ready, monkeypatch, rating_mode):
    session, request = new_request(ready, rating_mode=rating_mode)
    created = creation.create_generator_submission_player(ready, **request)
    args = review_args(session)
    args["player_ids"][request["participant_id"]] = created["player"]["id"]
    originals = copy.deepcopy(ready.db["players"][:-1])
    real_confirm = creation.confirm_starting_ratings
    def confirm(*a, **kw):
        assert ready.db["live_sessions"][0]["state"]["generator_submission"]["status"] == "processing"
        assert ready.db["live_sessions"][0]["pending_operation_key"]
        return real_confirm(*a, **kw)
    monkeypatch.setattr(creation, "confirm_starting_ratings", confirm)
    result = submissions.review_generator_submission(ready, **args)
    assert result["status"] == "approved"
    new_player = ready.db["players"][-1]
    if rating_mode == "unrated":
        assert new_player["matches_played"] == new_player["wins"] == new_player["losses"] == 0
        assert new_player["rating"] == new_player["starting_rating"]
        for old, current in zip(originals, ready.db["players"][:-1]):
            for field in ("rating", "starting_rating", "wins", "losses", "matches_played"):
                assert old[field] == current[field]
    else:
        assert new_player["matches_played"] > 0
    before_retry = copy.deepcopy(ready.db["players"])
    match_count = len(ready.db["matches"])
    assert submissions.review_generator_submission(ready, **args)["idempotent_replay"]
    assert ready.db["players"] == before_retry
    assert len(ready.db["matches"]) == match_count

def test_losing_session_reservation_never_changes_estimates(ready, monkeypatch):
    session, request = new_request(ready)
    request["player_ids"] = {}
    created = creation.create_generator_submission_player(ready, **request)
    args = review_args(session)
    args["player_ids"][request["participant_id"]] = created["player"]["id"]
    before = copy.deepcopy(ready.db["players"])
    def conflict(*args, **kwargs):
        raise ValueError("session changed")
    monkeypatch.setattr(submissions, "_save", conflict)
    with pytest.raises(ValueError, match="session changed"):
        submissions.review_generator_submission(ready, **args)
    assert ready.db["players"] == before
    assert not ready.db["matches"]

def test_manual_starting_edit_is_preserved_and_rated_history_not_reinitialized(ready):
    session, request = new_request(ready)
    created = creation.create_generator_submission_player(ready, **request)
    pid = created["player"]["id"]
    ready.db["players"][-1].update(rating=1800, starting_rating=1800)
    mapping = {**request["player_ids"], request["participant_id"]: pid}
    _, plans = creation.plan_generator_estimates(ready, club_id="club", session_key=session["session_key"], event=session["event"], player_ids=mapping)
    assert not plans
    queued = submissions.list_generator_submissions(ready, club_id="club", actor_role="administrator")["submissions"][0]
    assert request["participant_id"] not in queued["estimated_players"]
    assert queued["participants"][0]["player_id"] == pid
    ready.db["players"][-1].update(matches_played=1, wins=1)
    _, plans = creation.plan_generator_estimates(ready, club_id="club", session_key=session["session_key"], event=session["event"], player_ids=mapping)
    assert not plans

def test_singles_creation_and_reconciliation_preserve_independent_estimate(ready, monkeypatch):
    session, request = new_request(ready, play_format="singles")
    original_execute = CreationQuery.execute
    failed = False
    def timeout_after_insert(self):
        nonlocal failed
        result = original_execute(self)
        if self.name == "players" and self.payload is not None and not failed:
            failed = True
            raise TimeoutError("lost insert response")
        return result
    monkeypatch.setattr(CreationQuery, "execute", timeout_after_insert)
    with pytest.raises(GuardedWriteRecoveryRequired):
        creation.create_generator_submission_player(ready, **request)
    reconciled = reconcile_admin_player_editor_operation(ready, club_id="club", operation_key=request["idempotency_key"],
        confirmation_text="RECONCILE PLAYER OPERATION", actor_email=request["actor_email"], actor_role="administrator")
    assert reconciled["player"]["id"] == 9
    metadata = creation.generator_created_players(ready, club_id="club", session_keys=[session["session_key"]])
    assert metadata[session["session_key"]][request["participant_id"]]["ratings"]["singles"]["method"] == "score_estimate"
    player = ready.db["players"][-1]
    assert player["singles_rating"] == player["singles_replay_baseline"]["rating"]
    assert player["singles_matches_played"] == 0

def test_create_http_contract_uses_admin_and_write_gates(ready, monkeypatch):
    from services.api import admin_play_generator_routes as routes
    session, request = new_request(ready)
    role = "operator"
    monkeypatch.setattr(routes, "_resolve_role_or_403", lambda **kw: ("admin@example.invalid", role))
    monkeypatch.setattr(routes, "_require_write_gate", lambda: None)
    app = FastAPI()
    routes.install_admin_play_generator_routes(app, get_supabase_client=lambda: ready)
    client = TestClient(app)
    path = f"/admin/clubs/club/play-generators/sessions/{session['session_key']}/players"
    body = {k: request[k] for k in ("participant_id", "name", "player_ids", "expected_version", "idempotency_key")}
    assert client.post(path, json=body).status_code == 403
    role = "administrator"
    assert client.post(path, json={**body, "starting_jupr": 3.5}).status_code == 422
    assert client.post(path, json={**body, "expected_version": 1}).status_code == 400
    response = client.post(path, json=body)
    assert response.status_code == 200, response.text
    assert response.json()["rating_estimate"]["provisional"]
    def closed(): raise HTTPException(403, "Writes closed")
    monkeypatch.setattr(routes, "_require_write_gate", closed)
    assert client.post(path, json=body).status_code == 403

def test_create_route_staging_wave_boundaries():
    from scripts.staging_write_waves import wave_allows_request
    path = "/admin/clubs/club/play-generators/sessions/session/players"
    assert wave_allows_request("open", "POST", path)
    assert wave_allows_request("jupr-live", "POST", path)
    for wave in ("none", "communications", "public-live", "public-intake-auth"):
        assert not wave_allows_request(wave, "POST", path)


@pytest.mark.parametrize("current_only", [False, True])
def test_manual_override_is_not_replaced_by_plan_or_hidden_by_queue(ready, current_only):
    session, request = new_request(ready)
    created = creation.create_generator_submission_player(ready, **request)
    player = ready.db["players"][-1]
    player["rating"] = 1800
    if not current_only:
        player["starting_rating"] = 1800
    mapping = {**request["player_ids"], request["participant_id"]: created["player"]["id"]}
    _, plans = creation.plan_generator_estimates(ready, club_id="club", session_key=session["session_key"],
        event=session["event"], player_ids=mapping)
    assert not plans
    assert player["rating"] == 1800
    item = submissions.list_generator_submissions(ready, club_id="club", actor_role="administrator")["submissions"][0]
    assert request["participant_id"] not in item["estimated_players"]

@pytest.mark.parametrize("edited_field", ["rating", "starting_rating"])
def test_manual_rating_change_after_reservation_is_preserved_and_resumable(ready, monkeypatch, edited_field):
    session, request = new_request(ready, rating_mode="rated")
    created = creation.create_generator_submission_player(ready, **request)
    args = review_args(session)
    args["player_ids"][request["participant_id"]] = created["player"]["id"]
    real_save = submissions._save
    changed = False
    def reserve_then_edit(*a, **kw):
        nonlocal changed
        saved = real_save(*a, **kw)
        if kw.get("pending_operation_key") and not changed:
            changed = True
            ready.db["players"][-1][edited_field] = 1800
        return saved
    monkeypatch.setattr(submissions, "_save", reserve_then_edit)
    assert submissions.review_generator_submission(ready, **args)["status"] == "approved"
    assert ready.db["players"][-1]["starting_rating"] == 1800
    before = copy.deepcopy(ready.db["players"])
    assert submissions.review_generator_submission(ready, **args)["idempotent_replay"]
    assert ready.db["players"] == before

def test_estimate_apply_retry_preserves_saved_seed_and_rated_history(ready):
    session, request = new_request(ready)
    created = creation.create_generator_submission_player(ready, **request)
    mapping = {**request["player_ids"], request["participant_id"]: created["player"]["id"]}
    _, plans = creation.plan_generator_estimates(ready, club_id="club", session_key=session["session_key"],
        event=session["event"], player_ids=mapping)
    apply = dict(club_id="club", plans=plans, actor_email=request["actor_email"], actor_role="administrator")
    creation.apply_generator_estimates(ready, **apply)
    before = copy.deepcopy(ready.db["players"])
    creation.apply_generator_estimates(ready, **apply)
    assert ready.db["players"] == before
    ready.db["players"][-1].update(rating=1900, matches_played=2, wins=1, losses=1)
    after_history = copy.deepcopy(ready.db["players"])
    creation.apply_generator_estimates(ready, **apply)
    assert ready.db["players"] == after_history
