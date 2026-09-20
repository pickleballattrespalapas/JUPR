from __future__ import annotations

import copy
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from test_public_play_generator_service import Query as BaseQuery, matches
from jupr_app.services import generator_submission_service as service
from jupr_app.services import direct_match_entry_service as direct
from jupr_app.services.public_play_generator_service import (
    advance_public_play_generator_session, create_public_play_generator_session,
    save_public_play_generator_round, get_public_play_generator_session,
)
from jupr_app.services.public_live_operation_service import PublicLiveConflictError
from jupr_app.domain.recaps.weekly_recap import _compute_stats, _filter_matches
from jupr_app.services.public_player_service import _rating_projection


class Query(BaseQuery):
    def range(self, start, end):
        self.bounds = (start, end)
        return self

    def matches(self, row):
        expanded = dict(row)
        expanded["state->generator_submission->>status"] = ((row.get("state") or {}).get("generator_submission") or {}).get("status")
        return super().matches(expanded)

    def execute(self):
        if self.name == "live_sessions" and self.update_payload is not None:
            for row in self.db[self.name]:
                if self.matches(row) and row.get("pending_operation_key"):
                    assert self.update_payload.get("pending_operation_key") is None
                    assert self.update_payload.get("last_operation_key") == row["pending_operation_key"]
        response = super().execute()
        data = copy.deepcopy(response.data)
        if hasattr(self, "bounds"):
            data = data[self.bounds[0]:self.bounds[1] + 1]
        return SimpleNamespace(data=data)


class Database:
    def __init__(self):
        self.db = {"live_sessions": [], "public_live_operations": [], "matches": [], "players": [
            {"id": i, "club_id": "club", "name": f"Player {i}", "rating": 1400., "starting_rating": 1400.,
             "singles_rating": 1300., "singles_starting_rating": 1300., "wins": 0, "losses": 0,
             "matches_played": 0, "singles_wins": 0, "singles_losses": 0, "singles_matches_played": 0,
             "active": True, "inactive_at": None, "last_game_at": None, "singles_last_game_at": None}
            for i in range(1, 9)
        ]}
        self.rpc_calls = 0
        self.interrupt_after_commit = False

    def table(self, name):
        return Query(self.db, name)

    def rpc(self, name, params):
        assert name == direct.DIRECT_MATCH_RPC
        def execute():
            self.rpc_calls += 1
            match_ids = []
            # Mirror the deployed RPC's context whitelist and player CAS guard.
            for match in params["p_match_rows"]:
                assert (match.get("context_type") or "") in {"", "event", "league_live_session"}
            for update in params["p_player_updates"]:
                current = next(p for p in self.db["players"] if p["id"] == update["player_id"])
                assert all(current.get(key) == value for key, value in update["expected"].items())
            for match in params["p_match_rows"]:
                match_id = len(self.db["matches"]) + 1
                self.db["matches"].append({**copy.deepcopy(match), "id": match_id})
                match_ids.append(match_id)
            for update in params["p_player_updates"]:
                next(p for p in self.db["players"] if p["id"] == update["player_id"]).update(update["after"])
            result = {"ok": True, "committed": True, "request_fingerprint": params["p_request_fingerprint"],
                      "match_ids": match_ids, "player_updates": params["p_player_updates"],
                      "result_summary": params["p_result_summary"]}
            self.db.setdefault("admin_direct_match_entry_operations", []).append({
                "club_id": params["p_club_id"], "idempotency_key": params["p_idempotency_key"],
                "request_fingerprint": params["p_request_fingerprint"], "match_format": params["p_match_format"],
                "result_json": copy.deepcopy(result)})
            if self.interrupt_after_commit:
                self.interrupt_after_commit = False
                raise TimeoutError("response lost after commit")
            return SimpleNamespace(data=result)
        return SimpleNamespace(execute=execute)


@pytest.fixture
def db(monkeypatch):
    database = Database()
    monkeypatch.setattr(service, "load_data", lambda db, club: (
        pd.DataFrame(copy.deepcopy(db.db["players"])), None, pd.DataFrame(), None,
        pd.DataFrame(), None, None, {p["name"]: p["id"] for p in db.db["players"]}, None, False, None))
    monkeypatch.setattr(direct, "_post_commit_side_effects", lambda *a, **k: ({}, {}, []))
    monkeypatch.setattr("jupr_app.services.interclub_rating_service.reconcile_interclub_for_club", lambda *a, **k: {"status": "skipped"})
    return database


def completed_session(db, *, rating_mode="unrated", play_format="doubles", kind="round_robin", total_rounds=1):
    names = [f"Player {i}" for i in range(1, 9 if play_format != "singles" else 5)]
    created = create_public_play_generator_session(db, club_id="club", generator_kind=kind,
        play_format=play_format, title="Sunday play", participant_names=names,
        participant_player_ids={}, total_rounds=total_rounds, court_count=2,
        doubles_court_count=1 if play_format == "doubles_singles" else 0,
        singles_court_count=1 if play_format == "doubles_singles" else 0,
        preview_fingerprint=None, idempotency_key=f"create-{play_format}-{rating_mode}-{kind}",
        requester_hash="a" * 64, token_secret="t" * 48, rating_mode=rating_mode)
    session = created["session"]
    args = {"club_id": "club", "session_key": session["session_key"], "edit_token": created["edit_token"], "requester_hash": "a" * 64}
    for number in range(1, total_rounds + 1):
        scored = save_public_play_generator_round(db, **args, round_number=number, expected_version=session["version"],
            idempotency_key=f"scores-{session['session_key']}-{number}",
            scores=[{"match_id": m["id"], "score_a": 11, "score_b": 7} for m in matches(session["event"]["rounds"][number - 1])])
        session = advance_public_play_generator_session(db, **args, expected_version=scored["session"]["version"],
            idempotency_key=f"finish-{session['session_key']}-{number}")["session"]
    assert session["status"] == "completed"
    return session, created["edit_token"]


def submit(db, session, token):
    return service.submit_generator_session(db, club_id="club", session_key=session["session_key"],
        expected_version=session["version"], organizer_name="Guest organizer", match_date="2026-09-20",
        edit_token=token, idempotency_key=f"submit-{session['session_key']}", requester_hash="a" * 64)["session"]


def review_args(session):
    return {"club_id": "club", "session_key": session["session_key"], "expected_version": session["version"],
        "action": "approve", "player_ids": {p["id"]: int(p["name"].split()[-1]) for p in session["event"]["participants"]},
        "match_date": "2026-09-20", "reason": "", "actor_email": "admin@example.invalid", "actor_role": "administrator"}


@pytest.mark.parametrize("rating_mode", ["rated", "unrated"])
@pytest.mark.parametrize("play_format,kind", [("singles", "round_robin"), ("doubles", "round_robin"), ("doubles_singles", "round_robin"), ("doubles", "ladder")])
def test_organizer_choice_controls_approved_matches_ratings_stats_and_recap(db, rating_mode, play_format, kind):
    before = copy.deepcopy(db.db["players"])
    session, token = completed_session(db, rating_mode=rating_mode, play_format=play_format, kind=kind)
    session = submit(db, session, token)
    assert session["submission"]["status"] == "pending"
    assert session["submission"]["rating_mode"] == rating_mode
    assert db.db["matches"] == []
    assert db.db["players"] == before
    assert len(service.list_generator_submissions(db, club_id="club", actor_role="administrator")["submissions"]) == 1
    result = service.review_generator_submission(db, **review_args(session))
    assert result["session"]["submission"]["approved_mode"] == rating_mode
    records = db.db["matches"]
    assert records
    assert all(m["rating_scope"] == ("unrated" if rating_mode == "unrated" else "overall_only") for m in records)
    assert (db.db["players"] == before) == (rating_mode == "unrated")
    stats, _, _, _ = _compute_stats(_filter_matches(pd.DataFrame(records), include_tournaments=True), {})
    expected_games = sum(2 if m["match_format"] == "singles" else 4 for m in records)
    assert sum(p["games"] for p in stats.values()) == expected_games
    player_id = records[0]["t1_p1"]
    player_records = [m for m in records if player_id in [m.get(k) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]]
    projection = _rating_projection({"rating_jupr": 3.5}, player_records, player_id=player_id)
    assert sum(f["matches"] for f in projection["formats"]) == len(player_records)
    if rating_mode == "unrated":
        assert all(p["delta_jupr"] == 0 for p in stats.values())
        assert all(p["rating_delta_jupr"] == 0 for p in projection["history"])
    count = len(records)
    assert service.review_generator_submission(db, **review_args(session))["idempotent_replay"]
    assert len(db.db["matches"]) == count
    assert service.list_generator_submissions(db, club_id="club", actor_role="administrator")["submissions"] == []
    public = get_public_play_generator_session(db, club_id="club", session_key=session["session_key"])["session"]
    assert "review" not in public["submission"] and "submitted_by_email" not in public["submission"]


def test_reject_and_permissions_do_not_write_matches(db):
    session, token = completed_session(db)
    with pytest.raises(PermissionError):
        submit(db, session, "wrong-organizer-token")
    session = submit(db, session, token)
    args = review_args(session)
    with pytest.raises(PermissionError):
        service.review_generator_submission(db, **{**args, "actor_role": "operator"})
    with pytest.raises(ValueError):
        service.review_generator_submission(db, **{**args, "club_id": "different-club"})
    with pytest.raises(PublicLiveConflictError):
        service.review_generator_submission(db, **{**args, "expected_version": session["version"] - 1})
    with pytest.raises(ValueError, match="Choose approve"):
        service.review_generator_submission(db, **{**args, "action": "rated"})
    rejected = service.review_generator_submission(db, **{**args, "action": "reject", "reason": "Wrong scores"})
    assert rejected["status"] == "rejected"
    assert db.db["matches"] == []


def test_manual_names_must_map_to_distinct_players_in_same_club(db):
    session, token = completed_session(db)
    session = submit(db, session, token)
    args = review_args(session)
    with pytest.raises(ValueError, match="player ID"):
        service.review_generator_submission(db, **{**args, "player_ids": {}})
    with pytest.raises(ValueError, match="in this club"):
        service.review_generator_submission(db, **{**args, "player_ids": {k: 999 for k in args["player_ids"]}})
    with pytest.raises(ValueError, match="more than once"):
        service.review_generator_submission(db, **{**args, "player_ids": {k: 1 for k in args["player_ids"]}})
    assert db.db["matches"] == []
    assert not db.db["live_sessions"][0].get("pending_operation_key")


def test_interrupted_mixed_approval_resumes_without_duplicate_or_rating_mode_change(db):
    session, token = completed_session(db, rating_mode="rated", play_format="doubles_singles")
    session = submit(db, session, token)
    args = review_args(session)
    db.interrupt_after_commit = True
    with pytest.raises(direct.DirectMatchRecoveryRequiredError):
        service.review_generator_submission(db, **args)
    first_count = len(db.db["matches"])
    assert first_count > 0
    queue = service.list_generator_submissions(db, club_id="club", actor_role="administrator")
    assert queue["submissions"][0]["status"] == "processing"
    with pytest.raises(PublicLiveConflictError):
        service.review_generator_submission(db, **{**args, "action": "reject"})
    result = service.review_generator_submission(db, **args)
    assert result["status"] == "approved"
    assert db.rpc_calls == 2
    assert len({m["context_id"] for m in db.db["matches"]}) == len(db.db["matches"])
    assert not db.db["live_sessions"][0].get("pending_operation_key")


def test_rating_choice_is_in_preview_fingerprint_and_requires_scores(db):
    from jupr_app.domain.adaptive_play_engine import create_generator_preview
    setup = dict(generator_kind="round_robin", play_format="singles", title="Play", participant_names=["A", "B"], total_rounds=1)
    a = create_generator_preview(**setup, rating_mode="rated")
    b = create_generator_preview(**setup, rating_mode="unrated")
    assert a["previewFingerprint"] != b["previewFingerprint"]
    with pytest.raises(ValueError, match="require scores"):
        create_generator_preview(**setup, rating_mode="rated", scoring_mode="unscored")


def test_mixed_approval_refreshes_shared_activity_between_formats(db):
    session, token = completed_session(db, rating_mode="rated", play_format="doubles_singles", total_rounds=3)
    session = submit(db, session, token)
    result = service.review_generator_submission(db, **review_args(session))
    assert result["status"] == "approved"
    singles = {m[k] for m in db.db["matches"] if m["match_format"] == "singles" for k in ("t1_p1", "t2_p1")}
    doubles = {m[k] for m in db.db["matches"] if m["match_format"] == "doubles" for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")}
    assert singles & doubles
    assert len(db.db["matches"]) == 6


def test_public_submission_api_needs_organizer_token_but_no_account(db):
    from services.api.public_play_generator_routes import install_public_play_generator_routes
    session, token = completed_session(db, rating_mode="rated")
    app = FastAPI()
    def raise_error(exc):
        raise HTTPException(403 if isinstance(exc, PermissionError) else 400, str(exc))
    install_public_play_generator_routes(app, get_club=lambda slug: {"id": "club"}, get_supabase_client=lambda: db,
        public_club_payload=lambda club, slug: club, require_public_writes=lambda: None, require_service_role=lambda: None,
        requester_hash=lambda request: "a" * 64, raise_public_error=raise_error, public_writes_enabled=lambda: True, service_role_configured=lambda: True)
    client = TestClient(app)
    path = f"/clubs/test/play-generators/sessions/{session['session_key']}/submit"
    body = {"organizer_name": "Guest", "match_date": "2026-09-20", "edit_token": token,
            "expected_version": session["version"], "idempotency_key": "api-submit-0000001"}
    assert client.post(path, json={**body, "edit_token": "wrong"}).status_code == 403
    response = client.post(path, json=body)
    assert response.status_code == 200
    assert response.json()["session"]["submission"]["rating_mode"] == "rated"
    assert client.post(path, json=body).json()["idempotent_replay"]
    assert db.db["matches"] == []


def test_admin_api_denies_operators_and_cannot_override_organizer_mode(db, monkeypatch):
    from services.api import admin_play_generator_routes as routes
    session, token = completed_session(db, rating_mode="unrated")
    session = submit(db, session, token)
    app = FastAPI()
    role = ["operator"]
    monkeypatch.setattr(routes, "_require_write_gate", lambda: None)
    monkeypatch.setattr(routes, "authenticate_bearer", lambda authorization: SimpleNamespace(email="staff@example.invalid", user_id="test-staff"))
    monkeypatch.setattr(routes, "resolve_admin_role", lambda **kw: SimpleNamespace(role=role[0]))
    routes.install_admin_play_generator_routes(app, get_supabase_client=lambda: db)
    client = TestClient(app)
    path = f"/admin/clubs/club/play-generators/sessions/{session['session_key']}/review"
    body = {k: v for k, v in review_args(session).items() if k not in {"club_id", "session_key", "actor_email", "actor_role"}}
    assert client.post(path, json=body).status_code == 403
    role[0] = "administrator"
    assert client.post(path, json={**body, "rating_mode": "rated"}).status_code == 422
    assert client.post(path, json={**body, "action": "rated"}).status_code == 422
    response = client.post(path, json=body)
    assert response.status_code == 200
    assert response.json()["session"]["submission"]["approved_mode"] == "unrated"


def test_staff_session_submission_preserves_choice_and_seals_edits(db):
    from jupr_app.services.admin_play_generator_service import _persist_event
    session, _ = completed_session(db, rating_mode="rated")
    row = db.db["live_sessions"][0]
    row["state"]["mode"] = "admin_play_generator"
    submitted = service.submit_generator_session(db, club_id="club", session_key=session["session_key"],
        expected_version=row["updated_at"], organizer_name="Staff organizer", match_date="2026-09-20",
        actor_email="staff@example.invalid")
    assert submitted["session"]["submission"]["rating_mode"] == "rated"
    with pytest.raises(ValueError, match="locked"):
        _persist_event(db, before=db.db["live_sessions"][0], event=session["event"], expected_version=submitted["session"]["version"])
