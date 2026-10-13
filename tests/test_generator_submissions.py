from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from test_public_play_generator_service import Query as BaseQuery, matches
from jupr_app.data import load as data_loader
from jupr_app.services import generator_submission_service as service
from jupr_app.services import direct_match_entry_service as direct
from jupr_app.domain import player_activity
from jupr_app.services.public_play_generator_service import (
    advance_public_play_generator_session, complete_public_play_generator_session, create_public_play_generator_session,
    save_public_play_generator_round, get_public_play_generator_session,
)
from jupr_app.services.public_live_operation_service import PublicLiveConflictError
from jupr_app.domain.recaps.weekly_recap import _compute_stats, _filter_matches
from jupr_app.services.public_player_service import _rating_projection


class Query(BaseQuery):
    def upsert(self, payload, **kwargs):
        return self.insert(payload)

    def is_(self, field, value):
        return self.eq(field, None if value == "null" else value)

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
             "singles_rating": 1300., "singles_replay_baseline": {"rating": 1300., "wins": 0, "losses": 0, "matches_played": 0}, "wins": 0, "losses": 0,
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
                if any(current.get(key) != value for key, value in update["expected"].items()):
                    raise RuntimeError("JUPR_DIRECT_MATCH_PLAYER_STALE: player state changed before commit.")
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


def completed_session(db, *, rating_mode="unrated", play_format="doubles", kind="round_robin", total_rounds=1, player_count=None):
    count = player_count or (8 if play_format != "singles" else 4)
    names = [f"Player {i}" for i in range(1, count + 1)]
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
        finish = advance_public_play_generator_session if number < total_rounds else complete_public_play_generator_session
        session = finish(db, **args, expected_version=scored["session"]["version"],
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


def freeze_approval_clock(monkeypatch, now):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)
    monkeypatch.setattr(service, "datetime", Clock)
    monkeypatch.setattr(player_activity, "datetime", Clock)


@pytest.mark.parametrize("rating_mode,play_format", [
    ("rated", "doubles"),
    ("rated", "singles"), ("rated", "doubles_singles"),
])
def test_morning_approval_keeps_players_active_and_retry_keeps_frozen_time(
    db, monkeypatch, real_approval_data, rating_mode, play_format,
):
    now = datetime(2026, 10, 11, 15, 23, 50, tzinfo=timezone.utc)
    freeze_approval_clock(monkeypatch, now)
    session, token = completed_session(db, rating_mode=rating_mode, play_format=play_format)
    session = submit(db, session, token)
    args = {**review_args(session), "match_date": "2026-10-11"}
    db.interrupt_after_commit = True
    with pytest.raises(direct.DirectMatchRecoveryRequiredError):
        service.review_generator_submission(db, **args)
    committed = copy.deepcopy(db.db["matches"])
    assert committed
    for match in committed:
        assert datetime.fromisoformat(match["date"]).astimezone(timezone.utc) == now
    played_ids = {m[k] for m in committed for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2") if m.get(k) is not None}
    for player in db.db["players"]:
        if player["id"] in played_ids:
            assert player["active"] is True and player["inactive_at"] is None
            assert datetime.fromisoformat(player["last_game_at"]).astimezone(timezone.utc) == now
    freeze_approval_clock(monkeypatch, now + timedelta(hours=8))
    assert service.review_generator_submission(db, **args)["status"] == "approved"
    # Mixed sessions may publish the second format on retry; all games must use
    # the timestamp reserved by the original approval, with no duplicates.
    assert db.db["matches"][:len(committed)] == committed
    assert all(datetime.fromisoformat(m["date"]).astimezone(timezone.utc) == now for m in db.db["matches"])


@pytest.mark.parametrize("now,played_on,expected", [
    ("2026-10-11T20:00:00+00:00", "2026-10-11", "2026-10-11T12:00:00-07:00"),
    ("2026-10-11T15:00:00+00:00", "2026-10-10", "2026-10-10T12:00:00-07:00"),
    ("2026-10-12T02:00:00+00:00", "2026-10-11", "2026-10-11T12:00:00-07:00"),
    ("2026-10-12T07:01:00+00:00", "2026-10-12", "2026-10-12T00:01:00-07:00"),
])
def test_date_only_game_time_uses_club_day(monkeypatch, now, played_on, expected):
    freeze_approval_clock(monkeypatch, datetime.fromisoformat(now))
    assert service._played_at(played_on, ZoneInfo("America/Mazatlan")) == expected


def test_future_club_day_cannot_reserve_an_approval(db, monkeypatch):
    freeze_approval_clock(monkeypatch, datetime(2026, 10, 12, 2, tzinfo=timezone.utc))
    session, token = completed_session(db)
    session = submit(db, session, token)
    db.db.setdefault("pcs_club_sites", [])
    before = copy.deepcopy(db.db)
    with pytest.raises(ValueError, match="cannot be in the future"):
        service.review_generator_submission(db, **{**review_args(session), "match_date": "2026-10-12"})
    assert db.db == before


def title_args(session):
    return {"club_id": "club", "session_key": session["session_key"], "expected_version": session["version"],
            "title": "Saturday round robin", "actor_email": "admin@example.invalid", "actor_role": "administrator"}


@pytest.mark.parametrize("mode", ["public_play_generator", "admin_play_generator"])
@pytest.mark.parametrize("role", ["administrator", "club_owner", "super_admin"])
def test_pending_title_edit_preserves_results_and_publishes_corrected_label(db, mode, role):
    session, token = completed_session(db, total_rounds=3)
    session = submit(db, session, token)
    db.db["live_sessions"][0]["state"]["mode"] = mode
    before = copy.deepcopy(db.db["live_sessions"][0])
    players_before = copy.deepcopy(db.db["players"])
    operations_before = copy.deepcopy(db.db["public_live_operations"])
    result = service.update_generator_submission_title(db, **{
        **title_args(session), "title": "  Saturday\n round   robin  ", "actor_role": role})
    expected = copy.deepcopy(before)
    expected["title"] = expected["state"]["event_name"] = "Saturday round robin"
    expected["state"]["page_state"]["event_name"] = "Saturday round robin"
    expected["state"]["page_state"]["event"]["name"] = "Saturday round robin"
    expected["version"] += 1
    expected["updated_at"] = db.db["live_sessions"][0]["updated_at"]
    assert db.db["live_sessions"][0] == expected
    assert db.db["players"] == players_before
    assert db.db["public_live_operations"] == operations_before
    assert db.db["matches"] == []
    assert result["version"] == before["version"] + 1
    assert result["title"] == result["session"]["title"] == result["session"]["event"]["name"] == "Saturday round robin"
    assert [entry["action_type"] for entry in db.db["admin_activity_log"]][-2:] == [
        "update_generator_submission_title_intent", "update_generator_submission_title"]
    listed = service.list_generator_submissions(db, club_id="club", actor_role=role)["submissions"][0]
    assert listed["title"] == "Saturday round robin" and listed["version"] == result["version"]
    service.review_generator_submission(db, **{**review_args(session), "expected_version": result["version"]})
    assert {match["week_tag"] for match in db.db["matches"]} == {
        "Saturday round robin Round 1", "Saturday round robin Round 2", "Saturday round robin Round 3"}


@pytest.mark.parametrize("override,error", [
    ({"actor_role": "operator"}, PermissionError),
    ({"actor_role": "player"}, PermissionError),
    ({"club_id": "other-club"}, ValueError),
    ({"session_key": "absent"}, ValueError),
    ({"expected_version": 0}, PublicLiveConflictError),
    ({"title": "   \n\t  "}, ValueError),
    ({"title": "<>"}, ValueError),
    ({"title": "X" * 161}, ValueError),
])
def test_invalid_title_edits_leave_session_and_results_unchanged(db, override, error):
    session, token = completed_session(db)
    session = submit(db, session, token)
    before = copy.deepcopy(db.db)
    with pytest.raises(error):
        service.update_generator_submission_title(db, **{**title_args(session), **override})
    assert db.db == before


@pytest.mark.parametrize("locked_state", ["not_submitted", "active", "processing", "approved", "rejected", "pending_operation", "published"])
def test_title_edit_does_not_reopen_or_change_locked_sessions(db, locked_state):
    session, token = completed_session(db)
    session = submit(db, session, token)
    row = db.db["live_sessions"][0]
    if locked_state == "not_submitted":
        del row["state"]["generator_submission"]
    elif locked_state == "active":
        row["status"] = "active"
    elif locked_state == "pending_operation":
        row["pending_operation_key"] = "reserved-change"
    elif locked_state == "published":
        row["state"]["official_publish"] = {"published_match_ids": ["existing-match"]}
    else:
        row["state"]["generator_submission"]["status"] = locked_state
    before = copy.deepcopy(db.db)
    with pytest.raises(PublicLiveConflictError if locked_state == "pending_operation" else ValueError):
        service.update_generator_submission_title(db, **title_args(session))
    assert db.db == before


def test_title_edit_compare_and_swap_preserves_concurrent_approval(db, monkeypatch):
    session, token = completed_session(db)
    session = submit(db, session, token)
    execute = Query.execute
    def concurrent_approval(query):
        if query.name == "live_sessions" and query.update_payload:
            row = query.db["live_sessions"][0]
            row["version"] += 1
            row["state"]["generator_submission"]["status"] = "processing"
            row["pending_operation_key"] = "concurrent-approval"
        return execute(query)
    monkeypatch.setattr(Query, "execute", concurrent_approval)
    with pytest.raises(PublicLiveConflictError):
        service.update_generator_submission_title(db, **title_args(session))
    row = db.db["live_sessions"][0]
    assert row["title"] == row["state"]["page_state"]["event"]["name"] == "Sunday play"
    assert row["state"]["generator_submission"]["status"] == "processing"
    assert row["pending_operation_key"] == "concurrent-approval"


def test_title_edit_requires_audit_intent_before_changing_metadata(db, monkeypatch):
    session, token = completed_session(db)
    session = submit(db, session, token)
    before = copy.deepcopy(db.db["live_sessions"])
    execute = Query.execute
    def failed_audit(query):
        if query.name == "admin_activity_log":
            raise RuntimeError("Audit unavailable")
        return execute(query)
    monkeypatch.setattr(Query, "execute", failed_audit)
    with pytest.raises(RuntimeError, match="Required audit intent"):
        service.update_generator_submission_title(db, **title_args(session))
    assert db.db["live_sessions"] == before


def test_title_edit_http_uses_reviewer_club_write_validation_and_version_guards(db, monkeypatch):
    from services.api import admin_play_generator_routes as routes
    session, token = completed_session(db)
    session = submit(db, session, token)
    role = "operator"
    monkeypatch.setattr(routes, "_resolve_role_or_403", lambda **kw: ("admin@example.invalid", role))
    monkeypatch.setattr(routes, "_require_write_gate", lambda: None)
    app = FastAPI()
    routes.install_admin_play_generator_routes(app, get_supabase_client=lambda: db)
    client = TestClient(app)
    path = f"/admin/clubs/club/play-generators/sessions/{session['session_key']}/submission/title"
    body = {"expected_version": session["version"], "title": "Corrected title"}
    before = copy.deepcopy(db.db["live_sessions"])
    assert client.patch(path, json=body).status_code == 403
    role = "administrator"
    for invalid in ({"title": "  \n "}, {"title": "X" * 161}, {"expected_version": 0}, {"rating_mode": "rated"}):
        assert client.patch(path, json={**body, **invalid}).status_code == 422
    assert client.patch(path.replace("/clubs/club/", "/clubs/other-club/"), json=body).status_code == 400
    assert db.db["live_sessions"] == before
    response = client.patch(path, json={**body, "title": " " * 100 + "X" * 160 + "  "})
    assert response.status_code == 200, response.text
    assert response.json()["title"] == "X" * 160
    assert response.json()["version"] == session["version"] + 1
    assert client.patch(path, json=body).status_code == 409
    def closed_gate():
        raise HTTPException(403, "Writes closed")
    monkeypatch.setattr(routes, "_require_write_gate", closed_gate)
    assert client.patch(path, json={**body, "expected_version": response.json()["version"]}).status_code == 403


def test_title_edit_route_keeps_existing_staging_wave_boundaries():
    from scripts.staging_write_waves import wave_allows_request
    path = "/admin/clubs/club/play-generators/sessions/session/submission/title"
    assert wave_allows_request("open", "PATCH", path)
    assert wave_allows_request("jupr-live", "PATCH", path)
    for wave in ("none", "communications", "public-live", "public-intake-auth"):
        assert not wave_allows_request(wave, "PATCH", path)


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


@pytest.mark.parametrize("play_format,expected_matches", [("doubles_singles", 6), ("doubles", 202)])
def test_complete_session_email_aggregates_batches_and_resumes_failed_queue_without_duplicate_games(
    db, monkeypatch, play_format, expected_matches,
):
    session, token = completed_session(db, rating_mode="rated", play_format=play_format,
        total_rounds=3 if play_format == "doubles_singles" else 1)
    if play_format == "doubles":
        event = db.db["live_sessions"][0]["state"]["page_state"]["event"]
        first = copy.deepcopy(event["rounds"][0])
        event["rounds"] = []
        for number in range(1, 102):
            round_row = copy.deepcopy(first)
            round_row["roundNumber"] = number
            for index, match in enumerate(round_row["matches"]):
                match["id"] = f"round-{number}-match-{index}"
            event["rounds"].append(round_row)
    db.db["player_profile_update_subscriptions"] = [{"id": "subscription-1", "club_id": "club",
        "player_id": 1, "email": "player@example.invalid", "request_status": "active"}]
    session = submit(db, session, token)
    original_queue = service.queue_player_updates
    attempts = []
    def interrupted_queue(**kwargs):
        attempts.append(copy.deepcopy({key: value for key, value in kwargs.items() if key != "supabase"}))
        if len(attempts) == 1:
            return {"mode": "error", "failed": 1}
        return original_queue(**kwargs)
    monkeypatch.setattr(service, "queue_player_updates", interrupted_queue)
    args = review_args(session)
    with pytest.raises(RuntimeError, match="player updates could not be queued"):
        service.review_generator_submission(db, **args)
    assert len(db.db["matches"]) == expected_matches
    assert not db.db.get("player_profile_update_outbox")
    committed_players = copy.deepcopy(db.db["players"])
    rpc_calls = db.rpc_calls

    result = service.review_generator_submission(db, **args)

    assert result["status"] == "approved"
    assert db.rpc_calls == rpc_calls == 2
    assert len(db.db["matches"]) == expected_matches
    assert db.db["players"] == committed_players
    assert attempts[0]["match_ids"] == attempts[1]["match_ids"]
    assert attempts[0]["session_evidence"] == attempts[1]["session_evidence"]
    rows = db.db["player_profile_update_outbox"]
    assert len(rows) == 1
    snapshot = rows[0]["digest_snapshot_json"]
    assert set(snapshot["match_ids"]) == {str(match["id"]) for match in db.db["matches"]}
    assert snapshot["session_evidence"]["player_updates"]
    assert len(snapshot["badge_refs"]) == (2 if play_format == "doubles" else 1)


@pytest.fixture
def real_approval_data(db, monkeypatch):
    # Exercise the actual display loader and rating-plan builder. Only the
    # unrelated badge schema check and post-commit effects use test substitutes.
    monkeypatch.setattr(data_loader, "ensure_badge_schema_preflight", lambda db: None)
    snapshots = []
    def load(db, club_id):
        result = data_loader.load_data(db, club_id)
        snapshots.append(result[0].copy(deep=True))
        return result
    monkeypatch.setattr(service, "load_data", load)
    return snapshots


@pytest.mark.parametrize("last_game_at", [None, "2025-01-01T12:00:00+00:00"])
@pytest.mark.parametrize("rating_mode,play_format", [
    ("rated", "doubles"), ("unrated", "doubles"),
    ("rated", "singles"), ("rated", "doubles_singles"),
])
def test_approval_uses_persisted_activity_instead_of_leaderboard_projection(
    db, real_approval_data, last_game_at, rating_mode, play_format,
):
    for player in db.db["players"]:
        player["last_game_at"] = last_game_at
        player["active"] = True
    before = copy.deepcopy(db.db["players"])
    session, token = completed_session(db, rating_mode=rating_mode, play_format=play_format, total_rounds=3)
    session = submit(db, session, token)

    result = service.review_generator_submission(db, **review_args(session))

    assert result["status"] == "approved"
    assert real_approval_data and not real_approval_data[0]["active"].any()
    assert all(player["active"] for player in before)
    assert len(db.db["matches"]) == 6
    assert (db.db["players"] == before) == (rating_mode == "unrated")
    assert all(receipt["result_json"]["committed"] for receipt in db.db["admin_direct_match_entry_operations"])


def test_sealed_nine_player_approval_resumes_exact_eighteen_games_once(
    db, monkeypatch, real_approval_data,
):
    db.db["players"].append({**copy.deepcopy(db.db["players"][-1]), "id": 9, "name": "Player 9"})
    session, token = completed_session(db, rating_mode="rated", total_rounds=9, player_count=9)
    session = submit(db, session, token)
    args = review_args(session)
    players_before = copy.deepcopy(db.db["players"])
    original_rpc = db.rpc
    def stale_rpc(name, params):
        def execute():
            raise RuntimeError("JUPR_DIRECT_MATCH_PLAYER_STALE: doubles player state changed before commit.")
        return SimpleNamespace(execute=execute)
    monkeypatch.setattr(db, "rpc", stale_rpc)
    with pytest.raises(direct.DirectMatchConflictError):
        service.review_generator_submission(db, **args)
    row = db.db["live_sessions"][0]
    sealed = copy.deepcopy(row["state"]["generator_submission"]["review"])
    reserved_key = row["pending_operation_key"]
    assert row["state"]["generator_submission"]["status"] == "processing"
    assert len(sealed["payloads"]) == 18
    assert not db.db["matches"] and not db.db.get("admin_direct_match_entry_operations")
    assert db.db["players"] == players_before

    monkeypatch.setattr(db, "rpc", original_rpc)
    db.interrupt_after_commit = True
    with pytest.raises(direct.DirectMatchRecoveryRequiredError):
        service.review_generator_submission(db, **args)
    assert len(db.db["matches"]) == 18
    committed_players = copy.deepcopy(db.db["players"])
    assert db.db["live_sessions"][0]["pending_operation_key"] == reserved_key
    result = service.review_generator_submission(db, **args)

    assert result["status"] == "approved"
    assert db.rpc_calls == 1 and len(db.db["matches"]) == 18
    assert len(db.db["admin_direct_match_entry_operations"]) == 1
    assert db.db["players"] == committed_players
    assert all(player["matches_played"] == 8 for player in committed_players)
    assert db.db["live_sessions"][0]["state"]["generator_submission"]["review"] == sealed
    assert db.db["live_sessions"][0]["pending_operation_key"] is None


@pytest.mark.parametrize("failure,status,detail", [
    (RuntimeError("JUPR_DIRECT_MATCH_PLAYER_STALE: doubles player state changed before commit."),
     409, "Match or rating data changed before commit."),
    (TimeoutError("database transport interrupted"), 503, "The server response was interrupted."),
])
def test_review_http_reports_resumable_conflict_and_recovery_errors(
    db, monkeypatch, real_approval_data, failure, status, detail,
):
    from services.api import admin_play_generator_routes as routes
    session, token = completed_session(db, rating_mode="rated")
    session = submit(db, session, token)
    monkeypatch.setattr(routes, "_require_write_gate", lambda *args: None)
    monkeypatch.setattr(routes, "_resolve_role_or_403", lambda **kw: ("admin@example.invalid", "administrator"))
    app = FastAPI()
    routes.install_admin_play_generator_routes(app, get_supabase_client=lambda: db)
    client = TestClient(app)
    path = f"/admin/clubs/club/play-generators/sessions/{session['session_key']}/review"
    body = {key: value for key, value in review_args(session).items()
            if key in {"expected_version", "action", "player_ids", "match_date", "reason"}}
    original_rpc = db.rpc
    def failed_rpc(name, params):
        def execute():
            raise failure
        return SimpleNamespace(execute=execute)
    monkeypatch.setattr(db, "rpc", failed_rpc)

    response = client.post(path, json=body)

    assert response.status_code == status, response.text
    assert response.json()["detail"].startswith(detail)
    assert not db.db["matches"]
    assert db.db["live_sessions"][0]["state"]["generator_submission"]["status"] == "processing"
    monkeypatch.setattr(db, "rpc", original_rpc)
    resumed = client.post(path, json=body)
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["status"] == "approved"
    assert len(db.db["matches"]) == 2


@pytest.mark.parametrize("committed", [False, True])
def test_sealed_review_excludes_later_merged_player_but_can_finalize_existing_receipt(
    db, monkeypatch, real_approval_data, committed,
):
    session, token = completed_session(db, rating_mode="rated")
    session = submit(db, session, token)
    args = review_args(session)
    original_rpc = db.rpc
    if committed:
        db.interrupt_after_commit = True
    else:
        def failed_rpc(name, params):
            def execute():
                raise TimeoutError("response unavailable before commit")
            return SimpleNamespace(execute=execute)
        monkeypatch.setattr(db, "rpc", failed_rpc)
    with pytest.raises(direct.DirectMatchRecoveryRequiredError):
        service.review_generator_submission(db, **args)
    monkeypatch.setattr(db, "rpc", original_rpc)
    db.db["players"][0].update(name="Player 1 (MERGED into 2)", active=False)
    before = copy.deepcopy(db.db["players"])
    previous_rpc_calls = db.rpc_calls

    if committed:
        result = service.review_generator_submission(db, **args)
        assert result["status"] == "approved"
        assert len(db.db["matches"]) == 2
    else:
        with pytest.raises(RuntimeError, match="existing player snapshot for player 1"):
            service.review_generator_submission(db, **args)
        assert not db.db["matches"]
        assert db.db["live_sessions"][0]["state"]["generator_submission"]["status"] == "processing"
    assert db.db["players"] == before
    assert db.rpc_calls == previous_rpc_calls


def test_approval_still_rejects_a_rating_changed_after_its_persisted_snapshot(
    db, monkeypatch, real_approval_data,
):
    session, token = completed_session(db, rating_mode="rated")
    session = submit(db, session, token)
    original_rpc = db.rpc
    def concurrent_rating_change(name, params):
        player = db.db["players"][0]
        expected = next(update["expected"] for update in params["p_player_updates"]
                        if update["player_id"] == player["id"])
        assert expected["rating"] == player["rating"]
        player["rating"] += 20
        return original_rpc(name, params)
    monkeypatch.setattr(db, "rpc", concurrent_rating_change)

    with pytest.raises(direct.DirectMatchConflictError):
        service.review_generator_submission(db, **review_args(session))

    assert not db.db["matches"] and not db.db.get("admin_direct_match_entry_operations")
    assert db.db["players"][0]["rating"] == 1420
    assert all(player["matches_played"] == 0 for player in db.db["players"])
    assert db.db["live_sessions"][0]["state"]["generator_submission"]["status"] == "processing"


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


@pytest.fixture
def public_client(db, monkeypatch):
    from services.api.public_play_generator_routes import install_public_play_generator_routes

    monkeypatch.setenv("JUPR_PUBLIC_LIVE_TOKEN_SECRET", "test-generator-secret-" * 3)
    app = FastAPI()

    def raise_error(exc):
        status = 403 if isinstance(exc, PermissionError) else 409 if isinstance(exc, PublicLiveConflictError) else 400
        raise HTTPException(status, str(exc))

    install_public_play_generator_routes(app, get_club=lambda slug: {"id": "club"}, get_supabase_client=lambda: db,
        public_club_payload=lambda club, slug: club, require_public_writes=lambda: None, require_service_role=lambda: None,
        requester_hash=lambda request: "a" * 64, raise_public_error=raise_error, public_writes_enabled=lambda: True, service_role_configured=lambda: True)
    return TestClient(app)


@pytest.mark.parametrize("rating_mode", ["rated", "unrated"])
@pytest.mark.parametrize("kind", ["round_robin", "ladder"])
def test_public_http_rating_choice_survives_preview_play_and_approval(public_client, db, rating_mode, kind):
    path = "/clubs/test/play-generators"
    setup = {"generator_kind": kind, "play_format": "doubles", "title": "HTTP organizer session",
             "participant_names": [f"Player {i}" for i in range(1, 9)],
             "total_rounds": 2, "court_count": 2, "rating_mode": rating_mode}
    response = public_client.post(f"{path}/preview", json=setup)
    assert response.status_code == 200, response.text
    preview = response.json()["preview"]
    assert preview["ratingMode"] == rating_mode
    other_mode = "unrated" if rating_mode == "rated" else "rated"
    other_preview = public_client.post(f"{path}/preview", json={**setup, "rating_mode": other_mode}).json()["preview"]
    assert other_preview["ratingMode"] == other_mode
    assert other_preview["previewFingerprint"] != preview["previewFingerprint"]

    response = public_client.post(f"{path}/sessions", json={**setup,
        "preview_fingerprint": preview["previewFingerprint"], "idempotency_key": f"http-create-{kind}-{rating_mode}"})
    assert response.status_code == 200, response.text
    created = response.json()
    session = created["session"]
    session_path = f"{path}/sessions/{session['session_key']}"
    assert session["rating_mode"] == rating_mode
    assert session["unrated"] == (rating_mode == "unrated")

    for number in (1, 2):
        response = public_client.patch(f"{session_path}/rounds/{number}/scores", json={
            "edit_token": created["edit_token"], "expected_version": session["version"],
            "idempotency_key": f"http-scores-{number}",
            "scores": [{"match_id": match["id"], "score_a": 11, "score_b": 7}
                       for match in matches(session["event"]["rounds"][number - 1])]})
        assert response.status_code == 200, response.text
        session = response.json()["session"]
        assert session["rating_mode"] == rating_mode
        finish_action = "advance" if number < setup["total_rounds"] else "complete"
        response = public_client.post(f"{session_path}/{finish_action}", json={
            "edit_token": created["edit_token"], "expected_version": session["version"],
            "idempotency_key": f"http-{finish_action}-{number}"})
        assert response.status_code == 200, response.text
        session = response.json()["session"]
        assert session["rating_mode"] == rating_mode

    assert session["status"] == "completed"
    response = public_client.post(f"{session_path}/submit", json={
        "edit_token": created["edit_token"], "expected_version": session["version"],
        "idempotency_key": "http-submit-results", "organizer_name": "Guest organizer", "match_date": "2026-09-20"})
    assert response.status_code == 200, response.text
    session = response.json()["session"]
    assert session["submission"]["rating_mode"] == rating_mode
    assert db.db["matches"] == []
    service.review_generator_submission(db, **review_args(session))
    assert len(db.db["matches"]) == (12 if kind == "ladder" else 4)
    assert {match["rating_scope"] for match in db.db["matches"]} == {"overall_only" if rating_mode == "rated" else "unrated"}
    refreshed = public_client.get(session_path).json()["session"]
    assert refreshed["rating_mode"] == rating_mode
    assert refreshed["submission"]["approved_mode"] == rating_mode


def test_public_http_changed_rating_choice_requires_fresh_preview(public_client, db):
    path = "/clubs/test/play-generators"
    setup = {"generator_kind": "round_robin", "play_format": "singles",
             "participant_names": ["Player 1", "Player 2"], "rating_mode": "unrated"}
    preview = public_client.post(f"{path}/preview", json=setup).json()["preview"]
    response = public_client.post(f"{path}/sessions", json={**setup, "rating_mode": "rated",
        "preview_fingerprint": preview["previewFingerprint"], "idempotency_key": "http-changed-mode"})
    assert response.status_code == 400
    assert "Preview the schedule again" in response.json()["detail"]
    assert db.db["live_sessions"] == []


@pytest.mark.parametrize("endpoint", ["preview", "sessions"])
def test_public_http_rated_games_require_scores(public_client, db, endpoint):
    response = public_client.post(f"/clubs/test/play-generators/{endpoint}", json={
        "generator_kind": "round_robin", "play_format": "singles",
        "participant_names": ["Player 1", "Player 2"], "rating_mode": "rated", "scoring_mode": "unscored",
        "idempotency_key": "http-rated-unscored"})
    assert response.status_code == 400
    assert "require scores" in response.json()["detail"]
    assert db.db["live_sessions"] == []


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


def starting_level_setup(db, rating_mode="rated", play_format="doubles"):
    session, token = completed_session(db, play_format=play_format, rating_mode=rating_mode)
    session = submit(db, session, token)
    participant = session["event"]["participants"][0]
    player_id = int(participant["name"].split()[-1])
    queue = service.list_generator_submissions(db, club_id="club", actor_role="administrator")
    player = next(p for p in queue["players"] if p["id"] == player_id)
    args = dict(club_id="club", session_key=session["session_key"], participant_id=participant["id"],
                player_id=player_id, match_format="doubles", rating_jupr=4.25,
                expected_state_fingerprint=player["ratings"]["doubles"]["state_fingerprint"],
                actor_email="admin@example.invalid", actor_role="administrator")
    return session, args


def test_starting_level_is_used_by_approval_and_survives_retry(db):
    session, args = starting_level_setup(db)
    before = copy.deepcopy(db.db["players"])
    result = service.set_generator_starting_rating(db, **args)
    player = next(p for p in db.db["players"] if p["id"] == args["player_id"])
    assert player["rating"] == player["starting_rating"] == 1700
    assert db.db["matches"] == []
    assert player["matches_played"] == 0
    original = next(p for p in before if p["id"] == player["id"])
    assert {k: v for k, v in player.items() if k.startswith("singles_")} == {
        k: v for k, v in original.items() if k.startswith("singles_")}
    assert result["player"]["ratings"]["doubles"]["rating_jupr"] == 4.25
    assert any(row["action_type"] == "set_generator_starting_rating" for row in db.db["admin_activity_log"])
    with pytest.raises(PublicLiveConflictError):
        service.set_generator_starting_rating(db, **args)
    db.interrupt_after_commit = True
    with pytest.raises(direct.DirectMatchRecoveryRequiredError):
        service.review_generator_submission(db, **review_args(session))
    service.review_generator_submission(db, **review_args(session))
    assert player["starting_rating"] == 1700
    assert player["matches_played"] == 1
    assert abs(player["rating"] - 1700) < 100
    game = next(m for m in db.db["matches"] if player["id"] in [m.get(k) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")])
    position = next(k for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2") if game.get(k) == player["id"])
    assert game[f"{position}_r"] == 1700
    with pytest.raises(ValueError, match="before approving"):
        service.set_generator_starting_rating(db, **args)


@pytest.mark.parametrize("field", ["wins", "losses", "matches_played"])
def test_starting_level_cannot_reset_an_established_player(db, field):
    _, args = starting_level_setup(db)
    db.db["players"][0][field] = 1
    before = copy.deepcopy(db.db["players"])
    with pytest.raises(ValueError, match="already has rated games"):
        service.set_generator_starting_rating(db, **args)
    assert db.db["players"] == before


@pytest.mark.parametrize("override,error", [
    ({"actor_role": "operator"}, PermissionError),
    ({"player_id": 999}, ValueError),
    ({"club_id": "other-club"}, ValueError),
    ({"participant_id": "absent"}, ValueError),
    ({"match_format": "singles"}, ValueError),
    ({"rating_jupr": 0.5}, ValueError),
    ({"rating_jupr": 8}, ValueError),
    ({"rating_jupr": float("nan")}, ValueError),
    ({"rating_jupr": float("inf")}, ValueError),
    ({"expected_state_fingerprint": "0" * 64}, PublicLiveConflictError),
])
def test_invalid_starting_levels_leave_profiles_unchanged(db, override, error):
    _, args = starting_level_setup(db)
    before = copy.deepcopy(db.db["players"])
    with pytest.raises(error):
        service.set_generator_starting_rating(db, **{**args, **override})
    assert db.db["players"] == before


def test_unrated_and_singles_submissions_cannot_change_doubles_baselines(db):
    _, args = starting_level_setup(db, rating_mode="unrated")
    with pytest.raises(ValueError, match="rated results"):
        service.set_generator_starting_rating(db, **args)
    _, singles_args = starting_level_setup(db, play_format="singles")
    with pytest.raises(ValueError, match="scored doubles"):
        service.set_generator_starting_rating(db, **singles_args)


def test_starting_level_compare_and_swap_rejects_concurrent_game(db, monkeypatch):
    _, args = starting_level_setup(db)
    execute = Query.execute
    def concurrent_game(query):
        if query.name == "players" and query.update_payload:
            query.db["players"][0].update(rating=1450, wins=1, matches_played=1)
        return execute(query)
    monkeypatch.setattr(Query, "execute", concurrent_game)
    with pytest.raises(PublicLiveConflictError):
        service.set_generator_starting_rating(db, **args)
    assert db.db["players"][0]["rating"] == 1450
    assert db.db["players"][0]["starting_rating"] == 1400


def test_starting_level_audit_failure_prevents_change(db, monkeypatch):
    _, args = starting_level_setup(db)
    before = copy.deepcopy(db.db["players"])
    execute = Query.execute
    def failed_audit(query):
        if query.name == "admin_activity_log":
            raise RuntimeError("Audit unavailable")
        return execute(query)
    monkeypatch.setattr(Query, "execute", failed_audit)
    with pytest.raises(RuntimeError, match="Required audit intent"):
        service.set_generator_starting_rating(db, **args)
    assert db.db["players"] == before


def test_null_doubles_rating_needs_starting_level_before_approval(db):
    db.db["players"][0].update(rating=None, starting_rating=None)
    session, args = starting_level_setup(db)
    with pytest.raises(ValueError, match="Set the doubles starting level"):
        service.review_generator_submission(db, **review_args(session))
    assert db.db["matches"] == []
    assert not db.db["live_sessions"][0].get("pending_operation_key")
    service.set_generator_starting_rating(db, **args)
    service.review_generator_submission(db, **review_args(session))
    assert db.db["players"][0]["starting_rating"] == 1700


def test_singles_baseline_is_read_only_and_remains_a_valid_initial_seed(db):
    db.db["players"][0]["singles_rating"] = None
    queue = service.list_generator_submissions(db, club_id="club", actor_role="administrator")
    rating = next(p for p in queue["players"] if p["id"] == 1)["ratings"]["singles"]
    assert rating["rating_jupr"] == 3.25
    assert not rating["can_set_starting_rating"]
    session, token = completed_session(db, rating_mode="rated", play_format="singles")
    session = submit(db, session, token)
    service.review_generator_submission(db, **review_args(session))
    assert db.db["players"][0]["singles_replay_baseline"]["rating"] == 1300


def test_merged_profiles_are_not_available_for_starting_levels(db):
    _, args = starting_level_setup(db)
    db.db["players"][0]["name"] = "Old Player (MERGED into Player 2 #2)"
    queue = service.list_generator_submissions(db, club_id="club", actor_role="administrator")
    assert args["player_id"] not in [p["id"] for p in queue["players"]]
    with pytest.raises(ValueError, match="current player profile"):
        service.set_generator_starting_rating(db, **args)


def test_starting_level_http_route_uses_admin_and_write_gates(db, monkeypatch):
    from services.api import admin_play_generator_routes as routes
    session, args = starting_level_setup(db)
    role = "operator"
    monkeypatch.setattr(routes, "_resolve_role_or_403", lambda **kw: ("admin@example.invalid", role))
    monkeypatch.setattr(routes, "_require_write_gate", lambda: None)
    app = FastAPI()
    routes.install_admin_play_generator_routes(app, get_supabase_client=lambda: db)
    client = TestClient(app)
    path = f"/admin/clubs/club/play-generators/sessions/{session['session_key']}/starting-ratings/{args['player_id']}"
    body = {k: args[k] for k in ("participant_id", "match_format", "rating_jupr", "expected_state_fingerprint")}
    assert client.patch(path, json=body).status_code == 403
    role = "administrator"
    assert client.patch(path, json={**body, "rating_jupr": 7.1}).status_code == 422
    assert client.patch(path, json={**body, "match_format": "singles"}).status_code == 422
    response = client.patch(path, json=body)
    assert response.status_code == 200, response.text
    assert response.json()["player"]["ratings"]["doubles"]["rating_jupr"] == 4.25
    assert client.patch(path, json=body).status_code == 409
    def closed_gate():
        raise HTTPException(403, "Writes closed")
    monkeypatch.setattr(routes, "_require_write_gate", closed_gate)
    assert client.patch(path, json=body).status_code == 403


def test_starting_level_route_keeps_existing_staging_wave_boundaries():
    from scripts.staging_write_waves import wave_allows_request
    path = "/admin/clubs/club/play-generators/sessions/session/starting-ratings/1"
    assert wave_allows_request("open", "PATCH", path)
    assert wave_allows_request("jupr-live", "PATCH", path)
    for wave in ("none", "communications", "public-live", "public-intake-auth"):
        assert not wave_allows_request(wave, "PATCH", path)
