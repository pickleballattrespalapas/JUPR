"""Persisted regressions for restoring skipped games and continuing a round robin."""

from copy import deepcopy
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from test_public_play_generator_service import FakeSupabase, matches
from jupr_app.services import admin_play_generator_service as admin
from jupr_app.services import public_play_generator_service as public
from jupr_app.services.public_live_operation_service import PublicLiveConflictError, PublicLiveRecoveryRequiredError


class GeneratorSession:
    def __init__(self, audience, *, rounds=3, scoring_mode="scored"):
        self.audience = audience
        self.db = FakeSupabase()
        self.sequence = 0
        self.actor = dict(actor_email="staff@example.test", actor_role="admin", source="test")
        setup = dict(
            club_id="club", generator_kind="round_robin", play_format="doubles",
            title="Evening round robin", participant_names=[f"Player {i}" for i in range(1, 6)],
            total_rounds=rounds, court_count=1, scoring_mode=scoring_mode,
            rating_mode="unrated", preview_fingerprint=None,
        )
        if audience == "public":
            created = public.create_public_play_generator_session(
                self.db, **setup, participant_player_ids={}, idempotency_key="round-controls-create",
                requester_hash="a" * 64, token_secret="x" * 48,
            )
            self.edit_token = created["edit_token"]
        else:
            created = admin.create_play_generator_session(self.db, **setup, player_ids=[], **self.actor)
        self.session = created["session"]

    @property
    def stored(self):
        return self.db.db["live_sessions"][0]

    def reload(self):
        get = public.get_public_play_generator_session if self.audience == "public" else admin.get_play_generator_session
        self.session = get(self.db, club_id="club", session_key=self.session["session_key"])["session"]
        return self.session

    def call(self, action, **kwargs):
        self.sequence += 1
        names = {
            "skip": "skip_{prefix}play_generator_round",
            "reopen": "reopen_{prefix}play_generator_round",
            "save": "save_{prefix}play_generator_round",
            "played": "mark_{prefix}play_generator_round_played",
            "advance": "advance_{prefix}play_generator_session",
            "complete": "complete_{prefix}play_generator_session",
        }
        args = dict(club_id="club", session_key=self.session["session_key"], expected_version=self.session["version"])
        if self.audience == "public":
            service = public
            args.update(edit_token=self.edit_token, requester_hash="a" * 64,
                        idempotency_key=f"round-controls-{action}-{self.sequence}")
        else:
            service = admin
            args.update(self.actor)
        args.update(kwargs)
        result = getattr(service, names[action].format(prefix="public_" if self.audience == "public" else ""))(self.db, **args)
        self.session = result["session"]
        return result

    def score(self, round_number):
        row = next(row for row in self.session["event"]["rounds"] if row["number"] == round_number)
        return self.call("save", round_number=round_number,
                         scores=[{"match_id": match["id"], "score_a": 11, "score_b": 7} for match in matches(row)])

    def set_legacy_completed(self):
        # A row written by the previous eight-round auto-completion behavior.
        self.stored["status"] = "completed"
        self.stored["completed_at"] = "2026-10-08T19:25:00Z"
        event = self.stored["state"]["page_state"]["event"]
        event.update(status="completed", completedAt="2026-10-08T19:25:00Z")
        self.reload()


@pytest.mark.parametrize("audience", ["admin", "public"])
def test_legacy_completed_session_reopens_round_seven_with_original_games_and_accepts_scores(audience):
    game = GeneratorSession(audience, rounds=8)
    for number in range(1, 9):
        game.call("skip", round_number=number, reason="Looking for a usable matchup")
        if number < 8:
            game.call("advance")
    original = deepcopy(game.session["event"]["rounds"][6])
    game.set_legacy_completed()

    game.call("reopen", round_number=7)
    restored = game.reload()
    round_seven = restored["event"]["rounds"][6]
    assert restored["status"] == restored["event"]["status"] == "active"
    assert game.stored["completed_at"] is None
    assert not restored["event"].get("completedAt")
    assert restored["current_round_number"] == 8
    assert round_seven["status"] == "active"
    assert matches(round_seven) == matches(original)
    assert round_seven["byeParticipantIds"] == original["byeParticipantIds"]

    game.score(7)
    reloaded = game.reload()
    assert reloaded["current_round_number"] == 8
    assert reloaded["event"]["rounds"][6]["status"] == "saved"
    assert reloaded["event"]["rounds"][7]["status"] == "skipped"
    assert [(m["scoreA"], m["scoreB"]) for m in matches(reloaded["event"]["rounds"][6])] == [(11, 7)]


@pytest.mark.parametrize("audience", ["admin", "public"])
@pytest.mark.parametrize("scoring_mode", ["scored", "unscored"])
def test_finishing_reopened_earlier_round_does_not_advance_or_change_current_games(audience, scoring_mode):
    game = GeneratorSession(audience, scoring_mode=scoring_mode)
    for number in (1, 2):
        game.call("skip", round_number=number, reason="Skipped")
        game.call("advance")
    current = deepcopy(game.session["event"]["rounds"][2])
    game.call("reopen", round_number=1)
    if scoring_mode == "scored":
        game.score(1)
    else:
        game.call("played", round_number=1)
    session = game.reload()
    assert session["current_round_number"] == 3
    assert session["event"]["rounds"][2] == current
    assert session["event"]["rounds"][0]["status"] == ("saved" if scoring_mode == "scored" else "played")


@pytest.mark.parametrize("audience", ["admin", "public"])
@pytest.mark.parametrize("scoring_mode", ["scored", "unscored"])
def test_round_robin_continues_after_initial_plan_until_explicitly_completed(audience, scoring_mode):
    game = GeneratorSession(audience, rounds=1, scoring_mode=scoring_mode)
    for number in range(1, 4):
        if scoring_mode == "scored":
            game.score(number)
            game.call("advance")
        else:
            game.call("played", round_number=number)
        session = game.reload()
        assert session["current_round_number"] == number + 1
        assert session["status"] == session["event"]["status"] == "active"
        assert session["event"]["rounds"][number]["status"] == "active"
        assert matches(session["event"]["rounds"][number])
    game.call("skip", round_number=4, reason="Done for the evening")
    game.call("complete")
    assert game.reload()["status"] == "completed"


@pytest.mark.parametrize("audience", ["admin", "public"])
def test_advancing_a_completed_legacy_round_robin_resumes_both_persisted_statuses(audience):
    game = GeneratorSession(audience, rounds=1)
    game.score(1)
    game.set_legacy_completed()
    game.call("advance")
    session = game.reload()
    assert session["current_round_number"] == 2
    assert session["status"] == session["event"]["status"] == "active"
    assert game.stored["completed_at"] is None
    assert not session["event"].get("completedAt")


@pytest.mark.parametrize("audience", ["admin", "public"])
def test_unscored_session_can_finish_after_auto_advance_without_leaving_unplayed_games(audience):
    game = GeneratorSession(audience, rounds=1, scoring_mode="unscored")
    game.call("played", round_number=1)
    assert game.session["current_round_number"] == 2
    game.call("complete")
    session = game.reload()
    assert session["status"] == session["event"]["status"] == "completed"
    assert session["event"]["rounds"][0]["status"] == "played"
    assert session["event"]["rounds"][1]["status"] == "skipped"


@pytest.mark.parametrize("audience", ["admin", "public"])
def test_completion_waits_for_reopened_games_even_after_current_round_is_finished(audience):
    game = GeneratorSession(audience, rounds=2)
    game.call("skip", round_number=1, reason="Skipped")
    game.call("advance")
    game.call("reopen", round_number=1)
    game.score(2)
    before = deepcopy(game.stored)
    with pytest.raises(ValueError):
        game.call("complete")
    assert game.stored == before
    game.score(1)
    game.call("complete")
    assert game.reload()["status"] == "completed"


@pytest.mark.parametrize("audience", ["admin", "public"])
def test_reopen_rejects_stale_versions_and_submitted_results_without_changing_session(audience):
    game = GeneratorSession(audience, rounds=2)
    game.call("skip", round_number=1, reason="Skipped")
    game.call("advance")
    before = deepcopy(game.stored)
    stale = game.session["version"] - 1 if audience == "public" else "stale-version"
    with pytest.raises((ValueError, PublicLiveConflictError), match="changed"):
        game.call("reopen", round_number=1, expected_version=stale)
    assert game.stored == before

    game.stored["state"]["generator_submission"] = {"status": "pending"}
    game.set_legacy_completed()
    before = deepcopy(game.stored)
    with pytest.raises(ValueError, match="[Ss]ubmitted|locked|review"):
        game.call("reopen", round_number=1)
    assert game.stored == before


def test_public_reopen_requires_organizer_token_and_replays_without_duplicate_mutation():
    game = GeneratorSession("public", rounds=2)
    game.call("skip", round_number=1, reason="Skipped")
    game.call("advance")
    before = deepcopy(game.stored)
    with pytest.raises(PermissionError):
        game.call("reopen", round_number=1, edit_token="wrong-token")
    assert game.stored == before
    version = game.session["version"]
    first = game.call("reopen", round_number=1, expected_version=version, idempotency_key="reopen-replay-00001")
    replay = game.call("reopen", round_number=1, expected_version=version, idempotency_key="reopen-replay-00001")
    assert replay["idempotent_replay"] is True
    assert replay["session"]["version"] == first["session"]["version"] == version + 1


@pytest.mark.parametrize("acknowledgement_lost", [False, True])
def test_public_complete_retries_after_session_is_completed_without_writing_again(monkeypatch, acknowledgement_lost):
    game = GeneratorSession("public", rounds=1)
    game.score(1)
    version = game.session["version"]
    request = {"expected_version": version, "idempotency_key": "complete-replay-00001"}
    update_operation = public.update_public_live_operation
    failed = False

    def update_with_lost_acknowledgement(*args, **kwargs):
        nonlocal failed
        if acknowledgement_lost and not failed and kwargs.get("status") == "completed":
            failed = True
            raise PublicLiveRecoveryRequiredError("Simulated response loss after session completion")
        return update_operation(*args, **kwargs)

    monkeypatch.setattr(public, "update_public_live_operation", update_with_lost_acknowledgement)
    if acknowledgement_lost:
        with pytest.raises(PublicLiveRecoveryRequiredError):
            game.call("complete", **request)
    else:
        game.call("complete", **request)
    committed = deepcopy(game.stored)
    assert committed["status"] == "completed"

    replay = game.call("complete", **request)
    assert replay["idempotent_replay"] is True
    assert replay["session"]["status"] == "completed"
    assert replay["session"]["version"] == version + 1
    assert game.stored == committed
    with pytest.raises(PermissionError):
        game.call("complete", **request, edit_token="wrong-token")
    assert game.stored == committed


@pytest.mark.parametrize("action", ["reopen", "advance"])
def test_resuming_an_old_completed_public_round_robin_refreshes_expiry_and_stays_accessible(action):
    game = GeneratorSession("public", rounds=1)
    game.call("skip", round_number=1, reason="Skipped")
    game.set_legacy_completed()
    game.stored["expires_at"] = "2026-01-01T00:00:00Z"
    game.call(action, **({"round_number": 1} if action == "reopen" else {}))
    session = game.reload()
    assert session["status"] == session["event"]["status"] == "active"
    assert datetime.fromisoformat(session["expires_at"].replace("Z", "+00:00")) > datetime.now(timezone.utc)
    assert session["current_round_number"] == (1 if action == "reopen" else 2)


def test_public_reopen_http_route_requires_token_and_persists_original_round():
    from services.api.public_play_generator_routes import install_public_play_generator_routes

    game = GeneratorSession("public", rounds=2)
    game.call("skip", round_number=1, reason="Skipped")
    game.call("advance")
    original = deepcopy(matches(game.session["event"]["rounds"][0]))
    app = FastAPI()

    def raise_error(exc):
        status = 403 if isinstance(exc, PermissionError) else 409 if isinstance(exc, PublicLiveConflictError) else 400
        raise HTTPException(status, str(exc))

    install_public_play_generator_routes(
        app, get_club=lambda slug: {"id": "club"}, get_supabase_client=lambda: game.db,
        public_club_payload=lambda club, slug: club, require_public_writes=lambda: None,
        require_service_role=lambda: None, requester_hash=lambda request: "a" * 64,
        raise_public_error=raise_error, public_writes_enabled=lambda: True, service_role_configured=lambda: True,
    )
    client = TestClient(app)
    path = f"/clubs/test/play-generators/sessions/{game.session['session_key']}/rounds/1/reopen"
    body = {"edit_token": game.edit_token, "expected_version": game.session["version"], "idempotency_key": "http-reopen-000001"}
    assert client.post(path, json={**body, "edit_token": "invalid", "idempotency_key": "http-reopen-invalid"}).status_code == 403
    response = client.post(path, json=body)
    assert response.status_code == 200
    assert response.json()["session"]["event"]["rounds"][0]["status"] == "active"
    assert client.post(path, json=body).json()["idempotent_replay"] is True
    reloaded = game.reload()
    assert matches(reloaded["event"]["rounds"][0]) == original
    assert reloaded["current_round_number"] == 2
