from copy import deepcopy

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from test_generator_round_controls_service import GeneratorSession
from test_public_play_generator_service import matches
from jupr_app.services import generator_sharing_service as sharing
from jupr_app.services.public_live_operation_service import PublicLiveConflictError
from services.api.shared_play_generator_routes import install_shared_play_generator_routes


@pytest.fixture(params=["admin", "public"])
def game(request, monkeypatch):
    monkeypatch.setenv("JUPR_PUBLIC_LIVE_TOKEN_SECRET", "x" * 48)
    game = GeneratorSession(request.param, rounds=5)
    game.stored.setdefault("version", 1)  # Database default for legacy admin creates.
    game.db.db["clubs"] = [{"id": "club", "slug": "club-slug"}]
    return game


def share(game, action="handoff", **kwargs):
    game.reload()
    return sharing.manage_generator_sharing(game.db, club_id="club", session_key=game.session["session_key"],
        action=action, expected_version=game.session["version"],
        admin_authorized=game.audience == "admin", owner_token=getattr(game, "edit_token", None), **kwargs)


def operator(game, token, action="scores", body=None, key="shared-operator-0001", version=None):
    current = sharing.get_shared_generator(game.db, club_id="club", session_key=game.session["session_key"])["session"]
    body = body or {"round_number": 1, "scores": [{"match_id": matches(current["event"]["rounds"][0])[0]["id"], "score_a": 11, "score_b": 8}]}
    return sharing.mutate_shared_generator(game.db, club_id="club", session_key=game.session["session_key"],
        edit_token=token, expected_version=version or current["version"], idempotency_key=key,
        requester_hash="b" * 64, action=action, body=body)


def test_viewing_is_opt_in_and_does_not_leak_owner_or_handoff_secrets(game):
    with pytest.raises(LookupError):
        sharing.get_shared_generator(game.db, club_id="club", session_key=game.session["session_key"])
    original = deepcopy(game.session["event"])
    shared = share(game, "view")
    assert shared["sharing"]["club_slug"] == "club-slug"
    assert shared["sharing"]["handoff_token"] is None
    assert game.reload()["event"] == original
    token = share(game)["sharing"]["handoff_token"]
    result = sharing.get_shared_generator(game.db, club_id="club", session_key=game.session["session_key"])
    assert token not in str(result)
    assert "handoff_nonce" not in str(result)
    assert "edit_token_hash" not in str(result)
    assert "created_by_email" not in str(result)
    with pytest.raises(PermissionError):
        operator(game, "")


def test_handoff_runs_same_session_and_original_organizer_keeps_control(game):
    token = share(game)["sharing"]["handoff_token"]
    assert share(game)["sharing"]["handoff_token"] == token
    result = operator(game, token)
    shared = result["session"]
    assert shared["standings"] == game.reload()["standings"]
    assert shared["current_round_number"] == 1
    assert len(game.db.db["live_sessions"]) == 1
    assert len(game.db.db.get("matches", [])) == 0
    game.call("advance")
    advanced = operator(game, token, "roster", {"action": "add", "name": "Late arrival", "player_id": None,
        "participant_id": None, "substitute_scope": "rest", "roster_order": [], "participant_ids": [], "court_number": None}, key="shared-add-0001")["session"]
    assert advanced["current_round_number"] == 2
    assert advanced["event"]["participants"][-1]["name"] == "Late arrival"


def test_revoke_blocks_old_and_replacement_tokens_and_keeps_standings(game):
    old = share(game)["sharing"]["handoff_token"]
    share(game, "revoke")
    assert sharing.get_shared_generator(game.db, club_id="club", session_key=game.session["session_key"])["session"]
    with pytest.raises(PermissionError):
        operator(game, old)
    new = share(game)["sharing"]["handoff_token"]
    assert new != old
    with pytest.raises(PermissionError):
        operator(game, old)
    operator(game, new)


def test_operator_cannot_manage_sharing_or_use_another_club_or_session(game):
    token = share(game)["sharing"]["handoff_token"]
    game.reload()
    with pytest.raises(PermissionError):
        sharing.manage_generator_sharing(game.db, club_id="club", session_key=game.session["session_key"],
            action="handoff", expected_version=game.session["version"], owner_token=token)
    with pytest.raises(LookupError):
        sharing.get_shared_generator(game.db, club_id="other-club", session_key=game.session["session_key"])
    other = deepcopy(game.stored)
    other["session_key"] = "another-session"
    game.db.db["live_sessions"].append(other)
    with pytest.raises(PermissionError):
        sharing.mutate_shared_generator(game.db, club_id="club", session_key="another-session", edit_token=token,
            expected_version=other["updated_at"], idempotency_key="cross-session-0001", requester_hash="b" * 64,
            action="scores", body={})


@pytest.mark.parametrize("lock", ["submission", "published"])
def test_official_or_submitted_results_are_locked(game, lock):
    token = share(game)["sharing"]["handoff_token"]
    game.stored["state"]["generator_submission" if lock == "submission" else "official_publish"] = (
        {"status": "pending"} if lock == "submission" else {"published_match_ids": [123]})
    with pytest.raises(ValueError, match="locked"):
        operator(game, token)


def test_stale_write_rejected_and_exact_response_loss_retry_does_not_repeat(game):
    token = share(game)["sharing"]["handoff_token"]
    first = sharing.get_shared_generator(game.db, club_id="club", session_key=game.session["session_key"])["session"]
    body = {"round_number": 1, "scores": [{"match_id": matches(first["event"]["rounds"][0])[0]["id"], "score_a": 11, "score_b": 8}]}
    operator(game, token, body=body, version=first["version"])
    after = deepcopy(game.stored)
    replay = operator(game, token, body=body, version=first["version"])
    assert replay["idempotent_replay"] is True
    assert game.stored == after
    with pytest.raises(PublicLiveConflictError):
        operator(game, token, body=body, key="stale-write-0001", version=first["version"])


def test_revoke_racing_a_write_prevents_persisting_the_old_token(game, monkeypatch):
    token = share(game)["sharing"]["handoff_token"]
    mutate = sharing._mutate_event
    def revoke_before_save(*args, **kwargs):
        result = mutate(*args, **kwargs)
        share(game, "revoke")
        return result
    monkeypatch.setattr(sharing, "_mutate_event", revoke_before_save)
    with pytest.raises(PublicLiveConflictError):
        operator(game, token)
    assert game.reload()["event"]["rounds"][0]["status"] == "active"
    assert not game.stored["state"]["generator_sharing"].get("handoff_nonce")


def test_routes_enforce_token_gate_and_omit_submit_publish_and_access_management(game):
    token = share(game)["sharing"]["handoff_token"]
    app = FastAPI()
    def handle(exc):
        raise HTTPException(status_code=403 if isinstance(exc, PermissionError) else 400, detail=str(exc))
    def denied():
        raise HTTPException(status_code=403, detail="writes closed")
    install_shared_play_generator_routes(app, get_club=lambda _: {"id": "club"},
        get_supabase_client=lambda: game.db, require_public_writes=lambda: None,
        require_service_role=lambda: None, require_admin_writes=denied,
        requester_hash=lambda _: "b" * 64, raise_public_error=handle)
    client = TestClient(app)
    root = f"/clubs/club-slug/shared-play-generators/sessions/{game.session['session_key']}"
    assert client.get(root).status_code == 200
    body = {"edit_token": token, "expected_version": game.stored["updated_at"], "idempotency_key": "route-operator-0001"}
    for suffix in ["submit", "publish", "sharing", "review"]:
        assert client.post(f"{root}/{suffix}", json=body).status_code == 404
    if game.audience == "admin":
        assert client.post(f"{root}/complete", json=body).status_code == 403
    else:
        assert client.post(f"{root}/complete", json={**body, "edit_token": ""}).status_code == 422
