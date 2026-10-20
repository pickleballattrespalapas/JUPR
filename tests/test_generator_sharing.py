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
    def denied(club_id):
        assert club_id == "club"
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


@pytest.mark.parametrize("club_id", ["tres_palapas", "another_club"])
def test_production_owner_and_handoff_routes_use_the_session_club(monkeypatch, club_id):
    from services.api import admin_play_generator_routes as admin_routes

    settings = {"JUPR_ENV": "production", "FLY_APP_NAME": "juprleagues-api",
        "SUPABASE_URL": "https://dnoockbwfenunhcibwfn.supabase.co",
        "SUPABASE_SERVICE_ROLE_KEY": "server-fixture", "JUPR_PRODUCTION_WRITE_POLICY": "enabled",
        "JUPR_STAGING_WRITE_WAVE": "none", "JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE": "1",
        "JUPR_PUBLIC_LIVE_TOKEN_SECRET": "x" * 48}
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    game = GeneratorSession("admin", rounds=3)
    game.stored.update(club_id=club_id, version=1)
    game.db.db["clubs"] = [{"id": club_id, "slug": "club-slug"}]
    monkeypatch.setattr(admin_routes, "_resolve_role_or_403", lambda **kwargs: "admin")
    app = FastAPI()
    admin_routes.install_admin_play_generator_routes(app, get_supabase_client=lambda: game.db)
    install_shared_play_generator_routes(app, get_club=lambda _: {"id": club_id},
        get_supabase_client=lambda: game.db, require_public_writes=lambda: None,
        require_service_role=lambda: None, require_admin_writes=admin_routes._require_write_gate,
        requester_hash=lambda _: "b" * 64,
        raise_public_error=lambda exc: (_ for _ in ()).throw(exc))
    client = TestClient(app)
    key = game.session["session_key"]
    response = client.post(f"/admin/clubs/{club_id}/play-generators/sessions/{key}/sharing",
        json={"action": "handoff", "expected_version": game.stored["updated_at"]})
    if club_id == "tres_palapas":
        assert response.status_code == 200
        result = response.json()
    else:
        assert response.status_code == 403
        result = sharing.manage_generator_sharing(game.db, club_id=club_id, session_key=key,
            action="handoff", expected_version=game.stored["updated_at"], admin_authorized=True)
    root = f"/clubs/club-slug/shared-play-generators/sessions/{key}"
    session = client.get(root).json()["session"]
    match = matches(session["event"]["rounds"][0])[0]
    saved = client.patch(f"{root}/rounds/1/scores", json={
        "edit_token": result["sharing"]["handoff_token"], "expected_version": session["version"],
        "idempotency_key": "production-shared-scores-0001",
        "scores": [{"match_id": match["id"], "score_a": 11, "score_b": 8}]})
    assert saved.status_code == (200 if club_id == "tres_palapas" else 403)
    assert not game.db.db.get("matches")


def test_hub_public_rr_admin_sharing_uses_timestamp_and_owner_keeps_numeric_lease(monkeypatch):
    from test_admin_round_robin_hub_service import TransactionalHubDB, command, create, raw_session, session
    from jupr_app.services.admin_play_generator_service import get_play_generator_session
    from jupr_app.services.public_live_operation_service import hash_edit_token

    monkeypatch.setenv("JUPR_PUBLIC_LIVE_TOKEN_SECRET", "x" * 48)
    db = TransactionalHubDB()
    session(db, mode="public_play_generator")
    db.storage["clubs"] = [{"id": "club-a", "slug": "club-slug"}]
    raw_session(db)["edit_token_hash"] = hash_edit_token("original-owner")
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    fresh = get_play_generator_session(db, club_id="club-a", session_key="rr-a")["session"]
    before_event = deepcopy(fresh["event"])
    calls = len(db.calls)

    shared = sharing.manage_generator_sharing(db, club_id="club-a", session_key="rr-a",
        action="view", expected_version=fresh["version"], admin_authorized=True)
    assert isinstance(shared["session"]["version"], str)
    assert shared["session"]["version"] == raw_session(db)["updated_at"]
    assert shared["session"]["event"] == before_event
    assert shared["sharing"]["club_slug"] == "club-slug"
    assert len(db.calls) == calls + 1
    assert db.storage["round_robin_hubs"][0]["version"] == hub["hub"]["version"] + 1

    before = deepcopy(db.storage)
    with pytest.raises(PublicLiveConflictError):
        sharing.manage_generator_sharing(db, club_id="club-a", session_key="rr-a",
            action="handoff", expected_version=fresh["version"], admin_authorized=True)
    with pytest.raises(PermissionError):
        sharing.manage_generator_sharing(db, club_id="club-a", session_key="rr-a",
            action="handoff", expected_version=raw_session(db)["version"], owner_token="wrong-owner")
    assert db.storage == before

    owner = sharing.manage_generator_sharing(db, club_id="club-a", session_key="rr-a",
        action="handoff", expected_version=raw_session(db)["version"], owner_token="original-owner")
    assert isinstance(owner["session"]["version"], int)
    assert owner["session"]["version"] == raw_session(db)["version"]
    assert owner["sharing"]["handoff_token"]
    assert owner["session"]["event"] == before_event
    with pytest.raises(PublicLiveConflictError):
        sharing.manage_generator_sharing(db, club_id="club-a", session_key="rr-a",
            action="revoke", expected_version=raw_session(db)["updated_at"], owner_token="original-owner")


@pytest.mark.parametrize("mode", ["admin_play_generator", "public_play_generator"])
def test_shared_hub_standings_wait_for_whole_round_completion(mode):
    from test_admin_round_robin_hub_service import TransactionalHubDB, command, create, raw_session, session
    from jupr_app.services.admin_play_generator_service import get_play_generator_session

    db = TransactionalHubDB()
    session(db, mode=mode)
    db.storage["clubs"] = [{"id": "club-a", "slug": "club-slug"}]
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    sharing.manage_generator_sharing(db, club_id="club-a", session_key="rr-a",
        action="view", expected_version=raw_session(db)["updated_at"], admin_authorized=True)
    from jupr_app.services.admin_round_robin_hub_service import get_round_robin_hub
    hub = get_round_robin_hub(db, club_id="club-a", hub_id=hub["hub"]["id"])

    for completed in (1, 2):
        pending = hub["workspace"]["eligible_queue"][0]
        reference = {key: pending[key] for key in ("session_key", "round_number", "match_id")}
        hub = command(db, hub, "assign_game", {**reference, "court_id": "c1"})
        hub = command(db, hub, "score_and_release", {**reference, "court_id": "c1", "score_a": 11, "score_b": 8})
        shared = sharing.get_shared_generator(db, club_id="club-a", session_key="rr-a")["session"]
        admin = get_play_generator_session(db, club_id="club-a", session_key="rr-a")["session"]
        assert shared["standings"] == admin["standings"]
        assert {row["matches"] for row in shared["standings"]} == ({0} if completed == 1 else {1})
        assert shared["current_round_number"] == (1 if completed == 1 else 2)
        assert "handoff_nonce" not in str(shared)
        assert "edit_token_hash" not in str(shared)
