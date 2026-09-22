"""Club isolation and capability boundaries for season interest and meet RSVPs."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import re
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from services.api import admin_auth_routes, interclub_player_pool_routes as routes
from tests.test_api_contract_interclub_registration import Query
from tests.interclub_registration_fixtures import set_registration_phase


@pytest.fixture
def setup(monkeypatch):
    uid, sid, mid, member_id, response_id = [str(uuid4()) for _ in range(5)]
    user = SimpleNamespace(user_id=uid, email="admin@example.test")
    assignment = dict(club_id="alpha", user_id=uid, email=user.email, role="administrator")
    season = dict(id=sid, organizer_club_id="organizer", details=dict(name="Coastal League",
        start_date="2099-01-01", end_date="2099-03-31", timezone="America/Mazatlan",
        divisions=["3.5", "4.0"], club_ids=["alpha", "beta"]), rules={})
    set_registration_phase(season, "open")
    participation = dict(season_id=sid, club_id="alpha", status="accepted", revision=2)
    meet = dict(id=mid, season_id=sid, host_club_id="beta", club_ids=["alpha", "beta"],
        starts_at="2099-02-10T18:00:00Z", roster_deadline="2099-02-09T18:00:00Z", revision=1)
    member = dict(id=member_id, season_id=sid, club_id="alpha", name="Alex Alpha",
        email="alex@example.test", divisions=["3.5"], notes="Visiting in February", status="active",
        player_id=None, revision=1, created_at="2026-01-01T00:00:00Z", updated_at="2026-01-01T00:00:00Z",
        token_nonce=str(uuid4()))
    tables = {
        "admin_role_assignments": [assignment], "pcs_interclub_seasons": [season],
        "pcs_interclub_participations": [participation], "pcs_interclub_meets": [meet],
        "pcs_interclub_meet_workspaces": [meet],
        "clubs": [dict(id="alpha", name="Alpha", slug="alpha"), dict(id="beta", name="Beta", slug="beta")],
        "players": [dict(id=1, club_id="alpha", name="Alex Alpha", email="alex@example.test", active=True),
                    dict(id=2, club_id="beta", name="Private Beta", email="private@example.test", active=True)],
    }
    state = dict(user=user, assignment=assignment, season=season, participation=participation,
        meet=meet, member=member, response_id=response_id, tables=tables, reads=[], calls=[], error=None,
        result={})

    def table(name):
        state["reads"].append(name)
        return Query(tables.get(name, []))

    def rpc(name, params):
        if name == "pcs_interclub_pool_search_players":
            state["reads"].append(name)
            term = routes._name(params["p_query"])
            rows = [row for row in tables["players"] if row["club_id"] == params["p_club_id"] and row["active"] and term in routes._name(row["name"])]
            rows.sort(key=lambda row: (routes._name(row["name"]) != term, row["name"], row["id"]))
            safe = [{key: row.get(key) for key in routes.PLAYER_FIELDS.split(",")} for row in rows[:20]]
            return SimpleNamespace(execute=lambda: SimpleNamespace(data=safe))
        if name == "pcs_interclub_pool_player_details":
            state["reads"].append(name)
            rows = [dict(player_id=pid, league_rating=None, eligible_divisions=[]) for pid in params["p_player_ids"]]
            return SimpleNamespace(execute=lambda: SimpleNamespace(data=rows))
        state["calls"].append((name, params))
        def execute():
            if state["error"]:
                exc = RuntimeError("private database failure and contact details")
                exc.code = state["error"]
                raise exc
            return SimpleNamespace(data=state["result"])
        return SimpleNamespace(execute=execute)

    monkeypatch.setattr(admin_auth_routes, "authenticate_bearer", lambda _: user)
    monkeypatch.setattr(routes, "get_next_web_base_url", lambda: "https://staging.example.test")
    monkeypatch.setattr(routes, "get_email_mode", lambda: "dry_run")
    monkeypatch.setenv("JUPR_REGISTRATION_EDIT_SECRET", "pool-contract-test-secret-with-at-least-32-characters")
    state["db"] = SimpleNamespace(table=table, rpc=rpc)
    app = FastAPI()
    routes.install_interclub_player_pool_routes(app, get_supabase_client=lambda: state["db"])
    state["base"] = f"/admin/clubs/alpha/interclub/registrations/{sid}"
    state["availability"] = state["base"] + f"/meets/{mid}/availability"
    state["share_id"] = str(uuid4())
    state["settings"] = dict(season_id=sid, club_id="alpha", share_id=state["share_id"], open=True, revision=1)
    state["availability_settings"] = dict(season_id=sid, club_id="alpha", meet_id=mid, open=True,
        deadline="2099-02-01T12:00:00Z", revision=1)
    state["response"] = dict(id=response_id, member_id=member_id, season_id=sid, club_id="alpha", meet_id=mid,
        status="invited", revision=1, token_nonce=str(uuid4()), invited_at="2026-01-01T00:00:00Z", responded_at=None)
    tables[routes.POOL] = [state["settings"]]
    tables[routes.MEMBERS] = [member]
    tables[routes.SETTINGS] = [state["availability_settings"]]
    tables[routes.RESPONSES] = [state["response"]]
    state["signup"] = f"/public/interclub-signups/{state['share_id']}"
    state["signup_body"] = dict(name="Alex Alpha", email="ALEX@EXAMPLE.TEST", divisions=["3.5"],
        notes="Visiting in February", request_id=str(uuid4()), email_consent=True)
    return TestClient(app), state


@pytest.mark.parametrize("change", [dict(role="operator"), dict(club_id="beta"), dict(user_id="someone-else"),
    dict(revoked_at="2020-01-01T00:00:00Z"), dict(expires_at="2020-01-01T00:00:00Z")])
def test_pool_requires_current_administrator_assignment_for_this_club(setup, change):
    client, state = setup
    state["assignment"].update(change)
    for response in [client.get(state["base"] + "/pool"),
        client.put(state["base"] + "/pool", json=dict(expected_revision=1, rotate_link=True)),
        client.get(state["availability"])]:
        assert response.status_code == 403
    assert not state["calls"]


def test_unauthenticated_admin_cannot_read_or_modify_pool(setup, monkeypatch):
    client, state = setup
    def deny(_):
        raise HTTPException(401, "Sign in required")
    monkeypatch.setattr(admin_auth_routes, "authenticate_bearer", deny)
    assert client.get(state["base"] + "/pool").status_code == 401
    assert client.put(state["base"] + "/pool", json=dict(expected_revision=1, rotate_link=True)).status_code == 401
    assert not state["calls"]


@pytest.mark.parametrize("status", ["invited", "declined", "cancelled"])
def test_unaccepted_club_cannot_collect_or_read_season_interest(setup, status):
    client, state = setup
    state["participation"]["status"] = status
    assert client.get(state["base"] + "/pool").status_code == 404
    assert client.put(state["base"] + "/pool", json=dict(expected_revision=1, rotate_link=True)).status_code == 404
    assert not state["calls"]


def test_organizer_role_does_not_grant_another_clubs_pool(setup):
    client, state = setup
    state["assignment"]["club_id"] = "organizer"
    root = state["base"].replace("/alpha/", "/organizer/")
    assert client.get(root + "/pool").status_code == 404
    assert client.get(state["base"] + "/pool").status_code == 403
    assert not state["calls"]


@pytest.mark.parametrize("patch", [dict(expected_revision=-1), dict(open="invalid"), dict(club_id="beta"),
    dict(actor_id="forged"), dict(share_id=str(uuid4())), dict(season_id=str(uuid4()))])
def test_bad_pool_settings_do_not_reach_rpc(setup, patch):
    client, state = setup
    response = client.put(state["base"] + "/pool", json={"expected_revision": 1, "rotate_link": True, **patch})
    assert response.status_code == 422
    assert not state["calls"]


@pytest.mark.parametrize("patch", [dict(expected_revision=0), dict(player_id="bad"), dict(player_id="-1"),
    dict(status="accepted"), dict(club_id="beta"), dict(email="forged@example.test")])
def test_bad_member_link_or_scope_is_rejected_before_rpc(setup, patch):
    client, state = setup
    response = client.patch(state["base"] + "/pool/members/" + state["member"]["id"],
        json={"expected_revision": 1, "player_id": "1", "status": "active", **patch})
    assert response.status_code == 422
    assert not state["calls"]


@pytest.mark.parametrize("patch", [dict(deadline="not-a-date"), dict(deadline="2099-02-01T12:00:00"),
    dict(expected_revision=-1), dict(meet_id=str(uuid4())), dict(club_id="beta")])
def test_availability_settings_require_revision_and_aware_deadline(setup, patch):
    client, state = setup
    response = client.put(state["availability"], json={"expected_revision": 0, "open": True,
        "deadline": "2099-02-01T12:00:00Z", **patch})
    assert response.status_code == 422
    assert not state["calls"]


@pytest.mark.parametrize("patch", [dict(name=" "), dict(name="x" * 121), dict(email="invalid"),
    dict(email="x" * 255 + "@example.test"), dict(notes="x" * 1001), dict(request_id="bad"),
    dict(player_id="bad"), dict(player_id=-1), dict(club_id="beta"), dict(status="active"), dict(token="forged"),
    dict(email_consent=False), dict(divisions=["3.5", "3.5"])])
def test_public_signup_rejects_bad_fields_and_forged_scope(setup, patch):
    client, state = setup
    response = client.post(state["signup"], json={**state["signup_body"], **patch})
    assert response.status_code == 422
    assert not state["calls"]


@pytest.mark.parametrize("payload", [dict(token=""), dict(token="x" * 5000),
    dict(token="sample", club_id="beta"), dict(token="sample", member_id=str(uuid4()))])
def test_public_review_accepts_only_a_bounded_capability(setup, payload):
    client, state = setup
    response = client.post("/public/interclub-player-response/review", json=payload)
    assert response.status_code == 422
    assert not state["calls"]


@pytest.mark.parametrize("patch", [dict(expected_revision=0), dict(status="accepted"), dict(action="link_player"),
    dict(player_id="1"), dict(club_id="beta"), dict(member_id=str(uuid4()))])
def test_public_meet_response_cannot_forge_identity_or_profile(setup, patch):
    client, state = setup
    response = client.post("/public/interclub-player-response/respond", json={"token": "x" * 32,
        "expected_revision": 1, "action": "respond_meet", "status": "available", **patch})
    assert response.status_code == 422
    assert not state["calls"]


@pytest.mark.parametrize("code,status", [("42501", 403), ("P0002", 404), ("40001", 409), ("PT409", 409),
    ("23505", 409), ("22023", 422), ("PT423", 423), ("54000", 429), ("unknown", 503)])
def test_database_errors_are_safe_and_actionable(setup, code, status):
    _, state = setup
    state["error"] = code
    with pytest.raises(HTTPException) as exc:
        routes.pool_rpc(state["db"], "pcs_interclub_pool_action", {})
    assert exc.value.status_code == status
    assert "private" not in exc.value.detail and "contact details" not in exc.value.detail


def test_actor_scope_comes_from_verified_identity_and_route(setup):
    _, state = setup
    assert routes.pool_actor(state["user"], "alpha", state["season"]["id"]) == {
        "p_actor_id": state["user"].user_id, "p_actor_email": "admin@example.test",
        "p_club_id": "alpha", "p_season_id": state["season"]["id"],
    }


def private_headers(response):
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-robots-tag"] == "noindex, nofollow"
    assert response.headers["referrer-policy"] == "no-referrer"


def token_for(state, kind="season"):
    row = state["member"] if kind == "season" else state["response"]
    return routes._token(row, kind, datetime.now(timezone.utc) + timedelta(days=1))


def test_admin_pool_does_not_include_other_clubs_seasons_or_capability_nonces(setup):
    client, state = setup
    state["tables"][routes.MEMBERS].extend([
        {**state["member"], "id": str(uuid4()), "club_id": "beta", "name": "Other club private", "email": "private@example.test"},
        {**state["member"], "id": str(uuid4()), "season_id": str(uuid4()), "name": "Other season private"},
    ])
    response = client.get(state["base"] + "/pool")
    assert response.status_code == 200
    assert [row["id"] for row in response.json()["members"]] == [state["member"]["id"]]
    assert "private" not in response.text and "token_nonce" not in response.text
    assert response.json()["signup"]["url"] == "https://staging.example.test/interclub/signup/" + state["share_id"]
    manage = urlsplit(response.json()["members"][0]["manage_url"])
    assert manage.path == "/interclub/respond" and not manage.query
    claims = routes._verify_token(manage.fragment.removeprefix("token="))
    assert claims["kind"] == "season" and claims["id"] == state["member"]["id"] and claims["club_id"] == "alpha"
    private_headers(response)


def test_pool_and_availability_changes_forward_verified_scope_and_revision(setup):
    client, state = setup
    response = client.put(state["base"] + "/pool", json=dict(expected_revision=1, rotate_link=True))
    assert response.status_code == 200
    name, args = state["calls"][-1]
    assert name == "pcs_interclub_pool_action"
    assert args == {**routes.pool_actor(state["user"], "alpha", state["season"]["id"]),
        "p_action": "settings", "p_payload": {"expected_revision": 1, "rotate_link": True}}
    set_registration_phase(state["season"], "closed")
    response = client.put(state["availability"], json=dict(expected_revision=1, open=True, deadline="2099-02-01T12:00:00Z"))
    assert response.status_code == 200
    _, args = state["calls"][-1]
    assert args["p_action"] == "availability" and args["p_club_id"] == "alpha"
    assert args["p_payload"] == dict(expected_revision=1, open=True, deadline="2099-02-01T12:00:00Z", meet_id=state["meet"]["id"])


def test_member_link_forwards_current_club_and_member_and_redacts_nonce(setup):
    client, state = setup
    state["result"] = {**state["member"], "player_id": 1}
    response = client.patch(state["base"] + "/pool/members/" + state["member"]["id"],
        json=dict(expected_revision=1, player_id="1", status="active"))
    assert response.status_code == 200 and response.json()["member"]["player_id"] == "1"
    _, args = state["calls"][-1]
    assert args["p_club_id"] == "alpha" and args["p_season_id"] == state["season"]["id"]
    assert args["p_payload"] == dict(expected_revision=1, player_id=1, status="active", member_id=state["member"]["id"])
    assert "token_nonce" not in response.text
    assert response.json()["member"]["manage_url"].startswith("https://")


def test_meet_history_contains_only_own_members_and_scoped_private_links(setup):
    client, state = setup
    set_registration_phase(state["season"], "closed")
    state["tables"][routes.RESPONSES].extend([
        {**state["response"], "id": str(uuid4()), "club_id": "beta"},
        {**state["response"], "id": str(uuid4()), "meet_id": str(uuid4())},
        {**state["response"], "id": str(uuid4()), "season_id": str(uuid4())},
    ])
    response = client.get(state["availability"])
    assert response.status_code == 200
    rows = response.json()["responses"]
    assert [row["id"] for row in rows] == [state["response"]["id"]]
    link = urlsplit(rows[0]["response_url"])
    assert link.path == "/interclub/respond" and not link.query and link.fragment.startswith("token=")
    claims = routes._verify_token(link.fragment.removeprefix("token="))
    assert claims["club_id"] == "alpha" and claims["id"] == state["response"]["id"] and claims["kind"] == "meet"
    assert "token_nonce" not in response.text
    private_headers(response)


@pytest.mark.parametrize("patch", [dict(season_id=str(uuid4())), dict(club_ids=["beta"])])
def test_meet_outside_club_or_season_is_unavailable_before_rpc(setup, patch):
    client, state = setup
    set_registration_phase(state["season"], "closed")
    state["meet"].update(patch)
    assert client.get(state["availability"]).status_code == 404
    assert client.put(state["availability"], json=dict(expected_revision=1, open=True,
        deadline="2099-02-01T12:00:00Z")).status_code == 404
    assert not state["calls"]


def test_share_page_is_account_free_and_never_lists_members_or_other_clubs_meets(setup, monkeypatch):
    client, state = setup
    def no_login(_):
        raise AssertionError("Public signup must not require a player account")
    monkeypatch.setattr(admin_auth_routes, "authenticate_bearer", no_login)
    state["tables"]["pcs_interclub_meets"].extend([
        {**state["meet"], "id": str(uuid4()), "club_ids": ["beta"]},
        {**state["meet"], "id": str(uuid4()), "season_id": str(uuid4())},
    ])
    response = client.get(state["signup"])
    assert response.status_code == 200
    assert [row["id"] for row in response.json()["meets"]] == [state["meet"]["id"]]
    assert response.json()["signup"]["open"] is True
    assert "email" not in response.text and "Alex" not in response.text and "token" not in response.text
    assert routes.MEMBERS not in state["reads"] and "players" not in state["reads"]
    private_headers(response)


def test_duplicate_signup_does_not_return_or_overwrite_personal_data(setup):
    client, state = setup
    state["result"] = {"status": "already_registered", "member": state["member"]}
    response = client.post(state["signup"], json=state["signup_body"])
    assert response.status_code == 200 and response.json()["status"] == "already_registered"
    assert "manage_url" not in response.json() and "member" not in response.json()
    assert "email" not in response.json() and "token" not in response.text
    name, args = state["calls"][-1]
    assert name == "pcs_interclub_pool_public_action" and args["p_action"] == "signup"
    assert args["p_payload"]["email"] == "alex@example.test"
    assert args["p_payload"]["share_id"] == state["share_id"]
    assert len(args["p_payload"]["request_fingerprint"]) == 64 and len(args["p_requester_hash"]) == 64
    private_headers(response)


def test_new_signup_returns_only_own_season_link_in_fragment_and_stable_retry_fingerprint(setup):
    client, state = setup
    state["result"] = {"status": "registered", "member": state["member"]}
    first = client.post(state["signup"], json=state["signup_body"])
    assert first.status_code == 200
    first_args = state["calls"][-1][1]
    second = client.post(state["signup"], json=state["signup_body"])
    assert second.json() == first.json()
    assert state["calls"][-1][1]["p_payload"]["request_fingerprint"] == first_args["p_payload"]["request_fingerprint"]
    link = urlsplit(first.json()["manage_url"])
    assert not link.query and link.fragment.startswith("token=")
    claims = routes._verify_token(link.fragment.removeprefix("token="))
    assert claims["kind"] == "season" and claims["club_id"] == "alpha"
    assert claims["id"] == state["member"]["id"] and claims["season_id"] == state["season"]["id"]
    assert "email" not in first.json() and "member" not in first.json()
    private_headers(first)


@pytest.mark.parametrize("kind", ["season", "meet"])
def test_personal_review_returns_only_named_member_without_nonce(setup, kind):
    client, state = setup
    set_registration_phase(state["season"], "closed" if kind == "meet" else "open")
    response = client.post("/public/interclub-player-response/review", json={"token": token_for(state, kind)})
    assert response.status_code == 200
    assert response.json()["kind"] == kind and response.json()["member"]["id"] == state["member"]["id"]
    assert response.json()["can_respond"] is True
    assert "token_nonce" not in response.text and "members" not in response.json()
    private_headers(response)


@pytest.mark.parametrize("kind", ["season", "meet"])
def test_rotated_nonce_revokes_old_personal_link(setup, kind):
    client, state = setup
    token = token_for(state, kind)
    row = state["member"] if kind == "season" else state["response"]
    row["token_nonce"] = str(uuid4())
    assert client.post("/public/interclub-player-response/review", json={"token": token}).status_code == 404
    assert not state["calls"]


@pytest.mark.parametrize("change", ["expired", "tampered", "wrong_kind", "wrong_club", "wrong_season"])
def test_invalid_personal_capabilities_never_reach_rpc(setup, change):
    client, state = setup
    row = dict(state["member"])
    expiry = datetime.now(timezone.utc) + timedelta(days=1)
    kind = "season"
    if change == "expired": expiry = datetime.now(timezone.utc) - timedelta(seconds=1)
    if change == "wrong_kind": kind = "registration"
    if change == "wrong_club": row["club_id"] = "beta"
    if change == "wrong_season": row["season_id"] = str(uuid4())
    token = routes._token(row, kind, expiry)
    if change == "tampered": token = token[:-1] + ("a" if token[-1] != "a" else "b")
    response = client.post("/public/interclub-player-response/review", json={"token": token})
    assert response.status_code == 404
    assert state["member"]["email"] not in response.text and not state["calls"]


@pytest.mark.parametrize("status", ["declined", "cancelled"])
def test_club_cancellation_revokes_public_share_and_personal_access(setup, status):
    client, state = setup
    state["participation"]["status"] = status
    assert client.get(state["signup"]).status_code == 404
    for kind in ("season", "meet"):
        assert client.post("/public/interclub-player-response/review", json={"token": token_for(state, kind)}).status_code == 404
    assert not state["calls"]


def test_closed_pool_still_allows_withdrawal_but_not_reactivation_in_review(setup):
    client, state = setup
    set_registration_phase(state["season"], "closed")
    response = client.post("/public/interclub-player-response/review", json={"token": token_for(state)})
    assert response.status_code == 200 and not response.json()["can_respond"]
    assert response.json()["can_withdraw"] is True
    state["member"]["status"] = "withdrawn"
    response = client.post("/public/interclub-player-response/review", json={"token": token_for(state)})
    assert not response.json()["can_withdraw"]


@pytest.mark.parametrize("change", ["deadline", "closed", "withdrawn", "started"])
def test_meet_review_reports_response_closure_without_hiding_history(setup, change):
    client, state = setup
    set_registration_phase(state["season"], "closed")
    if change == "deadline": state["availability_settings"]["deadline"] = "2020-01-01T12:00:00Z"
    if change == "closed": state["availability_settings"]["open"] = False
    if change == "withdrawn": state["member"]["status"] = "withdrawn"
    if change == "started": state["meet"]["starts_at"] = "2020-01-01T12:00:00Z"
    response = client.post("/public/interclub-player-response/review", json={"token": token_for(state, "meet")})
    assert response.status_code == 200 and response.json()["can_respond"] is False
    assert response.json()["availability"]["status"] == "invited"


def test_meet_capability_cannot_update_season_and_season_capability_cannot_rsvp(setup):
    client, state = setup
    body = dict(expected_revision=1, action="respond_meet", status="available", token=token_for(state))
    assert client.post("/public/interclub-player-response/respond", json=body).status_code == 404
    body = dict(expected_revision=1, action="update_season", status="active", name="Alex", email="alex@example.test",
        divisions=["3.5"], notes="", token=token_for(state, "meet"))
    assert client.post("/public/interclub-player-response/respond", json=body).status_code == 404
    assert not state["calls"]


@pytest.mark.parametrize("kind", ["season", "meet"])
def test_personal_updates_send_capability_scope_nonce_and_expected_revision_to_atomic_rpc(setup, kind):
    client, state = setup
    set_registration_phase(state["season"], "closed" if kind == "meet" else "open")
    payload = dict(token=token_for(state, kind), expected_revision=1,
        action="update_season" if kind == "season" else "respond_meet", status="active" if kind == "season" else "available")
    if kind == "season": payload.update(name="Alex", email="ALEX@EXAMPLE.TEST", divisions=["3.5"], notes="February")
    response = client.post("/public/interclub-player-response/respond", json=payload)
    assert response.status_code == 200
    name, args = state["calls"][-1]
    assert name == "pcs_interclub_pool_public_action" and args["p_action"] == payload["action"]
    row = state["member"] if kind == "season" else state["response"]
    assert args["p_payload"]["id"] == row["id"] and args["p_payload"]["nonce"] == row["token_nonce"]
    assert args["p_payload"]["club_id"] == "alpha" and args["p_payload"]["season_id"] == state["season"]["id"]
    assert args["p_payload"]["expected_revision"] == 1
    assert "token" not in args["p_payload"] and "player_id" not in args["p_payload"]
    private_headers(response)


@pytest.mark.parametrize("code,status", [("40001", 409), ("PT409", 409), ("23505", 409), ("54000", 429)])
def test_public_signup_reports_atomic_retry_conflict_or_rate_limit_without_a_link(setup, code, status):
    client, state = setup
    state["error"] = code
    response = client.post(state["signup"], json=state["signup_body"])
    assert response.status_code == status
    assert "manage_url" not in response.text and "private database" not in response.text


def test_missing_signing_configuration_stops_signup_and_invitations_before_mutation(setup, monkeypatch):
    client, state = setup
    def unavailable():
        raise ValueError("Missing signing secret")
    monkeypatch.setattr(routes, "get_explicit_registration_edit_token_secret", unavailable)
    assert client.put(state["base"] + "/pool", json=dict(expected_revision=1, rotate_link=True)).status_code == 503
    assert client.post(state["signup"], json=state["signup_body"]).status_code == 503
    set_registration_phase(state["season"], "closed")
    with pytest.raises(HTTPException) as exc:
        routes.prepare_meet_invitations(state["db"], state["user"], "alpha", state["season"]["id"],
            state["meet"]["id"], [state["member"]["id"]])
    assert exc.value.status_code == 503 and not state["calls"]


def test_preparing_invites_uses_same_club_season_and_selected_member_ids(setup):
    _, state = setup
    set_registration_phase(state["season"], "closed")
    routes.prepare_meet_invitations(state["db"], state["user"], "alpha", state["season"]["id"],
        state["meet"]["id"], [state["member"]["id"]])
    name, args = state["calls"][-1]
    assert name == "pcs_interclub_pool_action"
    assert args == {**routes.pool_actor(state["user"], "alpha", state["season"]["id"]), "p_action": "invite",
        "p_payload": {"meet_id": state["meet"]["id"], "member_ids": [state["member"]["id"]]}}


def test_pool_schema_is_service_only_with_rls_and_atomic_invoker_functions():
    migration = Path(__file__).resolve().parents[1] / "supabase/migrations/20260919035344_interclub_player_interest_and_availability.sql"
    sql = migration.read_text().lower()
    tables = re.findall(r"create table public\.(pcs_interclub_\w+)\s*\(", sql)
    assert set(tables) == {"pcs_interclub_pool_settings", "pcs_interclub_pool_members",
        "pcs_interclub_availability_settings", "pcs_interclub_availability_responses", "pcs_interclub_pool_rate_buckets"}
    for table in tables:
        assert f"alter table public.{table} enable row level security" in sql
        assert re.search(r"revoke all on [^;]*public\." + table + r"[^;]*from public\s*,\s*anon\s*,\s*authenticated", sql)
    for function, signature in [("pcs_interclub_pool_action", "uuid,text,text,uuid,text,jsonb"),
        ("pcs_interclub_pool_public_action", "text,jsonb,text")]:
        declaration = sql.split("create function public." + function + "(", 1)[1].split("as $$", 1)[0]
        assert "security invoker" in declaration and "security definer" not in declaration
        assert f"revoke all on function public.{function}({signature}) from public,anon,authenticated" in sql
        assert f"grant execute on function public.{function}({signature}) to service_role" in sql


def test_private_response_errors_and_validation_failures_are_not_cacheable(setup):
    client, state = setup
    response = client.post("/public/interclub-player-response/review", json={"token": "invalid-but-long-enough-token"})
    assert response.status_code == 404
    private_headers(response)
    response = client.post(state["signup"], json={"name": "Alex"})
    assert response.status_code == 422
    private_headers(response)
    state["participation"]["status"] = "cancelled"
    response = client.get(state["base"] + "/pool")
    assert response.status_code == 404
    private_headers(response)


def test_pool_approvals_only_organizer_and_no_contact_details(setup):
    client, state = setup
    member = state["member"]
    member.update(approval_status="pending", late_join=True, approval_reason=None)
    assert client.get(state["base"] + "/pool/approvals").status_code == 403
    state["season"]["organizer_club_id"] = "alpha"
    # Organizers may operate the league without entering their own club.
    state["participation"]["status"] = "declined"
    response = client.get(state["base"] + "/pool/approvals")
    assert response.status_code == 200
    saved = response.json()["members"][0]
    assert saved["name"] == member["name"]
    assert saved["approval_status"] == "pending"
    assert not {"email", "notes", "token_nonce", "manage_url", "identity_key"}.intersection(saved)
    assert response.headers["cache-control"] == "no-store"


def test_pool_approval_uses_verified_actor_and_revision(setup):
    client, state = setup
    state["season"]["organizer_club_id"] = "alpha"
    state["result"] = {**state["member"], "approval_status": "approved"}
    body = dict(member_id=state["member"]["id"], expected_revision=3, approve=True, reason="Late traveler approved")
    response = client.post(state["base"] + "/pool/approvals", json=body)
    assert response.status_code == 200
    name, params = state["calls"][-1]
    assert name == "pcs_review_interclub_pool_member"
    assert params["p_actor_id"] == state["user"].user_id
    assert params["p_revision"] == 3
    assert params["p_reason"] == body["reason"]
    assert "email" not in response.json()["member"]
    assert client.post(state["base"] + "/pool/approvals", json={**body, "actor_id": str(uuid4())}).status_code == 422


@pytest.mark.parametrize("history_count", [0, 1001])
def test_approval_queue_excludes_ordinary_members_and_prioritizes_pending_over_history(setup, history_count):
    client, state = setup
    state["season"]["organizer_club_id"] = "alpha"
    common = {**state["member"], "approval_reason": None}
    ordinary = [{**common, "id": str(uuid4()), "name": f"A Ordinary {i}", "late_join": False,
                 "approval_status": "approved"} for i in range(1001)]
    history = [{**common, "id": str(uuid4()), "name": f"B Decided {i}", "late_join": True,
                "approval_status": "approved" if i % 2 else "rejected"} for i in range(history_count)]
    pending = [{**common, "id": str(uuid4()), "name": f"Z Pending {i}", "late_join": True,
                "approval_status": "pending"} for i in range(2)]
    hidden = [{**pending[0], "id": str(uuid4()), "status": "withdrawn"},
              {**pending[0], "id": str(uuid4()), "season_id": str(uuid4())}]
    state["tables"][routes.MEMBERS] = ordinary + history + hidden + pending
    state["tables"][routes.LATE_REQUESTS] = [{"member_id": pending[0]["id"],
        "season_id": state["season"]["id"], "club_id": "alpha", "reason": "Please review this late commitment.",
        "requested_at": "2026-09-22T12:00:00Z"}]

    response = client.get(state["base"] + "/pool/approvals")
    assert response.status_code == 200
    members = response.json()["members"]
    assert [row["id"] for row in members[:2]] == [row["id"] for row in pending]
    assert len(members) == min(1000, history_count + 2)
    assert all(row["late_join"] is True for row in members)
    assert all(row["approval_status"] in {"approved", "rejected"} for row in members[2:])
    assert members[0]["late_request_reason"] == "Please review this late commitment."
    assert not {row["id"] for row in hidden}.intersection(row["id"] for row in members)
    assert not {"email", "notes", "token_nonce", "manage_url"}.intersection(members[0])


def test_pool_approval_stale_conflict_and_nonadmin_rejected(setup):
    client, state = setup
    state["season"]["organizer_club_id"] = "alpha"
    body = dict(member_id=state["member"]["id"], expected_revision=1, approve=True, reason="Late traveler")
    state["error"] = "40001"
    assert client.post(state["base"] + "/pool/approvals", json=body).status_code == 409
    state["assignment"]["role"] = "operator"
    assert client.post(state["base"] + "/pool/approvals", json=body).status_code == 403


@pytest.mark.parametrize("choice", [None, 1, "1"])
def test_public_profile_selection_is_forwarded_without_account_ownership_changes(setup, choice):
    client, state = setup
    state["result"] = {"status": "registered", "member": state["member"]}
    response = client.post(state["signup"], json={**state["signup_body"], "player_id": choice})
    assert response.status_code == 200
    assert state["calls"][-1][1]["p_payload"]["player_id"] == (int(choice) if choice else None)
    assert len(state["calls"]) == 1
    assert "user_id" not in state["calls"][0][1]["p_payload"]


def test_omitting_public_profile_preserves_atomic_auto_match_and_stable_retry(setup):
    client, state = setup
    state["result"] = {"status": "registered", "member": state["member"]}
    assert client.post(state["signup"], json=state["signup_body"]).status_code == 200
    assert "player_id" not in state["calls"][-1][1]["p_payload"]


def test_public_candidate_search_is_club_scoped_normalized_and_contact_free(setup):
    client, state = setup
    state["tables"]["players"][0].update(rating=1480, gender="M")
    response = client.get(state["signup"] + "/players", params={"q": "Alex   Alpha"})
    assert response.status_code == 200
    assert response.json()["players"] == [dict(id="1", name="Alex Alpha", rating=3.7, gender="male", league_rating=None, eligible_divisions=[])]
    assert response.json()["linked_player"] is None
    assert "email" not in response.text and "token" not in response.text and "Private Beta" not in response.text
    assert not state["calls"]
    private_headers(response)
    assert client.get(state["signup"] + "/players").json()["players"] == []


def test_public_search_lists_exact_names_before_partial_results(setup):
    client, state = setup
    state["tables"]["players"] = [dict(id=i, club_id="alpha", name=f"A Alex Alpha {i}", active=True) for i in range(10, 35)] + [
        dict(id=1, club_id="alpha", name="Alex Alpha", active=True), dict(id=2, club_id="alpha", name="ALEX  ALPHA", active=True)]
    rows = client.get(state["signup"] + "/players", params={"q": "Alex Alpha"}).json()["players"]
    assert {row["id"] for row in rows[:2]} == {"1", "2"}
    assert len(rows) == 20


def test_optional_expired_account_does_not_block_public_search_or_signup(setup, monkeypatch):
    client, state = setup
    monkeypatch.setattr(routes, "authenticate_bearer", lambda _: (_ for _ in ()).throw(HTTPException(401, "Expired")))
    response = client.get(state["signup"] + "/players", params={"q": "Alex"}, headers={"Authorization": "Bearer expired"})
    assert response.status_code == 200 and response.json()["linked_player"] is None
    state["result"] = {"status": "registered", "member": state["member"]}
    response = client.post(state["signup"], json=state["signup_body"], headers={"Authorization": "Bearer expired"})
    assert response.status_code == 200


def test_account_prefill_requires_one_verified_contact_in_same_club(setup, monkeypatch):
    client, state = setup
    monkeypatch.setattr(routes, "authenticate_bearer", lambda _: state["user"])
    contact = dict(club_id="alpha", player_id=1, email_normalized=state["user"].email, request_status="active", verified_at="2026-01-01", unsubscribed_at=None)
    state["tables"]["player_profile_update_subscriptions"] = [contact,
        {**contact, "club_id": "beta", "player_id": 2}]
    response = client.get(state["signup"] + "/players", headers={"Authorization": "Bearer verified"})
    assert response.status_code == 200 and response.json()["linked_player"]["id"] == "1"
    contact["verified_at"] = None
    assert client.get(state["signup"] + "/players", headers={"Authorization": "Bearer verified"}).json()["linked_player"] is None
    assert not state["calls"]


def test_bulk_explicit_no_profile_choice_does_not_rematch_name(setup):
    client, state = setup
    state["tables"][routes.MEMBERS] = []
    state["tables"]["players"].append(dict(id=3, club_id="alpha", name="ALEX  ALPHA", active=True))
    body = {"members": [{"name": "Alex Alpha"}]}
    assert client.post(state["base"] + "/pool/bulk-preview", json=body).json()["rows"][0]["status"] == "ambiguous"
    body["members"][0]["player_id"] = None
    row = client.post(state["base"] + "/pool/bulk-preview", json=body).json()["rows"][0]
    assert row["status"] == "new" and row["player_id"] is None
    assert not state["calls"]


def test_admin_added_no_email_member_can_use_personal_link_to_withdraw(setup):
    client, state = setup
    state["member"]["email"] = ""
    response = client.post("/public/interclub-player-response/respond", json={"token": token_for(state),
        "expected_revision": 1, "action": "update_season", "status": "withdrawn", "name": state["member"]["name"],
        "email": "", "divisions": ["3.5"], "notes": ""})
    assert response.status_code == 200
    assert state["calls"][-1][1]["p_payload"]["email"] == ""


@pytest.mark.parametrize("phase", ["unconfigured", "scheduled", "closed"])
def test_intake_is_locked_for_all_clubs_outside_shared_registration_window(setup, phase):
    client, state = setup
    set_registration_phase(state["season"], phase)
    state["settings"]["open"] = True  # Legacy per-club flags cannot reopen intake.
    pool = client.get(state["base"] + "/pool")
    assert pool.status_code == 200 and pool.json()["season"]["registration"]["status"] == phase
    assert pool.json()["signup"]["open"] is False
    share = client.get(state["signup"])
    assert share.status_code == 200 and share.json()["signup"]["open"] is False
    assert client.post(state["signup"], json=state["signup_body"]).status_code == 423
    for action in ("bulk-preview", "bulk-add"):
        assert client.post(state["base"] + "/pool/" + action, json={"members": [{"name": "Verbal Commitment"}]}).status_code == 423
    assert client.post("/public/interclub-player-response/respond", json={
        "token": token_for(state), "action": "update_season", "expected_revision": 1, "status": "active",
        "name": state["member"]["name"], "email": state["member"]["email"], "divisions": ["3.5"], "notes": "Edit"
    }).status_code == 423
    assert not state["calls"]
    state["member"]["status"] = "withdrawn"
    assert client.patch(state["base"] + "/pool/members/" + state["member"]["id"], json={
        "expected_revision": 1, "player_id": "1", "status": "active"
    }).status_code == 423
    assert not state["calls"]


@pytest.mark.parametrize("phase", ["unconfigured", "scheduled", "open"])
def test_direct_availability_and_personal_meet_reply_wait_for_registration_close(setup, phase):
    client, state = setup
    set_registration_phase(state["season"], phase)
    assert client.get(state["availability"]).status_code == 423
    assert client.put(state["availability"], json={"expected_revision": 1, "open": True,
        "deadline": "2099-02-01T12:00:00Z"}).status_code == 423
    with pytest.raises(HTTPException) as exc:
        routes.prepare_meet_invitations(state["db"], state["user"], "alpha", state["season"]["id"],
                                       state["meet"]["id"], [state["member"]["id"]])
    assert exc.value.status_code == 423
    review = client.post("/public/interclub-player-response/review", json={"token": token_for(state, "meet")})
    assert review.status_code == 423
    assert client.post("/public/interclub-player-response/respond", json={"token": token_for(state, "meet"),
        "action": "respond_meet", "expected_revision": 1, "status": "available"}).status_code == 423
    assert not state["calls"]


@pytest.mark.parametrize("phase", ["unconfigured", "scheduled", "closed"])
def test_registration_lock_keeps_existing_member_linking_and_withdrawal_available(setup, phase):
    client, state = setup
    set_registration_phase(state["season"], phase)
    state["result"] = {**state["member"], "player_id": 1}
    linked = client.patch(state["base"] + "/pool/members/" + state["member"]["id"],
                          json={"expected_revision": 1, "player_id": "1", "status": "active"})
    assert linked.status_code == 200 and linked.json()["member"]["player_id"] == "1"
    response = client.post("/public/interclub-player-response/respond", json={"token": token_for(state),
        "action": "update_season", "expected_revision": 1, "status": "withdrawn", "name": "Sneaked identity edit",
        "email": "changed@example.test", "divisions": ["4.0"], "notes": "Changed while withdrawing"})
    assert response.status_code == 200
    assert state["calls"][-1][1]["p_payload"]["status"] == "withdrawn"
    assert {key: state["calls"][-1][1]["p_payload"][key] for key in ("name", "email", "divisions", "notes")} == {
        key: state["member"][key] for key in ("name", "email", "divisions", "notes")
    }, "Permitted withdrawal must not smuggle closed-season edits."


@pytest.mark.parametrize("open_value", [True, False])
def test_club_cannot_change_season_registration_using_legacy_pool_toggle(setup, open_value):
    client, state = setup
    response = client.put(state["base"] + "/pool", json={"expected_revision": 1, "open": open_value})
    assert response.status_code == 422 and not state["calls"]


def test_closing_shared_registration_switches_from_intake_to_meet_planning(setup):
    client, state = setup
    assert client.get(state["signup"]).json()["signup"]["open"] is True
    assert client.get(state["availability"]).status_code == 423
    set_registration_phase(state["season"], "closed")
    assert client.get(state["signup"]).json()["signup"]["open"] is False
    assert client.post(state["signup"], json=state["signup_body"]).status_code == 423
    assert client.get(state["availability"]).status_code == 200
    assert client.put(state["availability"], json={"expected_revision": 1, "open": True,
        "deadline": "2099-02-01T12:00:00Z"}).status_code == 200
    assert state["calls"][-1][1]["p_action"] == "availability"


@pytest.fixture
def late_request(setup):
    client, state = setup
    set_registration_phase(state["season"], "closed")
    state["tables"][routes.MEMBERS].clear()
    state["tables"]["pcs_interclub_late_player_requests"] = []
    state["tables"]["players"][0].update(rating=1160, gender="female")
    state["league_rating"] = 2.9
    state["eligible_divisions"] = ["3.5", "4.0"]
    state["late_body"] = {"player_id": 1, "reason": "Confirmed verbally after signup closed."}
    state["late_url"] = state["base"] + "/pool/late-requests"
    state["result"] = {
        **state["member"], "player_id": 1, "email": "", "notes": "",
        "approval_status": "pending", "late_join": True, "approval_reason": None,
    }
    original_rpc = state["db"].rpc

    def rpc(name, params):
        if name == "pcs_interclub_pool_player_details":
            state["reads"].append(name)
            return SimpleNamespace(execute=lambda: SimpleNamespace(data=[
                dict(player_id=pid, league_rating=state["league_rating"], eligible_divisions=state["eligible_divisions"])
                for pid in params["p_player_ids"]
            ]))
        result = original_rpc(name, params)
        if name != "pcs_request_interclub_late_player":
            return result

        def execute():
            member = {**result.execute().data, "divisions": params["p_divisions"]}
            state["tables"][routes.MEMBERS].append(member)
            state["tables"]["pcs_interclub_late_player_requests"].append({
                "member_id": member["id"], "season_id": member["season_id"],
                "club_id": member["club_id"], "player_id": member["player_id"],
                "reason": params["p_reason"], "requested_at": "2026-09-22T12:00:00Z",
                "requested_by": params["p_actor_id"],
            })
            return SimpleNamespace(data=member)

        return SimpleNamespace(execute=execute)

    state["db"].rpc = rpc
    return client, state


@pytest.mark.parametrize("divisions", [None, [], ["4.0"]])
def test_late_request_uses_verified_actor_and_remains_pending_without_contact_consent(late_request, divisions):
    client, state = late_request
    body = {**state["late_body"], "reason": "  Confirmed verbally after signup closed.  "}
    if divisions is not None:
        body["divisions"] = divisions
    response = client.post(state["late_url"], json=body)
    assert response.status_code == 200
    expected_divisions = divisions or state["eligible_divisions"]
    assert state["calls"] == [("pcs_request_interclub_late_player", {
        "p_actor_id": state["user"].user_id, "p_actor_email": state["user"].email,
        "p_club_id": "alpha", "p_season_id": state["season"]["id"], "p_player_id": 1,
        "p_divisions": expected_divisions, "p_reason": state["late_body"]["reason"],
    })]
    result = response.json()
    assert result["pool"]["can_request_late"] is True
    for member in (result["member"], result["pool"]["members"][0]):
        assert member["player_id"] == "1" and member["divisions"] == expected_divisions
        assert member["approval_status"] == "pending" and member["late_join"] is True
        assert member["late_request_reason"] == state["late_body"]["reason"]
        assert member["late_requested_at"] == "2026-09-22T12:00:00Z"
        assert member["email"] == "" and member["approval_reason"] is None
    assert "email_consent" not in response.text and "token_nonce" not in response.text
    assert state["tables"]["players"][0]["rating"] == 1160
    assert len(state["tables"][routes.RESPONSES]) == 1, "A late request does not create a meet RSVP."
    private_headers(response)
    reload = client.get(state["base"] + "/pool")
    assert reload.status_code == 200
    assert reload.json()["members"][0] == result["pool"]["members"][0]


@pytest.mark.parametrize("phase", ["unconfigured", "scheduled", "open"])
def test_late_requests_require_closed_registration(late_request, phase):
    client, state = late_request
    set_registration_phase(state["season"], phase)
    assert client.get(state["base"] + "/pool").json()["can_request_late"] is False
    response = client.post(state["late_url"], json=state["late_body"])
    assert response.status_code == 423
    private_headers(response)
    assert not state["calls"]
    assert not state["tables"][routes.MEMBERS]


def test_late_request_is_unavailable_after_the_season_ends(late_request):
    client, state = late_request
    state["season"]["details"].update(start_date="2000-01-01", end_date="2000-03-31")
    assert client.get(state["base"] + "/pool").json()["can_request_late"] is False
    response = client.post(state["late_url"], json=state["late_body"])
    assert response.status_code == 409
    assert not state["calls"]


@pytest.mark.parametrize("change", [dict(role="operator"), dict(club_id="beta"), dict(user_id="someone-else"),
    dict(revoked_at="2020-01-01T00:00:00Z"), dict(expires_at="2020-01-01T00:00:00Z")])
def test_late_request_requires_active_administrator_of_the_requesting_club(late_request, change):
    client, state = late_request
    state["assignment"].update(change)
    assert client.post(state["late_url"], json=state["late_body"]).status_code == 403
    assert not state["calls"]


def test_late_request_requires_authentication(late_request, monkeypatch):
    client, state = late_request

    def deny(_):
        raise HTTPException(401, "Sign in required")

    monkeypatch.setattr(admin_auth_routes, "authenticate_bearer", deny)
    assert client.post(state["late_url"], json=state["late_body"]).status_code == 401
    assert not state["calls"]


def test_late_request_checks_personal_link_configuration_before_saving(late_request, monkeypatch):
    client, state = late_request

    def unavailable():
        raise ValueError("Signing configuration unavailable")

    monkeypatch.setattr(routes, "get_explicit_registration_edit_token_secret", unavailable)
    response = client.post(state["late_url"], json=state["late_body"])
    assert response.status_code == 503
    assert not state["calls"]
    assert not state["tables"][routes.MEMBERS]


@pytest.mark.parametrize("status", ["invited", "declined", "cancelled"])
def test_late_request_requires_accepted_season_participation(late_request, status):
    client, state = late_request
    state["participation"]["status"] = status
    assert client.post(state["late_url"], json=state["late_body"]).status_code == 404
    assert not state["calls"]


@pytest.mark.parametrize("player_id,active", [(2, True), (999, True), (1, False)])
def test_late_request_requires_an_active_player_from_this_club(late_request, player_id, active):
    client, state = late_request
    state["tables"]["players"][0]["active"] = active
    response = client.post(state["late_url"], json={**state["late_body"], "player_id": player_id})
    assert response.status_code == 422
    assert "Private Beta" not in response.text
    assert not state["calls"]


@pytest.mark.parametrize("patch", [
    {"player_id": 0}, {"player_id": -1}, {"player_id": None}, {"player_id": "unknown"},
    {"reason": ""}, {"reason": "   "}, {"reason": "x" * 501},
    {"divisions": ["3.5", "3.5"]}, {"divisions": [""]}, {"divisions": ["not-a-season-division"]},
    {"actor_id": "forged"}, {"club_id": "beta"}, {"approval_status": "approved"},
    {"email_consent": True}, {"email": "injected@example.test"}, {"late_join": False},
])
def test_late_request_rejects_invalid_fields_and_forged_approval_or_consent(late_request, patch):
    client, state = late_request
    response = client.post(state["late_url"], json={**state["late_body"], **patch})
    assert response.status_code == 422
    private_headers(response)
    assert not state["calls"]
    assert not state["tables"][routes.MEMBERS]


def test_unrated_active_club_player_can_request_review_without_implied_eligibility(late_request):
    client, state = late_request
    state["eligible_divisions"] = []
    state["league_rating"] = None
    state["tables"]["players"][0]["rating"] = None
    response = client.post(state["late_url"], json=state["late_body"])
    assert response.status_code == 200
    assert state["calls"][0][1]["p_divisions"] == []
    member = response.json()["pool"]["members"][0]
    assert member["approval_status"] == "pending" and member["late_join"] is True
    assert member["divisions"] == member["eligible_divisions"] == []
    assert member["rating"] is None and member["league_rating"] is None


@pytest.mark.parametrize("code,status", [("PT423", 423), ("23505", 409), ("22023", 422)])
def test_late_request_surfaces_atomic_phase_duplicate_and_eligibility_rechecks(late_request, code, status):
    client, state = late_request
    state["error"] = code
    response = client.post(state["late_url"], json=state["late_body"])
    assert response.status_code == status
    assert "private database" not in response.text
    assert not state["tables"][routes.MEMBERS]
    private_headers(response)


def test_late_request_is_reviewable_only_by_the_season_organizer(late_request):
    client, state = late_request
    response = client.post(state["late_url"], json=state["late_body"])
    assert response.status_code == 200
    member = response.json()["member"]
    approval = {"member_id": member["id"], "expected_revision": member["revision"],
        "approve": True, "reason": "Approved for the remaining meets."}
    assert client.post(state["base"] + "/pool/approvals", json=approval).status_code == 403
    assert len(state["calls"]) == 1, "The requesting club cannot approve its own late player."

    state["assignment"]["club_id"] = "organizer"
    organizer_base = state["base"].replace("/alpha/", "/organizer/")
    review = client.get(organizer_base + "/pool/approvals")
    assert review.status_code == 200
    pending = review.json()["members"][0]
    assert pending["approval_status"] == "pending" and pending["late_join"] is True
    assert pending["late_request_reason"] == state["late_body"]["reason"]
    assert pending["late_requested_at"] == "2026-09-22T12:00:00Z"
    assert not {"email", "notes", "token_nonce", "manage_url"}.intersection(pending)

    state["result"] = {**state["tables"][routes.MEMBERS][0], "approval_status": "approved", "revision": 2}
    approved = client.post(organizer_base + "/pool/approvals", json=approval)
    assert approved.status_code == 200 and approved.json()["member"]["approval_status"] == "approved"
    name, params = state["calls"][-1]
    assert name == "pcs_review_interclub_pool_member"
    assert params["p_club_id"] == "organizer" and params["p_actor_id"] == state["user"].user_id
    assert params["p_revision"] == member["revision"] and params["p_approve"] is True
