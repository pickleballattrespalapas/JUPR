"""Capability privacy, club scope, deadlines and retry contracts for meet signup."""
from uuid import uuid4
from urllib.parse import urlsplit
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from services.api import interclub_meet_signup_routes as routes
from tests.test_api_contract_interclub_player_pool import setup  # noqa: F401
from tests.interclub_registration_fixtures import set_registration_phase


@pytest.fixture
def meet_signup(setup, monkeypatch):
    _, state = setup
    set_registration_phase(state["season"], "closed")
    monkeypatch.setattr(routes, "get_next_web_base_url", lambda: "https://staging.example.test")
    monkeypatch.setattr(routes, "require_public_intake_or_403", lambda: None)
    cfg = dict(**state["availability_settings"], share_id=str(uuid4()), meet_revision=state["meet"]["revision"])
    entry = dict(id=str(uuid4()), season_id=state["season"]["id"], club_id="alpha", meet_id=state["meet"]["id"],
                 member_id=state["member"]["id"], player_id=1, name="Alex Alpha", email="private@example.test", division="3.5",
                 gender="female", rating=3.6, registered_at="2026-01-01T00:00:00Z", registration_order=1,
                 status="active", placement="confirmed", priority="in_band", reason="Your spot is reserved.",
                 revision=1, token_nonce=str(uuid4()), request_id=str(uuid4()), fingerprint="private-fingerprint")
    state["tables"][routes.SETTINGS] = [cfg]
    state["tables"][routes.SIGNUPS] = [entry]
    state["tables"][routes.MEMBERS][0].update(player_id=1, approval_status="approved")
    state["cfg"], state["entry"] = cfg, entry
    state["public"] = routes.PUBLIC + "/" + cfg["share_id"]
    state["admin"] = state["availability"].replace("/availability", "/signup")
    state["body"] = dict(player_id=1, name="Alex Alpha", division="3.5", request_id=str(uuid4()), confirm_self=True)
    state["result"] = {"entry": entry}
    app = FastAPI()
    routes.install_interclub_meet_signup_routes(app, get_supabase_client=lambda: state["db"])
    return TestClient(app), state


def test_public_board_has_no_contacts_capabilities_or_database_metadata(meet_signup):
    client, state = meet_signup
    response = client.get(state["public"])
    assert response.status_code == 200
    assert response.json()["signup"]["open"] is True
    for secret in ("private@example.test", "token_nonce", "request_id", "fingerprint", "actor_id", "manage_url", "player_id"):
        assert secret not in response.text
    assert "no-store" in response.headers["cache-control"]
    assert not state["calls"], "GET never performs queue or roster writes"


def test_queue_positions_prioritize_band_then_fifo_separately_per_gender(meet_signup):
    client, state = meet_signup
    original = state["entry"]
    state["tables"][routes.SIGNUPS] = [dict(original, id=str(uuid4()), priority=priority, gender=gender, placement="waitlist")
        for priority, gender in [("play_up", "female"), ("in_band", "male"), ("in_band", "female"), ("in_band", "female"), ("play_up", "female")]]
    entries = client.get(state["public"]).json()["entries"]
    assert [entry["queue_position"] for entry in entries] == [3, 1, 1, 2, 4]


@pytest.mark.parametrize("patch", [{"open": False}, {"deadline": "2000-01-01T00:00:00Z"}, {"meet_revision": 99}])
def test_closed_and_rescheduled_signup_cannot_search_players(meet_signup, patch):
    client, state = meet_signup
    state["cfg"].update(patch)
    assert client.get(state["public"]).json()["signup"]["open"] is False
    assert client.get(state["public"] + "/players?q=Alex").status_code == 409


def test_search_only_approved_own_club_profiles(meet_signup):
    client, state = meet_signup
    response = client.get(state["public"] + "/players?q=Alex")
    assert response.status_code == 200
    assert [row["id"] for row in response.json()["players"]] == ["1"]
    assert "email" not in response.text
    state["member"]["approval_status"] = "pending"
    assert client.get(state["public"] + "/players?q=Alex").json()["players"] == []


def test_join_and_exact_retry_keep_request_and_private_fragment(meet_signup):
    client, state = meet_signup
    response = client.post(state["public"], json=state["body"])
    assert response.status_code == 200
    url = response.json()["entry"]["manage_url"]
    assert urlsplit(url).query == "" and urlsplit(url).fragment.startswith("token=")
    assert client.post(state["public"], json=state["body"]).json()["entry"]["manage_url"] == url
    params = [params for name, params in state["calls"] if name == "pcs_interclub_meet_signup_action"]
    assert params[0] == params[1]
    assert params[0]["p_actor_id"] is None
    state["result"] = {"duplicate": True}
    duplicate = client.post(state["public"], json={**state["body"], "request_id": str(uuid4())})
    assert duplicate.json()["duplicate"] is True
    assert "manage_url" not in duplicate.text and state["entry"]["id"] not in duplicate.text


def test_private_link_is_scoped_signed_revocable_and_never_in_url_query(meet_signup):
    client, state = meet_signup
    token = routes._private_url(state["entry"], state["season"]).split("#token=")[1]
    assert client.post(routes.PUBLIC + "/review", json={"token": token}).json()["entry"]["email"] == "private@example.test"
    assert client.post(routes.PUBLIC + "/review", json={"token": token[:-1] + "!"}).status_code == 404
    state["entry"]["token_nonce"] = str(uuid4())
    assert client.post(routes.PUBLIC + "/review", json={"token": token}).status_code == 404


@pytest.mark.parametrize("patch", [{"actor_id": str(uuid4())}, {"rating": 3.4}, {"gender": "male"}, {"club_id": "beta"}, {"confirm_self": False}, {"email": "bad"}])
def test_public_client_cannot_supply_scope_eligibility_or_actor(meet_signup, patch):
    client, state = meet_signup
    assert client.post(state["public"], json={**state["body"], **patch}).status_code == 422
    assert not state["calls"]


def test_admin_requires_exact_club_and_uses_authenticated_actor(meet_signup):
    client, state = meet_signup
    result = client.post(state["admin"] + "/actions", json={"action": "refresh"})
    assert result.status_code == 200
    assert state["calls"][-1][1]["p_actor_id"] == state["user"].user_id
    state["assignment"]["club_id"] = "organizer"
    assert client.get(state["admin"]).status_code == 403
    assert client.post(state["admin"] + "/actions", json={"action": "refresh"}).status_code == 403


def test_public_error_and_closed_intake_never_leak_database_details(meet_signup, monkeypatch):
    client, state = meet_signup
    state["error"] = "unexpected"
    response = client.post(state["public"], json=state["body"])
    assert response.status_code == 503 and "private database" not in response.text
    from fastapi import HTTPException
    def denied():
        raise HTTPException(503, "Closed")
    monkeypatch.setattr(routes, "require_public_intake_or_403", denied)
    state["calls"].clear()
    assert client.post(state["public"], json=state["body"]).status_code == 503
    assert not state["calls"]
