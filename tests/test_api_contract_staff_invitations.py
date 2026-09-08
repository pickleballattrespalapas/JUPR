from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from services.api import admin_auth_routes, staff_invitation_routes as routes
from tests.test_api_contract_club_settings import Query


class InvitationQuery(Query):
    def order(self, *_args, **_kwargs):
        return self


@pytest.fixture
def setup(monkeypatch):
    user = SimpleNamespace(user_id=str(uuid4()), email="admin@example.test")
    invite = dict(id=str(uuid4()), club_id="beta", email="invited@example.test", role="operator",
                  scopes=[dict(kind="program_type", program_type="leagues", resource_id="")],
                  status="pending", expires_at="2099-01-01T00:00:00Z", access_expires_at=None,
                  invited_by_email="private@example.test", target_before={"private":"value"})
    assignment = dict(email=user.email, user_id=user.user_id, club_id="beta", role="administrator")
    state = dict(user=user, invite=invite, assignment=assignment, calls=[], error=None, sent=[], generated=[])
    monkeypatch.setattr(admin_auth_routes, "authenticate_bearer", lambda _: user)
    monkeypatch.setattr(routes, "authenticate_bearer", lambda _: user)
    monkeypatch.setattr(routes, "get_email_mode", lambda: "dry_run")
    def table(name):
        return InvitationQuery({"admin_role_assignments":[assignment], "club_staff_invitations":[invite],
                                "clubs":[dict(id="beta",slug="beta-club",name="Beta Club")]}[name])
    def rpc(name, params):
        state["calls"].append((name, params))
        def execute():
            if state["error"]:
                exc = RuntimeError("private sql details")
                exc.code = state["error"]
                raise exc
            return SimpleNamespace(data=invite)
        return SimpleNamespace(execute=execute)
    def generate(params):
        state["generated"].append(params)
        return SimpleNamespace(properties=SimpleNamespace(hashed_token="credential-fixture"))
    db = SimpleNamespace(table=table, rpc=rpc, auth=SimpleNamespace(admin=SimpleNamespace(generate_link=generate)))
    state["db"] = db
    monkeypatch.setattr(routes, "send_email_with_inline_chart", lambda **kwargs: state["sent"].append(kwargs))
    app = FastAPI()
    routes.install_staff_invitation_routes(app, get_supabase_client=lambda: db)
    return TestClient(app), state


def payload(state, **overrides):
    return dict(invitation_id=state["invite"]["id"], email=" INVITED@EXAMPLE.TEST ", role="operator",
                scopes=[dict(kind="program_type", program_type="leagues")], **overrides)


def test_create_preserves_email_binding_and_never_grants_or_sends(setup):
    c, s = setup
    r = c.post("/admin/clubs/beta/staff/invitations", json=payload(s))
    assert r.status_code == 200
    name, args = s["calls"][0]
    assert name == "pcs_staff_invitation" and args["p_action"] == "create"
    assert args["p_actor_id"] == s["user"].user_id and args["p_actor_email"] == s["user"].email
    assert args["p_club_id"] == "beta" and args["p_email"] == "invited@example.test"
    assert args["p_id"] == s["invite"]["id"]
    assert args["p_scopes"] == s["invite"]["scopes"]
    assert not s["sent"] and not s["generated"]
    assert "target_before" not in r.text and "private" not in r.text


@pytest.mark.parametrize("change", [dict(role="operator"),dict(club_id="alpha"),dict(revoked_at="2020-01-01"),dict(user_id="other"),dict(expires_at="2020-01-01T00:00:00Z")])
def test_only_current_club_admin_can_list_create_or_cancel(setup, change):
    c, s = setup
    s["assignment"].update(change)
    assert c.get("/admin/clubs/beta/staff/invitations").status_code == 403
    assert c.post("/admin/clubs/beta/staff/invitations", json=payload(s)).status_code == 403
    assert c.post(f"/admin/clubs/beta/staff/invitations/{s['invite']['id']}/cancel").status_code == 403
    assert not s["calls"]


@pytest.mark.parametrize("field,value", [("role","super_admin"),("scopes",[]),("email","bad@email"),("actor_id","forged"),("club_id","alpha"),
                                       ("access_expires_at","2026-01-01T00:00:00Z"),("access_expires_at","2099-01-01T00:00:00"),("invitation_id","not-uuid")])
def test_invalid_or_privileged_fields_are_rejected(setup, field, value):
    c, s = setup
    assert c.post("/admin/clubs/beta/staff/invitations", json={**payload(s),field:value}).status_code == 422
    assert not s["calls"]


def test_no_invitation_disclosure_to_wrong_account_and_no_roles_needed_to_review(setup):
    c, s = setup
    path = f"/staff-invitations/{s['invite']['id']}"
    assert c.get(path).status_code == 404
    s["user"].email = "invited@example.test"
    response = c.get(path)
    assert response.status_code == 200 and response.json()["club"]["id"] == "beta"
    assert "private" not in response.text and not s["calls"]
    assert c.post(path + "/accept").status_code == 200
    assert s["calls"][-1][1] == dict(p_action="accept",p_id=s["invite"]["id"],p_actor_id=s["user"].user_id,p_actor_email="invited@example.test")


def test_cancel_passes_route_club_to_transaction_and_list_marks_expiry(setup):
    c, s = setup
    s["invite"]["access_expires_at"] = "2020-01-01T00:00:00Z"
    assert c.get("/admin/clubs/beta/staff/invitations").json()["invitations"][0]["status"] == "expired"
    assert c.post(f"/admin/clubs/beta/staff/invitations/{s['invite']['id']}/cancel").status_code == 200
    assert s["calls"][-1][1]["p_club_id"] == "beta"


@pytest.mark.parametrize("code,status", [("42501",403),("40001",409),("P0002",404),("22023",422),("unknown",503)])
def test_database_rejections_do_not_expose_details(setup, code, status):
    c, s = setup
    s["error"] = code
    r = c.post(f"/staff-invitations/{s['invite']['id']}/accept")
    assert r.status_code == status and "private" not in r.text


@pytest.mark.parametrize("mode", ["dry_run", "staging_redirect"])
@pytest.mark.parametrize("setup_password", [False, True])
def test_nonlive_email_never_creates_auth_user_token_or_mail(setup, monkeypatch, mode, setup_password):
    c, s = setup
    monkeypatch.setattr(routes, "get_email_mode", lambda: mode)
    r = c.post(f"/staff-invitations/{s['invite']['id']}/sign-in",json={"email":"invited@example.test", "setup_password":setup_password})
    assert r.status_code == 200 and not r.json()["email_enabled"]
    assert routes.send_invitation_sign_in(s["db"],s["invite"]) is False
    assert not s["calls"] and not s["generated"] and not s["sent"]


@pytest.mark.parametrize("setup_password", [False, True])
def test_live_link_is_only_delivered_to_bound_recipient_after_claim(setup, monkeypatch, setup_password):
    c, s = setup
    monkeypatch.setattr(routes, "get_email_mode", lambda: "live")
    monkeypatch.setattr(routes, "get_next_web_base_url", lambda **_: "https://web.example.test")
    r = c.post(f"/staff-invitations/{s['invite']['id']}/sign-in",json={"email":"invited@example.test", "setup_password":setup_password})
    assert r.status_code == 200 and "credential-fixture" not in r.text and "web.example.test" not in r.text
    assert s["calls"][0][1]["p_action"] == "email_claim"
    assert s["generated"] == [{"type":"magiclink","email":"invited@example.test"}]
    sent = s["sent"][0]
    assert sent["to_email"] == "invited@example.test"
    setup_query = "&setup=password" if setup_password else ""
    assert f"https://web.example.test/admin/accept-invitation?invitation={s['invite']['id']}{setup_query}#staff_token_hash=credential-fixture" in sent["text_body"]
    if setup_password:
        assert "choose a password" in sent["text_body"]


@pytest.mark.parametrize("mode", ["dry_run", "staging_redirect", "live"])
def test_sign_in_options_are_public_and_never_disclose_invitation_or_account(setup, monkeypatch, mode):
    c, s = setup
    monkeypatch.setattr(routes, "get_email_mode", lambda: mode)
    assert c.get(f"/staff-invitations/{uuid4()}/sign-in").json() == {"email_enabled": mode == "live"}
    assert not s["calls"] and not s["generated"] and not s["sent"]


@pytest.mark.parametrize("code", ["40001", "P0002", "42501"])
def test_unavailable_email_requests_are_generic_and_send_nothing(setup, monkeypatch, code):
    c, s = setup
    monkeypatch.setattr(routes, "get_email_mode", lambda: "live")
    s["error"] = code
    r = c.post(f"/staff-invitations/{s['invite']['id']}/sign-in", json={"email":"wrong@example.test", "setup_password":True})
    assert r.status_code == 200 and "If the email matches" in r.json()["message"]
    assert not s["sent"] and not s["generated"]


@pytest.mark.parametrize("origin", ["", "http://web.example.test", "https://web.example.test/redirect", "https://user:pass@web.example.test", "https://web.example.test?next=other"])
def test_bad_callback_configuration_fails_before_generating_credentials(setup, monkeypatch, origin):
    _, s = setup
    monkeypatch.setattr(routes, "get_email_mode", lambda: "live")
    monkeypatch.setattr(routes, "get_next_web_base_url", lambda **_: origin)
    with pytest.raises(ValueError): routes.send_invitation_sign_in(s["db"], s["invite"])
    assert not s["generated"] and not s["sent"]


def test_unauthenticated_review_and_accept_are_denied(setup, monkeypatch):
    c, s = setup
    def denied(_): raise HTTPException(401, "Sign in required")
    monkeypatch.setattr(routes, "authenticate_bearer", denied)
    assert c.get(f"/staff-invitations/{s['invite']['id']}").status_code == 401
    assert c.post(f"/staff-invitations/{s['invite']['id']}/accept").status_code == 401
    assert not s["calls"]
