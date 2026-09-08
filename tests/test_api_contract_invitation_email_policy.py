from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import pytest

from services.api import invitation_email_policy as policy
from services.api import staff_invitation_routes as staff, club_join_invitation_routes as clubs


@pytest.fixture
def test_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(policy, "CONFIG_PATH", tmp_path / "email-test.json")
    monkeypatch.setattr(policy, "get_jupr_env", lambda: "staging")
    values = {"FLY_APP_NAME": "juprleagues-api-staging", "SUPABASE_URL": policy.STAGING_AUTH}
    monkeypatch.setattr(policy, "get_env_or_default", lambda name: values.get(name, ""))
    monkeypatch.setattr(policy, "get_next_web_base_url", lambda **_: policy.STAGING_WEB)
    monkeypatch.setattr(policy, "get_smtp_config_status", lambda: {"ok": True, "use_tls": True})
    now = datetime.now(timezone.utc)
    config = {"enabled": True, "recipients": ["tester@example.com"],
              "approved_at": (now - timedelta(minutes=1)).isoformat(),
              "expires_at": (now + timedelta(days=1)).isoformat()}
    policy.CONFIG_PATH.write_text(json.dumps(config))
    return config, values


def test_restricted_email_does_not_enable_global_mail_or_disclose_recipients(test_settings):
    p = policy.invitation_email_policy("dry_run")
    assert p.allows(" Tester@example.com ")
    assert not p.allows("other@example.com") and not p.allows("tester+other@example.com")
    assert p.public_options() == {"email_enabled": True, "email_test_mode": True}
    assert "tester@" not in json.dumps(policy.invitation_email_test_status("dry_run"))
    # The restricted path never accepts staging_redirect or broad live mode.
    assert not policy.invitation_email_policy("staging_redirect").enabled
    assert not policy.invitation_email_policy("live").enabled


@pytest.mark.parametrize("change", [
    {"enabled": False}, {"enabled": "true"}, {"recipients": []},
    {"recipients": ["*@example.com"]}, {"recipients": ["staging@x.invalid"]},
    {"recipients": ["tester@example.com\nother@example.com"]},
    {"recipients": [f"tester{i}@example.com" for i in range(4)]},
    {"expires_at": "2000-01-01T00:00:00Z"}, {"expires_at": "2099-01-01T00:00:00Z"},
    {"approved_at": "2099-01-01T00:00:00Z"}, {"approved_at": "2026-09-08T00:00:00"},
    {"approved_at": None},
])
def test_invalid_disabled_expired_or_overbroad_config_sends_nothing(test_settings, change):
    config, _ = test_settings
    policy.CONFIG_PATH.write_text(json.dumps({**config, **change}))
    assert not policy.invitation_email_policy("dry_run").allows("tester@example.com")


@pytest.mark.parametrize("field,value", [("FLY_APP_NAME", "another-app"), ("SUPABASE_URL", "https://another-project.supabase.co")])
def test_exact_staging_target_required(test_settings, field, value):
    _, values = test_settings; values[field] = value
    assert not policy.invitation_email_policy("dry_run").enabled


def test_config_cannot_enable_email_in_other_environments(test_settings, monkeypatch):
    monkeypatch.setattr(policy, "get_jupr_env", lambda: "production")
    assert not policy.invitation_email_policy("dry_run").enabled
    # Existing live production behavior is independent of the staging test file.
    assert policy.invitation_email_policy("live").allows("another@example.com")


def test_bad_origin_missing_smtp_or_no_tls_blocks_before_auth(test_settings, monkeypatch):
    monkeypatch.setattr(policy, "get_next_web_base_url", lambda **_: "https://other.example.com")
    assert not policy.invitation_email_policy("dry_run").enabled
    monkeypatch.setattr(policy, "get_next_web_base_url", lambda **_: policy.STAGING_WEB)
    for smtp in ({"ok": False, "use_tls": True}, {"ok": True, "use_tls": False}):
        monkeypatch.setattr(policy, "get_smtp_config_status", lambda: smtp)
        assert not policy.invitation_email_policy("dry_run").enabled


@pytest.mark.parametrize("club_join", [False, True])
def test_test_email_still_claims_matching_invitation_and_delivers_only_to_recipient(test_settings, monkeypatch, club_join):
    route = clubs if club_join else staff
    monkeypatch.setattr(route, "get_email_mode", lambda: "dry_run")
    monkeypatch.setattr(staff, "get_email_mode", lambda: "dry_run")
    monkeypatch.setattr(staff, "get_next_web_base_url", lambda **_: policy.STAGING_WEB)
    calls, generated, sent = [], [], []
    invite = {"id": str(uuid4()), "email": "tester@example.com"}
    def rpc(db, **params):
        calls.append(params)
        return invite
    monkeypatch.setattr(route, "rpc" if club_join else "invitation_rpc", rpc)
    monkeypatch.setattr(staff, "send_email_with_inline_chart", lambda **kw: sent.append(kw))
    db = SimpleNamespace(auth=SimpleNamespace(admin=SimpleNamespace(generate_link=lambda p: generated.append(p) or SimpleNamespace(properties=SimpleNamespace(hashed_token="token-fixture")))))
    app = FastAPI()
    install = route.install_club_join_invitation_routes if club_join else route.install_staff_invitation_routes
    install(app, get_supabase_client=lambda: db)
    client = TestClient(app)
    path = f"/{'club' if club_join else 'staff'}-invitations/{invite['id']}/sign-in"
    assert client.get(path).json() == {"email_enabled": True, "email_test_mode": True}
    other = client.post(path, json={"email": "other@example.com", "setup_password": True})
    assert not calls and not generated and not sent
    good = client.post(path, json={"email": "tester@example.com", "setup_password": True})
    assert good.json() == other.json(), "No account or allowlist enumeration in the response"
    assert calls[0]["p_action"] == "email_claim" and calls[0]["p_email"] == invite["email"]
    assert generated == [{"type": "magiclink", "email": invite["email"]}]
    assert sent[0]["to_email"] == invite["email"]
    assert sent[0]["subject"].startswith("[PCS staging test]")
    assert "&setup=password#staff_token_hash=token-fixture" in sent[0]["text_body"]
    assert "token-fixture" not in good.text
    assert ("&kind=club" in sent[0]["text_body"]) is club_join
    # Sender rechecks the actual bound address, even if a returned row changes.
    assert staff.send_invitation_sign_in(db, {**invite, "email": "other@example.com"}) is False
    assert len(generated) == len(sent) == 1
    def reject(db, **params):
        raise HTTPException(409, "Invitation is no longer valid")
    monkeypatch.setattr(route, "rpc" if club_join else "invitation_rpc", reject)
    rejected = client.post(path, json={"email": invite["email"]})
    assert rejected.json() == good.json()
    assert len(generated) == len(sent) == 1


def test_repository_test_delivery_is_disabled_until_recipient_approval():
    # Activation is a separate, reviewed configuration change after Joe selects
    # an actual mailbox. Do not bake test recipients into a functional change.
    config = json.loads(policy.CONFIG_PATH.read_text())
    if config["enabled"]:
        assert config["recipients"] and config["approved_at"] and config["expires_at"]
    else:
        assert config == {"enabled": False, "recipients": [], "approved_at": None, "expires_at": None}
