import base64
import copy
import json
import time
from urllib.parse import parse_qs, urlsplit

import pytest

from scripts import prepare_staging_qa as qa

USER = "c69be83f-b1db-4b1b-bc4c-99c8df45f623"
ACTOR = "9170a273-a8ed-469e-8f25-30f3f53e9abb"


def user(uid=USER, email=qa.EMAIL):
    return {"id": uid, "email": email, "email_confirmed_at": "2026-09-16T12:00:00Z",
            "app_metadata": {"pcs_staging_qa": qa.MARKER}, "is_anonymous": False}


def role(club):
    return {"id": club, "club_id": club, "email": qa.EMAIL, "user_id": USER,
            "role": "administrator", "revoked_at": None, "expires_at": None}


class Server:
    def __init__(self, exists=True):
        self.user = user() if exists else None
        self.roles = [role(c) for c in qa.CLUBS] if exists else []
        self.calls = []
        self.bad_workspaces = False
        self.bad_identity = False
        now = int(time.time())
        claims = {"sub": USER, "iss": qa.auth.EXPECTED_SUPABASE_ISSUER,
                  "aud": "authenticated", "email": qa.EMAIL,
                  "session_id": "07f80228-38f9-4cbc-899c-420e30aa4483", "iat": now, "exp": now + 3600}
        encode = lambda data: base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")
        self.token = encode({"alg": "HS256"}) + "." + encode(claims) + ".fake_signature"
        self.auth_payload = {"access_token": self.token, "token_type": "bearer", "expires_in": 3600, "user": user()}

    def __call__(self, request):
        url = urlsplit(request.full_url)
        method = request.get_method()
        payload = json.loads(request.data) if request.data else None
        self.calls.append((method, url.path, payload))
        data = None
        if url.path == "/rest/v1/clubs":
            data = [{"id": c, "is_active": True} for c in qa.CLUBS]
        elif url.path == "/auth/v1/admin/users" and method == "GET":
            data = {"users": [self.user] if self.user else []}
        elif url.path == "/auth/v1/admin/users" and method == "POST":
            self.user = user()
            data = self.user
        elif url.path == f"/auth/v1/admin/users/{ACTOR}":
            data = user(ACTOR, "fixture-admin@example.invalid")
        elif url.path == "/rest/v1/admin_role_assignments":
            query = parse_qs(url.query)
            if query["email"] == [f"neq.{qa.EMAIL}"]:
                data = [{"club_id": c, "user_id": ACTOR, "email": "fixture-admin@example.invalid"} for c in qa.CLUBS]
            elif method == "PATCH":
                for row in self.roles:
                    if query["club_id"] == [f"eq.{row['club_id']}"]:
                        row["user_id"] = payload["user_id"]
                data = []
            else:
                data = self.roles
        elif url.path == "/rest/v1/rpc/pcs_save_staff":
            row = role(payload["p_club_id"])
            row["user_id"] = None
            self.roles.append(row)
            data = row
        elif url.path == "/auth/v1/admin/generate_link":
            data = {"id": USER, "email": qa.EMAIL, "hashed_token": "d" * 64, "verification_type": "magiclink"}
        elif url.path == "/auth/v1/verify":
            data = copy.deepcopy(self.auth_payload)
            if self.bad_identity:
                data["user"]["id"] = ACTOR
        elif url.path == "/admin/auth/workspaces":
            data = {"workspaces": [{"club_id": c, "roles": ["administrator"]} for c in qa.CLUBS]}
            if self.bad_workspaces:
                data["workspaces"].append({"club_id": "tres_palapas", "roles": ["super_admin"]})
        elif url.path == "/auth/v1/logout":
            return 204, b""
        else:
            raise AssertionError("Unexpected request route")
        return 200, json.dumps(data).encode()


@pytest.fixture
def env(tmp_path):
    return {"GITHUB_ACTIONS": "true", "GITHUB_REF": "refs/heads/staging",
            "GITHUB_ENV": str(tmp_path / "private-env"),
            "STAGING_SUPABASE_URL": qa.auth.EXPECTED_SUPABASE_ORIGIN,
            "STAGING_API_BASE_URL": qa.auth.EXPECTED_API_ORIGIN,
            "STAGING_SUPABASE_SERVICE_ROLE_KEY": "service-secret",
            "STAGING_SUPABASE_ANON_KEY": "anon-key"}


@pytest.mark.parametrize("key,value", [
    ("STAGING_SUPABASE_URL", "https://production.supabase.co"),
    ("STAGING_API_BASE_URL", "https://api.juprleagues.com"),
    ("GITHUB_REF", "refs/heads/rollback-feb8"), ("GITHUB_ACTIONS", "false"),
])
def test_refuses_wrong_environment_before_network(env, key, value):
    server = Server(False)
    with pytest.raises(qa.Error, match="canonical staging"):
        qa.prepare(env={**env, key: value}, provision=True, transport=server)
    assert not server.calls


def test_provisions_without_password_or_email_and_audits_only_three_grants(env, capsys):
    server = Server(False)
    result = qa.prepare(env=env, provision=True, transport=server)
    creation = [p for m, path, p in server.calls if m == "POST" and path == "/auth/v1/admin/users"]
    assert len(creation) == 1
    assert creation[0]["email_confirm"] is True
    assert "password" not in creation[0]
    grants = [p for _, path, p in server.calls if path == "/rest/v1/rpc/pcs_save_staff"]
    assert {p["p_club_id"] for p in grants} == set(qa.CLUBS)
    assert all(p["p_role"] == "administrator" and p["p_actor_id"] == ACTOR for p in grants)
    assert all(r["user_id"] == USER for r in server.roles)
    assert result["email_sent"] is False
    assert server.token not in json.dumps(result) + capsys.readouterr().out
    assert not any(path.endswith(("/otp", "/invite", "/signup")) for _, path, _ in server.calls)


def test_reuses_identity_without_account_or_permission_writes(env):
    server = Server()
    qa.prepare(env=env, transport=server)
    assert all(method == "GET" for method, path, _ in server.calls
               if path.startswith(("/rest/", "/auth/v1/admin/users")))


def test_ordinary_run_cannot_create_missing_account(env):
    server = Server(False)
    with pytest.raises(qa.Error, match="explicit provisioning"):
        qa.prepare(env=env, transport=server)
    assert all(m == "GET" for m, _, _ in server.calls)


def test_ordinary_run_cannot_restore_deleted_assignment(env):
    server = Server()
    server.roles.pop()
    with pytest.raises(qa.Error, match="never grant"):
        qa.prepare(env=env, transport=server)
    assert all(m == "GET" for m, _, _ in server.calls)


def test_does_not_adopt_an_existing_account_with_same_email(env):
    server = Server()
    server.user["app_metadata"] = {}
    with pytest.raises(qa.Error, match="unrelated"):
        qa.prepare(env=env, provision=True, transport=server)
    assert all(m == "GET" for m, _, _ in server.calls)


@pytest.mark.parametrize("patch", [
    {"revoked_at": "2026-09-16"}, {"role": "super_admin"},
    {"club_id": "tres_palapas"}, {"user_id": ACTOR}, {"expires_at": "2026-12-01"},
])
def test_refuses_changed_or_revoked_assignments_even_during_provision(env, patch):
    server = Server()
    server.roles[0].update(patch)
    with pytest.raises(qa.Error, match="assignments have changed"):
        qa.prepare(env=env, provision=True, transport=server)
    assert all(m == "GET" for m, _, _ in server.calls)


@pytest.mark.parametrize("field", ["bad_identity", "bad_workspaces"])
def test_ends_new_session_when_identity_or_live_scope_is_wrong(env, field):
    server = Server()
    setattr(server, field, True)
    with pytest.raises(qa.Error):
        qa.prepare(env=env, transport=server)
    assert any(path == "/auth/v1/logout" for _, path, _ in server.calls)
