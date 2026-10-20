"""Exercise real JWT verification/key caching with an in-memory HTTP boundary."""
import io
import json
import time
import urllib.error
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException

from services.api import auth

URL = "https://configured-project.supabase.co/auth/v1/.well-known/jwks.json"


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    auth._jwks_client.cache_clear()
    monkeypatch.setenv("JUPR_SUPABASE_JWT_MODE", "jwks")
    monkeypatch.setenv("SUPABASE_JWKS_URL", URL)
    monkeypatch.setenv("SUPABASE_JWT_AUDIENCE", "authenticated")
    yield
    auth._jwks_client.cache_clear()


def signing_key(kid):
    private = ec.generate_private_key(ec.SECP256R1())
    public = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(private.public_key()))
    return private, {**public, "kid": kid, "use": "sig", "alg": "ES256"}


def token(private, kid, **claims):
    payload = {
        "sub": "user-1", "email": "admin@example.test",
        "exp": int(time.time()) + 3600, "aud": "authenticated", **claims,
    }
    return jwt.encode(payload, private, algorithm="ES256", headers={"kid": kid})


def key_endpoint(monkeypatch, keys):
    state = {"keys": keys, "down": False, "requests": []}

    def open_url(request, **kwargs):
        state["requests"].append((request.full_url, kwargs.get("timeout")))
        if state["down"]:
            raise urllib.error.URLError("simulated unavailable key endpoint")
        return io.BytesIO(json.dumps({"keys": state["keys"]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    monkeypatch.setattr(
        "urllib.request.build_opener", lambda *_handlers: SimpleNamespace(open=open_url)
    )
    return state


def authenticate(value):
    return auth.authenticate_bearer("Bearer " + value)


def test_verified_requests_reuse_keys_but_never_cache_claim_validation(monkeypatch):
    private, public = signing_key("first")
    endpoint = key_endpoint(monkeypatch, [public])
    assert authenticate(token(private, "first")).user_id == "user-1"
    assert authenticate(token(private, "first", sub="user-2")).user_id == "user-2"
    for invalid in [
        token(private, "first", exp=int(time.time()) - 1),
        token(private, "first", aud="another-audience"),
    ]:
        with pytest.raises(HTTPException) as err:
            authenticate(invalid)
        assert err.value.status_code == 401
    assert endpoint["requests"] == [(URL, 5)]


def test_rotation_refreshes_keys_without_a_new_client(monkeypatch):
    first, first_public = signing_key("first")
    second, second_public = signing_key("second")
    endpoint = key_endpoint(monkeypatch, [first_public])
    authenticate(token(first, "first"))
    endpoint["keys"] = [second_public]
    # Preserve the library's unknown-key cooldown on versions that implement it.
    advanced = time.monotonic() + 31
    monkeypatch.setattr("jwt.jwk_set_cache.time.monotonic", lambda: advanced)
    assert authenticate(token(second, "second")).user_id == "user-1"
    assert len(endpoint["requests"]) == 2


def test_cached_keys_survive_short_outage_but_expired_cache_fails_closed(monkeypatch):
    private, public = signing_key("first")
    endpoint = key_endpoint(monkeypatch, [public])
    signed = token(private, "first")
    authenticate(signed)
    endpoint["down"] = True
    assert authenticate(signed).user_id == "user-1"
    assert len(endpoint["requests"]) == 1
    advanced = time.monotonic() + 301
    monkeypatch.setattr("jwt.jwk_set_cache.time.monotonic", lambda: advanced)
    with pytest.raises(HTTPException) as err:
        authenticate(signed)
    assert err.value.status_code == 401
    assert len(endpoint["requests"]) == 2


def test_configuration_change_does_not_reuse_a_different_projects_keys(monkeypatch):
    first, first_public = signing_key("shared-id")
    second, second_public = signing_key("shared-id")
    endpoint = key_endpoint(monkeypatch, [first_public])
    authenticate(token(first, "shared-id"))
    other_url = "https://other-configured-project.supabase.co/auth/v1/.well-known/jwks.json"
    endpoint["keys"] = [second_public]
    monkeypatch.setenv("SUPABASE_JWKS_URL", other_url)
    assert authenticate(token(second, "shared-id")).user_id == "user-1"
    with pytest.raises(HTTPException) as err:
        authenticate(token(first, "shared-id"))
    assert err.value.status_code == 401
    assert endpoint["requests"] == [(URL, 5), (other_url, 5)]
