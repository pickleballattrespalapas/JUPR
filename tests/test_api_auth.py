import base64
import json

from services.api import auth


def _token_with_alg(alg: str) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": alg, "kid": "test"}).encode()).rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(json.dumps({"sub": "user-1", "email": "ADMIN@example.com", "exp": 4_102_444_800}).encode()).rstrip(b"=").decode()
    signature = base64.urlsafe_b64encode(b"signature").rstrip(b"=").decode()
    return f"{header}.{body}.{signature}"


def test_auto_decoder_uses_jwks_for_asymmetric_token_even_when_secret_exists(monkeypatch):
    token = _token_with_alg("ES256")
    calls = {"secret": 0, "jwks": 0}
    monkeypatch.delenv("JUPR_SUPABASE_JWT_MODE", raising=False)
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "legacy-secret")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")

    def fake_secret(*_args, **_kwargs):
        calls["secret"] += 1
        raise AssertionError("secret decoder should not be used for ES256")

    def fake_jwks(*_args, **_kwargs):
        calls["jwks"] += 1
        return {"sub": "user-1", "email": "ADMIN@example.com", "exp": 4_102_444_800}

    monkeypatch.setattr(auth, "_decode_with_secret", fake_secret)
    monkeypatch.setattr(auth, "_decode_with_jwks", fake_jwks)

    user = auth.authenticate_bearer(f"Bearer {token}")

    assert user.email == "admin@example.com"
    assert calls == {"secret": 0, "jwks": 1}


def test_auto_decoder_uses_secret_for_hs256_token(monkeypatch):
    token = _token_with_alg("HS256")
    calls = {"secret": 0, "jwks": 0}
    monkeypatch.delenv("JUPR_SUPABASE_JWT_MODE", raising=False)
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "legacy-secret")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")

    def fake_secret(*_args, **_kwargs):
        calls["secret"] += 1
        return {"sub": "user-1", "email": "ADMIN@example.com", "exp": 4_102_444_800}

    def fake_jwks(*_args, **_kwargs):
        calls["jwks"] += 1
        raise AssertionError("jwks decoder should not be used for HS256")

    monkeypatch.setattr(auth, "_decode_with_secret", fake_secret)
    monkeypatch.setattr(auth, "_decode_with_jwks", fake_jwks)

    user = auth.authenticate_bearer(f"Bearer {token}")

    assert user.email == "admin@example.com"
    assert calls == {"secret": 1, "jwks": 0}


def test_auto_mode_reports_configured_when_secret_and_jwks_available(monkeypatch):
    monkeypatch.delenv("JUPR_SUPABASE_JWT_MODE", raising=False)
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "legacy-secret")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")

    assert auth.jwt_verification_mode() == "auto"
    assert auth.jwt_verification_configured() is True
