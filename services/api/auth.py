from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse

import jwt
from fastapi import Header, HTTPException
from jwt import InvalidTokenError


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    email: str
    claims: dict[str, Any]


JWKS_ALGORITHMS = ("RS256", "RS384", "RS512", "ES256", "ES384", "ES512", "EdDSA")
SECRET_ALGORITHMS = ("HS256",)


def _unauthorized(detail: str = "invalid bearer token") -> HTTPException:
    return HTTPException(status_code=401, detail=detail)


def parse_bearer_token(authorization: str | None) -> str:
    value = str(authorization or "").strip()
    if not value:
        raise _unauthorized("missing bearer token")
    scheme, _, token = value.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise _unauthorized("malformed bearer token")
    return token.strip()


def _unverified_alg(token: str) -> str:
    try:
        return str(jwt.get_unverified_header(token).get("alg") or "").strip()
    except Exception:
        return ""


def _decode_with_secret(token: str, *, secret: str, audience: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "algorithms": list(SECRET_ALGORITHMS),
        "options": {"require": ["exp", "sub"]},
    }
    if audience:
        kwargs["audience"] = audience
    return jwt.decode(token, secret, **kwargs)


def _decode_with_jwks(token: str, *, jwks_url: str, audience: str | None) -> dict[str, Any]:
    signing_key = jwt.PyJWKClient(jwks_url).get_signing_key_from_jwt(token)
    kwargs: dict[str, Any] = {"algorithms": list(JWKS_ALGORITHMS), "options": {"require": ["exp", "sub"]}}
    if audience:
        kwargs["audience"] = audience
    return jwt.decode(token, signing_key.key, **kwargs)


def _clean_base_url(value: str | None) -> str:
    return str(value or "").strip().rstrip("/")


def _configured_supabase_url() -> str:
    for name in ("SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_URL", "JUPR_SUPABASE_URL"):
        value = _clean_base_url(os.getenv(name))
        if value:
            return value
    return ""


def get_supabase_jwks_url() -> str:
    explicit = _clean_base_url(os.getenv("SUPABASE_JWKS_URL"))
    if explicit:
        return explicit
    supabase_url = _configured_supabase_url()
    if not supabase_url:
        return ""
    return f"{supabase_url}/auth/v1/.well-known/jwks.json"


def _configured_jwt_inputs() -> tuple[str, str, str]:
    mode = os.getenv("JUPR_SUPABASE_JWT_MODE", "").strip().lower()
    secret = os.getenv("SUPABASE_JWT_SECRET", "").strip()
    jwks_url = get_supabase_jwks_url()
    return mode, secret, jwks_url


def jwt_verification_mode() -> str:
    mode, secret, jwks_url = _configured_jwt_inputs()
    if mode == "secret" and secret:
        return "secret"
    if mode == "jwks" and jwks_url:
        return "jwks"
    if mode in {"", "auto"}:
        if secret and jwks_url:
            return "auto"
        if secret:
            return "secret"
        if jwks_url:
            return "jwks"
    return "unconfigured"


def jwt_verification_configured() -> bool:
    return jwt_verification_mode() != "unconfigured"


def jwt_verification_project_ref() -> str | None:
    """Return the public Supabase project ref used for asymmetric JWT checks."""

    if jwt_verification_mode() not in {"jwks", "auto"}:
        return None
    host = (urlparse(get_supabase_jwks_url()).hostname or "").strip().lower()
    if not host.endswith(".supabase.co"):
        return None
    return host.split(".", 1)[0] or None


def get_token_decoder() -> Callable[[str], dict[str, Any]]:
    mode, secret, jwks_url = _configured_jwt_inputs()
    audience = os.getenv("SUPABASE_JWT_AUDIENCE", "authenticated").strip() or None

    if mode == "secret":
        if not secret:
            raise RuntimeError("Supabase JWT secret mode is enabled, but SUPABASE_JWT_SECRET is not set.")
        return lambda token: _decode_with_secret(token, secret=secret, audience=audience)

    if mode == "jwks":
        if not jwks_url:
            raise RuntimeError("Supabase JWT JWKS mode is enabled, but SUPABASE_JWKS_URL or SUPABASE_URL is not set.")
        return lambda token: _decode_with_jwks(token, jwks_url=jwks_url, audience=audience)

    if mode not in {"", "auto"}:
        raise RuntimeError(f"Unsupported JUPR_SUPABASE_JWT_MODE={mode!r}. Use auto, secret, or jwks.")

    def decode_auto(token: str) -> dict[str, Any]:
        alg = _unverified_alg(token)
        if alg in SECRET_ALGORITHMS:
            if not secret:
                raise RuntimeError("Supabase JWT uses HS256, but SUPABASE_JWT_SECRET is not set.")
            return _decode_with_secret(token, secret=secret, audience=audience)
        if alg in JWKS_ALGORITHMS:
            if not jwks_url:
                raise RuntimeError("Supabase JWT uses an asymmetric algorithm, but SUPABASE_JWKS_URL or SUPABASE_URL is not set.")
            return _decode_with_jwks(token, jwks_url=jwks_url, audience=audience)
        if secret and not jwks_url:
            return _decode_with_secret(token, secret=secret, audience=audience)
        if jwks_url:
            return _decode_with_jwks(token, jwks_url=jwks_url, audience=audience)
        raise RuntimeError(
            "Supabase JWT verification is not configured. Set SUPABASE_JWT_SECRET for HS256 tokens "
            "or SUPABASE_JWKS_URL/SUPABASE_URL for asymmetric Supabase tokens."
        )

    return decode_auto


def authenticate_bearer(
    authorization: str | None,
    *,
    decode_token: Callable[[str], dict[str, Any]] | None = None,
) -> AuthenticatedUser:
    token = parse_bearer_token(authorization)
    try:
        decoder = decode_token or get_token_decoder()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        claims = decoder(token)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InvalidTokenError:
        raise _unauthorized("invalid bearer token")
    except Exception:
        raise _unauthorized("invalid bearer token")

    user_id = str(claims.get("sub") or "").strip()
    email = str(claims.get("email") or "").strip().lower()
    if not user_id or not email:
        raise _unauthorized("invalid bearer token")
    return AuthenticatedUser(user_id=user_id, email=email, claims=claims)


def auth_header(authorization: str | None = Header(default=None)) -> str | None:
    return authorization
