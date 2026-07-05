from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import jwt
from fastapi import Header, HTTPException
from jwt import InvalidTokenError


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    email: str
    claims: dict[str, Any]


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


def _decode_with_secret(token: str, *, secret: str, audience: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "algorithms": ["HS256"],
        "options": {"require": ["exp", "sub"]},
    }
    if audience:
        kwargs["audience"] = audience
    return jwt.decode(token, secret, **kwargs)


def _decode_with_jwks(token: str, *, jwks_url: str, audience: str | None) -> dict[str, Any]:
    signing_key = jwt.PyJWKClient(jwks_url).get_signing_key_from_jwt(token)
    kwargs: dict[str, Any] = {"algorithms": ["RS256"], "options": {"require": ["exp", "sub"]}}
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


def jwt_verification_mode() -> str:
    mode = os.getenv("JUPR_SUPABASE_JWT_MODE", "").strip().lower()
    secret = os.getenv("SUPABASE_JWT_SECRET", "").strip()
    jwks_url = get_supabase_jwks_url()
    if mode in {"", "secret"} and secret:
        return "secret"
    if mode in {"jwks", ""} and jwks_url:
        return "jwks"
    return "unconfigured"


def jwt_verification_configured() -> bool:
    return jwt_verification_mode() != "unconfigured"


def get_token_decoder() -> Callable[[str], dict[str, Any]]:
    mode = os.getenv("JUPR_SUPABASE_JWT_MODE", "").strip().lower()
    secret = os.getenv("SUPABASE_JWT_SECRET", "").strip()
    audience = os.getenv("SUPABASE_JWT_AUDIENCE", "authenticated").strip() or None
    jwks_url = get_supabase_jwks_url()

    if mode in {"", "secret"} and secret:
        return lambda token: _decode_with_secret(token, secret=secret, audience=audience)
    if mode in {"jwks", ""} and jwks_url:
        return lambda token: _decode_with_jwks(token, jwks_url=jwks_url, audience=audience)

    raise RuntimeError(
        "Supabase JWT verification is not configured. Set SUPABASE_JWT_SECRET for secret mode "
        "or SUPABASE_JWKS_URL with JUPR_SUPABASE_JWT_MODE=jwks. If SUPABASE_URL is set, "
        "JWKS mode can be derived automatically for projects that use asymmetric signing keys."
    )


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
