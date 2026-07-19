from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class SMTPConfig:
    host: str
    port: int
    username: str
    password: str
    from_email: str
    from_name: str
    reply_to: str
    use_tls: bool


EMAIL_MODE_LIVE = "live"
EMAIL_MODE_DRY_RUN = "dry_run"
EMAIL_MODE_STAGING_REDIRECT = "staging_redirect"
SUPPORTED_EMAIL_MODES = {EMAIL_MODE_LIVE, EMAIL_MODE_DRY_RUN, EMAIL_MODE_STAGING_REDIRECT}


def _streamlit_secret_value(*path: str) -> str:
    try:
        import streamlit as st
    except Exception:
        return ""

    try:
        current: object = st.secrets
    except Exception:
        return ""

    for key in path:
        if not isinstance(current, Mapping):
            return ""
        try:
            if key not in current:
                return ""
            current = current[key]
        except Exception:
            return ""
    return str(current or "").strip()


def get_env_or_default(name: str, default: str = "") -> str:
    value = str(os.getenv(name, "")).strip()
    if value:
        return value
    secret_value = _streamlit_secret_value(name)
    if secret_value:
        return secret_value
    return str(default).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    value = get_env_or_default(name).lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on"}


def get_public_base_url(default: str = "http://localhost:8501") -> str:
    for env_name in ("JUPR_PUBLIC_BASE_URL", "PUBLIC_BASE_URL"):
        value = get_env_or_default(env_name)
        if value:
            return value.rstrip("/")
    return str(default).strip().rstrip("/")


def get_smtp_config() -> SMTPConfig:
    host = get_env_or_default("SMTP_HOST")
    port_raw = get_env_or_default("SMTP_PORT")
    username = get_env_or_default("SMTP_USERNAME")
    password = get_env_or_default("SMTP_PASSWORD")
    from_email = get_env_or_default("SMTP_FROM_EMAIL")
    from_name = get_env_or_default("SMTP_FROM_NAME", "JUPR Notifications")
    reply_to = get_env_or_default("SMTP_REPLY_TO", "joe@juprleagues.com")
    use_tls = _env_bool("SMTP_USE_TLS", default=True)

    missing = [
        key
        for key, value in [
            ("SMTP_HOST", host),
            ("SMTP_PORT", port_raw),
            ("SMTP_USERNAME", username),
            ("SMTP_PASSWORD", password),
            ("SMTP_FROM_EMAIL", from_email),
        ]
        if not value
    ]
    if missing:
        raise ValueError(f"Missing SMTP configuration: {', '.join(missing)}")

    try:
        port = int(port_raw)
    except Exception as exc:
        raise ValueError("SMTP_PORT must be an integer") from exc

    return SMTPConfig(
        host=host,
        port=port,
        username=username,
        password=password,
        from_email=from_email,
        from_name=from_name,
        reply_to=reply_to,
        use_tls=use_tls,
    )


def get_jupr_env() -> str:
    return get_env_or_default("JUPR_ENV", "production").lower()


def get_email_mode() -> str:
    env = get_jupr_env()
    configured = get_env_or_default("JUPR_EMAIL_MODE").lower()
    if configured:
        if configured not in SUPPORTED_EMAIL_MODES:
            raise ValueError(f"Invalid JUPR_EMAIL_MODE: {configured}")
        if (
            env == "staging"
            and configured == EMAIL_MODE_LIVE
            and get_env_or_default("JUPR_ALLOW_STAGING_LIVE_EMAIL") != "1"
        ):
            raise ValueError("Staging live email blocked. Set JUPR_ALLOW_STAGING_LIVE_EMAIL=1 to allow live sends.")
        return configured
    if env == "staging":
        return EMAIL_MODE_DRY_RUN
    return EMAIL_MODE_LIVE


def _registration_edit_secret_fallback() -> str:
    for candidate in (
        get_env_or_default("SUPABASE_SERVICE_ROLE_KEY"),
        _streamlit_secret_value("supabase", "service_role_key"),
        get_env_or_default("SUPABASE_ANON_KEY"),
        _streamlit_secret_value("supabase", "anon_key"),
        _streamlit_secret_value("supabase", "key"),
    ):
        if candidate:
            return f"registration-edit-token:{candidate}"
    return ""


def get_explicit_registration_edit_token_secret() -> str:
    """Return only an operator-managed, rotation-stable edit-token secret."""

    for candidate in (
        get_env_or_default("JUPR_REGISTRATION_EDIT_SECRET"),
        _streamlit_secret_value("registration", "edit_token_secret"),
        _streamlit_secret_value("registration", "edit_secret"),
        _streamlit_secret_value("jupr", "registration_edit_secret"),
    ):
        if candidate:
            if len(candidate.encode("utf-8")) < 32:
                raise ValueError(
                    "JUPR_REGISTRATION_EDIT_SECRET must contain at least 32 bytes "
                    "of operator-managed secret material."
                )
            return candidate
    raise ValueError(
        "Public registration edit links require an explicit, stable "
        "JUPR_REGISTRATION_EDIT_SECRET. Supabase credential fallbacks are not "
        "accepted by the public edit API because key rotation would invalidate "
        "outstanding links."
    )


def get_registration_edit_token_secret() -> str:
    for candidate in (
        get_env_or_default("JUPR_REGISTRATION_EDIT_SECRET"),
        _streamlit_secret_value("registration", "edit_token_secret"),
        _streamlit_secret_value("registration", "edit_secret"),
        _streamlit_secret_value("jupr", "registration_edit_secret"),
        _registration_edit_secret_fallback(),
    ):
        if candidate:
            return candidate
    raise ValueError(
        "JUPR_REGISTRATION_EDIT_SECRET is required for registration edit links. "
        "Set it directly, or configure Supabase credentials so the app can derive a stable signing secret."
    )


def get_registration_confirmation_token_secret() -> str:
    """Return a server-only signing secret for confirmation access tokens.

    Confirmation tokens protect a registrant-specific projection. Never derive
    them from the publishable/anonymous key because that key is intentionally
    delivered to browsers.
    """

    for candidate in (
        get_env_or_default("JUPR_REGISTRATION_CONFIRMATION_SECRET"),
        _streamlit_secret_value("registration", "confirmation_token_secret"),
        get_env_or_default("JUPR_REGISTRATION_EDIT_SECRET"),
        _streamlit_secret_value("registration", "edit_token_secret"),
    ):
        if candidate:
            return candidate

    for candidate in (
        get_env_or_default("SUPABASE_SERVICE_ROLE_KEY"),
        _streamlit_secret_value("supabase", "service_role_key"),
    ):
        if candidate:
            return f"registration-confirmation-token:{candidate}"

    raise ValueError(
        "JUPR_REGISTRATION_CONFIRMATION_SECRET is required for registration "
        "confirmation links. Set it directly, reuse the server-only registration "
        "edit secret, or configure the Supabase service-role credential."
    )
